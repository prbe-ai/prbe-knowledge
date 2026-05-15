"""Manual entity-cluster merging — internal API.

Mounted into the ingestion service alongside admin_routes.py. Gated by
X-Internal-Knowledge-Key (caller is prbe-backend BFF after JWT validation).

Endpoints:

  POST   /api/entity-clusters/merge
         Run the full merge transaction: validate → lock → snapshot →
         rewrite edges → drop self-loops → merge provenance → delete
         alias nodes → recompute degree → INSERT routing + audit.

  DELETE /api/entity-clusters/{label}/{primary}/aliases/{alias}
         Run the unmerge transaction: re-INSERT alias node from snapshot,
         UPDATE edges via aliased_from/to back to alias, re-INSERT
         snapshotted self-loops, recompute degree, drop routing row,
         flip audit status if last alias under merge_id.

Design + plan:
  docs/superpowers/specs/2026-05-13-entity-clusters-design.md
  docs/superpowers/specs/2026-05-13-entity-clusters-phase1-plan.md
"""

from __future__ import annotations

import hmac
import logging
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.config import get_settings
from shared.db import with_tenant

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/entity-clusters", tags=["internal-api"])


# ---------------------------------------------------------------------------
# Auth: X-Internal-Knowledge-Key (same gate as admin_routes.py)
# ---------------------------------------------------------------------------


def _require_internal_key(
    x_internal_knowledge_key: str | None = Header(default=None),
) -> None:
    expected = get_settings().internal_knowledge_api_key
    if expected is None or not expected.get_secret_value():
        raise HTTPException(
            status_code=503,
            detail="disabled — set INTERNAL_KNOWLEDGE_API_KEY",
        )
    if not x_internal_knowledge_key or not hmac.compare_digest(
        x_internal_knowledge_key, expected.get_secret_value()
    ):
        raise HTTPException(status_code=401, detail="invalid internal key")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MergeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id:          str            = Field(..., min_length=1, max_length=128)
    performed_by_user_id: uuid.UUID
    label:                str            = Field(..., min_length=1, max_length=64)
    primary_canonical_id: str            = Field(..., min_length=1, max_length=512)
    alias_canonical_ids:  list[str]      = Field(..., min_length=1, max_length=64)
    reason:               str | None     = Field(default=None, max_length=2000)

    @field_validator("alias_canonical_ids")
    @classmethod
    def _unique_non_blank(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("alias_canonical_ids must be unique")
        for a in v:
            if not a or not a.strip():
                raise ValueError("alias_canonical_ids may not contain blanks")
        return v


class MergeResponse(BaseModel):
    merge_id:                   uuid.UUID
    label:                      str
    primary_canonical_id:       str
    merged_alias_canonical_ids: list[str]


# ---------------------------------------------------------------------------
# POST /api/entity-clusters/merge
# ---------------------------------------------------------------------------


@router.post(
    "/merge",
    response_model=MergeResponse,
    dependencies=[Depends(_require_internal_key)],
)
async def merge_cluster(body: MergeRequest) -> MergeResponse:
    """Run the merge transaction described in the design doc."""
    customer_id = body.customer_id

    if body.primary_canonical_id in body.alias_canonical_ids:
        raise HTTPException(
            status_code=400,
            detail="primary_canonical_id must not appear in alias_canonical_ids",
        )

    all_ids = [body.primary_canonical_id, *body.alias_canonical_ids]
    merge_id = uuid.uuid4()

    async with with_tenant(customer_id) as conn:
        # 1. Existence check.
        existing_rows = await conn.fetch(
            """
            SELECT canonical_id, node_id FROM graph_nodes
            WHERE customer_id = $1 AND label = $2 AND canonical_id = ANY($3::text[])
            """,
            customer_id, body.label, all_ids,
        )
        existing = {r["canonical_id"]: r["node_id"] for r in existing_rows}
        missing = [c for c in all_ids if c not in existing]
        if missing:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "unknown canonical_ids for label",
                    "label": body.label,
                    "missing": missing,
                },
            )
        primary_node_id = existing[body.primary_canonical_id]
        alias_node_ids = [existing[a] for a in body.alias_canonical_ids]

        # 2. None of the aliases are already in another cluster.
        already = await conn.fetch(
            """
            SELECT alias_canonical_id, primary_canonical_id FROM entity_aliases
            WHERE customer_id = $1 AND label = $2
              AND alias_canonical_id = ANY($3::text[])
            """,
            customer_id, body.label, body.alias_canonical_ids,
        )
        if already:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "one or more aliases already belong to a cluster",
                    "conflicting_aliases": {
                        r["alias_canonical_id"]: r["primary_canonical_id"]
                        for r in already
                    },
                },
            )

        # 3. Primary not itself an alias.
        primary_as_alias = await conn.fetchrow(
            """
            SELECT primary_canonical_id FROM entity_aliases
            WHERE customer_id = $1 AND label = $2 AND alias_canonical_id = $3
            LIMIT 1
            """,
            customer_id, body.label, body.primary_canonical_id,
        )
        if primary_as_alias is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "primary_canonical_id is already an alias of another cluster",
                    "actual_primary": primary_as_alias["primary_canonical_id"],
                },
            )

        # 4. Lock every edge touching any alias node.
        await conn.execute(
            """
            SELECT edge_id FROM graph_edges
            WHERE customer_id = $1
              AND (from_node_id = ANY($2::bigint[]) OR to_node_id = ANY($2::bigint[]))
            FOR UPDATE
            """,
            customer_id, alias_node_ids,
        )

        # 5. Insert audit row.
        await conn.execute(
            """
            INSERT INTO entity_merge_audit
              (merge_id, customer_id, label, primary_canonical_id,
               merged_alias_canonical_ids, performed_by_user_id, reason)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            merge_id, customer_id, body.label, body.primary_canonical_id,
            body.alias_canonical_ids, body.performed_by_user_id, body.reason,
        )

        # 6. Snapshot alias nodes (incl. inlined provenance).
        await conn.execute(
            """
            INSERT INTO entity_merge_node_snapshot
              (merge_id, customer_id, label, canonical_id, properties,
               degree, community_id, created_at, provenance)
            SELECT $1, gn.customer_id, gn.label, gn.canonical_id,
                   gn.properties, gn.degree, gn.community_id, gn.created_at,
                   COALESCE(
                     (SELECT jsonb_agg(jsonb_build_object(
                        'source_system', p.source_system,
                        'first_seen_at', p.first_seen_at,
                        'last_seen_at',  p.last_seen_at))
                      FROM graph_node_provenance p
                      WHERE p.node_id = gn.node_id),
                     '[]'::jsonb
                   )
              FROM graph_nodes gn
             WHERE gn.customer_id = $2 AND gn.node_id = ANY($3::bigint[])
            """,
            merge_id, customer_id, alias_node_ids,
        )

        # 7. Merge alias provenance into canonical.
        await conn.execute(
            """
            INSERT INTO graph_node_provenance
              (node_id, customer_id, source_system, first_seen_at, last_seen_at)
            SELECT $1, $2, p.source_system,
                   MIN(p.first_seen_at), MAX(p.last_seen_at)
              FROM graph_node_provenance p
             WHERE p.node_id = ANY($3::bigint[]) AND p.customer_id = $2
             GROUP BY p.source_system
            ON CONFLICT (node_id, source_system) DO UPDATE
              SET first_seen_at = LEAST(graph_node_provenance.first_seen_at, EXCLUDED.first_seen_at),
                  last_seen_at  = GREATEST(graph_node_provenance.last_seen_at, EXCLUDED.last_seen_at)
            """,
            primary_node_id, customer_id, alias_node_ids,
        )

        # 8. Classify each touched edge and apply.
        #    First fetch them so we can classify in Python.
        edges_to_rewrite = await conn.fetch(
            """
            SELECT edge_id, edge_type, from_node_id, to_node_id,
                   properties, confidence, valid_from, valid_to,
                   source_system, extractor_id, extracted_at,
                   aliased_from_canonical_id, aliased_to_canonical_id
            FROM graph_edges
            WHERE customer_id = $1
              AND (from_node_id = ANY($2::bigint[]) OR to_node_id = ANY($2::bigint[]))
            """,
            customer_id, alias_node_ids,
        )
        node_id_to_canonical = {existing[c]: c for c in body.alias_canonical_ids}
        snapshot_seq = 0
        for e in edges_to_rewrite:
            from_aliased = e["from_node_id"] in alias_node_ids
            to_aliased   = e["to_node_id"]   in alias_node_ids
            new_from = primary_node_id if from_aliased else e["from_node_id"]
            new_to   = primary_node_id if to_aliased   else e["to_node_id"]
            new_aliased_from = (
                node_id_to_canonical[e["from_node_id"]]
                if from_aliased else e["aliased_from_canonical_id"]
            )
            new_aliased_to = (
                node_id_to_canonical[e["to_node_id"]]
                if to_aliased else e["aliased_to_canonical_id"]
            )
            # Self-loop after rewrite → snapshot + DELETE.
            if new_from == new_to:
                snapshot_seq += 1
                await conn.execute(
                    """
                    INSERT INTO entity_merge_edge_snapshot
                      (merge_id, snapshot_seq, customer_id, operation,
                       pre_edge_type,
                       pre_from_canonical_id, pre_from_label,
                       pre_to_canonical_id,   pre_to_label,
                       pre_properties, pre_confidence,
                       pre_valid_from, pre_valid_to,
                       pre_source_system, pre_extractor_id, pre_extracted_at,
                       pre_aliased_from_canonical_id, pre_aliased_to_canonical_id)
                    SELECT $1, $2, $3, 'deleted_self_loop',
                           $4,
                           gn_from.canonical_id, gn_from.label,
                           gn_to.canonical_id,   gn_to.label,
                           $5::jsonb, $6, $7, $8, $9, $10, $11, $12, $13
                      FROM graph_nodes gn_from, graph_nodes gn_to
                     WHERE gn_from.node_id = $14 AND gn_to.node_id = $15
                    """,
                    merge_id, snapshot_seq, customer_id,
                    e["edge_type"],
                    e["properties"], e["confidence"],
                    e["valid_from"], e["valid_to"],
                    e["source_system"], e["extractor_id"], e["extracted_at"],
                    e["aliased_from_canonical_id"], e["aliased_to_canonical_id"],
                    e["from_node_id"], e["to_node_id"],
                )
                await conn.execute(
                    "DELETE FROM graph_edges WHERE edge_id = $1", e["edge_id"]
                )
                continue
            # Clean rewrite: UPDATE endpoints + stamp aliased_from/to.
            await conn.execute(
                """
                UPDATE graph_edges
                   SET from_node_id = $1,
                       to_node_id   = $2,
                       aliased_from_canonical_id = $3,
                       aliased_to_canonical_id   = $4
                 WHERE edge_id = $5
                """,
                new_from, new_to, new_aliased_from, new_aliased_to,
                e["edge_id"],
            )

        # 9. Hard-delete alias graph_nodes (CASCADE drops their remaining
        #    provenance rows — already merged into canonical at step 7).
        await conn.execute(
            "DELETE FROM graph_nodes "
            "WHERE customer_id = $1 AND node_id = ANY($2::bigint[])",
            customer_id, alias_node_ids,
        )

        # 10. Recompute degree on canonical.
        await conn.execute(
            """
            UPDATE graph_nodes
               SET degree = (
                   SELECT COUNT(*) FROM graph_edges
                    WHERE customer_id = $1
                      AND (from_node_id = $2 OR to_node_id = $2)
               )
             WHERE customer_id = $1 AND node_id = $2
            """,
            customer_id, primary_node_id,
        )

        # 11. INSERT routing rows.
        await conn.execute(
            """
            INSERT INTO entity_aliases
              (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
            SELECT $1, $2, alias, $3, $4
              FROM UNNEST($5::text[]) AS alias
            """,
            customer_id, body.label, body.primary_canonical_id, merge_id,
            body.alias_canonical_ids,
        )

    log.info(
        "entity_clusters.merge: customer=%s label=%s primary=%s aliases=%d merge_id=%s",
        customer_id, body.label, body.primary_canonical_id,
        len(body.alias_canonical_ids), merge_id,
    )
    return MergeResponse(
        merge_id=merge_id,
        label=body.label,
        primary_canonical_id=body.primary_canonical_id,
        merged_alias_canonical_ids=list(body.alias_canonical_ids),
    )


# ---------------------------------------------------------------------------
# DELETE /api/entity-clusters/{label}/{primary}/aliases/{alias}
# ---------------------------------------------------------------------------


@router.delete(
    "/{label}/{primary_canonical_id}/aliases/{alias_canonical_id}",
    status_code=204,
    dependencies=[Depends(_require_internal_key)],
)
async def unmerge_alias(
    label: str = Path(..., min_length=1, max_length=64),
    primary_canonical_id: str = Path(..., min_length=1, max_length=512),
    alias_canonical_id: str = Path(..., min_length=1, max_length=512),
    x_prbe_customer: str | None = Header(default=None, alias="X-Prbe-Customer"),
) -> None:
    """Remove a single alias from its cluster (inverse of merge)."""
    if not x_prbe_customer:
        raise HTTPException(
            status_code=400,
            detail="X-Prbe-Customer header is required",
        )
    customer_id = x_prbe_customer

    async with with_tenant(customer_id) as conn:
        # 1. Look up the merge_id; 404 if no routing row.
        existing = await conn.fetchrow(
            """
            SELECT merge_id FROM entity_aliases
            WHERE customer_id = $1 AND label = $2
              AND alias_canonical_id = $3 AND primary_canonical_id = $4
            """,
            customer_id, label, alias_canonical_id, primary_canonical_id,
        )
        if existing is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no alias '{alias_canonical_id}' under cluster "
                    f"'{primary_canonical_id}' for label '{label}'"
                ),
            )
        merge_id = existing["merge_id"]

        # 2. Re-INSERT the alias node from snapshot. Gets a fresh node_id.
        new_alias = await conn.fetchrow(
            """
            INSERT INTO graph_nodes
              (customer_id, label, canonical_id, properties,
               degree, community_id, created_at, updated_at)
            SELECT customer_id, label, canonical_id, properties,
                   degree, community_id, created_at, NOW()
              FROM entity_merge_node_snapshot
             WHERE merge_id = $1 AND label = $2 AND canonical_id = $3
             RETURNING node_id
            """,
            merge_id, label, alias_canonical_id,
        )
        if new_alias is None:
            # Should not happen — snapshot is captured on merge.
            raise HTTPException(
                status_code=500,
                detail="missing node snapshot for alias",
            )
        new_alias_node_id = new_alias["node_id"]

        # 3. Restore alias provenance from inlined JSONB.
        await conn.execute(
            """
            INSERT INTO graph_node_provenance
              (node_id, customer_id, source_system, first_seen_at, last_seen_at)
            SELECT $1, customer_id,
                   p->>'source_system',
                   (p->>'first_seen_at')::timestamptz,
                   (p->>'last_seen_at')::timestamptz
              FROM entity_merge_node_snapshot,
                   LATERAL jsonb_array_elements(provenance) AS p
             WHERE merge_id = $2 AND label = $3 AND canonical_id = $4
            ON CONFLICT (node_id, source_system) DO NOTHING
            """,
            new_alias_node_id, merge_id, label, alias_canonical_id,
        )

        # 4. Rewrite edges back: from-side and to-side independently.
        await conn.execute(
            """
            UPDATE graph_edges
               SET from_node_id = $1,
                   aliased_from_canonical_id = NULL
             WHERE customer_id = $2 AND aliased_from_canonical_id = $3
            """,
            new_alias_node_id, customer_id, alias_canonical_id,
        )
        await conn.execute(
            """
            UPDATE graph_edges
               SET to_node_id = $1,
                   aliased_to_canonical_id = NULL
             WHERE customer_id = $2 AND aliased_to_canonical_id = $3
            """,
            new_alias_node_id, customer_id, alias_canonical_id,
        )

        # 5. Re-INSERT snapshotted self-loops involving this alias.
        await conn.execute(
            """
            INSERT INTO graph_edges
              (customer_id, edge_type, from_node_id, to_node_id,
               properties, valid_from, valid_to, source_system,
               confidence, extractor_id, extracted_at,
               aliased_from_canonical_id, aliased_to_canonical_id)
            SELECT s.customer_id, s.pre_edge_type,
                   $1, $1,  -- self-loop on the restored alias node
                   s.pre_properties, s.pre_valid_from, s.pre_valid_to,
                   s.pre_source_system, s.pre_confidence,
                   s.pre_extractor_id, s.pre_extracted_at,
                   s.pre_aliased_from_canonical_id, s.pre_aliased_to_canonical_id
              FROM entity_merge_edge_snapshot s
             WHERE s.merge_id = $2
               AND s.operation = 'deleted_self_loop'
               AND (s.pre_from_canonical_id = $3 OR s.pre_to_canonical_id = $3)
            """,
            new_alias_node_id, merge_id, alias_canonical_id,
        )

        # 6. Recompute degree on primary + restored alias.
        primary_row = await conn.fetchrow(
            """
            SELECT node_id FROM graph_nodes
            WHERE customer_id = $1 AND label = $2 AND canonical_id = $3
            """,
            customer_id, label, primary_canonical_id,
        )
        if primary_row is not None:
            await conn.execute(
                """
                UPDATE graph_nodes
                   SET degree = (
                       SELECT COUNT(*) FROM graph_edges
                        WHERE customer_id = $1
                          AND (from_node_id = graph_nodes.node_id
                               OR to_node_id = graph_nodes.node_id)
                   )
                 WHERE customer_id = $1 AND node_id IN ($2, $3)
                """,
                customer_id, primary_row["node_id"], new_alias_node_id,
            )

        # 7. Drop routing row + flip audit status if last alias remaining.
        await conn.execute(
            """
            DELETE FROM entity_aliases
            WHERE customer_id = $1 AND label = $2 AND alias_canonical_id = $3
            """,
            customer_id, label, alias_canonical_id,
        )
        await conn.execute(
            """
            UPDATE entity_merge_audit SET status = 'reversed'
            WHERE merge_id = $1
              AND NOT EXISTS (
                  SELECT 1 FROM entity_aliases WHERE merge_id = $1
              )
            """,
            merge_id,
        )

    log.info(
        "entity_clusters.unmerge: customer=%s label=%s primary=%s alias=%s merge_id=%s",
        customer_id, label, primary_canonical_id, alias_canonical_id, merge_id,
    )
