"""Upsert graph_nodes + graph_edges from connector output.

Runs inside a with_tenant() transaction -- RLS is set before this is called.
"""

from __future__ import annotations

from datetime import datetime

import asyncpg
import orjson

from engine.shared.constants import EdgeType, NodeLabel
from engine.shared.logging import get_logger
from engine.shared.models import GraphEdgeSpec, GraphNodeSpec

log = get_logger(__name__)


async def _fetch_aliases(
    conn: asyncpg.Connection,
    customer_id: str,
    keys: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Bulk-resolve `(label, canonical_id) → primary_canonical_id` for aliased keys.

    Returned dict only contains entries for keys that ARE aliases. Non-aliased
    keys are absent; callers should treat absence as "no rewrite needed."

    One bulk query per call regardless of input size — entity_aliases is
    typically O(100s) of rows per tenant; the (customer_id, label,
    alias_canonical_id) PK answers this with an index-only scan.
    """
    if not keys:
        return {}
    labels = [k[0] for k in keys]
    aliases = [k[1] for k in keys]
    rows = await conn.fetch(
        """
        SELECT label, alias_canonical_id, primary_canonical_id
        FROM entity_aliases
        WHERE customer_id = $1
          AND (label, alias_canonical_id) IN (
                SELECT * FROM UNNEST($2::text[], $3::text[])
              )
        """,
        customer_id, labels, aliases,
    )
    return {(r["label"], r["alias_canonical_id"]): r["primary_canonical_id"] for r in rows}


async def upsert_nodes(
    conn: asyncpg.Connection,
    customer_id: str,
    nodes: list[GraphNodeSpec],
    source_system: str,
) -> dict[tuple[str, str], int]:
    """Upsert nodes. Returns map (label, canonical_id) → node_id.

    Also records provenance: which source_system asserted this node, so
    disconnect-integration can correctly handle nodes touched by multiple
    connectors (delete only when the last source disconnects).

    One INSERT for nodes + one INSERT for provenance, regardless of input
    size. Per-node round-trips (the prior shape) serialized on row locks for
    hot canonical_ids — e.g. a Slack channel referenced by hundreds of
    backfilled messages — and pushed Phase B past the 30s
    db_statement_timeout under heavy worker fan-out, sending rows to DLQ.
    Sorting by (label, canonical_id) makes every transaction acquire row
    locks in the same order, which prevents the lock-wait staircase that
    independent orderings produce when concurrent batches overlap.
    """
    if not nodes:
        return {}

    # Alias resolution: if any inbound (label, canonical_id) is an alias of
    # a merged cluster, rewrite to the primary BEFORE the existing dedup
    # logic. Empty entity_aliases → no-op.
    alias_map = await _fetch_aliases(
        conn,
        customer_id,
        keys=[(n.label.value, n.canonical_id) for n in nodes],
    )
    if alias_map:
        for n in nodes:
            primary = alias_map.get((n.label.value, n.canonical_id))
            if primary is not None:
                n.canonical_id = primary

    # Dedupe: ON CONFLICT DO UPDATE cannot affect the same row twice in one
    # statement, so collapse repeated (label, canonical_id) entries here.
    # Property merge matches the prior loop's semantics: each repeat would
    # have run `properties = graph_nodes.properties || EXCLUDED.properties`,
    # i.e. a shallow JSONB merge with later keys winning on collision.
    deduped: dict[tuple[str, str], dict] = {}
    for n in nodes:
        key = (n.label.value, n.canonical_id)
        if key in deduped:
            deduped[key] = {**deduped[key], **n.properties}
        else:
            deduped[key] = dict(n.properties)

    sorted_keys = sorted(deduped.keys())
    labels = [k[0] for k in sorted_keys]
    canonical_ids = [k[1] for k in sorted_keys]
    properties_json = [
        orjson.dumps(deduped[k]).decode("utf-8") for k in sorted_keys
    ]

    rows = await conn.fetch(
        """
        INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, updated_at)
        SELECT $1, label, canonical_id, properties::jsonb, NOW()
        FROM unnest($2::text[], $3::text[], $4::text[])
            AS t(label, canonical_id, properties)
        ON CONFLICT (customer_id, label, canonical_id)
        DO UPDATE SET
            properties = graph_nodes.properties || EXCLUDED.properties,
            updated_at = NOW()
        RETURNING node_id, label, canonical_id
        """,
        customer_id,
        labels,
        canonical_ids,
        properties_json,
    )

    results: dict[tuple[str, str], int] = {
        (r["label"], r["canonical_id"]): r["node_id"] for r in rows
    }

    # Provenance: one INSERT for the same set of node_ids. Sorting node_ids
    # keeps lock-acquisition order stable here too.
    provenance_node_ids = sorted(results.values())
    await conn.execute(
        """
        INSERT INTO graph_node_provenance
            (node_id, customer_id, source_system, first_seen_at, last_seen_at)
        SELECT node_id, $2, $3, NOW(), NOW()
        FROM unnest($1::bigint[]) AS t(node_id)
        ON CONFLICT (node_id, source_system) DO UPDATE
            SET last_seen_at = NOW()
        """,
        provenance_node_ids,
        customer_id,
        source_system,
    )

    # Enqueue post-write processing (entity auto-merge, future analyzers).
    # Same transaction as the upsert so queue rows track committed nodes
    # exactly: a rollback drops both the node and the enqueue. ON CONFLICT
    # resets analyzer_status to '{}' on re-write so an updated node gets
    # re-analyzed against any new property values.
    await conn.execute(
        """
        INSERT INTO node_post_write_queue (customer_id, node_id, analyzer_status)
        SELECT $2, node_id, '{}'::jsonb
        FROM unnest($1::bigint[]) AS t(node_id)
        ON CONFLICT (customer_id, node_id) DO UPDATE
            SET analyzer_status = '{}'::jsonb,
                enqueued_at     = NOW(),
                locked_until    = NULL
        """,
        provenance_node_ids,
        customer_id,
    )
    return results


async def upsert_edges(
    conn: asyncpg.Connection,
    customer_id: str,
    edges: list[GraphEdgeSpec],
    node_ids: dict[tuple[str, str], int],
    source_system: str,
    extractor_id: str | None = None,
    extracted_at: datetime | None = None,
) -> int:
    """Upsert edges. Returns count upserted.

    Edges whose endpoints aren't in `node_ids` are silently skipped -- the
    normalizer is responsible for including the full node set in the same
    NormalizationResult.

    `source_system` is recorded on initial insert and preserved on conflict
    (first asserting source wins; edges are not multi-sourced today).

    `extractor_id` and `extracted_at` are optional provenance fields written
    by the inferred-edges pipeline (Lane B). Both default to NULL for all
    existing callers (back-compat). When set, they identify which prompt
    version produced these edges and when.

    One INSERT regardless of edge count -- same shape as upsert_nodes. The
    per-edge loop produced the same kind of row-lock-staircase contention
    on hot edges (every PR creates a `repo_contains_pr` edge anchored on
    the same repo node, etc.) that drove willow's DLQ flood.
    """
    if not edges:
        return 0

    # Alias resolution for both endpoints. Populate aliased_from/to on
    # the spec object so the INSERT row carries the original alias.
    endpoint_keys: list[tuple[str, str]] = []
    for e in edges:
        endpoint_keys.append((e.from_label.value, e.from_canonical_id))
        endpoint_keys.append((e.to_label.value, e.to_canonical_id))
    alias_map = await _fetch_aliases(conn, customer_id, endpoint_keys)

    if alias_map:
        kept: list[GraphEdgeSpec] = []
        for e in edges:
            from_primary = alias_map.get((e.from_label.value, e.from_canonical_id))
            to_primary = alias_map.get((e.to_label.value, e.to_canonical_id))
            if from_primary is not None:
                e.aliased_from_canonical_id = e.from_canonical_id
                e.from_canonical_id = from_primary
            if to_primary is not None:
                e.aliased_to_canonical_id = e.to_canonical_id
                e.to_canonical_id = to_primary
            # Self-loop after resolution → drop (matches Lane B Rule 5 +
            # the system-wide "no self-edges" convention).
            if (
                e.from_label == e.to_label
                and e.from_canonical_id == e.to_canonical_id
            ):
                continue
            kept.append(e)
        edges = kept

    # Resolve endpoints + dedupe. Composite UNIQUE means different alias
    # lanes (same edge_type/from/to but different aliased_from_canonical_id)
    # coexist; the dedup key must reflect that so two webhook events from
    # different alias origins don't collapse into one entry.
    # Merge semantics match the prior loop: shallow JSONB merge on
    # properties (later wins on key collision), `LEAST` on valid_from,
    # last-seen on valid_to.
    # Confidence semantics on dedupe + conflict: never demote. If the
    # batch contains both an EXTRACTED and an AMBIGUOUS assertion of the
    # same edge (one extractor resolved it, a sibling didn't), EXTRACTED
    # wins. Same rule on ON CONFLICT — an existing EXTRACTED row stays
    # EXTRACTED even if a later AMBIGUOUS write touches it.
    # Order: EXTRACTED > INFERRED > AMBIGUOUS.
    deduped: dict[tuple[str, int, int, str, str], dict] = {}
    for edge in edges:
        from_id = node_ids.get((edge.from_label.value, edge.from_canonical_id))
        to_id = node_ids.get((edge.to_label.value, edge.to_canonical_id))
        if from_id is None or to_id is None:
            continue
        aliased_from = edge.aliased_from_canonical_id or ""
        aliased_to = edge.aliased_to_canonical_id or ""
        key = (edge.edge_type.value, from_id, to_id, aliased_from, aliased_to)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = {
                "properties": dict(edge.properties),
                "valid_from": edge.valid_from,
                "valid_to": edge.valid_to,
                "confidence": edge.confidence,
                "aliased_from": edge.aliased_from_canonical_id,
                "aliased_to": edge.aliased_to_canonical_id,
            }
        else:
            existing["properties"] = {**existing["properties"], **edge.properties}
            if edge.valid_from is not None and (
                existing["valid_from"] is None
                or edge.valid_from < existing["valid_from"]
            ):
                existing["valid_from"] = edge.valid_from
            existing["valid_to"] = edge.valid_to
            existing["confidence"] = _stronger_confidence(
                existing["confidence"], edge.confidence
            )

    if not deduped:
        return 0

    sorted_keys = sorted(deduped.keys())
    edge_types = [k[0] for k in sorted_keys]
    from_ids = [k[1] for k in sorted_keys]
    to_ids = [k[2] for k in sorted_keys]
    properties_json = [
        orjson.dumps(deduped[k]["properties"]).decode("utf-8") for k in sorted_keys
    ]
    valid_from_list = [deduped[k]["valid_from"] for k in sorted_keys]
    valid_to_list = [deduped[k]["valid_to"] for k in sorted_keys]
    confidences = [deduped[k]["confidence"] for k in sorted_keys]
    aliased_from_list = [deduped[k]["aliased_from"] for k in sorted_keys]
    aliased_to_list = [deduped[k]["aliased_to"] for k in sorted_keys]

    # Use RETURNING to detect genuine INSERTs vs ON CONFLICT merges.
    # xmax = 0 on the returning row means a fresh insert (no existing tuple
    # was updated); xmax != 0 means an existing row was touched by ON CONFLICT
    # DO UPDATE. We only bump degree for new edges -- a conflict on an
    # already-existing edge must not double-count the endpoints.
    inserted_rows = await conn.fetch(
        """
        INSERT INTO graph_edges (
            customer_id, edge_type, from_node_id, to_node_id,
            properties, valid_from, valid_to, source_system, confidence,
            extractor_id, extracted_at,
            aliased_from_canonical_id, aliased_to_canonical_id
        )
        SELECT $1, edge_type, from_node_id, to_node_id,
               properties::jsonb, COALESCE(valid_from, NOW()), valid_to, $2, confidence,
               $10, $11,
               aliased_from, aliased_to
        FROM unnest(
            $3::text[], $4::bigint[], $5::bigint[],
            $6::text[], $7::timestamptz[], $8::timestamptz[], $9::text[],
            $12::text[], $13::text[]
        ) AS t(edge_type, from_node_id, to_node_id,
               properties, valid_from, valid_to, confidence,
               aliased_from, aliased_to)
        ON CONFLICT (
            customer_id, edge_type, from_node_id, to_node_id,
            COALESCE(aliased_from_canonical_id, ''),
            COALESCE(aliased_to_canonical_id, '')
        )
        DO UPDATE SET
            properties = graph_edges.properties || EXCLUDED.properties,
            valid_from = LEAST(graph_edges.valid_from, EXCLUDED.valid_from),
            valid_to   = EXCLUDED.valid_to,
            confidence = CASE
                WHEN graph_edges.confidence = 'EXTRACTED' THEN graph_edges.confidence
                WHEN EXCLUDED.confidence = 'EXTRACTED' THEN EXCLUDED.confidence
                WHEN graph_edges.confidence = 'INFERRED' THEN graph_edges.confidence
                ELSE EXCLUDED.confidence
            END,
            extractor_id  = COALESCE(EXCLUDED.extractor_id, graph_edges.extractor_id),
            extracted_at  = COALESCE(EXCLUDED.extracted_at, graph_edges.extracted_at)
        RETURNING from_node_id, to_node_id, (xmax = 0) AS inserted
        """,
        customer_id,
        source_system,
        edge_types,
        from_ids,
        to_ids,
        properties_json,
        valid_from_list,
        valid_to_list,
        confidences,
        extractor_id,
        extracted_at,
        aliased_from_list,
        aliased_to_list,
    )

    # Collect all node_ids that are endpoints of genuinely-inserted edges.
    # Each new edge increments both its from_node and to_node by 1.
    # Build a counter mapping node_id -> increment_amount so we can update
    # degree in a single UPDATE statement rather than N round-trips.
    degree_increments: dict[int, int] = {}
    for row in inserted_rows:
        if row["inserted"]:
            degree_increments[row["from_node_id"]] = (
                degree_increments.get(row["from_node_id"], 0) + 1
            )
            degree_increments[row["to_node_id"]] = (
                degree_increments.get(row["to_node_id"], 0) + 1
            )

    if degree_increments:
        inc_node_ids = list(degree_increments.keys())
        inc_amounts = [degree_increments[nid] for nid in inc_node_ids]
        await conn.execute(
            """
            UPDATE graph_nodes
            SET degree = graph_nodes.degree + delta
            FROM unnest($1::bigint[], $2::int[]) AS t(node_id, delta)
            WHERE graph_nodes.node_id = t.node_id
              AND graph_nodes.customer_id = $3
            """,
            inc_node_ids,
            inc_amounts,
            customer_id,
        )

    return len(deduped)


async def delete_edges_from_node(
    conn: asyncpg.Connection,
    customer_id: str,
    *,
    from_label: NodeLabel,
    from_canonical_id: str,
    edge_type: EdgeType,
    keep_to_node_ids: list[int] | None = None,
) -> int:
    """Delete edges of one type leaving one node, decrementing degree.

    The graph is otherwise append-only: nothing in the ingest path has ever
    removed an edge, and `GraphEdgeSpec.valid_to` is inert end-to-end (written
    by upsert_edges, set by no connector, and read by NO retriever -- so
    closing an edge would not hide it). Without a delete path, a relationship
    that MOVES leaves the old edge live forever: reparent a run to a different
    experiment and it surfaces under both, permanently, with nothing marking
    which is current.

    `keep_to_node_ids` spares endpoints that are still current, so a caller
    can re-assert its full edge set and have only the departed ones removed
    in a single statement -- no delete-then-reinsert window.

    Degree is decremented in the SAME statement via a data-modifying CTE.
    Doing it in two statements is what silently drained degree in
    kb/code_graph/cross_repo_deps.py: its DELETE decremented, its re-INSERT
    never incremented, so repo-node degree ratcheted toward the
    GREATEST(..., 0) floor on every refresh.

    Returns the number of edges deleted.
    """
    row = await conn.fetchrow(
        """
        SELECT node_id FROM graph_nodes
        WHERE customer_id = $1 AND label = $2 AND canonical_id = $3
        """,
        customer_id,
        from_label.value,
        from_canonical_id,
    )
    if row is None:
        return 0
    from_node_id = int(row["node_id"])

    deleted = await conn.fetch(
        """
        WITH deleted AS (
            DELETE FROM graph_edges
            WHERE customer_id = $1
              AND edge_type = $2
              AND from_node_id = $3
              AND NOT (to_node_id = ANY($4::bigint[]))
            RETURNING from_node_id, to_node_id
        ),
        endpoint_decs AS (
            SELECT node_id, COUNT(*) AS dec FROM (
                SELECT from_node_id AS node_id FROM deleted
                UNION ALL
                SELECT to_node_id FROM deleted
            ) e
            GROUP BY node_id
        ),
        bumped AS (
            UPDATE graph_nodes gn
            SET degree = GREATEST(gn.degree - ed.dec, 0)
            FROM endpoint_decs ed
            WHERE gn.customer_id = $1 AND gn.node_id = ed.node_id
            RETURNING 1
        )
        SELECT count(*) AS n FROM deleted
        """,
        customer_id,
        edge_type.value,
        from_node_id,
        keep_to_node_ids or [],
    )
    removed = int(deleted[0]["n"]) if deleted else 0
    if removed:
        log.info(
            "graph_writer.edges_deleted",
            customer=customer_id,
            edge_type=edge_type.value,
            from_canonical_id=from_canonical_id,
            removed=removed,
        )
    return removed


_CONFIDENCE_RANK: dict[str, int] = {"AMBIGUOUS": 0, "INFERRED": 1, "EXTRACTED": 2}


def _stronger_confidence(a: str, b: str) -> str:
    """Return the stronger of two confidence tiers (never demote)."""
    return a if _CONFIDENCE_RANK.get(a, 0) >= _CONFIDENCE_RANK.get(b, 0) else b
