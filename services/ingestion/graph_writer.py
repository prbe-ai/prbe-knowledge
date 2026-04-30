"""Upsert graph_nodes + graph_edges from connector output.

Runs inside a with_tenant() transaction — RLS is set before this is called.
"""

from __future__ import annotations

import asyncpg
import orjson

from shared.models import GraphEdgeSpec, GraphNodeSpec


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
    return results


async def upsert_edges(
    conn: asyncpg.Connection,
    customer_id: str,
    edges: list[GraphEdgeSpec],
    node_ids: dict[tuple[str, str], int],
    source_system: str,
) -> int:
    """Upsert edges. Returns count upserted.

    Edges whose endpoints aren't in `node_ids` are silently skipped — the
    normalizer is responsible for including the full node set in the same
    NormalizationResult.

    `source_system` is recorded on initial insert and preserved on conflict
    (first asserting source wins; edges are not multi-sourced today).
    """
    if not edges:
        return 0

    inserted = 0
    for edge in edges:
        from_id = node_ids.get((edge.from_label.value, edge.from_canonical_id))
        to_id = node_ids.get((edge.to_label.value, edge.to_canonical_id))
        if from_id is None or to_id is None:
            # Endpoint missing — skip rather than insert a dangling edge.
            continue
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type, from_node_id, to_node_id,
                properties, valid_from, valid_to, source_system
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, COALESCE($6, NOW()), $7, $8)
            ON CONFLICT (customer_id, edge_type, from_node_id, to_node_id)
            DO UPDATE SET
                properties = graph_edges.properties || EXCLUDED.properties,
                valid_from = LEAST(graph_edges.valid_from, EXCLUDED.valid_from),
                valid_to   = EXCLUDED.valid_to
            """,
            customer_id,
            edge.edge_type.value,
            from_id,
            to_id,
            orjson.dumps(edge.properties).decode("utf-8"),
            edge.valid_from,
            edge.valid_to,
            source_system,
        )
        inserted += 1
    return inserted
