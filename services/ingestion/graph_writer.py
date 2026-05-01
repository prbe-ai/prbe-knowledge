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

    One INSERT regardless of edge count — same shape as upsert_nodes. The
    per-edge loop produced the same kind of row-lock-staircase contention
    on hot edges (every PR creates a `repo_contains_pr` edge anchored on
    the same repo node, etc.) that drove willow's DLQ flood.
    """
    if not edges:
        return 0

    # Resolve endpoints + dedupe. ON CONFLICT DO UPDATE can't touch the same
    # row twice in one statement, so collapse repeated
    # (edge_type, from_node_id, to_node_id) entries here. Merge semantics
    # match the prior loop: shallow JSONB merge on properties (later wins on
    # key collision), `LEAST` on valid_from, last-seen on valid_to.
    deduped: dict[tuple[str, int, int], dict] = {}
    for edge in edges:
        from_id = node_ids.get((edge.from_label.value, edge.from_canonical_id))
        to_id = node_ids.get((edge.to_label.value, edge.to_canonical_id))
        if from_id is None or to_id is None:
            continue
        key = (edge.edge_type.value, from_id, to_id)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = {
                "properties": dict(edge.properties),
                "valid_from": edge.valid_from,
                "valid_to": edge.valid_to,
            }
        else:
            existing["properties"] = {**existing["properties"], **edge.properties}
            if edge.valid_from is not None and (
                existing["valid_from"] is None
                or edge.valid_from < existing["valid_from"]
            ):
                existing["valid_from"] = edge.valid_from
            existing["valid_to"] = edge.valid_to

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

    await conn.execute(
        """
        INSERT INTO graph_edges (
            customer_id, edge_type, from_node_id, to_node_id,
            properties, valid_from, valid_to, source_system
        )
        SELECT $1, edge_type, from_node_id, to_node_id,
               properties::jsonb, COALESCE(valid_from, NOW()), valid_to, $2
        FROM unnest(
            $3::text[], $4::bigint[], $5::bigint[],
            $6::text[], $7::timestamptz[], $8::timestamptz[]
        ) AS t(edge_type, from_node_id, to_node_id, properties, valid_from, valid_to)
        ON CONFLICT (customer_id, edge_type, from_node_id, to_node_id)
        DO UPDATE SET
            properties = graph_edges.properties || EXCLUDED.properties,
            valid_from = LEAST(graph_edges.valid_from, EXCLUDED.valid_from),
            valid_to   = EXCLUDED.valid_to
        """,
        customer_id,
        source_system,
        edge_types,
        from_ids,
        to_ids,
        properties_json,
        valid_from_list,
        valid_to_list,
    )
    return len(deduped)
