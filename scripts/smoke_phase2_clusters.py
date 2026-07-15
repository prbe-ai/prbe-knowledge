"""One-off seed for the Phase 2 cluster-awareness smoke test.

Idempotent: drops the smoke customer first, then re-seeds. Safe to
re-run between iterations. Never imported by application code.

Seed shape:
  - Customer: smoke-phase2-cust
  - Person nodes:    richardwei6, mahit@prbe.ai, U07ABC123
  - Provenance:      richardwei6 -> github
                     mahit@prbe.ai -> slack
                     U07ABC123 -> linear
  - Document:        d-1 (authored by richardwei6) + d-2 (authored by mahit@prbe.ai)
  - Graph edges:     richardwei6 -AUTHORED-> Document:d-1
                     mahit@prbe.ai -AUTHORED-> Document:d-2
  - No entity_aliases rows yet -- /api/entity-clusters/merge writes those.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from engine.shared.db import close_pool, init_pool, raw_conn

CUSTOMER = "smoke-phase2-cust"
NOW = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)


async def main() -> None:
    await init_pool()
    try:
        async with raw_conn() as conn:
            await conn.execute("DELETE FROM customers WHERE customer_id = $1", CUSTOMER)
            await conn.execute(
                """
                INSERT INTO customers (customer_id, display_name, api_key_hash)
                VALUES ($1::text, 'phase2 smoke', 'h-' || $1::text)
                """,
                CUSTOMER,
            )
            for doc_id, author in [("d-1", "richardwei6"), ("d-2", "mahit@prbe.ai")]:
                await conn.execute(
                    """
                    INSERT INTO documents (
                        doc_id, version, customer_id,
                        source_system, source_id, source_url,
                        doc_class, doc_type, content_type,
                        content_hash, title, body_size_bytes, body_token_count,
                        created_at, updated_at, valid_from, ingested_at, acl,
                        author_id
                    ) VALUES (
                        $1, 1, $2,
                        'github', $3, 'https://example/' || $1,
                        'raw_source', 'github.commit', 'text/plain',
                        'h-' || $1, 'doc-' || $1, 100, 0,
                        $4, $4, $4, $4, '{}'::jsonb,
                        $5
                    )
                    """,
                    doc_id, CUSTOMER, f"commit:{doc_id}", NOW, author,
                )
                await conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, customer_id,
                        chunk_index, content, content_hash, token_count,
                        embedding, first_seen_version, last_seen_version
                    ) VALUES (
                        $1, $2, $3, 0, $4, $5, 5,
                        array_fill(0::real, ARRAY[3072])::halfvec,
                        1, 1
                    )
                    """,
                    f"{doc_id}:c0", doc_id, CUSTOMER,
                    f"body of {doc_id}", f"chash-{doc_id}",
                )
            for canonical, source in [
                ("richardwei6", "github"),
                ("mahit@prbe.ai", "slack"),
                ("U07ABC123", "linear"),
            ]:
                await conn.execute(
                    """
                    INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
                    VALUES ($1::text, 'Person', $2::text, jsonb_build_object('name', $2::text), 1)
                    """,
                    CUSTOMER, canonical,
                )
                await conn.execute(
                    """
                    INSERT INTO graph_node_provenance (
                        customer_id, node_id, source_system,
                        first_seen_at, last_seen_at
                    )
                    SELECT $1, gn.node_id, $3, $4, $4
                    FROM graph_nodes gn
                    WHERE gn.customer_id = $1 AND gn.label = 'Person' AND gn.canonical_id = $2
                    """,
                    CUSTOMER, canonical, source, NOW,
                )
            for doc_id in ("d-1", "d-2"):
                await conn.execute(
                    """
                    INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
                    VALUES ($1, 'Document', $2, '{}'::jsonb, 1)
                    """,
                    CUSTOMER, doc_id,
                )
            for author, doc_id in [("richardwei6", "d-1"), ("mahit@prbe.ai", "d-2")]:
                await conn.execute(
                    """
                    INSERT INTO graph_edges (
                        customer_id, edge_type,
                        from_node_id, to_node_id,
                        confidence, properties
                    )
                    SELECT $1, 'AUTHORED',
                           p.node_id, d.node_id,
                           'EXTRACTED', '{}'::jsonb
                    FROM graph_nodes p, graph_nodes d
                    WHERE p.customer_id = $1 AND p.label = 'Person'   AND p.canonical_id = $2
                      AND d.customer_id = $1 AND d.label = 'Document' AND d.canonical_id = $3
                    """,
                    CUSTOMER, author, doc_id,
                )
        print(f"Seeded {CUSTOMER}: 3 Person nodes + 2 docs + AUTHORED edges + provenance.")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
