"""Phase 2 cluster awareness in the related-entities walker.

Covers:
1. member_count = primary + alias count
2. member_sources = DISTINCT source_systems from graph_node_provenance
3. Unmerged neighbor reports member_count=1 + its own source
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from services.retrieval.retrievers.related_entities import (
    walk_result_doc_neighbors,
)
from shared.db import raw_conn

pytestmark = pytest.mark.asyncio


CUSTOMER_ID = "rel-ents-cluster-cust"
PRIMARY = "richardwei6"
ALIAS_A = "mahit@prbe.ai"
ALIAS_B = "U07ABC123"
DOC_ID = "d-1"


async def _seed_full_cluster(customer_id: str) -> None:
    """Seed: customer + doc/chunk/Document node + Person:PRIMARY + AUTHORED
    edge + entity_aliases (2 aliases routing to PRIMARY) + provenance with
    3 distinct source_systems on the PRIMARY's node."""
    now = datetime(2026, 4, 28, tzinfo=UTC)
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at, acl
            ) VALUES (
                $2, 1, $1,
                'github', 'commit:' || $2, 'https://example/' || $2,
                'raw_source', 'github.commit', 'text/plain',
                'h-' || $2, 'doc', 100, 0,
                $3, $3, $3, $3, '{}'::jsonb
            )
            """,
            customer_id, DOC_ID, now,
        )
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count,
                embedding, first_seen_version, last_seen_version
            ) VALUES (
                $1, $2, $3, 0, 'body', 'chash', 5,
                array_fill(0::real, ARRAY[3072])::halfvec,
                1, 1
            )
            """,
            f"{DOC_ID}:c0", DOC_ID, customer_id,
        )
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
            VALUES
              ($1, 'Document', $2, '{}'::jsonb, 1),
              ($1, 'Person',   $3, '{"name":"Richard"}'::jsonb, 1)
            """,
            customer_id, DOC_ID, PRIMARY,
        )
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
            WHERE p.customer_id = $1 AND p.label = 'Person'   AND p.canonical_id = $3
              AND d.customer_id = $1 AND d.label = 'Document' AND d.canonical_id = $2
            """,
            customer_id, DOC_ID, PRIMARY,
        )
        # Post-merge provenance: alias source_systems consolidated to primary.
        await conn.execute(
            """
            INSERT INTO graph_node_provenance (
                customer_id, node_id, source_system,
                first_seen_at, last_seen_at
            )
            SELECT $1, p.node_id, src, $2, $2
            FROM graph_nodes p, UNNEST(ARRAY['github','slack','linear']) AS src
            WHERE p.customer_id = $1 AND p.label = 'Person' AND p.canonical_id = $3
            """,
            customer_id, now, PRIMARY,
        )
        # Cluster metadata: 2 aliases routing to PRIMARY.
        merge_id = str(uuid.uuid4())
        await conn.execute(
            """
            INSERT INTO entity_merge_audit (
                merge_id, customer_id, label, primary_canonical_id,
                merged_alias_canonical_ids, performed_by_user_id, status
            ) VALUES ($1, $2, 'Person', $3, ARRAY[$4, $5]::text[],
                      '11111111-1111-1111-1111-111111111111', 'active')
            """,
            merge_id, customer_id, PRIMARY, ALIAS_A, ALIAS_B,
        )
        await conn.executemany(
            """
            INSERT INTO entity_aliases (
                customer_id, label, alias_canonical_id,
                primary_canonical_id, merge_id
            ) VALUES ($1, 'Person', $2, $3, $4)
            """,
            [
                (customer_id, ALIAS_A, PRIMARY, merge_id),
                (customer_id, ALIAS_B, PRIMARY, merge_id),
            ],
        )


async def _seed_unmerged_person(customer_id: str) -> None:
    """Seed: customer + Document:d-loner + Person:loner-id + AUTHORED edge
    + single-source provenance. NO entity_aliases rows."""
    now = datetime(2026, 4, 28, tzinfo=UTC)
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'test', 'h-' || $1) ON CONFLICT (customer_id) DO NOTHING",
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at, acl
            ) VALUES (
                'd-loner', 1, $1,
                'github', 'commit:d-loner', 'https://example/d-loner',
                'raw_source', 'github.commit', 'text/plain',
                'h-d-loner', 'doc', 100, 0,
                $2, $2, $2, $2, '{}'::jsonb
            )
            """,
            customer_id, now,
        )
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count,
                embedding, first_seen_version, last_seen_version
            ) VALUES (
                'd-loner:c0', 'd-loner', $1, 0, 'body', 'chash', 5,
                array_fill(0::real, ARRAY[3072])::halfvec, 1, 1
            )
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
            VALUES
              ($1, 'Document', 'd-loner', '{}'::jsonb, 1),
              ($1, 'Person',   'loner-id', '{"name":"Loner"}'::jsonb, 1)
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type, from_node_id, to_node_id,
                confidence, properties
            )
            SELECT $1, 'AUTHORED', p.node_id, d.node_id, 'EXTRACTED', '{}'::jsonb
            FROM graph_nodes p, graph_nodes d
            WHERE p.customer_id = $1 AND p.label = 'Person'   AND p.canonical_id = 'loner-id'
              AND d.customer_id = $1 AND d.label = 'Document' AND d.canonical_id = 'd-loner'
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO graph_node_provenance (
                customer_id, node_id, source_system,
                first_seen_at, last_seen_at
            )
            SELECT $1, p.node_id, 'github', $2, $2
            FROM graph_nodes p
            WHERE p.customer_id = $1 AND p.label = 'Person' AND p.canonical_id = 'loner-id'
            """,
            customer_id, now,
        )


async def test_walker_populates_member_count_and_sources(live_db):
    await _seed_full_cluster(CUSTOMER_ID)
    rels = await walk_result_doc_neighbors(
        customer_id=CUSTOMER_ID,
        ranked_result_docs=[(DOC_ID, 1)],
        exclude_node_keys=set(),
        min_confidence=None,
        top_n=10,
    )
    # Only one neighbor: the Person primary.
    people = [r for r in rels if r.label == "Person"]
    assert len(people) == 1
    person = people[0]
    assert person.canonical_id == PRIMARY
    # member_count = primary (1) + 2 aliases = 3.
    assert person.member_count == 3
    # member_sources from consolidated provenance.
    assert sorted(person.member_sources) == ["github", "linear", "slack"]


async def test_walker_member_count_one_for_unmerged_node(live_db):
    customer_id = "rel-ents-unmerged-cust"
    await _seed_unmerged_person(customer_id)
    rels = await walk_result_doc_neighbors(
        customer_id=customer_id,
        ranked_result_docs=[("d-loner", 1)],
        exclude_node_keys=set(),
        min_confidence=None,
        top_n=10,
    )
    people = [r for r in rels if r.label == "Person"]
    assert len(people) == 1
    person = people[0]
    assert person.member_count == 1
    assert person.member_sources == ["github"]
