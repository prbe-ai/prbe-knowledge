"""Unit tests for the inferred-edges bundle builder.

The most important test here (CRITICAL) is the cross-tenant isolation test:
given two tenants with overlapping content, build_bundle for tenant A must
never return a doc belonging to tenant B.

These tests run against a live Postgres instance (docker-compose up -d).
They are skipped if the DB is unavailable.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import pytest_asyncio

from engine.ingest.inferred_edges.bundle import (
    build_bundle,
)
from engine.shared.db import with_tenant

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_customer(conn, customer_id: str) -> None:
    # api_key_hash is NOT NULL on customers; pass an arbitrary placeholder
    # for tests (matches the pattern in test_idempotency.py / test_chunk_diff.py).
    await conn.execute(
        """
        INSERT INTO customers (customer_id, display_name, api_key_hash, status)
        VALUES ($1, $1, 'test-' || $1, 'active')
        ON CONFLICT (customer_id) DO NOTHING
        """,
        customer_id,
    )


async def _insert_doc(conn, customer_id: str, doc_id: str, source_system: str = "slack") -> None:
    now = datetime.now(UTC)
    await conn.execute(
        """
        INSERT INTO documents (
            doc_id, customer_id, version, source_system, source_id, source_url,
            doc_class, doc_type, content_hash, title, body_size_bytes, body_token_count,
            author_id, created_at, updated_at, valid_from, ingested_at,
            acl
        ) VALUES (
            $1, $2, 1, $3, $1, 'https://example.com/' || $1,
            'raw_source', 'slack.thread', md5($1),
            'Test doc ' || $1, 100, 20,
            NULL, $4, $4, $4, $4,
            '{"principals": [], "captured_at": "2026-01-01T00:00:00Z"}'::jsonb
        )
        ON CONFLICT (customer_id, doc_id, version) DO NOTHING
        """,
        doc_id,
        customer_id,
        source_system,
        now,
    )


async def _insert_chunk(conn, customer_id: str, doc_id: str, content: str, idx: int = 0) -> None:
    # chunks.embedding is NOT NULL halfvec(3072); use a zero-vector
    # placeholder. Test doesn't exercise vector similarity so values
    # don't matter — only the fact that the INSERT satisfies the schema.
    await conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, doc_id, customer_id, chunk_index, content, content_hash,
            token_count, embedding, embedding_model, embedding_dim,
            chunker_version, first_seen_version, last_seen_version, valid_from
        ) VALUES (
            $1, $2, $3, $4, $5, md5($5),
            length($5) / 4,
            array_fill(0::real, ARRAY[3072])::halfvec,
            'openai/text-embedding-3-large', 3072,
            'naive-v1', 1, 1, NOW()
        )
        ON CONFLICT (customer_id, chunk_id) DO NOTHING
        """,
        f"{doc_id}:chunk:{idx}",
        doc_id,
        customer_id,
        idx,
        content,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TENANT_A = "cust-bundle-test-tenant-a"
TENANT_B = "cust-bundle-test-tenant-b"


@pytest_asyncio.fixture
async def db_with_tenants(live_db):
    """Insert two tenants with overlapping content. Yield (TENANT_A, TENANT_B)."""
    from engine.shared.db import raw_conn

    async with raw_conn() as conn:
        await _insert_customer(conn, TENANT_A)
        await _insert_customer(conn, TENANT_B)

        # Tenant A docs
        await _insert_doc(conn, TENANT_A, "docA1", "slack")
        await _insert_chunk(conn, TENANT_A, "docA1", "Tenant A doc 1 content about authentication", 0)

        await _insert_doc(conn, TENANT_A, "docA2", "linear")
        await _insert_chunk(conn, TENANT_A, "docA2", "Tenant A doc 2 about auth service bug", 0)

        # Tenant B docs with similar content (potential cross-tenant bleed)
        await _insert_doc(conn, TENANT_B, "docB1", "slack")
        await _insert_chunk(conn, TENANT_B, "docB1", "Tenant B doc 1 content about authentication", 0)

        await _insert_doc(conn, TENANT_B, "docB2", "github")
        await _insert_chunk(conn, TENANT_B, "docB2", "Tenant B doc 2 about auth service fix", 0)

    yield TENANT_A, TENANT_B


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bundle_anchor_only(live_db) -> None:
    """An anchor doc with no neighbors produces a bundle with just the anchor."""
    from engine.shared.db import raw_conn

    async with raw_conn() as conn:
        await _insert_customer(conn, "cust-anchor-only")
        await _insert_doc(conn, "cust-anchor-only", "anchor-only-doc", "slack")
        await _insert_chunk(conn, "cust-anchor-only", "anchor-only-doc", "Only content here", 0)

    async with with_tenant("cust-anchor-only") as conn:
        bundle = await build_bundle("cust-anchor-only", "anchor-only-doc", conn)

    assert bundle.customer_id == "cust-anchor-only"
    assert bundle.anchor_doc_id == "anchor-only-doc"
    assert len(bundle.docs) == 1
    assert bundle.docs[0].doc_id == "anchor-only-doc"
    assert bundle.total_tokens > 0


@pytest.mark.asyncio
async def test_bundle_missing_anchor_returns_empty(live_db) -> None:
    """If the anchor doc doesn't exist, return an empty bundle (don't crash)."""
    from engine.shared.db import raw_conn

    async with raw_conn() as conn:
        await _insert_customer(conn, "cust-missing-anchor")

    async with with_tenant("cust-missing-anchor") as conn:
        bundle = await build_bundle("cust-missing-anchor", "nonexistent-doc-xyz", conn)

    assert bundle.docs == []
    assert bundle.total_tokens == 0


@pytest.mark.asyncio
async def test_bundle_token_budget_enforced(live_db) -> None:
    """Bundle content is trimmed to fit within the token budget."""
    from engine.shared.db import raw_conn

    customer_id = "cust-token-budget"
    async with raw_conn() as conn:
        await _insert_customer(conn, customer_id)
        await _insert_doc(conn, customer_id, "big-doc", "slack")
        # Insert a chunk with a lot of content
        big_content = "x" * 10_000  # ~2857 estimated tokens at 3.5 chars/token
        await _insert_chunk(conn, customer_id, "big-doc", big_content, 0)

    tiny_budget = 100  # tokens
    async with with_tenant(customer_id) as conn:
        bundle = await build_bundle(customer_id, "big-doc", conn, token_budget=tiny_budget)

    # Bundle should not exceed the budget
    assert bundle.total_tokens <= tiny_budget + 50  # small rounding tolerance
    if bundle.docs:
        total_chars = sum(len(d.content) for d in bundle.docs)
        # At 3.5 chars/token, 100 tokens => ~350 chars budget
        assert total_chars <= tiny_budget * 4  # rough bound


# ---------------------------------------------------------------------------
# CRITICAL: Cross-tenant isolation test                                      #
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bundle_never_returns_cross_tenant_docs(db_with_tenants) -> None:
    """CRITICAL: build_bundle for tenant A must never return a tenant B doc.

    This test covers the highest-blast-radius bug we could ship.
    Both tenants have documents with similar content. The bundle builder
    must only return docs for the requested tenant.
    """
    tenant_a, _tenant_b = db_with_tenants

    async with with_tenant(tenant_a) as conn:
        bundle = await build_bundle(tenant_a, "docA1", conn)

    # Every doc in the bundle must belong to tenant A
    for doc in bundle.docs:
        assert doc.customer_id == tenant_a, (
            f"CROSS-TENANT LEAK: bundle for {tenant_a} contains doc "
            f"{doc.doc_id!r} belonging to {doc.customer_id!r}"
        )
        # Double-check the doc_id prefix (docA* vs docB*)
        assert not doc.doc_id.startswith("docB"), (
            f"CROSS-TENANT LEAK: doc {doc.doc_id!r} from tenant B found in "
            f"bundle for tenant A"
        )

    # Sanity: the bundle should contain the anchor doc
    doc_ids = {d.doc_id for d in bundle.docs}
    assert "docA1" in doc_ids


@pytest.mark.asyncio
async def test_bundle_tenant_b_does_not_see_tenant_a(db_with_tenants) -> None:
    """Mirror of the cross-tenant test: tenant B's bundle must not leak tenant A."""
    _tenant_a, tenant_b = db_with_tenants

    async with with_tenant(tenant_b) as conn:
        bundle = await build_bundle(tenant_b, "docB1", conn)

    for doc in bundle.docs:
        assert doc.customer_id == tenant_b, (
            f"CROSS-TENANT LEAK: bundle for {tenant_b} contains doc "
            f"{doc.doc_id!r} belonging to {doc.customer_id!r}"
        )
        assert not doc.doc_id.startswith("docA"), (
            f"CROSS-TENANT LEAK: doc {doc.doc_id!r} from tenant A found in "
            f"bundle for tenant B"
        )


@pytest.mark.asyncio
async def test_bundle_customer_id_on_bundle_object(live_db) -> None:
    """bundle.customer_id always equals the requested customer_id."""
    from engine.shared.db import raw_conn

    async with raw_conn() as conn:
        await _insert_customer(conn, "cust-id-check")
        await _insert_doc(conn, "cust-id-check", "id-check-doc", "slack")
        await _insert_chunk(conn, "cust-id-check", "id-check-doc", "some content", 0)

    async with with_tenant("cust-id-check") as conn:
        bundle = await build_bundle("cust-id-check", "id-check-doc", conn)

    assert bundle.customer_id == "cust-id-check"


# ---------------------------------------------------------------------------
# _fetch_vector_similar_cross_source (REGRESSION, 2026-08-20)
#
# This function had ZERO tests, and that is exactly why it shipped broken: its
# CTE was named `similar`, a reserved word in Postgres (SIMILAR TO), so the
# statement was a syntax error and the query NEVER ran once in production --
# 752 failures a day, from the day it shipped.
#
# Nothing caught it because the failure is swallowed and the function returns
# [], which is indistinguishable from "this document has no similar neighbours".
# Every end-to-end bundle test above still passed; they just silently received
# fewer candidate docs, and nothing asserted on the count.
#
# The assertion that matters is `!= []` on a corpus that HAS a cross-source
# neighbour. Both tests below fail against the pre-rename code.
#
# Note these seed `embedding_v2`, not `embedding`. _insert_chunk above fills the
# legacy v1 column with a zero vector and says outright that values don't matter
# because it never exercises similarity -- the production query reads v2, and a
# zero vector has no defined cosine direction anyway.
# ---------------------------------------------------------------------------

VEC_TENANT = "cust-vector-similar-tenant"


async def _insert_chunk_with_v2(
    conn, customer_id: str, doc_id: str, content: str, fill: float
) -> None:
    """Seed a chunk whose embedding_v2 is a non-zero constant vector.

    `fill` differentiates documents: cosine distance between two identical
    constant vectors is 0, and between different constants it is small but
    non-zero, which is enough to make ORDER BY deterministic without pretending
    these are meaningful embeddings.
    """
    await conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, doc_id, customer_id, chunk_index, content, content_hash,
            token_count, embedding_v2, embedding_v2_model, embedding_v2_dim,
            chunker_version, first_seen_version, last_seen_version, valid_from
        ) VALUES (
            $1, $2, $3, 0, $4, md5($4),
            length($4) / 4,
            (SELECT array_agg(v)::halfvec
               FROM (SELECT $5::real AS v FROM generate_series(1, 3072)) s),
            'gemini-embedding-2', 3072,
            'naive-v1', 1, 1, NOW()
        )
        ON CONFLICT (customer_id, chunk_id) DO NOTHING
        """,
        f"{doc_id}:chunk:0", doc_id, customer_id, content, fill,
    )


@pytest.mark.asyncio
async def test_vector_similar_cross_source_actually_returns_neighbours(
    live_db,
) -> None:
    """The query executes and finds a cross-source neighbour.

    Pre-fix this returned [] because the SQL did not parse.
    """
    from engine.ingest.inferred_edges.bundle import _fetch_vector_similar_cross_source
    from engine.shared.db import raw_conn

    async with raw_conn() as conn:
        await _insert_customer(conn, VEC_TENANT)
        await _insert_doc(conn, VEC_TENANT, "vec-anchor", "slack")
        await _insert_chunk_with_v2(conn, VEC_TENANT, "vec-anchor", "auth service", 0.5)
        await _insert_doc(conn, VEC_TENANT, "vec-neighbour", "github")
        await _insert_chunk_with_v2(conn, VEC_TENANT, "vec-neighbour", "auth fix", 0.5)

    async with with_tenant(VEC_TENANT) as conn:
        found = await _fetch_vector_similar_cross_source(
            conn, VEC_TENANT, "vec-anchor", "slack", set()
        )

    assert found != [], "query did not execute — the reserved-word bug is back"
    assert "vec-neighbour" in found
    assert "vec-anchor" not in found, "anchor must exclude itself"


@pytest.mark.asyncio
async def test_vector_similar_cross_source_excludes_same_source(live_db) -> None:
    """A same-source neighbour is filtered out — the whole point is CROSS source.

    Guards the `d.source_system <> $3` predicate, which a careless rewrite of
    the CTE could drop while still parsing.
    """
    from engine.ingest.inferred_edges.bundle import _fetch_vector_similar_cross_source
    from engine.shared.db import raw_conn

    async with raw_conn() as conn:
        await _insert_customer(conn, VEC_TENANT)
        await _insert_doc(conn, VEC_TENANT, "same-anchor", "slack")
        await _insert_chunk_with_v2(conn, VEC_TENANT, "same-anchor", "auth", 0.5)
        await _insert_doc(conn, VEC_TENANT, "same-sibling", "slack")
        await _insert_chunk_with_v2(conn, VEC_TENANT, "same-sibling", "auth too", 0.5)

    async with with_tenant(VEC_TENANT) as conn:
        found = await _fetch_vector_similar_cross_source(
            conn, VEC_TENANT, "same-anchor", "slack", set()
        )

    assert "same-sibling" not in found
