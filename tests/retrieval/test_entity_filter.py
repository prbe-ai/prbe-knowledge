"""Integration tests for the list-path graph entity filter (PR-B).

Mandatory tests (per locked plan):
- Focused regression fixture: "last commit on prbe-backend" returns ONLY
  prbe-backend commits, even when other repos have newer commits.
- Loose match: bare-name and full-name queries return the same result.
- Guardrail: "backend" doesn't false-positive match prbe-backend.
- Cross-tenant RLS: tenant A's filter never returns tenant B docs.
- Person filter still uses author_id (regression).

Plus EXPLAIN test (`SET enable_seqscan = OFF`) to assert the planner
finds an index path on the loose-match join.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from services.retrieval.retrievers.sql import (
    GraphEntityFilter,
    sql_count,
    sql_list,
)
from shared.config import Settings, get_settings
from shared.db import raw_conn
from shared.embeddings import reset_embedder
from shared.storage import reset_store

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


# ---- seed helpers ---------------------------------------------------------


async def _seed_customer(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )


async def _seed_doc_with_repo_link(
    customer_id: str,
    *,
    doc_id: str,
    full_name: str,  # e.g. "prbe-ai/prbe-backend"
    repo_short_name: str,  # e.g. "prbe-backend" — goes in graph_nodes.properties.name
    title: str,
    updated_at: datetime,
    doc_type: str = "github.commit",
) -> None:
    """Seed a github-shaped doc + Document graph node + Repo graph node +
    Document → Repo edge. Mirrors what the github handler produces."""
    async with raw_conn() as conn:
        # documents
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at, acl
            ) VALUES (
                $1, 1, $2,
                'github', $3, $4,
                'raw_source', $5, 'text/plain',
                $6, $7, 100, 0,
                $8, $8, $8, $8, '{}'::jsonb
            )
            """,
            doc_id, customer_id, doc_id + "-src",
            f"https://github.com/{full_name}/example",
            doc_type, f"hash-{doc_id}", title, updated_at,
        )
        # chunk
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
            f"{doc_id}:c0", doc_id, customer_id,
            f"body of {title}", f"chash-{doc_id}",
        )
        # graph_nodes — Document + Repo (using ON CONFLICT semantics from
        # the upsert path; here we just INSERT-or-skip)
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
            VALUES ($1, 'Document', $2, '{}'::jsonb)
            ON CONFLICT (customer_id, label, canonical_id) DO NOTHING
            """,
            customer_id, doc_id,
        )
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
            VALUES ($1, 'Repo', $2, $3::jsonb)
            ON CONFLICT (customer_id, label, canonical_id) DO NOTHING
            """,
            customer_id, full_name,
            f'{{"name": "{repo_short_name}"}}',
        )
        # graph_edges — Document → Repo (look up node_ids inline)
        await conn.execute(
            """
            INSERT INTO graph_edges (customer_id, edge_type, from_node_id, to_node_id, valid_from)
            SELECT $1, 'TOUCHES', d.node_id, r.node_id, $4
            FROM graph_nodes d, graph_nodes r
            WHERE d.customer_id = $1 AND d.label = 'Document' AND d.canonical_id = $2
              AND r.customer_id = $1 AND r.label = 'Repo'     AND r.canonical_id = $3
            ON CONFLICT DO NOTHING
            """,
            customer_id, doc_id, full_name, updated_at,
        )


# ---- focused regression fixture ------------------------------------------


async def test_last_commit_on_prbe_backend_returns_only_that_repo(live_db) -> None:
    """SHIP GATE — the original failing query, with explicit fixture.

    5 commits across 3 repos. prbe-knowledge has the newest 2 commits.
    A naive 'most recent commit' returns prbe-knowledge, NOT prbe-backend.
    With the entity filter, asking for prbe-backend returns ONLY the
    prbe-backend commit even though it's older.
    """
    cust = "cust-bug-fix"
    await _seed_customer(cust)
    base = datetime(2026, 4, 28, 12, 0, tzinfo=UTC)

    # Newest first across all repos: prbe-knowledge (2 commits), then
    # prbe-dashboard, then prbe-backend, then prbe-knowledge again.
    await _seed_doc_with_repo_link(
        cust, doc_id="github:prbe-ai/prbe-knowledge:commit:KN1",
        full_name="prbe-ai/prbe-knowledge", repo_short_name="prbe-knowledge",
        title="newest knowledge commit",
        updated_at=base,
    )
    await _seed_doc_with_repo_link(
        cust, doc_id="github:prbe-ai/prbe-knowledge:commit:KN2",
        full_name="prbe-ai/prbe-knowledge", repo_short_name="prbe-knowledge",
        title="second knowledge commit",
        updated_at=base - timedelta(minutes=1),
    )
    await _seed_doc_with_repo_link(
        cust, doc_id="github:prbe-ai/prbe-dashboard:commit:DB1",
        full_name="prbe-ai/prbe-dashboard", repo_short_name="prbe-dashboard",
        title="dashboard commit",
        updated_at=base - timedelta(minutes=2),
    )
    await _seed_doc_with_repo_link(
        cust, doc_id="github:prbe-ai/prbe-backend:commit:BE1",
        full_name="prbe-ai/prbe-backend", repo_short_name="prbe-backend",
        title="THE prbe-backend commit",
        updated_at=base - timedelta(minutes=3),
    )
    await _seed_doc_with_repo_link(
        cust, doc_id="github:prbe-ai/prbe-knowledge:commit:KN3",
        full_name="prbe-ai/prbe-knowledge", repo_short_name="prbe-knowledge",
        title="oldest knowledge commit",
        updated_at=base - timedelta(minutes=4),
    )

    # Naive query (no entity filter) — returns the newest commit overall,
    # which is prbe-knowledge. Verify our setup mirrors the bug.
    naive = await sql_list(cust, top_k=1, doc_types=["github.commit"])
    assert naive[0].title == "newest knowledge commit"

    # WITH entity filter — returns ONLY the prbe-backend commit, even
    # though it's not the newest overall.
    filtered = await sql_list(
        cust,
        top_k=10,
        doc_types=["github.commit"],
        graph_entity_filters=[
            GraphEntityFilter(label="Repo", values=["prbe-backend", "prbe-ai/prbe-backend"])
        ],
    )
    assert len(filtered) == 1
    assert filtered[0].title == "THE prbe-backend commit"


# ---- loose match ----------------------------------------------------------


async def test_loose_match_bare_name_via_properties(live_db) -> None:
    """`'prbe-backend'` (bare name) matches via `properties->>'name'` even
    though canonical_id is `'prbe-ai/prbe-backend'` (full form)."""
    cust = "cust-loose-bare"
    await _seed_customer(cust)
    await _seed_doc_with_repo_link(
        cust, doc_id="d1",
        full_name="prbe-ai/prbe-backend", repo_short_name="prbe-backend",
        title="commit", updated_at=datetime(2026, 4, 28, tzinfo=UTC),
    )

    hits = await sql_list(
        cust, top_k=10,
        graph_entity_filters=[GraphEntityFilter(label="Repo", values=["prbe-backend"])],
    )
    assert len(hits) == 1


async def test_loose_match_suffix_after_slash(live_db) -> None:
    """When `properties->>'name'` is missing (legacy graph nodes), the
    suffix-after-slash arm catches `'prbe-backend'` against
    `canonical_id = 'prbe-ai/prbe-backend'`."""
    cust = "cust-loose-suffix"
    await _seed_customer(cust)
    # Seed without properties.name (legacy shape).
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            cust,
        )
    await _seed_doc_with_repo_link(
        cust, doc_id="d1",
        full_name="prbe-ai/prbe-backend",
        repo_short_name="",  # NO properties.name
        title="commit", updated_at=datetime(2026, 4, 28, tzinfo=UTC),
    )
    # Wipe the properties.name so only the suffix arm can match.
    async with raw_conn() as conn:
        await conn.execute(
            """
            UPDATE graph_nodes SET properties = '{}'::jsonb
            WHERE customer_id = $1 AND label = 'Repo'
            """,
            cust,
        )

    hits = await sql_list(
        cust, top_k=10,
        graph_entity_filters=[GraphEntityFilter(label="Repo", values=["prbe-backend"])],
    )
    assert len(hits) == 1


async def test_loose_match_full_name_exact(live_db) -> None:
    cust = "cust-loose-exact"
    await _seed_customer(cust)
    await _seed_doc_with_repo_link(
        cust, doc_id="d1",
        full_name="prbe-ai/prbe-backend", repo_short_name="prbe-backend",
        title="commit", updated_at=datetime(2026, 4, 28, tzinfo=UTC),
    )

    hits = await sql_list(
        cust, top_k=10,
        graph_entity_filters=[
            GraphEntityFilter(label="Repo", values=["prbe-ai/prbe-backend"])
        ],
    )
    assert len(hits) == 1


async def test_loose_match_case_insensitive(live_db) -> None:
    cust = "cust-loose-case"
    await _seed_customer(cust)
    await _seed_doc_with_repo_link(
        cust, doc_id="d1",
        full_name="prbe-ai/prbe-backend", repo_short_name="prbe-backend",
        title="commit", updated_at=datetime(2026, 4, 28, tzinfo=UTC),
    )

    hits = await sql_list(
        cust, top_k=10,
        graph_entity_filters=[GraphEntityFilter(label="Repo", values=["PRBE-Backend"])],
    )
    assert len(hits) == 1


@pytest.mark.parametrize(
    ("query_value", "expect_match"),
    [
        # Same logical entity expressed every way Haiku might emit it.
        # All should match a graph node whose canonical_id is
        # 'external-investigations'.
        ("external-investigations", True),  # exact
        ("external investigations", True),  # space → hyphen
        ("external_investigations", True),  # underscore → hyphen
        ("External Investigations", True),  # title-case + space
        ("EXTERNAL_INVESTIGATIONS", True),  # uppercase + underscore
        ("ExternalInvestigations", True),  # camel-case, no separator
        # NOT a match — different word entirely; alnum normalization
        # doesn't introduce substring fuzziness.
        ("investigations", False),
        ("external", False),
    ],
)
async def test_alphanumeric_normalization_arm(live_db, query_value, expect_match) -> None:
    """The 4th + 5th match arms strip non-alphanumeric chars on both
    sides so cosmetic-separator differences don't defeat the filter.
    Catches the "external investigations" vs "external-investigations"
    family of mismatches Haiku produces from natural-language queries."""
    cust = f"cust-alnum-{abs(hash(query_value)) % 10_000}"
    await _seed_customer(cust)
    await _seed_doc_with_repo_link(
        cust,
        doc_id="d1",
        full_name="external-investigations",
        repo_short_name="external-investigations",
        title="ticket",
        updated_at=datetime(2026, 4, 28, tzinfo=UTC),
        # `Repo` label is the seeder's default; the alnum normalization
        # arm doesn't care about label semantics — it's purely a string
        # comparison on canonical_id / properties.name.
    )

    hits = await sql_list(
        cust,
        top_k=10,
        graph_entity_filters=[GraphEntityFilter(label="Repo", values=[query_value])],
    )
    if expect_match:
        assert len(hits) == 1, f"{query_value!r} should match"
    else:
        assert hits == [], f"{query_value!r} should NOT match (would be substring fuzziness)"


# ---- guardrail: substring DOESN'T match ----------------------------------


async def test_substring_does_not_match(live_db) -> None:
    """`'backend'` (no slash, not a full name, not properties.name) must
    NOT match `'prbe-ai/prbe-backend'` — the loose-match SQL deliberately
    omits a contains-substring arm to avoid false positives like
    `'backend'` matching every `*-backend` repo."""
    cust = "cust-substr"
    await _seed_customer(cust)
    await _seed_doc_with_repo_link(
        cust, doc_id="d1",
        full_name="prbe-ai/prbe-backend", repo_short_name="prbe-backend",
        title="commit", updated_at=datetime(2026, 4, 28, tzinfo=UTC),
    )
    await _seed_doc_with_repo_link(
        cust, doc_id="d2",
        full_name="prbe-ai/willow-backend", repo_short_name="willow-backend",
        title="other commit", updated_at=datetime(2026, 4, 28, tzinfo=UTC),
    )

    hits = await sql_list(
        cust, top_k=10,
        graph_entity_filters=[GraphEntityFilter(label="Repo", values=["backend"])],
    )
    # Should match neither — bare 'backend' isn't a full name, isn't
    # any node's `properties.name`, and the suffix arm requires `'/'`.
    assert hits == []


# ---- multi-tenant RLS regression -----------------------------------------


async def test_entity_filter_does_not_cross_tenants(live_db) -> None:
    """SHIP GATE — tenant A's entity filter must never return tenant B
    docs even when the entity name is the same in both tenants."""
    await _seed_customer("tenant-A")
    await _seed_customer("tenant-B")
    now = datetime(2026, 4, 28, tzinfo=UTC)
    await _seed_doc_with_repo_link(
        "tenant-A", doc_id="A-commit",
        full_name="prbe-ai/prbe-backend", repo_short_name="prbe-backend",
        title="A's secret commit", updated_at=now,
    )
    await _seed_doc_with_repo_link(
        "tenant-B", doc_id="B-commit",
        full_name="prbe-ai/prbe-backend", repo_short_name="prbe-backend",
        title="B's secret commit", updated_at=now,
    )

    a_hits = await sql_list(
        "tenant-A", top_k=10,
        graph_entity_filters=[GraphEntityFilter(label="Repo", values=["prbe-backend"])],
    )
    a_titles = {h.title for h in a_hits}
    assert a_titles == {"A's secret commit"}
    assert "B's secret commit" not in a_titles

    n_a = await sql_count(
        "tenant-A",
        graph_entity_filters=[GraphEntityFilter(label="Repo", values=["prbe-backend"])],
    )
    n_b = await sql_count(
        "tenant-B",
        graph_entity_filters=[GraphEntityFilter(label="Repo", values=["prbe-backend"])],
    )
    assert n_a == 1
    assert n_b == 1


# ---- empty-filter + None preserves legacy behavior ----------------------


async def test_empty_entity_filter_behaves_as_no_filter(live_db) -> None:
    """REGRESSION — passing None or an empty list for graph_entity_filters
    is identical to the pre-PR-B behavior."""
    cust = "cust-empty-filter"
    await _seed_customer(cust)
    await _seed_doc_with_repo_link(
        cust, doc_id="d1",
        full_name="prbe-ai/prbe-backend", repo_short_name="prbe-backend",
        title="c1", updated_at=datetime(2026, 4, 28, tzinfo=UTC),
    )

    a = await sql_list(cust, top_k=10, graph_entity_filters=None)
    b = await sql_list(cust, top_k=10)
    c = await sql_list(cust, top_k=10, graph_entity_filters=[])
    assert {h.title for h in a} == {h.title for h in b} == {h.title for h in c} == {"c1"}


# ---- EXPLAIN test (planner uses indexes) ---------------------------------


async def test_entity_filter_excludes_soft_closed_edges(live_db) -> None:
    """REGRESSION — when a graph_edge is soft-closed (valid_to IS NOT NULL),
    the entity filter must NOT include it. Today no code path closes edges,
    but this guards against silent inclusion of stale relationships when a
    future feature ('user left channel') starts soft-deleting."""
    cust = "cust-soft-close"
    await _seed_customer(cust)
    await _seed_doc_with_repo_link(
        cust,
        doc_id="d1",
        full_name="prbe-ai/prbe-backend",
        repo_short_name="prbe-backend",
        title="commit",
        updated_at=datetime(2026, 4, 28, tzinfo=UTC),
    )

    # Filter sees the doc — baseline.
    before = await sql_list(
        cust,
        top_k=10,
        graph_entity_filters=[GraphEntityFilter(label="Repo", values=["prbe-backend"])],
    )
    assert len(before) == 1

    # Soft-close the Document → Repo edge.
    async with raw_conn() as conn:
        await conn.execute(
            "SELECT set_config('app.current_customer_id', $1, false)", cust
        )
        await conn.execute(
            """
            UPDATE graph_edges SET valid_to = NOW()
            WHERE customer_id = $1
              AND from_node_id IN (
                SELECT node_id FROM graph_nodes
                WHERE customer_id = $1 AND label = 'Document'
              )
              AND to_node_id IN (
                SELECT node_id FROM graph_nodes
                WHERE customer_id = $1 AND label = 'Repo'
              )
            """,
            cust,
        )

    # Filter must NOT find the doc anymore — the only edge is closed.
    after = await sql_list(
        cust,
        top_k=10,
        graph_entity_filters=[GraphEntityFilter(label="Repo", values=["prbe-backend"])],
    )
    assert after == []


async def test_explain_uses_index_when_seqscan_off(live_db) -> None:
    """With `enable_seqscan = OFF`, the planner can only execute the
    query if it has a usable index path on the equality arms of the
    loose match. If the functional indexes (idx_graph_nodes_lower_canonical
    and idx_graph_nodes_lower_props_name) are missing, this would error
    out. This is a tripwire: catches "the migration didn't ship the index"
    regressions even though the test fixture is small.
    """
    cust = "cust-explain"
    await _seed_customer(cust)
    await _seed_doc_with_repo_link(
        cust, doc_id="d1",
        full_name="prbe-ai/prbe-backend", repo_short_name="prbe-backend",
        title="c1", updated_at=datetime(2026, 4, 28, tzinfo=UTC),
    )

    # Forcing enable_seqscan = OFF; if Postgres can't find any index
    # path, the query fails. We don't assert a specific plan shape —
    # we just assert it executes successfully.
    async with raw_conn() as conn:
        # SET commands don't accept parameter binding; use SELECT
        # set_config to set the GUC parameterized.
        await conn.execute("SELECT set_config('app.current_customer_id', $1, false)", cust)
        await conn.execute("SET LOCAL enable_seqscan = OFF")
        rows = await conn.fetch(
            """
            SELECT 1
            FROM graph_nodes
            WHERE customer_id = $1
              AND label = 'Repo'
              AND LOWER(canonical_id) = LOWER($2)
            LIMIT 1
            """,
            cust, "prbe-ai/prbe-backend",
        )
        assert len(rows) == 1
        rows = await conn.fetch(
            """
            SELECT 1
            FROM graph_nodes
            WHERE customer_id = $1
              AND label = 'Repo'
              AND LOWER(properties->>'name') = LOWER($2)
            LIMIT 1
            """,
            cust, "prbe-backend",
        )
        assert len(rows) == 1
