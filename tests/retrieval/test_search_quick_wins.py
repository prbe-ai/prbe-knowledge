"""Search latency quick wins (plan T2, T3, T6).

* T2: pool connections never receive NOTICEs. A stopword probe makes
  plainto_tsquery raise one NOTICE per row it is evaluated against; the
  multi-probe title match sent 74,234 of them per search on the research
  plane.
* T3: the doc-title channel orders ties by doc_id, so the documents that cross
  the cap are the same on every run.
* T6: graph_nodes has an index for a canonical-id lookup without a label, and
  pending_edges one for the tenant-purge FK.
"""

from __future__ import annotations

from datetime import UTC, datetime

import asyncpg
import pytest

import engine.shared.db as db_module
from engine.retrieval.grounding import (
    _fuzzy_match_document_titles,
    _fuzzy_match_document_titles_multi,
)

# Depends on g, so it is evaluated per row (a constant argument is folded once).
_STOPWORD_SQL = "SELECT plainto_tsquery('english', 'does' || repeat('', g)) FROM generate_series(1, 5) g"


@pytest.mark.integration
async def test_pool_connections_receive_no_notices(live_db, settings):
    notices: list[str] = []
    listener = lambda _c, msg: notices.append(str(msg))  # noqa: E731
    # Twice: the second acquire gets a connection the pool has already reset
    # (RESET ALL on release), which is what reverted a plain SET.
    for _ in range(2):
        async with db_module.raw_conn() as conn:
            conn.add_log_listener(listener)
            try:
                await conn.fetch(_STOPWORD_SQL)
                assert await conn.fetchval("SHOW client_min_messages") == "warning"
            finally:
                conn.remove_log_listener(listener)
    assert notices == []


@pytest.mark.integration
async def test_a_bare_connection_would_receive_them(live_db, settings):
    """Negative control: without the pool's setup the same query notifies,
    so the test above proves the setting, not an absent stopword."""
    notices: list[str] = []
    conn = await asyncpg.connect(settings.database_url)
    try:
        conn.add_log_listener(lambda _c, msg: notices.append(str(msg)))
        await conn.fetch(_STOPWORD_SQL)
    finally:
        await conn.close()
    assert len(notices) == 5


_TIED_CUSTOMER = "test-cust-title-ties"
_TIED_DOC_IDS = [f"custom:tie:{i:02d}" for i in range(12)]


async def _seed_tied_titles() -> None:
    now = datetime(2026, 9, 29, tzinfo=UTC)
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1, $1, $2)",
            _TIED_CUSTOMER,
            "hash-ties",
        )
        # Identical title and updated_at: every sort key but doc_id ties.
        # Inserted in reverse so scan order disagrees with doc_id order.
        await conn.executemany(
            """
            INSERT INTO documents (
                doc_id, version, customer_id, source_system, source_id, source_url,
                doc_class, doc_type, content_type, content_hash, title, body_preview,
                body_size_bytes, body_token_count, created_at, updated_at, valid_from,
                valid_to, ingested_at, acl
            ) VALUES (
                $1, 1, $2, 'custom_ingest', $1, '/x', 'raw_source', 'custom.document',
                'text/markdown', 'h-' || $1, 'Quarterly tiebreak review', 'body',
                10, 0, $3, $3, $3, NULL, $3, '{}'::jsonb
            )
            """,
            [(d, _TIED_CUSTOMER, now) for d in reversed(_TIED_DOC_IDS)],
        )


@pytest.mark.integration
async def test_single_probe_title_ties_cross_the_cap_by_doc_id(live_db):
    await _seed_tied_titles()
    runs = []
    for _ in range(3):
        cands = await _fuzzy_match_document_titles(
            customer_id=_TIED_CUSTOMER, tokens=["tiebreak"], cap=5
        )
        runs.append([c.canonical_id for c in cands])
    assert runs[0] == sorted(_TIED_DOC_IDS)[:5]
    assert runs[0] == runs[1] == runs[2]


@pytest.mark.integration
async def test_multi_probe_title_ties_cross_the_cap_by_doc_id(live_db):
    await _seed_tied_titles()
    per_probe = await _fuzzy_match_document_titles_multi(
        customer_id=_TIED_CUSTOMER, probes=["tiebreak", "quarterly"], cap=4
    )
    for cands in per_probe:
        assert [c.canonical_id for c in cands] == sorted(_TIED_DOC_IDS)[:4]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("name", "table", "columns"),
    [
        ("idx_graph_nodes_customer_canonical", "graph_nodes", "(customer_id, canonical_id)"),
        ("idx_pending_edges_customer", "pending_edges", "(customer_id)"),
    ],
)
async def test_0143_indexes_are_declared(live_db, name, table, columns):
    async with db_module.raw_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT pg_get_indexdef(c.oid) AS def, i.indisvalid
            FROM pg_class c JOIN pg_index i ON i.indexrelid = c.oid
            WHERE c.relname = $1
            """,
            name,
        )
    assert row is not None, f"{name} missing from db/schema.sql"
    assert row["indisvalid"]
    assert f"ON public.{table} USING btree {columns}" in row["def"]


@pytest.mark.integration
async def test_label_free_canonical_lookup_can_use_the_index(live_db):
    """The plan shape anchor_exists / the adapter need: an index condition on
    both columns, no label."""
    async with db_module.raw_conn() as conn, conn.transaction():
        await conn.execute("SET LOCAL enable_seqscan = off")
        plan = "\n".join(
            r[0]
            for r in await conn.fetch(
                "EXPLAIN SELECT node_id FROM graph_nodes "
                "WHERE customer_id = 'probe' AND canonical_id = 'richardwei6'"
            )
        )
    assert "idx_graph_nodes_customer_canonical" in plan
