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

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest
import sqlalchemy as sa

import engine.shared.db as db_module
from engine.retrieval.grounding import (
    _fuzzy_match_document_titles,
    _fuzzy_match_document_titles_multi,
    _fuzzy_match_entities,
    _fuzzy_match_entities_multi,
)

# Depends on g, so it is evaluated per row (a constant argument is folded once).
_STOPWORD_SQL = "SELECT plainto_tsquery('english', 'does' || repeat('', g)) FROM generate_series(1, 5) g"


@pytest.mark.integration
async def test_tenant_transactions_receive_no_notices(live_db):
    notices: list[str] = []
    listener = lambda _c, msg: notices.append(str(msg))  # noqa: E731
    # Twice: the second acquire gets a connection the pool has already reset
    # (RESET ALL on release).
    for _ in range(2):
        async with db_module.with_tenant("test-cust-notices") as conn:
            conn.add_log_listener(listener)
            try:
                await conn.fetch(_STOPWORD_SQL)
                assert await conn.fetchval("SHOW client_min_messages") == "warning"
            finally:
                conn.remove_log_listener(listener)
    assert notices == []


@pytest.mark.integration
async def test_the_setting_is_scoped_to_the_tenant_transaction(live_db):
    """Negative control, and proof it does not leak: outside with_tenant the
    same pooled connections still notify, so the test above proves the
    setting rather than an absent stopword."""
    notices: list[str] = []
    listener = lambda _c, msg: notices.append(str(msg))  # noqa: E731
    async with db_module.with_tenant("test-cust-notices"):
        pass
    async with db_module.raw_conn() as conn:
        conn.add_log_listener(listener)
        try:
            assert await conn.fetchval("SHOW client_min_messages") == "notice"
            await conn.fetch(_STOPWORD_SQL)
        finally:
            conn.remove_log_listener(listener)
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


_ENTITY_CUSTOMER = "test-cust-entity-ties"
_ENTITY_IDS = [f"tie-repo-{i:02d}" for i in range(9)]


async def _seed_tied_entities() -> None:
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1, $1, $1)",
            _ENTITY_CUSTOMER,
        )
        # Same name (same score), no last_seen_at: only canonical_id differs.
        await conn.executemany(
            "INSERT INTO graph_nodes (customer_id, label, canonical_id, properties) "
            """VALUES ($1, 'Service', $2, '{"name": "Pipeline Tiebreak"}'::jsonb)""",
            [(_ENTITY_CUSTOMER, c) for c in reversed(_ENTITY_IDS)],
        )


@pytest.mark.integration
async def test_entity_ties_cross_the_cap_by_canonical_id(live_db):
    await _seed_tied_entities()
    runs = []
    for _ in range(3):
        cands = await _fuzzy_match_entities(
            _ENTITY_CUSTOMER, ["pipeline tiebreak"], per_type_cap=4, total_cap=4
        )
        runs.append([c.canonical_id for c in cands])
    assert runs[0] == sorted(_ENTITY_IDS)[:4]
    assert runs[0] == runs[1] == runs[2]


@pytest.mark.integration
async def test_multi_entity_ties_cross_the_cap_by_canonical_id(live_db):
    await _seed_tied_entities()
    per_probe = await _fuzzy_match_entities_multi(
        _ENTITY_CUSTOMER, ["pipeline", "tiebreak"], per_type_cap=3, total_cap=3
    )
    for cands in per_probe:
        assert [c.canonical_id for c in cands] == sorted(_ENTITY_IDS)[:3]


def _load_0143():
    path = (
        Path(__file__).resolve().parents[2]
        / "db/migrations/versions/20260929_0143_canonical_and_purge_indexes.py"
    )
    spec = importlib.util.spec_from_file_location("m0143", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.integration
async def test_0143_replaces_a_wrong_same_named_index_and_keeps_a_right_one(live_db, settings):
    mod = _load_0143()
    sync_dsn = settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = sa.create_engine(sync_dsn, isolation_level="AUTOCOMMIT")
    name, table, columns = mod.INDEXES[0]
    with engine.connect() as bind:
        # A hand-built index with the right name and the WRONG key order.
        bind.execute(sa.text(f"DROP INDEX IF EXISTS public.{name}"))
        bind.execute(sa.text(f"CREATE INDEX {name} ON public.{table} (canonical_id, customer_id)"))
        assert mod.ensure_index(bind, name, table, columns) is True
        oid = bind.execute(sa.text(f"SELECT 'public.{name}'::regclass::oid")).scalar()
        # Now correct: left alone.
        assert mod.ensure_index(bind, name, table, columns) is False
        assert bind.execute(sa.text(f"SELECT 'public.{name}'::regclass::oid")).scalar() == oid
        definition = bind.execute(sa.text(f"SELECT pg_get_indexdef('public.{name}'::regclass)")).scalar()
        assert definition == mod.expected_definition(name, table, columns)
    engine.dispose()


_ROLE = "quickwins_app"


@pytest.mark.integration
async def test_label_free_lookup_uses_the_index_under_force_rls(live_db, settings):
    """As a non-superuser table reader with the tenant GUC set, so FORCE RLS
    applies: the index condition carries both the policy's customer_id and the
    canonical id (texteq is LEAKPROOF)."""
    import asyncpg

    async with db_module.raw_conn() as conn:
        await conn.execute(
            f"""
            DO $$ BEGIN
              IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{_ROLE}') THEN
                CREATE ROLE {_ROLE} LOGIN PASSWORD 'pw' NOSUPERUSER NOBYPASSRLS;
              END IF;
            END $$;
            """
        )
        await conn.execute(f"GRANT USAGE ON SCHEMA public TO {_ROLE}")
        await conn.execute(f"GRANT SELECT ON graph_nodes TO {_ROLE}")
    parts = urlsplit(settings.database_url)
    dsn = urlunsplit(
        (parts.scheme, f"{_ROLE}:pw@{parts.hostname}:{parts.port or 5432}",
         parts.path, parts.query, parts.fragment)
    )
    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction():
            await conn.execute("SELECT set_config('app.current_customer_id', 'probe', true)")
            await conn.execute("SET LOCAL enable_seqscan = off")
            plan = "\n".join(
                r[0]
                for r in await conn.fetch(
                    "EXPLAIN SELECT node_id FROM graph_nodes WHERE canonical_id = 'richardwei6'"
                )
            )
    finally:
        await conn.close()
    assert "idx_graph_nodes_customer_canonical" in plan
    assert "canonical_id = 'richardwei6'" in plan
