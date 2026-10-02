"""Live chunks carry the sentinel (migration 0145) and BM25 asks the index for it (plan T8).

Pins: the CHECK holds in both directions; 0145's `run` adds and validates it,
is idempotent, and refuses to record a constraint it could not validate; a
partition provisioned afterwards carries it; the BM25 pool filters "live" on
the indexed `last_seen_version` instead of the heap-only `valid_to`, and
still returns exactly the live rows (closed versions of the same text never
appear; AS_OF still reaches them).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import asyncpg
import pytest
import sqlalchemy as sa

import engine.shared.db as db_module
from engine.shared.constants import LIVE_CHUNK_LAST_SEEN
from engine.shared.partitions import ensure_tenant_partition, is_partitioned, partition_of

CONSTRAINT = "chunks_live_sentinel_chk"


def _load_0145():
    path = (
        Path(__file__).resolve().parents[2]
        / "db/migrations/versions/20261002_0145_chunks_live_sentinel.py"
    )
    spec = importlib.util.spec_from_file_location("m0145", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def _tenant_with_doc(conn, tenant: str, doc_id: str, versions: int = 1) -> None:
    await conn.execute(
        "INSERT INTO customers (customer_id, display_name, api_key_hash) "
        "VALUES ($1, $1, $1) ON CONFLICT DO NOTHING",
        tenant,
    )
    for v in range(1, versions + 1):
        await conn.execute(
            """
            INSERT INTO documents (customer_id, doc_id, version, source_system,
                                   source_id, source_url, doc_type, content_hash,
                                   created_at, updated_at, valid_from, valid_to, acl,
                                   title, body_preview)
            VALUES ($2, $1, $3::int, 'custom_ingest', $1, 'https://x', 'custom.note',
                    'dh' || $3::int::text, NOW(), NOW(), NOW() - interval '2 days',
                    CASE WHEN $3::int < $4::int THEN NOW() - interval '1 day' END,
                    '{}'::jsonb, 'T', 'p')
            """,
            doc_id,
            tenant,
            v,
            versions,
        )


async def _chunk(conn, tenant, doc_id, chunk_id, content_hash, first, last, *, closed: bool):
    await conn.execute(
        """
        INSERT INTO chunks (chunk_id, doc_id, customer_id, chunk_index, content,
                            content_hash, token_count, chunker_version,
                            first_seen_version, last_seen_version, kind, visibility,
                            valid_from, valid_to)
        VALUES ($1, $2, $3, 0, 'sentinelword ' || $1, $4, 3, 'v1', $5, $6, 'content',
                'approved', NOW() - interval '2 days',
                CASE WHEN $7 THEN NOW() - interval '1 day' END)
        """,
        chunk_id,
        doc_id,
        tenant,
        content_hash,
        first,
        last,
        closed,
    )


@pytest.mark.integration
async def test_the_check_holds_in_both_directions(live_db):
    async with db_module.raw_conn() as conn:
        await _tenant_with_doc(conn, "lss-a", "lss-a:d1", versions=2)
        await ensure_tenant_partition(conn, "lss-a")
        # Allowed: live + sentinel, closed + exact.
        await _chunk(conn, "lss-a", "lss-a:d1", "lss-a:live", "h-live", 2, LIVE_CHUNK_LAST_SEEN, closed=False)
        await _chunk(conn, "lss-a", "lss-a:d1", "lss-a:old", "h-old", 1, 1, closed=True)
        # Refused: a live row with an exact version, a closed row left open-ended.
        with pytest.raises(asyncpg.exceptions.CheckViolationError, match=CONSTRAINT):
            await _chunk(conn, "lss-a", "lss-a:d1", "lss-a:legacy", "h-legacy", 1, 2, closed=False)
        with pytest.raises(asyncpg.exceptions.CheckViolationError, match=CONSTRAINT):
            await _chunk(
                conn, "lss-a", "lss-a:d1", "lss-a:ghost", "h-ghost", 1, LIVE_CHUNK_LAST_SEEN, closed=True
            )
        # A close that leaves the sentinel behind is refused too.
        with pytest.raises(asyncpg.exceptions.CheckViolationError, match=CONSTRAINT):
            await conn.execute(
                "UPDATE chunks SET valid_to = NOW() WHERE customer_id = 'lss-a' AND chunk_id = 'lss-a:live'"
            )


@pytest.mark.integration
async def test_a_partition_provisioned_later_carries_the_check(live_db):
    async with db_module.raw_conn() as conn:
        await _tenant_with_doc(conn, "lss-late", "lss-late:d1")
        await ensure_tenant_partition(conn, "lss-late")
        leaf = await partition_of(conn, "lss-late")
        if leaf is None:
            pytest.skip("chunks is not partitioned on this database")
        assert await conn.fetchval(
            "SELECT convalidated FROM pg_constraint WHERE conrelid = $1::regclass AND conname = $2",
            leaf,
            CONSTRAINT,
        )


@pytest.mark.integration
async def test_0145_adds_validates_and_is_idempotent(live_db, settings):
    mod = _load_0145()
    sync_dsn = settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = sa.create_engine(sync_dsn, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as bind:
            bind.execute(sa.text(f"ALTER TABLE chunks DROP CONSTRAINT IF EXISTS {CONSTRAINT}"))
            mod.run(bind)
            assert mod._state(bind) == (True, True)
            mod.run(bind)  # nothing left to do
            assert mod._state(bind) == (True, True)
    finally:
        engine.dispose()


@pytest.mark.integration
async def test_0145_refuses_to_finish_over_a_legacy_live_row(live_db, settings):
    """VALIDATE fails while a pre-#604 row remains: the deploy must stop there,
    not record a constraint the BM25 change would then trust."""
    mod = _load_0145()
    sync_dsn = settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = sa.create_engine(sync_dsn, isolation_level="AUTOCOMMIT")
    async with db_module.raw_conn() as conn:
        await _tenant_with_doc(conn, "lss-legacy", "lss-legacy:d1")
        await ensure_tenant_partition(conn, "lss-legacy")
    try:
        with engine.connect() as bind:
            bind.execute(sa.text(f"ALTER TABLE chunks DROP CONSTRAINT IF EXISTS {CONSTRAINT}"))
            bind.execute(
                sa.text(
                    "INSERT INTO chunks (chunk_id, doc_id, customer_id, chunk_index, content, "
                    "content_hash, token_count, first_seen_version, last_seen_version) "
                    "VALUES ('lss-legacy:c', 'lss-legacy:d1', 'lss-legacy', 0, 'x', 'h', 1, 1, 1)"
                )
            )
            with pytest.raises(sa.exc.IntegrityError, match=CONSTRAINT):
                mod.run(bind)
            assert mod._state(bind) == (True, False)
            bind.execute(
                sa.text(
                    "UPDATE chunks SET last_seen_version = 2147483647 "
                    "WHERE customer_id = 'lss-legacy' AND chunk_id = 'lss-legacy:c'"
                )
            )
            mod.run(bind)
            assert mod._state(bind) == (True, True)
    finally:
        engine.dispose()


def test_bm25_pool_filters_live_on_the_indexed_column():
    """The pool asks the index for the sentinel and drops the two heap-only
    predicates; AS_OF keeps its valid_to window."""
    import inspect

    from engine.retrieval.retrievers import bm25

    src = inspect.getsource(bm25.bm25_search)
    assert 'f"AND c.last_seen_version = {int(LIVE_CHUNK_LAST_SEEN)}"' in src
    assert "AND c.visibility = 'approved'" not in src
    assert "pool_chunk_sql = pred.chunk_sql" in src


@pytest.mark.integration
async def test_bm25_returns_live_rows_only_and_as_of_still_reaches_old_ones(pg_search_db):
    """Same text in a closed v1 chunk and a live v2 chunk: LATEST returns only
    the live one (the sentinel term replaced valid_to IS NULL), AS_OF a moment
    inside v1's window returns the closed one."""
    from datetime import UTC, datetime, timedelta

    from engine.retrieval.retrievers.bm25 import bm25_search
    from engine.shared.models import TemporalMode, TemporalSpec

    tenant = "lss-bm25"
    async with db_module.raw_conn() as conn:
        if not await is_partitioned(conn):
            pytest.skip("chunks is not partitioned on this database")
        await _tenant_with_doc(conn, tenant, f"{tenant}:d1", versions=2)
        await ensure_tenant_partition(conn, tenant)
        await _chunk(conn, tenant, f"{tenant}:d1", f"{tenant}:v1", "h1", 1, 1, closed=True)
        await _chunk(conn, tenant, f"{tenant}:d1", f"{tenant}:v2", "h2", 2, LIVE_CHUNK_LAST_SEEN, closed=False)

    live = await bm25_search(tenant, "sentinelword", top_k=10)
    assert [h.chunk_id for h in live] == [f"{tenant}:v2"]

    past = datetime.now(UTC) - timedelta(hours=36)
    old = await bm25_search(
        tenant, "sentinelword", top_k=10, temporal=TemporalSpec(mode=TemporalMode.AS_OF, as_of=past)
    )
    assert f"{tenant}:v1" in [h.chunk_id for h in old]
