"""Guardian against a REAL pg_search database.

    docker run -d --name kb-guardian-pg \
      -e POSTGRES_USER=prbe -e POSTGRES_PASSWORD=prbe -e POSTGRES_DB=prbe_knowledge \
      -p 5460:5432 paradedb/paradedb:latest
    PRBE_PGSEARCH_TEST_DSN=postgresql://prbe:prbe@localhost:5460/prbe_knowledge \
      pytest tests/test_pg_search_guardian_integration.py

Skipped without that DSN. It is a SEPARATE variable from the usual test
database on purpose: neither the compose stack (`pgvector/pgvector:pg16`) nor
CI installs pg_search, so these cannot ride the normal `live_db` fixture and
must not silently no-op inside it.

WHY THE DANGEROUS DIRECTION IS THE ONE TESTED HERE
--------------------------------------------------
The unit tests cover "does it detect a broken index" with fabricated rows. What
they cannot cover is whether the detection SQL is VALID against real pg_search
catalogs -- and a detector with a bad column reference or a wrong join does not
fail loudly, it finds nothing, forever, while looking healthy. That is the same
silent-failure shape the guardian exists to end, so it gets a real database.

The other half is the false positive: the guardian holds DDL rights and runs
unattended, so "healthy index is left alone" matters more than "broken index is
found". Verified here against a genuinely healthy index.

REPRODUCING THE DESTRUCTIVE HALF (verified 2026-08-26, pg_search 0.25.4)
------------------------------------------------------------------------
Emptying the index file needs filesystem access to PGDATA, which pytest does
not have against an arbitrary DSN, so it is not automated here. The procedure,
run against the container above:

    # 1. table with rows + a real index
    CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, content TEXT,
                         title TEXT, customer_id TEXT);
    INSERT INTO chunks SELECT 'c'||i, 'content '||i, 'title '||i, 'probe'
      FROM generate_series(1,500) i;
    CREATE INDEX idx_chunks_bm25_v2 ON chunks
      USING bm25 (chunk_id, content, title, customer_id) WITH (key_field=chunk_id);

    # 2. truncate the index file with Postgres STOPPED. Truncating a running
    #    instance does nothing lasting -- shutdown flushes the still-dirty
    #    buffers straight back over the hole.
    docker stop kb-guardian-pg
    docker run --rm --volumes-from kb-guardian-pg alpine \
      truncate -s 0 /var/lib/postgresql/18/docker/base/16384/25486
    docker start kb-guardian-pg

Observed after that, which is the 2026-08-25 incident byte for byte:

    idx_chunks_bm25_v2 | bm25 | 0 bytes | indisvalid=t | reltuples=500
    ERROR: could not read blocks 0..0 in file "base/16384/25486":
           read only 0 of 8192 bytes

and the guardian then detected it, dropped it, and the table planned again.
"""

from __future__ import annotations

import os

import asyncpg
import pytest

from engine.shared import pg_search_guardian as guardian

DSN = os.environ.get("PRBE_PGSEARCH_TEST_DSN")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="PRBE_PGSEARCH_TEST_DSN unset; needs a pg_search database"),
]


@pytest.fixture
async def conn():
    c = await asyncpg.connect(DSN)
    await c.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    await c.execute("DROP TABLE IF EXISTS chunks CASCADE")
    await c.execute(
        """
        CREATE TABLE chunks (
            chunk_id     TEXT PRIMARY KEY,
            content      TEXT,
            title        TEXT,
            customer_id  TEXT
        )
        """
    )
    await c.execute(
        "INSERT INTO chunks SELECT 'c'||i, 'content '||i, 'title '||i, 'probe' "
        "FROM generate_series(1,500) i"
    )
    await c.execute(
        "CREATE INDEX idx_chunks_bm25_v2 ON chunks "
        "USING bm25 (chunk_id, content, title, customer_id) WITH (key_field=chunk_id)"
    )
    await c.execute("ANALYZE chunks")
    try:
        yield c
    finally:
        await c.execute("DROP TABLE IF EXISTS chunks CASCADE")
        await c.close()


async def test_detection_sql_is_valid_against_real_catalogs(conn) -> None:
    """The query runs. A silently-malformed detector is this module's own
    version of the bug it exists to catch."""
    assert await guardian.find_broken_pg_search_indexes(conn) == []
    assert await guardian.find_invalid_index_debris(conn) == []


async def test_healthy_index_is_never_flagged(conn) -> None:
    """The false positive is the expensive mistake: the guardian holds DDL
    rights, so wrongly dropping a healthy index would be a self-inflicted
    outage on a live search corpus."""
    size = await conn.fetchval("SELECT pg_relation_size('idx_chunks_bm25_v2')")
    assert size > 0
    assert await guardian.find_broken_pg_search_indexes(conn) == []


async def test_a_healthy_index_on_an_empty_table_is_not_zero_bytes(conn) -> None:
    """Measured, because the whole detector rests on it.

    If pg_search deferred its first write, a healthy index on a table that had
    not been flushed yet would read as 0 bytes and the guardian would drop it.
    It does not: on 0.25.4 an index over an EMPTY table is ~2.8 MB of metadata
    from the moment it is built. So 0 bytes is genuinely pathological and only
    a truncated or never-replicated file produces it.

    `reltuples = -1` here (never analyzed) is the second half of the same
    point, and is why the detector treats -1 as nonempty: a freshly promoted
    standby has no statistics, and reading -1 as "empty" would switch the
    guardian off on exactly the instance it is for."""
    await conn.execute("DROP TABLE IF EXISTS fresh CASCADE")
    await conn.execute("CREATE TABLE fresh (chunk_id TEXT PRIMARY KEY, content TEXT)")
    await conn.execute(
        "CREATE INDEX idx_fresh_bm25 ON fresh USING bm25 (chunk_id, content) "
        "WITH (key_field=chunk_id)"
    )
    try:
        size = await conn.fetchval("SELECT pg_relation_size('idx_fresh_bm25')")
        reltuples = await conn.fetchval("SELECT reltuples FROM pg_class WHERE relname='fresh'")
        assert size > 0, "a healthy pg_search index must never read as 0 bytes"
        assert reltuples == -1
    finally:
        await conn.execute("DROP TABLE IF EXISTS fresh CASCADE")


async def test_drop_refuses_an_unlisted_index_against_a_real_db(conn) -> None:
    """The allowlist holds with a real connection in hand, not just a fake."""
    await conn.execute("DROP TABLE IF EXISTS other CASCADE")
    await conn.execute("CREATE TABLE other (id TEXT PRIMARY KEY, body TEXT)")
    await conn.execute(
        "CREATE INDEX idx_other_bm25 ON other USING bm25 (id, body) WITH (key_field=id)"
    )
    try:
        with pytest.raises(ValueError, match="allowlist"):
            await guardian.drop_broken_index(conn, "idx_other_bm25")
        assert await conn.fetchval("SELECT to_regclass('idx_other_bm25')") is not None
    finally:
        await conn.execute("DROP TABLE IF EXISTS other CASCADE")


async def test_timeline_round_trip(conn) -> None:
    """The promotion detector's state, against a real instance.

    An absent row must read as None -- the guardian treats that as "first tick,
    record and do nothing" rather than as a promotion, so installing it does
    not alert on every cluster the day it ships."""
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pg_search_guardian_state (
            id                SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            last_timeline_id  BIGINT NOT NULL,
            observed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    await conn.execute("DELETE FROM pg_search_guardian_state")
    try:
        assert await guardian.read_last_timeline(conn) is None

        timeline = await guardian.current_timeline_id(conn)
        assert timeline >= 1

        await guardian.record_timeline(conn, timeline)
        assert await guardian.read_last_timeline(conn) == timeline

        # UPSERT, not INSERT: two rows would mean two disagreeing memories of
        # the same fact.
        await guardian.record_timeline(conn, timeline + 1)
        assert await guardian.read_last_timeline(conn) == timeline + 1
        assert await conn.fetchval("SELECT count(*) FROM pg_search_guardian_state") == 1
    finally:
        await conn.execute("DELETE FROM pg_search_guardian_state")
