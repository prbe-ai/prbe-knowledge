"""Detect and repair a pg_search index that a failover left unusable.

THE FAILURE THIS EXISTS FOR
---------------------------
2026-08-25, research plane. A CNPG failover promoted a standby whose
`idx_chunks_bm25_v2` was a 0-byte file. pg_search Community does not replicate
its index storage to physical standbys, so that standby had held an empty copy
since the index was built on 2026-08-20. The catalog still said
`indisvalid = true`, so the planner kept choosing it -- and pg_search's planner
hook reads the index at PLAN time, so the failure was not "BM25 queries return
nothing", it was `XX001` on every statement that planned against `chunks`. That
includes the ingestion worker's ordinary `SELECT ... WHERE customer_id = $1 AND
doc_id = $2`, which has nothing to do with search. ~10 errors/sec for forty
minutes, ended by a human running DROP INDEX.

Dropping the index is the whole repair. Without it every plan on the table
dies; with it the table works and search degrades to vector + graph + exact,
which is a serviceable product. Rebuilding is deliberately NOT done here -- see
"WHY THIS NEVER REBUILDS" below.

TWO FAILURE SHAPES, ONLY ONE OF THEM LOUD
-----------------------------------------
  1. EMPTY (0 bytes). The standby never received the index's contents at all.
     Loud: every plan on the table raises. Detected by size.

  2. STALE (nonzero). The standby was cloned via pg_basebackup from a healthy
     primary, so it copied real index files -- and then pg_search's subsequent
     writes never replicated. After promotion the index plans fine and quietly
     omits every document ingested since the clone. Silent. Size cannot see it.

Shape 2 is why this module tracks the Postgres timeline ID rather than only
scanning for empty files. The timeline increments on promotion, and a promotion
is the only reliable evidence that this instance's pg_search indexes are
suspect. See migration 0120.

WHY THE DETECTION PREDICATE IS THREE CONJUNCTS
----------------------------------------------
`DROP INDEX` unattended is a destructive act, so the trigger has to be
impossible to satisfy by accident:

  * ALLOWLIST -- the index name must be one this repository already declares in
    `engine/retrieval/index_contracts.py`. An index nobody declared is not ours
    to drop.
  * ACCESS METHOD -- `pg_am.amname IN ('bm25', 'paradedb')`. Never match on the
    index NAME for this half: pg_search 0.25 renamed the access method from
    `bm25` to `paradedb`, and a name-matched query would go quietly blind after
    that upgrade, which is the same class of silent failure this module exists
    to end.
  * NONEMPTY TABLE -- an index on an empty table is legitimately 0 bytes. On a
    fresh or truncated `chunks` the first two conjuncts are both true and the
    index is perfectly healthy; dropping it there would be a self-inflicted
    outage.

WHY THIS NEVER REBUILDS
-----------------------
`CREATE INDEX` is the dangerous operation, not the drop. Building a pg_search
index emits WAL that a Community standby's replay may refuse (paradedb#6007,
filed against 0.25.2), and `CREATE INDEX CONCURRENTLY` waits indefinitely on
transactions older than the build -- which wedged this project's production for
two hours on 2026-08-16 (see migration 0105). Neither belongs in an unattended
one-minute cron. The guardian restores SERVICE and alerts; a human rebuilds.

FAILURE POSTURE
---------------
Alerting must never be able to prevent the repair: a PostHog capture that
raises is swallowed, because a database that plans is worth more than a
notification. A database error is NOT swallowed -- the caller exits nonzero so
the CronJob's own history shows red, which is the fallback signal when the
alerting path is the thing that is broken.
"""

from __future__ import annotations

import asyncpg

from engine.retrieval.index_contracts import INDEX_CONTRACTS
from engine.shared.logging import get_logger

log = get_logger(__name__)

# Access methods a pg_search index can be built with. 0.25 renamed `bm25` to
# `paradedb` and kept `bm25` as a backwards-compatible alias, so both are live
# in the wild and an upgrade must not silently narrow what this matches.
PG_SEARCH_ACCESS_METHODS: tuple[str, ...] = ("bm25", "paradedb")

# Only indexes this repository declares a contract for are droppable. Derived
# rather than hardcoded so a new pg_search index is covered by declaring it in
# the one place the codebase already declares indexes.
ALLOWED_INDEX_NAMES: frozenset[str] = frozenset(c.index for c in INDEX_CONTRACTS)

# How long a half-built index may sit `indisvalid = false` before it is treated
# as abandoned debris worth alerting on. A live CREATE INDEX CONCURRENTLY on
# `chunks` runs for minutes, not an hour, so this is comfortably above a
# healthy build and comfortably below "nobody noticed".
INVALID_INDEX_ALERT_AFTER_SECONDS = 3600

# Ceiling on the DROP's wait for its ACCESS EXCLUSIVE lock. The guardian must
# never queue behind live ingestion holding the table: it would pin a lock
# request that blocks every reader arriving after it, converting a degraded
# search into a stalled database. Skipping costs one tick.
DROP_LOCK_TIMEOUT = "5s"


async def current_timeline_id(conn: asyncpg.Connection) -> int:
    """The instance's current Postgres timeline ID.

    Increments on every promotion, which is the signal that this instance's
    pg_search indexes may be stale-but-nonzero (see module docstring).
    """
    return int(await conn.fetchval("SELECT timeline_id FROM pg_control_checkpoint()"))


async def find_broken_pg_search_indexes(conn: asyncpg.Connection) -> list[dict[str, object]]:
    """Indexes that are valid, allowlisted, pg_search-backed, 0 bytes, and on a
    table that has rows.

    Catalog-only by construction: it reads `pg_class` / `pg_index` / `pg_am`
    and `pg_class.reltuples`, and never plans a query against the damaged table
    itself. That matters more than it looks -- when this fires, ANY statement
    that plans against `chunks` raises, so a detector that touched the table
    would be taken out by the exact fault it is meant to find.

    `reltuples` is an estimate maintained by ANALYZE, and `-1` means "never
    analyzed". Both are treated as nonempty: the question here is only "is this
    table plausibly empty", and refusing to act on an unanalyzed table would
    disable the guardian on precisely the freshly-promoted instance it is for.
    """
    rows = await conn.fetch(
        """
        SELECT
            i.indexrelid::regclass::text  AS index_name,
            c.relname                     AS table_name,
            am.amname                     AS access_method,
            pg_relation_size(i.indexrelid) AS index_bytes,
            c.reltuples                   AS table_reltuples
        FROM pg_index i
        JOIN pg_class ic ON ic.oid = i.indexrelid
        JOIN pg_class c  ON c.oid  = i.indrelid
        JOIN pg_am am    ON am.oid = ic.relam
        WHERE am.amname = ANY($1::text[])
          AND i.indisvalid
          AND pg_relation_size(i.indexrelid) = 0
        """,
        list(PG_SEARCH_ACCESS_METHODS),
    )
    broken: list[dict[str, object]] = []
    for r in rows:
        # `regclass` renders schema-qualified only when the index is outside
        # search_path; compare on the bare name the contracts declare.
        bare_name = r["index_name"].split(".")[-1].strip('"')
        if bare_name not in ALLOWED_INDEX_NAMES:
            log.info(
                "guardian.skip_unlisted_index",
                index=r["index_name"],
                reason="no index contract declares this index",
            )
            continue
        if r["table_reltuples"] == 0:
            # Legitimately empty table -> legitimately empty index.
            log.info(
                "guardian.skip_empty_table", index=r["index_name"], table=r["table_name"]
            )
            continue
        broken.append(
            {
                "index": bare_name,
                "table": r["table_name"],
                "access_method": r["access_method"],
                "index_bytes": int(r["index_bytes"]),
            }
        )
    return broken


async def find_invalid_index_debris(conn: asyncpg.Connection) -> list[str]:
    """Allowlisted pg_search indexes stuck `indisvalid = false`.

    A failed `CREATE INDEX CONCURRENTLY` leaves one of these behind. It is
    inert -- the planner will not choose an invalid index, so it breaks
    nothing -- but it also blocks a retry of the build under the same name, and
    it is invisible unless something looks.

    Deliberately alert-only. An in-progress CONCURRENTLY build looks exactly
    like abandoned debris from the catalog's point of view, and dropping the
    index out from under a human's attended rebuild would be the guardian
    causing the incident.
    """
    rows = await conn.fetch(
        """
        SELECT i.indexrelid::regclass::text AS index_name
        FROM pg_index i
        JOIN pg_class ic ON ic.oid = i.indexrelid
        JOIN pg_am am    ON am.oid = ic.relam
        WHERE am.amname = ANY($1::text[])
          AND NOT i.indisvalid
        """,
        list(PG_SEARCH_ACCESS_METHODS),
    )
    names = [r["index_name"].split(".")[-1].strip('"') for r in rows]
    return [n for n in names if n in ALLOWED_INDEX_NAMES]


async def drop_broken_index(conn: asyncpg.Connection, index_name: str) -> bool:
    """Drop one broken index. True if dropped, False if the lock was unavailable.

    Bounded by `lock_timeout` so the guardian yields to live traffic rather
    than queueing ahead of it. A skip is not a failure: the next tick retries,
    and the table is no worse off in the meantime than it already was.

    The identifier is interpolated because DDL cannot take a bind parameter. It
    is safe here for a reason worth stating rather than assuming: `index_name`
    reaches this function only from `ALLOWED_INDEX_NAMES`, which is built from
    string literals in `index_contracts.py` -- it never carries a value derived
    from user input or from the database.
    """
    if index_name not in ALLOWED_INDEX_NAMES:
        raise ValueError(f"refusing to drop index outside the allowlist: {index_name!r}")
    try:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL lock_timeout = '{DROP_LOCK_TIMEOUT}'")
            await conn.execute(f'DROP INDEX IF EXISTS "{index_name}"')
        return True
    except asyncpg.LockNotAvailableError:
        log.warning(
            "guardian.drop_skipped_lock_timeout",
            index=index_name,
            reason="table busy; next tick retries",
        )
        return False


async def analyze_tables(conn: asyncpg.Connection, tables: list[str]) -> None:
    """ANALYZE after a promotion, because planner statistics do not replicate.

    A promoted standby starts with none. On 2026-08-25 that alone cost 19.3s of
    grounding latency per query until someone ran ANALYZE by hand -- 2.6s
    afterwards. This runs on timeline change rather than on a stats heuristic:
    "n_live_tup = 0 and last_analyze IS NULL" also describes a genuinely empty
    table and drifts after any stats reset, whereas a timeline bump means
    exactly one thing.

    Best-effort, and that is a deliberate posture rather than laziness. The
    REPAIR is the drop; this is an optimization on top of it. An ANALYZE that
    raises must not abort the tick, because the tick ends by recording the
    timeline -- and a tick that dies before that point never advances the
    marker, so the next one re-detects the same promotion and alerts again.
    Every minute. Forever. A missing table (a self-host schema, a rename, a
    fresh database) would turn one alert into an unbounded alert storm, which
    is how a signal stops being read.

    Found by a smoke run against a database that happened not to have
    `documents`: the tick failed, the timeline stayed at 0, and the promotion
    alert fired on every subsequent tick.
    """
    for table in tables:
        if not table.replace("_", "").isalnum():
            raise ValueError(f"refusing to ANALYZE suspicious identifier: {table!r}")
        try:
            if await conn.fetchval("SELECT to_regclass($1)", table) is None:
                log.info("guardian.analyze_skipped", table=table, reason="table does not exist")
                continue
            await conn.execute(f'ANALYZE "{table}"')
            log.info("guardian.analyzed", table=table)
        except Exception as exc:
            log.warning("guardian.analyze_failed", table=table, error=str(exc))


async def read_last_timeline(conn: asyncpg.Connection) -> int | None:
    """The timeline the guardian last recorded, or None on its first ever tick."""
    return await conn.fetchval("SELECT last_timeline_id FROM pg_search_guardian_state WHERE id = 1")


async def record_timeline(conn: asyncpg.Connection, timeline_id: int) -> None:
    """Persist the observed timeline (singleton UPSERT)."""
    await conn.execute(
        """
        INSERT INTO pg_search_guardian_state (id, last_timeline_id, observed_at)
        VALUES (1, $1, NOW())
        ON CONFLICT (id) DO UPDATE
            SET last_timeline_id = EXCLUDED.last_timeline_id,
                observed_at      = EXCLUDED.observed_at
        """,
        timeline_id,
    )


# Indexes that MUST exist for the product to be whole, mapped to their table.
# This is deliberately its own list rather than the drop-allowlist above: the
# allowlist bounds what an unattended job may DESTROY (and includes non-
# pg_search indexes like the HNSW one, which replicate fine); this names what
# the guardian should MISS when it is gone. Today that is exactly the BM25
# index -- the one whose absence makes lexical search silently return nothing
# while every tick reads `broken_count: 0`.
REQUIRED_PG_SEARCH_INDEXES: dict[str, str] = {"idx_chunks_bm25_v2": "chunks"}


async def find_absent_required_indexes(conn: asyncpg.Connection) -> list[str]:
    """Required pg_search indexes that do not exist at all.

    The guardian's own repair produces this state: it DROPS a broken index,
    after which the table plans fine, BM25 quietly returns zero hits, and the
    broken-index detector -- which can only see indexes that exist -- reports
    a false all-clear. Observed live on 2026-08-26: the 04:14 tick correctly
    dropped the corrupted index, and every subsequent tick read
    `broken_count: 0` while lexical search was dead. Nothing prompted the
    rebuild for half an hour until a human went looking.

    A required index over a table that does not exist is NOT reported: on a
    fresh or partially-migrated database the missing piece is the table, and
    that is the migration chain's problem, not a search outage.
    """
    absent: list[str] = []
    for index_name, table_name in REQUIRED_PG_SEARCH_INDEXES.items():
        if await conn.fetchval("SELECT to_regclass($1)", index_name) is not None:
            continue
        if await conn.fetchval("SELECT to_regclass($1)", table_name) is None:
            continue
        absent.append(index_name)
    return absent


async def read_known_absent(conn: asyncpg.Connection) -> frozenset[str]:
    """The absences already reported, so alerts fire on transitions only.

    An absence persists for hours BY DESIGN -- rebuilds are attended -- so
    alerting on the state rather than the change would fire every minute for
    the whole window: the exact alert-storm shape the timeline marker's
    record-LAST discipline exists to avoid on the promotion path.
    """
    value = await conn.fetchval(
        "SELECT known_absent FROM pg_search_guardian_state WHERE id = 1"
    )
    # List-guarded, not just truthiness-guarded: a string is iterable and
    # every character passes an element check, so a malformed scalar here
    # would silently become a set of letters (the exact bug class the
    # lost_channels forwarding shipped with).
    if not isinstance(value, (list, tuple)):
        return frozenset()
    return frozenset(v for v in value if isinstance(v, str))


async def record_known_absent(conn: asyncpg.Connection, absent: frozenset[str]) -> None:
    """Persist the currently-known absences (singleton UPSERT, timeline kept).

    The INSERT arm needs a timeline value for a first-ever tick; 0 is the
    "never seen" sentinel `read_last_timeline` already treats as None-like
    (a first real timeline is always >= 1, so the first comparison records
    rather than alerts).
    """
    await conn.execute(
        """
        INSERT INTO pg_search_guardian_state (id, last_timeline_id, observed_at, known_absent)
        VALUES (1, 0, NOW(), $1)
        ON CONFLICT (id) DO UPDATE
            SET known_absent = EXCLUDED.known_absent,
                observed_at  = EXCLUDED.observed_at
        """,
        sorted(absent),
    )


# ---------------------------------------------------------------------------
# Unattended rebuild
# ---------------------------------------------------------------------------

# The DDL for every index in REQUIRED_PG_SEARCH_INDEXES, so the rebuild does not
# have to parse db/schema.sql at runtime.
#
# THIS IS A COPY, AND COPIES DRIFT. tests/test_pg_search_rebuild.py asserts that
# every statement here still matches the one schema.sql declares, normalised for
# whitespace and comments -- the same shape index_contracts.py uses, and for the
# same reason: a rebuild that silently recreates last month's index definition
# is worse than no rebuild, because the result LOOKS healthy.
#
# `IF NOT EXISTS` is deliberate. The rebuild only runs when the index is already
# known absent, but two ticks racing is not worth a crash.
REQUIRED_INDEX_DDL: dict[str, str] = {
    "idx_chunks_bm25_v2": """
        CREATE INDEX IF NOT EXISTS idx_chunks_bm25_v2
        ON chunks USING bm25 (
            chunk_id, content, title, customer_id, doc_id, kind,
            chunk_index, first_seen_version, last_seen_version, visibility
        )
        WITH (
            key_field=chunk_id,
            text_fields='{"title": {"tokenizer": {"type": "source_code"}}}'
        )
    """,
}

# Advisory lock key. Two rebuilds of the same index at once would each hold a
# SHARE lock and build a 500MB+ index; the second is pure waste at best. The
# CronJob's concurrencyPolicy already prevents overlap WITHIN one cluster, but
# nothing stops a human running the attended recipe while a tick is mid-build --
# and that is the case worth defending against, because it is the case where
# somebody is watching the wrong terminal.
REBUILD_ADVISORY_LOCK_KEY = 0x7042_5F42_4D32_3501

# Plain CREATE INDEX, never CONCURRENTLY. CIC on pg_search 0.23.4 dies at 99%
# with an uninitialized tablespace OID, and `lock_timeout` kills its
# wait-for-writers phase outright. Plain build takes an ACCESS SHARE-blocking
# ShareLock: readers are unaffected, writers to the table queue.
#
# 256MB, not the instance default of 512MB, and no parallel workers. This runs
# on an instance whose whole container limit is 4Gi with 2GB of that pinned in
# shared_buffers -- the budget that produced two OOM group kills on 2026-08-28
# and 2026-08-29. A rebuild that takes six minutes and finishes beats one that
# takes three and kills the primary.
REBUILD_MAINTENANCE_WORK_MEM = "256MB"
REBUILD_PARALLEL_WORKERS = 0

# Ceiling on the wait for the table lock. Same reasoning as DROP_LOCK_TIMEOUT
# but far longer: the drop is a catalog flick that must never queue, while the
# build is the thing we actually came to do, so it is worth waiting out a
# transient ingestion batch. Still bounded -- a skip costs one tick, a pinned
# lock request blocks every writer arriving behind it.
REBUILD_LOCK_TIMEOUT = "60s"


async def rebuild_absent_index(
    conn: asyncpg.Connection,
    index_name: str,
    *,
    dry_run: bool = False,
) -> bool:
    """Rebuild one absent required index. True if built, False if it skipped.

    Returns False rather than raising for the two conditions that are a normal
    part of a busy database -- the lock was unavailable, or the index already
    exists -- because both are resolved by the next tick and neither is worth a
    red CronJob.

    The identifier and the DDL are interpolated because DDL takes no bind
    parameters. Safe for the same reason `drop_broken_index` is: `index_name`
    reaches here only from REQUIRED_PG_SEARCH_INDEXES, and the statement itself
    comes from REQUIRED_INDEX_DDL. Neither carries a value derived from user
    input or from the database.
    """
    if index_name not in REQUIRED_INDEX_DDL:
        raise ValueError(f"no rebuild DDL declared for index: {index_name!r}")

    ddl = REQUIRED_INDEX_DDL[index_name]
    if dry_run:
        log.info("rebuild.would_build", index=index_name)
        return False

    try:
        # NOT in a transaction block, and not by accident: a 500MB index build
        # inside an explicit transaction holds its locks and its WAL for the
        # whole duration with nothing to gain. Each SET is session-scoped and
        # this connection is used for nothing else.
        await conn.execute(f"SET lock_timeout = '{REBUILD_LOCK_TIMEOUT}'")
        await conn.execute(f"SET maintenance_work_mem = '{REBUILD_MAINTENANCE_WORK_MEM}'")
        await conn.execute(
            f"SET max_parallel_maintenance_workers = {REBUILD_PARALLEL_WORKERS}"
        )
        # The build outlives any sane statement_timeout. Unset for this session
        # only -- the guardian's own connection, never a request-path one.
        await conn.execute("SET statement_timeout = 0")
        await conn.execute(ddl)
        return True
    except asyncpg.LockNotAvailableError:
        log.warning(
            "rebuild.skipped_lock_timeout",
            index=index_name,
            reason="table busy with writers; next tick retries",
        )
        return False
