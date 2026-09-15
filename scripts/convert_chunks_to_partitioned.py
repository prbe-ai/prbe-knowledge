"""Convert `chunks` into a table PARTITIONED BY LIST (customer_id). ATTENDED.

    .venv/bin/python -m scripts.convert_chunks_to_partitioned --status
    .venv/bin/python -m scripts.convert_chunks_to_partitioned --plan
    .venv/bin/python -m scripts.convert_chunks_to_partitioned --run
    .venv/bin/python -m scripts.convert_chunks_to_partitioned --run --only anthrogen

WHY THIS IS NOT A MIGRATION
---------------------------
`charts/research-os/templates/engine-migrate-job.yaml` runs `alembic upgrade
head` as a `pre-upgrade` helm hook with `activeDeadlineSeconds: 1800` and a
`backoffLimit`, unattended, on both planes, from the PINNED engine image. This
conversion rebuilds a 2,716 MB heap and ~23 GB of indexes (HNSW 4,831 MB, GIN
tsv 2,560 MB, BM25 1,104 MB). It cannot finish in 30 minutes, and a job that is
killed at minute 30 and then RETRIED is how a half-converted `chunks` becomes
permanent on a customer plane.

So: a human runs this, per plane, watching it. Alembic is stamped afterwards by
a separate migration that only records the state this script produced.

WHY IT IS SAFE TO STOP AT ANY POINT
------------------------------------
Every step is idempotent and checks the world before acting, so `--run` can be
killed and re-run and converges to the same end state:

    step 1  create `chunks_part` (partitioned, correct index set)  -> skipped if present
    step 2  create one partition per tenant + DEFAULT              -> skipped per tenant
    step 3  copy rows, one tenant at a time, in batches            -> resumes per tenant
    step 4  build indexes on each partition                        -> skipped if present
    step 5  swap: rename old -> chunks_old, new -> chunks          -> single txn, lock_timeout
    step 6  verify counts                                          -> read-only

Until step 5 the live `chunks` is untouched and serving. Step 5 is the only
moment anything changes for a reader, and it is one transaction: it either
happens or it does not.

    live chunks ──(read)──► chunks_part ──┐
         │                                │ step 5, one txn, ACCESS EXCLUSIVE
         │                                ▼
         └────────────────────► RENAME chunks -> chunks_old
                                RENAME chunks_part -> chunks

ROLLBACK is `chunks_old` -- kept, not dropped. Dropping it is a separate,
deliberate, later command once the new table has served real traffic.

THE WRITE FENCE
---------------
Rows written to the OLD table after this script copied a tenant would be lost at
the swap. `--run` therefore refuses to proceed unless ingestion is stopped
(`--i-have-stopped-ingestion`), and step 6 re-counts both tables and refuses the
swap on any drift it did not expect. This is the one thing the operator must get
right, and the script will not do it for them.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import asyncpg

OLD_TABLE = "chunks"
NEW_TABLE = "chunks_part"
RETIRED_TABLE = "chunks_old"
BATCH_ROWS = 50_000

#: Index builds are the bulk of the wall clock. pgvector builds an HNSW graph in
#: memory only while it fits; prod's default `maintenance_work_mem` is 512MB and
#: a 340k-row partition of 6 KB vectors does not fit, which silently drops it to
#: a much slower on-disk build. Raised per session, never globally.
BUILD_MAINTENANCE_WORK_MEM = "4GB"
BUILD_PARALLEL_WORKERS = 4

#: Every DDL step waits this long for its lock, then gives up and is retried.
#: Prod runs with `lock_timeout = 0`, so without this the swap would queue behind
#: a long search holding ACCESS SHARE and then block every reader behind itself.
DDL_LOCK_TIMEOUT = "5s"
SWAP_ATTEMPTS = 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def _tenants(conn: asyncpg.Connection) -> list[str]:
    rows = await conn.fetch(
        f"SELECT DISTINCT customer_id FROM {OLD_TABLE} ORDER BY 1"
    )
    return [r["customer_id"] for r in rows]


async def _table_kind(conn: asyncpg.Connection, name: str) -> str | None:
    return await conn.fetchval(
        "SELECT relkind FROM pg_class WHERE oid = to_regclass($1)", name
    )


async def _column_list(conn: asyncpg.Connection, table: str) -> list[str]:
    """Insertable columns: everything except GENERATED ones.

    `chunks.content_tsv` is `GENERATED ALWAYS`, and naming it in an INSERT is a
    hard error (`cannot insert a non-DEFAULT value into column "content_tsv"`),
    not a warning. It is recomputed on the target anyway, so excluding it is
    both necessary and lossless. `attgenerated = ''` is the filter.
    """
    rows = await conn.fetch(
        """
        SELECT attname FROM pg_attribute
        WHERE attrelid = to_regclass($1) AND attnum > 0 AND NOT attisdropped
          AND attgenerated = ''
        ORDER BY attnum
        """,
        table,
    )
    return [r["attname"] for r in rows]


async def status(conn: asyncpg.Connection) -> None:
    old_kind = await _table_kind(conn, OLD_TABLE)
    new_kind = await _table_kind(conn, NEW_TABLE)
    log(f"{OLD_TABLE}: relkind={old_kind!r}  (p = already partitioned, r = plain)")
    log(f"{NEW_TABLE}: relkind={new_kind!r}")
    if old_kind == "p":
        log("ALREADY CONVERTED. Nothing to do.")
        return
    total = await conn.fetchval(f"SELECT count(*) FROM {OLD_TABLE}")
    log(f"{OLD_TABLE} rows: {total:,}")
    for t in await _tenants(conn):
        n = await conn.fetchval(
            f"SELECT count(*) FROM {OLD_TABLE} WHERE customer_id = $1", t
        )
        copied = 0
        if new_kind == "p":
            copied = (
                await conn.fetchval(
                    f"SELECT count(*) FROM {NEW_TABLE} WHERE customer_id = $1", t
                )
                or 0
            )
        flag = "" if copied == n else "  <-- incomplete"
        log(f"  {t:28} old={n:>9,}  new={copied:>9,}{flag}")


async def _create_parent(conn: asyncpg.Connection) -> None:
    if await _table_kind(conn, NEW_TABLE) is not None:
        log(f"step 1: {NEW_TABLE} already exists, skipping")
        return
    log(f"step 1: creating {NEW_TABLE} (partitioned)")
    # LIKE carries columns, defaults, NOT NULLs, checks and storage. Indexes and
    # constraints are declared explicitly below so the new table gets the
    # POST-partition index set (tenant-qualified uniques, no chunks_chunk_id_unique)
    # rather than a copy of the old one, which a partitioned table would reject.
    await conn.execute(
        f"""
        CREATE TABLE {NEW_TABLE} (
            LIKE {OLD_TABLE}
                INCLUDING DEFAULTS
                INCLUDING CONSTRAINTS
                INCLUDING STORAGE
                INCLUDING COMMENTS
                INCLUDING GENERATED
        ) PARTITION BY LIST (customer_id)
        """
    )
    # NAMES ARE DATABASE-WIDE, so the new table cannot reuse the live table's
    # index names while the live table still exists. Everything is built under
    # the `chunks_part%` prefix and renamed to the canonical names inside the
    # swap transaction -- otherwise the converted table would permanently carry
    # `chunks_part_*` index names, which `db/schema.sql`, the pg_search guardian
    # (`REQUIRED_PG_SEARCH_INDEXES`) and the IndexContracts all name explicitly.
    await conn.execute(
        f"ALTER TABLE {NEW_TABLE} ADD CONSTRAINT {NEW_TABLE}_pkey "
        "PRIMARY KEY (customer_id, chunk_id)"
    )
    await conn.execute(
        f"ALTER TABLE {NEW_TABLE} "
        f"ADD CONSTRAINT {NEW_TABLE}_customer_doc_hash_key "
        "UNIQUE (customer_id, doc_id, content_hash)"
    )
    await conn.execute(
        f"ALTER TABLE {NEW_TABLE} ADD FOREIGN KEY (customer_id) "
        "REFERENCES customers(customer_id) ON DELETE CASCADE"
    )
    await conn.execute(f"ALTER TABLE {NEW_TABLE} ENABLE ROW LEVEL SECURITY")
    await conn.execute(f"ALTER TABLE {NEW_TABLE} FORCE ROW LEVEL SECURITY")
    await conn.execute(
        f"CREATE POLICY tenant_isolation ON {NEW_TABLE} "
        "USING (customer_id = current_setting('app.current_customer_id', true))"
    )
    log(f"step 1: {NEW_TABLE} created")


async def _partition_name(customer_id: str) -> str:
    from engine.shared.partitions import partition_name_for

    return partition_name_for(customer_id)


async def _create_partitions(conn: asyncpg.Connection, tenants: list[str]) -> None:
    default_name = f"{NEW_TABLE}_default"
    if await _table_kind(conn, default_name) is None:
        await conn.execute(
            f"CREATE TABLE {default_name} PARTITION OF {NEW_TABLE} DEFAULT"
        )
        log(f"step 2: DEFAULT partition {default_name} created")
    for t in tenants:
        part = await _partition_name(t)
        if await _table_kind(conn, part) is not None:
            continue
        literal = t.replace("'", "''")
        # CREATE + ATTACH, not `PARTITION OF`: the latter takes ACCESS EXCLUSIVE
        # on the parent. Irrelevant here (nothing reads the new table yet) but
        # kept identical to the runtime path so there is one way this is done.
        await conn.execute(f'CREATE TABLE "{part}" (LIKE {NEW_TABLE} INCLUDING ALL)')
        await conn.execute(
            f"ALTER TABLE {NEW_TABLE} ATTACH PARTITION \"{part}\" "
            f"FOR VALUES IN ('{literal}')"
        )
        await conn.execute(f'ALTER TABLE "{part}" ENABLE ROW LEVEL SECURITY')
        await conn.execute(f'ALTER TABLE "{part}" FORCE ROW LEVEL SECURITY')
        await conn.execute(
            f'CREATE POLICY tenant_isolation ON "{part}" '
            "USING (customer_id = current_setting('app.current_customer_id', true))"
        )
        log(f"step 2: partition {part} for {t}")


async def _copy_tenant(conn: asyncpg.Connection, customer_id: str) -> int:
    """Copy one tenant's rows, resuming from whatever is already there.

    Keyed on `chunk_id` rather than an offset: an offset resumes wrongly if the
    source shifts, and `(customer_id, chunk_id)` is the primary key so it is
    both unique and indexed on each side.
    """
    cols = await _column_list(conn, OLD_TABLE)
    collist = ", ".join(f'"{c}"' for c in cols)
    total = await conn.fetchval(
        f"SELECT count(*) FROM {OLD_TABLE} WHERE customer_id = $1", customer_id
    )
    copied = 0
    while True:
        cursor = await conn.fetchval(
            f"SELECT max(chunk_id) FROM {NEW_TABLE} WHERE customer_id = $1",
            customer_id,
        )
        moved = await conn.fetchval(
            f"""
            WITH batch AS (
                SELECT {collist} FROM {OLD_TABLE}
                WHERE customer_id = $1
                  AND ($2::text IS NULL OR chunk_id > $2)
                ORDER BY chunk_id
                LIMIT {BATCH_ROWS}
            ), ins AS (
                INSERT INTO {NEW_TABLE} ({collist})
                SELECT {collist} FROM batch
                ON CONFLICT (customer_id, chunk_id) DO NOTHING
                RETURNING 1
            )
            SELECT count(*) FROM ins
            """,
            customer_id,
            cursor,
        )
        if not moved:
            break
        copied += moved
        log(f"    {customer_id}: {copied:,}/{total:,}")
    return copied


async def _build_partition_indexes(conn: asyncpg.Connection) -> None:
    """Create the non-constraint indexes on the PARENT once rows are in place.

    Building after the copy rather than before is the whole reason this step is
    separate: an index maintained row-by-row during a 1.6M-row copy costs far
    more than one built in bulk at the end. Creating on the parent cascades to
    every partition, and pgvector/pg_search both support that (verified on
    paradedb 0.23.4-pg16).
    """
    await conn.execute("SET statement_timeout = 0")
    await conn.execute(f"SET maintenance_work_mem = '{BUILD_MAINTENANCE_WORK_MEM}'")
    await conn.execute(
        f"SET max_parallel_maintenance_workers = {BUILD_PARALLEL_WORKERS}"
    )
    rows = await conn.fetch(
        """
        SELECT indexname, indexdef FROM pg_indexes
        WHERE tablename = $1
          AND indexname NOT IN ('chunks_pkey', 'chunks_chunk_id_unique',
                                'chunks_doc_id_content_hash_key')
        ORDER BY indexname
        """,
        OLD_TABLE,
    )
    for r in rows:
        new_name = r["indexname"].replace("chunks", NEW_TABLE, 1)
        if await conn.fetchval("SELECT to_regclass($1)", new_name) is not None:
            continue
        # CONCURRENTLY is unavailable on a partitioned table and unnecessary:
        # nothing reads this table until the swap.
        ddl = (
            r["indexdef"]
            .replace(f" INDEX {r['indexname']} ", f" INDEX {new_name} ")
            .replace(f" ON public.{OLD_TABLE} ", f" ON public.{NEW_TABLE} ")
            .replace(f" ON {OLD_TABLE} ", f" ON {NEW_TABLE} ")
        )
        log(f"step 4: building {new_name}")
        started = time.time()
        await conn.execute(ddl)
        log(f"step 4: {new_name} done in {time.time() - started:.0f}s")


async def _verify(conn: asyncpg.Connection) -> bool:
    old_total = await conn.fetchval(f"SELECT count(*) FROM {OLD_TABLE}")
    new_total = await conn.fetchval(f"SELECT count(*) FROM {NEW_TABLE}")
    log(f"step 6: {OLD_TABLE}={old_total:,}  {NEW_TABLE}={new_total:,}")
    if old_total != new_total:
        log("step 6: COUNT MISMATCH -- refusing to swap.")
        return False
    in_default = await conn.fetchval(
        f"SELECT count(*) FROM ONLY {NEW_TABLE}_default"
    )
    if in_default:
        log(f"step 6: {in_default:,} rows landed in DEFAULT -- a tenant has no "
            "partition. Refusing to swap.")
        return False
    return True


async def _rename_indexes(conn: asyncpg.Connection, table: str, old: str, new: str) -> None:
    """Rename every index on `table` whose name starts with `old` to start with `new`.

    Partitioned parents and their children are both renamed: the children carry
    the data and their names are what EXPLAIN shows, while the parent's name is
    what the guardian and the index contracts declare.
    """
    rows = await conn.fetch(
        """
        SELECT c.relname
        FROM pg_class c
        JOIN pg_index i ON i.indexrelid = c.oid
        WHERE i.indrelid = to_regclass($1) AND c.relname LIKE $2
        """,
        table,
        f"%{old}%",
    )
    for r in rows:
        src = r["relname"]
        # SUBSTRING replace, not a prefix strip. Index names here come in two
        # shapes -- `chunks_pkey` (constraint-derived, prefixed) and
        # `idx_chunks_content_tsv` (declared, INFIXED) -- and a prefix-only
        # rename silently leaves every `idx_*` index carrying the build-time
        # name. That is not cosmetic: `db/schema.sql`, the pg_search guardian's
        # `REQUIRED_PG_SEARCH_INDEXES`, and the IndexContracts all name these
        # indexes literally, so a missed rename means the guardian reports a
        # required index absent on a healthy database, forever.
        dst = src.replace(old, new, 1)
        if dst == src:
            continue
        if await conn.fetchval("SELECT to_regclass($1)", dst) is not None:
            continue
        await conn.execute(f'ALTER INDEX "{src}" RENAME TO "{dst}"')


async def _swap(conn: asyncpg.Connection) -> None:
    """Rename old out and new in, in ONE transaction, under a lock timeout.

    Indexes move with their tables but keep their names, and index names are
    database-wide. So the transaction renames BOTH: the retiring table's indexes
    get an `_old` marker to free the canonical names, and the incoming table's
    `chunks_part%` indexes take them. Anything that names an index -- the
    pg_search guardian, `db/schema.sql`, the IndexContracts -- is then correct
    without a follow-up.

    Retried rather than waited out: prod runs `lock_timeout = 0`, so a bare
    ALTER would queue behind a long search holding ACCESS SHARE and then block
    every query that arrived after it. Failing fast and retrying keeps the
    blocking window bounded by the timeout instead of by the slowest search.
    """
    for attempt in range(1, SWAP_ATTEMPTS + 1):
        try:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL lock_timeout = '{DDL_LOCK_TIMEOUT}'")
                await conn.execute(
                    f"ALTER TABLE {OLD_TABLE} RENAME TO {RETIRED_TABLE}"
                )
                await _rename_indexes(
                    conn, RETIRED_TABLE, OLD_TABLE, RETIRED_TABLE
                )
                await conn.execute(f"ALTER TABLE {NEW_TABLE} RENAME TO {OLD_TABLE}")
                await _rename_indexes(conn, OLD_TABLE, NEW_TABLE, OLD_TABLE)
            log(f"step 5: SWAPPED on attempt {attempt}. Old table kept as "
                f"{RETIRED_TABLE} -- drop it deliberately, later.")
            return
        except asyncpg.LockNotAvailableError:
            log(f"step 5: lock busy, retry {attempt}/{SWAP_ATTEMPTS}")
            await asyncio.sleep(2)
    raise RuntimeError("could not acquire the swap lock; nothing was changed")


async def run(conn: asyncpg.Connection, only: str | None) -> int:
    if await _table_kind(conn, OLD_TABLE) == "p":
        log("already partitioned; nothing to do")
        return 0
    tenants = await _tenants(conn)
    if only:
        tenants = [t for t in tenants if t == only]
        if not tenants:
            log(f"no such tenant in {OLD_TABLE}: {only}")
            return 1

    await _create_parent(conn)
    await _create_partitions(conn, tenants)

    log("step 3: copying rows")
    # Smallest tenants first: they finish fast, so a run that has to be stopped
    # still leaves most TENANTS complete rather than most ROWS.
    sized = sorted(
        [
            (
                await conn.fetchval(
                    f"SELECT count(*) FROM {OLD_TABLE} WHERE customer_id = $1", t
                ),
                t,
            )
            for t in tenants
        ]
    )
    for n, t in sized:
        log(f"  {t} ({n:,} rows)")
        await _copy_tenant(conn, t)

    await _build_partition_indexes(conn)

    if only:
        log("partial run (--only): stopping before the swap")
        return 0
    if not await _verify(conn):
        return 1
    await _swap(conn)
    await conn.execute("SET statement_timeout = 0")
    log("post-swap: ANALYZE chunks (the partitioned parent is never autoanalyzed)")
    await conn.execute(f"ANALYZE {OLD_TABLE}")
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true", help="report progress, change nothing")
    ap.add_argument("--plan", action="store_true", help="alias for --status")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", help="copy a single tenant and stop before the swap")
    ap.add_argument(
        "--i-have-stopped-ingestion",
        action="store_true",
        help="required for --run: rows written to the old table during the copy "
        "are LOST at the swap",
    )
    args = ap.parse_args()

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is required", file=sys.stderr)
        return 2
    conn = await asyncpg.connect(dsn)
    try:
        if args.run:
            if not args.i_have_stopped_ingestion:
                print(
                    "Refusing to run: pass --i-have-stopped-ingestion.\n"
                    "Rows written to the old table after their tenant is copied "
                    "are silently lost at the swap.",
                    file=sys.stderr,
                )
                return 2
            return await run(conn, args.only)
        await status(conn)
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
