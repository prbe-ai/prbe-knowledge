"""Swap the BM25 index from v2 to v3 (adds `project_id` as a fast field).

ATTENDED. Run this in a window you chose, watching the output. It is not a
CronJob and must not become one.

WHY IT IS NOT AUTOMATIC
-----------------------
pg_search permits exactly ONE `USING bm25` index per relation, so moving
generations is DROP then CREATE, and between those two statements the tenant
has NO lexical search at all. On the research primary the build took 226s on
2026-08-29 against a table that has only grown since. That is a real outage
window, deliberately taken, with somebody watching -- not something a tick
does at 3am because a column appeared.

The guardian (`engine/shared/pg_search_guardian.py`) knows about both
generations and will REFUSE to build one while the other stands, precisely so
no unattended path can do what this script does.

WHAT THE STANDBY DOES
---------------------
Nothing good, and nothing new. pg_search Community does not replicate BM25
contents, so the standby ends up with a 0-byte index it believes is valid --
the same state every rebuild leaves and the same one the guardian repairs
after a promotion. This swap re-arms that trap exactly once more. It does not
make it worse, and it is NOT a reason to upgrade pg_search: 0.25.6 FATALs the
Community standby on the INIT_INDEX WAL record and takes the cluster's HA with
it. That measurement is in `cron_pg_search_rebuild.py` and it has not changed.

ORDER OF OPERATIONS
-------------------
Migration 0131 must have run first: it adds `chunks.project_id`, backfills it
and installs the triggers that keep it true. Creating v3 without the column
fails, and creating it while the column is still NULL everywhere would index a
column of nothing -- a scoped query would then match zero rows INDEX-side and
return an empty result that looks like a legitimately empty project. This
script checks both before it touches anything.

    python -m scripts.swap_bm25_index [--dry-run]

EXIT CODES
----------
  0  swapped, or nothing to do (already on v3).
  1  refused (preconditions unmet) or the build failed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from engine.shared.db import close_pool, get_pool, init_pool
from engine.shared.logging import configure_logging, get_logger
from engine.shared.pg_search_guardian import (
    BM25_INDEX_V2,
    BM25_INDEX_V3,
    REBUILD_ADVISORY_LOCK_KEY,
    REBUILD_COMMAND_TIMEOUT_SECONDS,
    REBUILD_MAINTENANCE_WORK_MEM,
    REBUILD_PARALLEL_WORKERS,
    REQUIRED_INDEX_DDL,
    find_invalid_index_debris,
)

log = get_logger(__name__)

#: Below this, the backfill did not really run. A database with the column but
#: no values in it would build a v3 index over nothing, and every scoped query
#: would then return zero rows index-side -- indistinguishable, to a caller,
#: from a project that genuinely has no documents. Refusing is the only safe
#: answer; the fix is to run 0131's backfill, not to lower this number.
_MIN_BACKFILLED_ROWS = 1


async def _preconditions(conn) -> str | None:
    """None when it is safe to swap, else the reason to refuse."""
    has_column = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
             WHERE table_name = 'chunks' AND column_name = 'project_id'
        )
        """
    )
    if not has_column:
        return "chunks.project_id does not exist -- run migration 0131 first"

    if await conn.fetchval("SELECT to_regclass($1)", BM25_INDEX_V3) is not None:
        return None  # already swapped; caller treats this as nothing-to-do

    if await conn.fetchval("SELECT to_regclass($1)", BM25_INDEX_V2) is None:
        return (
            f"{BM25_INDEX_V2} is absent -- there is nothing to swap FROM. "
            "Let the guardian's rebuild job build v3 instead; with the column "
            "present it already targets v3."
        )

    debris = await find_invalid_index_debris(conn)
    if debris:
        return (
            f"invalid index debris present ({sorted(debris)}) -- a build is in "
            "flight or one failed. Resolve by hand; this will not race it."
        )

    backfilled = await conn.fetchval(
        "SELECT count(*) FROM chunks WHERE project_id IS NOT NULL"
    )
    if backfilled < _MIN_BACKFILLED_ROWS:
        return (
            "chunks.project_id is NULL on every row -- 0131's backfill has not "
            "run. Building v3 now would index an empty column and every scoped "
            "query would return nothing."
        )
    return None


async def run_once(*, dry_run: bool = False) -> int:
    pool = get_pool()
    async with pool.acquire() as conn:
        got_lock = await conn.fetchval(
            "SELECT pg_try_advisory_lock($1)", REBUILD_ADVISORY_LOCK_KEY
        )
        if not got_lock:
            log.error("swap.refused_locked", reason="a rebuild holds the lock")
            return 1

        if await conn.fetchval("SELECT to_regclass($1)", BM25_INDEX_V3) is not None:
            log.info("swap.nothing_to_do", reason=f"{BM25_INDEX_V3} already exists")
            return 0

        refusal = await _preconditions(conn)
        if refusal is not None:
            log.error("swap.refused", reason=refusal)
            return 1

        if dry_run:
            log.info(
                "swap.would_swap",
                drop=BM25_INDEX_V2,
                create=BM25_INDEX_V3,
                note="lexical search is DOWN between the two statements",
            )
            return 0

        await conn.execute(f"SET maintenance_work_mem = '{REBUILD_MAINTENANCE_WORK_MEM}'")
        await conn.execute(
            f"SET max_parallel_maintenance_workers = {REBUILD_PARALLEL_WORKERS}"
        )
        await conn.execute("SET statement_timeout = 0")

        # NOT in a transaction. Wrapping DROP + CREATE together would hold the
        # lock and the WAL for the whole build to buy an atomicity nobody can
        # use: a reader inside the transaction still sees no index, and a
        # rollback after a failed CREATE leaves v2 back in place only if the
        # DROP has not already been replicated -- which it has. The honest
        # shape is two statements and a loud log line between them saying
        # lexical search is down.
        t0 = time.perf_counter()
        log.warning(
            "swap.dropping",
            index=BM25_INDEX_V2,
            note="lexical search is DOWN from here until the build completes",
        )
        await conn.execute(f"DROP INDEX IF EXISTS {BM25_INDEX_V2}")

        log.info("swap.building", index=BM25_INDEX_V3)
        try:
            await conn.execute(
                REQUIRED_INDEX_DDL[BM25_INDEX_V3],
                timeout=REBUILD_COMMAND_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            # The window is open and the build failed. Say so at the top of the
            # log rather than letting a traceback bury it: somebody is watching
            # this output and their next move is to rebuild, by hand, now.
            log.error(
                "swap.build_failed_search_is_down",
                index=BM25_INDEX_V3,
                error=str(exc) or type(exc).__name__,
                next_step=(
                    "lexical search has NO index. Rebuild immediately: the "
                    "guardian's rebuild job targets v3 once the column exists, "
                    "or run the DDL from REQUIRED_INDEX_DDL by hand."
                ),
            )
            return 1

        log.info(
            "swap.done",
            index=BM25_INDEX_V3,
            elapsed_s=round(time.perf_counter() - t0, 1),
            note="the standby now holds a 0-byte copy it believes is valid; "
            "the guardian repairs that on promotion, as it always has",
        )
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_logging()
    await init_pool()
    try:
        return await run_once(dry_run=args.dry_run)
    finally:
        await close_pool()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
