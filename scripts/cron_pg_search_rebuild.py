"""Rebuild the BM25 index the guardian dropped, without waiting for a human.

WHY THIS IS A SEPARATE JOB
--------------------------
`cron_pg_search_guardian` runs every minute with `activeDeadlineSeconds: 240`.
The rebuild it exists to trigger took 226s on 2026-08-29 -- fourteen seconds of
margin, against a table that only grows. Putting the build inside that tick
means the deadline eventually kills a half-finished CREATE INDEX, and a killed
build leaves `indisvalid = false` debris that the guardian deliberately refuses
to touch (it cannot tell debris from a human's in-flight rebuild). So the build
gets its own schedule, its own deadline and its own exit-code semantics, and the
guardian goes back to being a thing that finishes in seconds.

WHAT CHANGED THE ANSWER
-----------------------
The guardian's docstring says rebuilding is "the dangerous half" because
`CREATE INDEX` emits WAL a Community standby's replay may refuse. Measured on
2026-08-29 against the live cluster: a 577MB build left replication `streaming`
at 2.6ms replay lag, sent_lsn == replay_lsn. The observed failure mode is not
"the standby refuses replay" -- it is "the standby ends up with a 0-byte index
it believes is valid", which is precisely the state the guardian already
detects and repairs on promotion. That is a materially weaker objection than
the one the docstring records, and it is what makes this job defensible.

WHAT HAS NOT CHANGED
--------------------
The rebuild does NOT fix the standby. pg_search Community does not replicate
BM25 contents (streaming-replica reads are an Enterprise feature), so every
rebuild re-arms the trap for the next failover. This job automates the
treadmill; it does not stop it.

DO NOT "FIX" THAT BY UPGRADING pg_search. That was the obvious next move and it
is actively harmful on Community with a physical standby. Measured 2026-08-29,
primary + streaming standby in Docker, index built while the standby streamed:

  0.23.4 (ours)  standby index 0 bytes, standby STAYS UP, BM25 broken after
                 promotion -- the degradation this job repairs.
  0.25.6         standby's startup process takes the pg_search/INIT_INDEX WAL
                 record and FATALs:
                   "replicas are not supported on community and require
                    paradedb enterprise"
                 then "shutting down due to startup process failure". The
                 standby is GONE, and with it the cluster's HA.

Combined with this job the newer version is worse still: every automatic
rebuild would kill the standby. paradedb#4901 ports WAL integration for
SINGLE-NODE crash recovery; halting a Community standby is its intended
behaviour, not an edge case. The supported fix is ParadeDB Enterprise.

Cheaper mitigation that does work on Community, same measurement run: a standby
RE-CLONED (pg_basebackup) after the index exists receives a full-size, working
copy and serves BM25 correctly once promoted. So recycling the replica after a
rebuild converts the next failover from degraded to clean.

DELIBERATELY SILENT ON SUCCESS
------------------------------
No new event names. The guardian's next tick sees the index return and fires the
`kb_pg_search_index_restored` it already owns, which is already wired into the
PostHog destination -- an event added here but not to that destination's filter
chain renders as an unrecognised alert, so the cheapest correct thing is to emit
nothing and let the existing closing alert close the loop.

EXIT CODES
----------
  0  nothing to do, built successfully, or skipped on a busy table.
  1  a database operation failed.

    python -m scripts.cron_pg_search_rebuild [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from engine.shared.db import close_pool, get_pool, init_pool
from engine.shared.logging import configure_logging, get_logger
from engine.shared.pg_search_guardian import (
    REBUILD_ADVISORY_LOCK_KEY,
    find_absent_required_indexes,
    find_invalid_index_debris,
    rebuild_absent_index,
)

log = get_logger(__name__)


async def run_once(*, dry_run: bool = False) -> int:
    """One tick. Returns the process exit code."""
    pool = get_pool()
    async with pool.acquire() as conn:
        # Session-scoped advisory lock, taken before anything else and released
        # when the connection closes. `try_` rather than a blocking acquire: if
        # another rebuild holds it, the right move is to exit immediately, not
        # to queue a second builder behind the first.
        got_lock = await conn.fetchval(
            "SELECT pg_try_advisory_lock($1)", REBUILD_ADVISORY_LOCK_KEY
        )
        if not got_lock:
            log.info("rebuild.skipped_locked", reason="another rebuild holds the lock")
            return 0

        # Debris means somebody -- or something -- has a build in flight or a
        # failed one on the floor. Either way this job must not start a second
        # one under the same name: CREATE INDEX would fail on the name clash,
        # and if it did not, it would be racing a human. Alert-only is the
        # guardian's stance on debris and it is the right stance here too.
        debris = await find_invalid_index_debris(conn)
        if debris:
            log.warning(
                "rebuild.skipped_debris",
                indexes=sorted(debris),
                reason="invalid index present; a build is in flight or failed. "
                "Resolve by hand -- this job will not drop a human's work.",
            )
            return 0

        absent = await find_absent_required_indexes(conn)
        if not absent:
            log.info("rebuild.nothing_to_do")
            return 0

        built: list[str] = []
        for index_name in sorted(absent):
            log.info("rebuild.starting", index=index_name)
            if await rebuild_absent_index(conn, index_name, dry_run=dry_run):
                built.append(index_name)
                log.info("rebuild.built", index=index_name)

    if built:
        # No alert on purpose -- see the module docstring. The guardian's next
        # tick fires kb_pg_search_index_restored, which is the event the Slack
        # destination already knows how to render.
        log.info("rebuild.done", indexes=built)
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and log, build nothing.",
    )
    args = parser.parse_args()

    configure_logging()
    await init_pool()
    try:
        return await run_once(dry_run=args.dry_run)
    except Exception as exc:
        # Nonzero so the CronJob history shows red. A failed rebuild leaves
        # search degraded, which is exactly the state somebody has to know about.
        log.error("rebuild.failed", error=str(exc), exc_info=True)
        return 1
    finally:
        await close_pool()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
