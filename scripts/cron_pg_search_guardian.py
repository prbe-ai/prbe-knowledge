"""Keep a failover from taking the database down with the search index.

Runs every minute. Each tick:

  1. Reads the Postgres timeline ID. A change since last tick means a promotion
     happened -- so ANALYZE (statistics do not replicate) and flag every
     pg_search index as suspect, because a promoted standby's index is either
     empty or silently stale (see engine/shared/pg_search_guardian.py).
  2. Finds allowlisted pg_search indexes that are valid, 0 bytes, and on a
     nonempty table, and DROPS them. That is the repair: while such an index
     exists, EVERY statement that plans against its table raises XX001 --
     including ordinary lookups that have nothing to do with search.
  3. Alerts, so a human knows to rebuild.

WHAT "SUCCESS" MEANS HERE
-------------------------
Search comes back DEGRADED, not whole: vector + graph + exact serve, BM25 does
not. That is the intended end state for an unattended job. Rebuilding the index
is the dangerous half -- `CREATE INDEX` emits WAL a Community standby's replay
may refuse, and CONCURRENTLY can wedge behind old transactions for hours (this
project lost two hours to exactly that on 2026-08-16, migration 0105) -- so a
human does it in an attended window.

EXIT CODES
----------
  0  nothing to do, or a repair succeeded.
  1  a database operation failed.

Alerting failures do NOT affect the exit code: the CronJob's red history is the
fallback signal for when PostHog is the thing that is down, so it has to mean
"the database work failed" and nothing else.

    python -m scripts.cron_pg_search_guardian [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from engine.shared.db import close_pool, get_pool, init_pool
from engine.shared.logging import configure_logging, get_logger
from engine.shared.ops_alert import capture
from engine.shared.pg_search_guardian import (
    analyze_tables,
    current_timeline_id,
    drop_broken_index,
    find_absent_required_indexes,
    find_broken_pg_search_indexes,
    find_invalid_index_debris,
    read_known_absent,
    read_last_timeline,
    record_known_absent,
    record_timeline,
)

log = get_logger(__name__)

# Tables worth re-ANALYZE-ing after a promotion. Deliberately a short explicit
# list rather than "every table": ANALYZE on the whole database would be a
# surprise workload on an instance that has just been promoted and is already
# serving cold. These are the large ones on the retrieval hot path.
ANALYZE_AFTER_PROMOTION = ["chunks", "documents"]


async def run_once(*, dry_run: bool = False) -> int:
    """One tick. Returns the process exit code."""
    pool = get_pool()
    async with pool.acquire() as conn:
        timeline = await current_timeline_id(conn)
        last_timeline = await read_last_timeline(conn)
        # First tick after install: record and take no action. Treating an
        # absent row as a change would fire a promotion alert on every cluster
        # the day this ships, training everyone to ignore it.
        promoted = last_timeline is not None and timeline != last_timeline

        broken = await find_broken_pg_search_indexes(conn)
        debris = await find_invalid_index_debris(conn)
        # ABSENT is its own axis, because the guardian's own repair produces
        # it: after a drop, the broken-index detector -- which can only see
        # indexes that exist -- reads a false all-clear while BM25 quietly
        # returns nothing. Observed 2026-08-26: the 04:14 tick correctly
        # dropped the corrupted index and every later tick reported healthy
        # while lexical search was dead.
        absent = frozenset(await find_absent_required_indexes(conn))
        known_absent = await read_known_absent(conn)

        log.info(
            "guardian.tick",
            timeline_id=timeline,
            last_timeline_id=last_timeline,
            promoted=promoted,
            broken_count=len(broken),
            debris_count=len(debris),
            absent_count=len(absent),
            dry_run=dry_run,
        )

        # Transitions only. An absence lasts hours by design (rebuilds are
        # attended), so alerting on the STATE would fire every minute for the
        # whole window and train everyone to ignore it. The restored side
        # closes the loop: the rebuild's completion is worth telling too.
        newly_absent = absent - known_absent
        newly_restored = known_absent - absent
        if newly_absent:
            capture(
                "kb_pg_search_index_absent",
                {
                    "indexes": sorted(newly_absent),
                    "timeline_id": timeline,
                    "state": "lexical search returns nothing for these; "
                    "an attended rebuild is required",
                },
            )
        if newly_restored:
            capture(
                "kb_pg_search_index_restored",
                {"indexes": sorted(newly_restored), "timeline_id": timeline},
            )
        if absent != known_absent and not dry_run:
            await record_known_absent(conn, absent)

        if promoted:
            capture(
                "kb_pg_search_promotion_detected",
                {
                    "timeline_id": timeline,
                    "previous_timeline_id": last_timeline,
                    # The loud half is reported separately below. This says the
                    # QUIET half is now possible: a nonzero index frozen at
                    # clone time plans fine and returns incomplete results, and
                    # nothing but this promotion signal can tell you.
                    "action_required": "verify/rebuild pg_search indexes; they may be stale",
                },
            )
            if not dry_run:
                await analyze_tables(conn, ANALYZE_AFTER_PROMOTION)

        if debris:
            # Alert-only, never dropped: an in-progress CONCURRENTLY build is
            # indistinguishable from abandoned debris here, and dropping one
            # would mean the guardian broke a human's attended rebuild.
            capture(
                "kb_pg_search_invalid_index_debris",
                {"indexes": debris, "timeline_id": timeline},
            )

        dropped: list[str] = []
        skipped: list[str] = []
        for item in broken:
            index_name = str(item["index"])
            if dry_run:
                log.info("guardian.would_drop", **item)
                continue
            if await drop_broken_index(conn, index_name):
                dropped.append(index_name)
            else:
                skipped.append(index_name)

        announced = True
        if dropped:
            # The repair alert already says BM25 is down; pre-record the
            # absence so the next tick's transition check does not re-alert
            # for the same event.
            if not dry_run:
                await record_known_absent(conn, known_absent | frozenset(dropped))
            announced = capture(
                "kb_pg_search_index_repaired",
                {
                    "indexes": dropped,
                    "timeline_id": timeline,
                    "promoted": promoted,
                    # Say plainly what the operator is now looking at, so the
                    # Slack message does not read as "all clear".
                    "state": "search degraded (vector+graph+exact); BM25 needs an attended rebuild",
                },
            )
        if skipped:
            capture(
                "kb_pg_search_repair_deferred",
                {"indexes": skipped, "reason": "lock timeout; retrying next tick"},
            )

        # Record LAST. If anything above raised, the timeline is not advanced
        # and the next tick re-detects the promotion rather than concluding it
        # already handled one.
        if not dry_run:
            await record_timeline(conn, timeline)

    # A repair nobody was told about is the one outcome worse than no repair:
    # BM25 silently stays off until somebody notices search got worse. Exiting
    # nonzero puts the tick in the CronJob's FAILED history, which the chart
    # keeps ten of precisely so it can serve as the fallback alert channel
    # when the alerting path is what is unavailable.
    #
    # This is not hypothetical on the target cluster. `engine-secrets` in the
    # research namespace carries no POSTHOG_API_KEY (verified 2026-08-26), so
    # `capture` short-circuits and every repair would otherwise exit 0, land
    # under successfulJobsHistoryLimit: 1, and be gone within the minute.
    #
    # Deliberately ONLY for a repair. A promotion or debris alert that goes
    # undelivered is worth a log, not a red job -- neither changed the
    # database, and reporting them the same way would make red mean "something
    # happened" instead of "a change was made and nobody was told".
    if dropped and not announced:
        log.error(
            "guardian.repair_unannounced",
            indexes=dropped,
            reason="alert delivery failed or was not configured; failing the job so the "
            "repair is visible in CronJob history",
        )
        return 1

    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and log, change nothing. Alerts still fire.",
    )
    args = parser.parse_args()

    configure_logging()
    await init_pool()
    try:
        return await run_once(dry_run=args.dry_run)
    except Exception as exc:
        # Nonzero so the CronJob history shows red. This is the signal that
        # survives when the alerting path is itself broken.
        log.error("guardian.failed", error=str(exc), exc_info=True)
        return 1
    finally:
        await close_pool()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
