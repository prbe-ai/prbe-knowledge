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
    analyze_partitioned_parents,
    analyze_tables,
    current_timeline_id,
    drop_broken_index,
    find_absent_required_indexes,
    find_broken_pg_search_indexes,
    find_invalid_index_debris,
    find_nonempty_default_partitions,
    find_partitions_missing_required_index,
    find_unpartitioned_tables,
    prewarm_indexes,
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

# Indexes worth pg_prewarm'ing after a promotion -- what the DEFAULT search
# path actually walks, in the order a cold search hits them. The LIVE partial
# HNSW index (0124), not the full one: TemporalMode.LATEST's planner choice
# is the live index, and the full one goes cold in cache by design -- warming
# 7.7GB of it would evict hot pages to speed up queries nobody runs. Same
# explicit-list discipline as ANALYZE_AFTER_PROMOTION: warming everything
# would be a surprise I/O storm on an instance already serving cold.
PREWARM_AFTER_PROMOTION = [
    "idx_chunks_embedding_v2_hnsw_live",
    "idx_chunks_bm25_v2",
]


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
        # PARTITION-ERA CHECKS. Both are no-ops on an unpartitioned database,
        # so this same code ships before and after the conversion.
        #
        # A partition without its child index is `absent` scoped to one tenant:
        # lexical search dies for them alone while every global check stays
        # green -- the same blind spot the absent-detector was added for, one
        # level down.
        partitions_missing_index = await find_partitions_missing_required_index(conn)
        # Non-empty DEFAULT means a tenant is back on a shared index, which is
        # the fault partitioning removed, restored silently for that tenant.
        default_rows = await find_nonempty_default_partitions(conn)
        # The conversion is attended and out of band, so a plane can sit at
        # alembic head with a flat `chunks`. Expected, briefly; invisible, never.
        unpartitioned = await find_unpartitioned_tables(conn)

        log.info(
            "guardian.tick",
            timeline_id=timeline,
            last_timeline_id=last_timeline,
            promoted=promoted,
            broken_count=len(broken),
            debris_count=len(debris),
            absent_count=len(absent),
            partitions_missing_index=len(partitions_missing_index),
            default_partition_bytes=sum(int(d["heap_bytes"]) for d in default_rows),
            unpartitioned=unpartitioned,
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

        if unpartitioned:
            capture(
                "kb_chunks_not_partitioned",
                {
                    "tables": unpartitioned,
                    "timeline_id": timeline,
                    "state": "db/schema.sql declares these partitioned but this "
                    "database has them flat; every tenant is still sharing one "
                    "ANN index. Run scripts/convert_chunks_to_partitioned.py",
                },
            )

        if partitions_missing_index:
            capture(
                "kb_pg_search_partition_index_missing",
                {
                    "partitions": partitions_missing_index,
                    "timeline_id": timeline,
                    "state": "lexical search returns nothing for these tenants; "
                    "the rebuild cron must build the missing child index",
                },
            )

        if default_rows:
            capture(
                "kb_chunks_default_partition_nonempty",
                {
                    "partitions": default_rows,
                    "timeline_id": timeline,
                    "state": "a tenant has no partition of its own and is back on a "
                    "shared ANN index; run split_default() for the named tenants",
                },
            )

        # Statistics on a PARTITIONED PARENT are nobody else's job: PG16
        # autovacuum analyzes leaves only. Retrieval plans against the parent,
        # so stale parent stats reproduce the mis-costing this partitioning was
        # done to fix. Runs on its own age check, not on promotion.
        if not dry_run:
            await analyze_partitioned_parents(conn)

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
            # The CHILD is dropped; the ROOT is what an index contract declares
            # and therefore what the allowlist can validate. Equal when the
            # index is not partitioned.
            root_name = str(item.get("root_index", index_name))
            if dry_run:
                log.info("guardian.would_drop", **item)
                continue
            if await drop_broken_index(conn, index_name, allowlist_as=root_name):
                dropped.append(root_name)
            else:
                skipped.append(root_name)

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

        # Prewarm AFTER the repair and AFTER the timeline is recorded, on
        # purpose and against the reading order. The warm is minutes of
        # sequential I/O on a freshly promoted instance, and the CronJob has
        # an activeDeadlineSeconds it can hit mid-read -- if that happened
        # BEFORE record_timeline, the killed tick would never advance the
        # marker, the next tick would re-detect the same promotion, and the
        # repair would livelock behind a cache warm forever (with
        # concurrencyPolicy: Forbid skipping ticks in between). Here, a
        # deadline kill costs only the tail of an optimization: the repair
        # is done, the timeline is recorded, and search is already serving.
        if promoted and not dry_run:
            await prewarm_indexes(conn, PREWARM_AFTER_PROMOTION)

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
