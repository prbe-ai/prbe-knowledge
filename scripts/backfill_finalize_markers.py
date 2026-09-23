"""One-time: put the end signal back on sessions the worker has already mined.

Before the one-rule change (engine/shared/session_signals.py), the worker
deleted a session's finalize key after an authoritative mine, and the sweep
treated "no finalize key" as "never ended". Every idle protocol-1 session was
therefore re-ended and re-mined daily. Under the new rule the sweep skips a
session whose newest key is an end signal -- but the mined sessions no longer
have one. Without this script the first sweep after the deploy would mine all
of them once more (~5,000 on research, ~$100, ~40k document rewrites).

A row gets the marker key re-attached ONLY with evidence that its current
transcript was mined by a complete, authoritative pass:

  1. protocol 1, `status='done'`, newest key not already an end signal. NOT
     filtered on idleness: the old loop re-ended every row within the last
     day, so an idle filter would skip exactly the rows this is for;
  2. its LAST upsert was not a batch: `enqueued_at` falls after the end of the
     UTC day in its newest batch key (`raw/<src>/<cust>/YYYY/MM/DD/...`), with
     a 10-minute margin for the two clocks involved.
     Only three writers move `enqueued_at` on an agent row -- a v1 batch, a v2
     batch, and an end-signal append -- so the last write was an end signal;
  3. `completed_at >= enqueued_at`: the worker finished a pass after that write,
     and its commit is version-checked, so that pass saw every key on the row
     and was therefore a COMPLETE pass;
  4. the key the old worker deleted only after an AUTHORITATIVE pass is gone
     (1), and the sweep's `finalize.marker` object for the session still
     exists in R2. The object is referenced again, never written: a missing
     object is missing evidence, and that row is left for the sweep to mine
     once.

Every write is conditioned on the row being exactly as it was read (version,
status, completed_at, newest key); a row that changed is skipped and counted,
never forced. Nothing about the row's version or status changes, so no worker
pass follows.

Run it with the sweep SUSPENDED, dry-run first:

    python -m scripts.backfill_finalize_markers --dry-run --report /tmp/r.json
    python -m scripts.backfill_finalize_markers --customer <one tenant>
    python -m scripts.backfill_finalize_markers --report /tmp/r.json

`--report` also lists protocol-1 rows that are done AND already end in an end
signal: their last pass kept the key, which the old worker did only for a
non-authoritative pass. That list is the retry population for the follow-up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from engine.shared.db import get_pool, init_pool
from engine.shared.logging import get_logger
from engine.shared.session_signals import (
    cron_marker_key,
    has_v2_key_sql,
    last_key_ends_v1_session_sql,
)
from engine.shared.storage import get_store
from kb.session_completer import AGENT_SOURCES
from kb.session_receipts import _lock

log = get_logger(__name__)

#: Newest date in any non-marker key. NULL when no key carries a date.
_NEWEST_BATCH_DAY = """(
    SELECT max(to_date(substring(_b FROM '/(\\d{4}/\\d{2}/\\d{2})/'), 'YYYY/MM/DD'))
      FROM unnest(payload_s3_keys) AS _b
     WHERE right(_b, 16) <> '/finalize.marker'
)"""

#: The key's date comes from the app's clock and `enqueued_at` from the
#: database's, so a batch landing a second before midnight must not read as a
#: later day. Erring this way costs a few extra mines, never an unmined session.
_DAY_EDGE_MARGIN = "10 minutes"

_V1_DONE = f"""
       source_system = $1
   AND ($2::text[] IS NULL OR customer_id = ANY($2::text[]))
   AND status = 'done'
   AND cardinality(payload_s3_keys) > 0
   AND NOT {has_v2_key_sql()}
"""

_EVIDENCE = f"""
       NOT {last_key_ends_v1_session_sql()}
   AND completed_at >= enqueued_at
   AND (enqueued_at AT TIME ZONE 'UTC') >= {_NEWEST_BATCH_DAY} + 1 + interval '{_DAY_EDGE_MARGIN}'
"""

_CANDIDATES_SQL = f"""
SELECT queue_id, customer_id, source_event_id AS session_id, version, completed_at
  FROM ingestion_queue
 WHERE {_V1_DONE}
   AND {_EVIDENCE}
 ORDER BY queue_id
"""

_SWEEP_ONCE_SQL = f"""
SELECT count(*) FROM ingestion_queue
 WHERE {_V1_DONE}
   AND NOT {last_key_ends_v1_session_sql()}
   AND NOT ({_EVIDENCE})
"""

_ALREADY_ENDED_SQL = f"""
SELECT queue_id FROM ingestion_queue
 WHERE {_V1_DONE}
   AND {last_key_ends_v1_session_sql()}
 ORDER BY queue_id
"""

#: Re-attach the marker only if the row is exactly as read.
_RELINK_SQL = f"""
UPDATE ingestion_queue
   SET payload_s3_keys = payload_s3_keys || ARRAY[$2]::text[]
 WHERE queue_id = $1
   AND version = $3
   AND status = 'done'
   AND completed_at = $4
   AND NOT {last_key_ends_v1_session_sql()}
RETURNING queue_id
"""


@dataclass
class Report:
    dry_run: bool
    candidates: int = 0
    relinked: int = 0
    would_relink: int = 0
    skipped_changed: int = 0
    skipped_stream: int = 0
    missing_marker_object: int = 0
    sweep_will_mine_once: int = 0
    #: Done rows already ending in an end signal: kept by a non-authoritative pass.
    already_ended_queue_ids: list[int] = field(default_factory=list)
    by_source: dict[str, dict[str, int]] = field(default_factory=dict)


async def _relink_one(row: Any, source: str, *, dry_run: bool, sem: asyncio.Semaphore) -> str:
    customer_id, session_id = row["customer_id"], row["session_id"]
    key = cron_marker_key(source, customer_id, session_id)
    store = get_store()
    async with sem:
        bucket = await store.bucket_for(customer_id)
        present = await store.exists(bucket, key)
    if not present:
        return "missing_marker_object"
    async with get_pool().acquire() as conn, conn.transaction():
        await conn.execute("SELECT set_config('app.current_customer_id', $1, true)", customer_id)
        await _lock(conn, customer_id, source, session_id)
        if await conn.fetchval(
            "SELECT 1 FROM session_streams WHERE customer_id=$1 AND source_system=$2 AND session_id=$3",
            customer_id,
            source,
            session_id,
        ):
            return "skipped_stream"
        if dry_run:
            return "would_relink"
        done = await conn.fetchval(_RELINK_SQL, row["queue_id"], key, row["version"], row["completed_at"])
    return "relinked" if done is not None else "skipped_changed"


async def backfill(
    *,
    dry_run: bool = True,
    concurrency: int = 16,
    limit: int | None = None,
    customers: list[str] | None = None,
) -> Report:
    report = Report(dry_run=dry_run)
    sem = asyncio.Semaphore(concurrency)
    async with get_pool().acquire() as conn:
        plan = []
        for source in AGENT_SOURCES:
            rows = await conn.fetch(_CANDIDATES_SQL, source.value, customers)
            if limit is not None:
                rows = rows[:limit]
            once = await conn.fetchval(_SWEEP_ONCE_SQL, source.value, customers)
            ended = await conn.fetch(_ALREADY_ENDED_SQL, source.value, customers)
            report.candidates += len(rows)
            report.sweep_will_mine_once += once
            report.already_ended_queue_ids.extend(r["queue_id"] for r in ended)
            report.by_source[source.value] = {
                "candidates": len(rows),
                "sweep_will_mine_once": once,
                "already_ended": len(ended),
            }
            plan.append((source.value, rows))
    for source, rows in plan:
        outcomes = await asyncio.gather(
            *(_relink_one(r, source, dry_run=dry_run, sem=sem) for r in rows)
        )
        for outcome in outcomes:
            setattr(report, outcome, getattr(report, outcome) + 1)
            report.by_source[source][outcome] = report.by_source[source].get(outcome, 0) + 1
    # The whole point of the report: every candidate is accounted for.
    accounted = (
        report.relinked
        + report.would_relink
        + report.skipped_changed
        + report.skipped_stream
        + report.missing_marker_object
    )
    if accounted != report.candidates:
        raise RuntimeError(f"accounted for {accounted} of {report.candidates} candidates")
    log.info(
        "backfill_finalize_markers.done",
        **{k: v for k, v in asdict(report).items() if k != "already_ended_queue_ids"},
        already_ended=len(report.already_ended_queue_ids),
    )
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Count and HEAD only; write nothing.")
    parser.add_argument("--concurrency", type=int, default=16, help="Concurrent R2 HEADs.")
    parser.add_argument("--limit", type=int, default=None, help="Per source, for a trial run.")
    parser.add_argument("--report", default=None, help="Write the full report as JSON here.")
    parser.add_argument(
        "--customer", action="append", default=None,
        help="Only this tenant (repeatable). Run one tenant first as a canary.",
    )
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    return args


async def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    await init_pool()
    try:
        report = await backfill(
            dry_run=args.dry_run,
            concurrency=args.concurrency,
            limit=args.limit,
            customers=args.customer,
        )
    finally:
        from engine.shared.db import close_pool

        await close_pool()
    summary = {k: v for k, v in asdict(report).items() if k != "already_ended_queue_ids"}
    summary["already_ended"] = len(report.already_ended_queue_ids)
    print(json.dumps(summary, indent=2, default=str))
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(asdict(report), fh, indent=2, default=str)


if __name__ == "__main__":
    asyncio.run(_main())
