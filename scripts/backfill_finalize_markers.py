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
  2. its LAST upsert was not a batch: `enqueued_at` is at least a full day
     after the end of the UTC day in its newest batch key
     (`raw/<src>/<cust>/YYYY/MM/DD/...`). The key's date comes from the app's
     clock when a request arrives and `enqueued_at` from the database when it
     commits; no upload request lives for a day, so a batch can never pass for
     a later write. (A 10-minute margin was not enough: a request stalled in
     redaction or on a lock across midnight would have. On research the full
     day costs one row.)
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
signal. Run BEFORE the new worker has mined anything, that is the old worker's
partial passes (it kept an end signal only then); afterwards it also includes
sessions the new worker mined completely, which keep theirs by design.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from engine.shared.db import get_pool, init_pool
from engine.shared.logging import get_logger
from engine.shared.session_signals import (
    cron_marker_key,
    ends_v1_session_sql,
    has_v2_key_sql,
    is_cron_marker_key_sql,
    last_key_sql,
)
from engine.shared.storage import get_store
from kb.session_completer import AGENT_SOURCES, has_v2_stream
from kb.session_receipts import _lock

log = get_logger(__name__)


class Outcome(StrEnum):
    RELINKED = "relinked"
    WOULD_RELINK = "would_relink"
    SKIPPED_CHANGED = "skipped_changed"
    SKIPPED_STREAM = "skipped_stream"
    MISSING_MARKER_OBJECT = "missing_marker_object"
    ERRORED = "errored"


#: Newest date in any non-marker key. NULL when no key carries a date.
_NEWEST_BATCH_DAY = f"""(
    SELECT max(to_date(substring(_b FROM '/(\\d{{4}}/\\d{{2}}/\\d{{2}})/'), 'YYYY/MM/DD'))
      FROM unnest(q.payload_s3_keys) AS _b
     WHERE NOT {is_cron_marker_key_sql("_b")}
)"""

_FROM = f"""
  FROM ingestion_queue q
  CROSS JOIN LATERAL (SELECT {last_key_sql("q.payload_s3_keys")} AS last_key OFFSET 0) lk
"""

_V1_DONE = f"""
       q.source_system = $1
   AND ($2::text[] IS NULL OR q.customer_id = ANY($2::text[]))
   AND q.status = 'done'
   AND cardinality(q.payload_s3_keys) > 0
   AND strpos(q.source_event_id, ':') = 0
   AND NOT {has_v2_key_sql("q.payload_s3_keys")}
"""

_ENDED = ends_v1_session_sql("lk.last_key", "q.source_event_id")

_EVIDENCE = f"""
       NOT {_ENDED}
   AND q.completed_at >= q.enqueued_at
   AND (q.enqueued_at AT TIME ZONE 'UTC') >= {_NEWEST_BATCH_DAY} + 2
"""

_CANDIDATES_SQL = f"""
SELECT q.queue_id, q.customer_id, q.source_event_id AS session_id, q.version, q.completed_at
{_FROM}
 WHERE {_V1_DONE}
   AND {_EVIDENCE}
 ORDER BY q.queue_id
"""

_SWEEP_ONCE_SQL = f"""
SELECT count(*)
{_FROM}
 WHERE {_V1_DONE}
   AND NOT {_ENDED}
   AND NOT ({_EVIDENCE})
"""

_ALREADY_ENDED_SQL = f"""
SELECT q.queue_id
{_FROM}
 WHERE {_V1_DONE}
   AND {_ENDED}
 ORDER BY q.queue_id
"""

#: Re-attach the marker only if the row is exactly as read.
_RELINK_SQL = f"""
UPDATE ingestion_queue
   SET payload_s3_keys = payload_s3_keys || ARRAY[$2]::text[]
 WHERE queue_id = $1
   AND version = $3
   AND status = 'done'
   AND completed_at = $4
   AND NOT {ends_v1_session_sql(last_key_sql())}
RETURNING queue_id
"""


@dataclass
class Report:
    dry_run: bool
    candidates: int = 0
    sweep_will_mine_once: int = 0
    outcomes: Counter = field(default_factory=Counter)
    #: Done rows already ending in an end signal (see the module docstring).
    already_ended_queue_ids: list[int] = field(default_factory=list)
    by_source: dict[str, dict[str, int]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        out = {k: v for k, v in asdict(self).items() if k not in ("already_ended_queue_ids", "outcomes")}
        out.update({o.value: self.outcomes[o] for o in Outcome})
        out["already_ended"] = len(self.already_ended_queue_ids)
        return out


async def _relink_one(row: Any, source: str, *, dry_run: bool, sem: asyncio.Semaphore) -> Outcome:
    """One row, end to end, under the concurrency cap (R2 and a pooled
    connection alike). A failure is counted, never allowed to abort the run."""
    customer_id, session_id = row["customer_id"], row["session_id"]
    key = cron_marker_key(source, customer_id, session_id)
    async with sem:
        try:
            store = get_store()
            if not await store.exists(await store.bucket_for(customer_id), key):
                return Outcome.MISSING_MARKER_OBJECT
            async with get_pool().acquire() as conn, conn.transaction():
                await conn.execute("SELECT set_config('app.current_customer_id', $1, true)", customer_id)
                await _lock(conn, customer_id, source, session_id)
                if await has_v2_stream(conn, customer_id, source, session_id):
                    return Outcome.SKIPPED_STREAM
                if dry_run:
                    return Outcome.WOULD_RELINK
                done = await conn.fetchval(
                    _RELINK_SQL, row["queue_id"], key, row["version"], row["completed_at"]
                )
            return Outcome.RELINKED if done is not None else Outcome.SKIPPED_CHANGED
        except Exception as exc:
            log.warning(
                "backfill_finalize_markers.row_failed",
                queue_id=row["queue_id"],
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
            return Outcome.ERRORED


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
            report.outcomes[outcome] += 1
            report.by_source[source][outcome.value] = report.by_source[source].get(outcome.value, 0) + 1
    # The whole point of the report: every candidate is accounted for.
    accounted = sum(report.outcomes.values())
    if accounted != report.candidates:
        raise RuntimeError(f"accounted for {accounted} of {report.candidates} candidates")
    log.info("backfill_finalize_markers.done", **report.summary())
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Count and HEAD only; write nothing.")
    parser.add_argument("--concurrency", type=int, default=16, help="Rows in flight (R2 and DB).")
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
    print(json.dumps(report.summary(), indent=2, default=str))
    if args.report:
        with open(args.report, "w") as fh:
            json.dump({**report.summary(), "already_ended_queue_ids": report.already_ended_queue_ids},
                      fh, indent=2, default=str)


if __name__ == "__main__":
    asyncio.run(_main())
