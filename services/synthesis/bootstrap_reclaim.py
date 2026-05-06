"""Reclaim stale bootstrap runs.

Bootstrap runs (`wiki_synthesis_runs.kind='bootstrap'`) open with
`status='running'` and are flipped to a terminal state by the
orchestrator on completion. If the bootstrap fly machine crashes
mid-run (OOM, deploy mid-flight, infra blip), the row is orphaned
in `running` indefinitely. The status endpoint
`GET /api/wiki/bootstrap/status` then reports an in-flight bootstrap
forever — the dashboard "Bootstrapping..." pill stays up until
someone notices and manually fixes the row.

This module runs as a periodic asyncio task inside the bootstrap fly
app (wiring lands as a follow-up to PR #118 once that branch merges
— this module is purely additive on origin/main). Every 15 minutes
it scans for `kind='bootstrap'` rows that have been `running` for
> 6 hours and flips them to `failed` with an `error` marker so the
audit trail captures what happened.

Design notes
------------
- No `heartbeat_at` on `wiki_synthesis_runs` (unlike
  `wiki_synthesis_queue`). Bootstrap runs are coarse-grained — one
  row per source per trigger — so we use `started_at` + a duration
  threshold instead of a per-step heartbeat.
- Threshold sized at 6 hours: bootstrap of largest customer (Pebble)
  takes ~30 min today; 6h is 12x that, well outside any normal run.
- Loop interval: 15 minutes. This is cleanup, not heartbeat tracking
  — there's no point sweeping more often than the threshold's
  granularity warrants.
- Mirrors the structural pattern of `services/synthesis/reclaim.py`
  (WikiReclaimLoop): asyncio.Event for shutdown, contextlib.suppress
  on the wait_for, exception swallow on the inner tick so a
  transient DB blip doesn't kill the whole loop.
"""

from __future__ import annotations

import asyncio
import contextlib

from shared.db import raw_conn
from shared.logging import get_logger

log = get_logger(__name__)


# 6 hours — see module docstring for sizing rationale.
BOOTSTRAP_RECLAIM_THRESHOLD_HOURS = 6

# 15 minutes — cleanup cadence, not heartbeat tracking.
BOOTSTRAP_RECLAIM_INTERVAL_SECONDS = 15 * 60.0


# Marker appended to wiki_synthesis_runs.error so the audit trail
# captures *why* a row was flipped to 'failed'. Tests grep for the
# "reclaimed:" prefix.
RECLAIM_ERROR_MARKER = "reclaimed: stale running row, machine likely crashed"


async def reclaim_stale_bootstrap_runs(
    *,
    threshold_hours: int = BOOTSTRAP_RECLAIM_THRESHOLD_HOURS,
) -> int:
    """One pass: flip stale `running` bootstrap rows to `failed`.

    Returns the number of rows reclaimed. Idempotent — running a
    second time immediately after returns 0 because the WHERE
    clause filters on `status='running'`.
    """
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            UPDATE wiki_synthesis_runs
               SET status = 'failed',
                   finished_at = NOW(),
                   error = COALESCE(error, '')
                       || (CASE
                             WHEN error IS NULL OR error = ''
                               THEN ''
                             ELSE ' | '
                           END)
                       || $2
             WHERE kind = 'bootstrap'
               AND status = 'running'
               AND started_at < NOW() - make_interval(hours => $1)
            RETURNING run_id, customer_id, source
            """,
            threshold_hours,
            RECLAIM_ERROR_MARKER,
        )
    if rows:
        log.warning(
            "bootstrap_reclaim.stale_runs_flipped",
            count=len(rows),
            run_ids=[r["run_id"] for r in rows],
            customers=sorted({r["customer_id"] for r in rows}),
        )
    return len(rows)


class BootstrapReclaimLoop:
    """Periodically reclaim stale bootstrap runs.

    Mirrors `WikiReclaimLoop` — asyncio.Event for shutdown,
    contextlib.suppress on wait_for, exception swallow on the inner
    tick so transient DB blips don't kill the loop.
    """

    def __init__(
        self,
        *,
        threshold_hours: int = BOOTSTRAP_RECLAIM_THRESHOLD_HOURS,
        interval_seconds: float = BOOTSTRAP_RECLAIM_INTERVAL_SECONDS,
    ) -> None:
        self._threshold_hours = threshold_hours
        self._interval = interval_seconds
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        log.info(
            "bootstrap_reclaim_loop.start",
            threshold_hours=self._threshold_hours,
            interval_seconds=self._interval,
        )
        # Brief initial delay so the loop doesn't fire mid-boot
        # before the bootstrap orchestrator has had a chance to
        # open its first run row.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._shutdown.wait(), timeout=self._interval)
        while not self._shutdown.is_set():
            try:
                await reclaim_stale_bootstrap_runs(
                    threshold_hours=self._threshold_hours,
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover — keep loop alive
                log.exception("bootstrap_reclaim_loop.tick_failed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._shutdown.wait(), timeout=self._interval)
        log.info("bootstrap_reclaim_loop.stop")

    def shutdown(self) -> None:
        self._shutdown.set()
