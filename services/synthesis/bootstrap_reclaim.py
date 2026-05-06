"""Reclaim stale bootstrap runs.

Bootstrap runs (`wiki_synthesis_runs.kind='bootstrap'`) open with
`status='pending'`, are claimed by a ``BootstrapWorker`` (flips to
`running`), and the worker writes a terminal state on completion. If
the worker's fly machine crashes mid-run (OOM, deploy mid-flight,
infra blip), the row is orphaned in `running` indefinitely.

This module runs as a periodic asyncio task inside the bootstrap fly
app. Every 15 minutes it scans for `kind='bootstrap'` rows that have
been `running` for > 6 hours and flips them back to `pending` with an
`error` marker so the audit trail captures what happened, AND so a
peer worker re-claims the row on its next tick. This is the
self-healing path for machine death — without it, a stuck row would
sit in `running` until manual intervention.

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
- Reclaim flips `running` -> `pending` (NOT `failed`). Bounded retry
  via an `attempts` cap is deferred to v2; in v1 a poison row that
  fails repeatedly will loop here every 6h, which surfaces as wasted
  compute (visible on the dashboard via the row's accumulated error
  text) rather than corruption.
- Scoping: `WHERE kind='bootstrap' AND status='running'`. v4
  daily-replay runs use kind in ('wake','scheduled','onboarding') and
  are untouched by this loop.
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
# captures *why* a row was flipped back to 'pending'. Tests grep for
# the "reclaimed:" prefix.
RECLAIM_ERROR_MARKER = "reclaimed: stale running row released for retry"


async def reclaim_stale_bootstrap_runs(
    *,
    threshold_hours: int = BOOTSTRAP_RECLAIM_THRESHOLD_HOURS,
) -> int:
    """One pass: flip stale `running` bootstrap rows back to `pending`.

    Returns the number of rows reclaimed. Idempotent — running a
    second time immediately after returns 0 because the WHERE
    clause filters on `status='running'`. Once a row is back at
    `pending`, the BootstrapWorker on any machine claims it via
    ``FOR UPDATE SKIP LOCKED`` on the next tick.

    `finished_at` is intentionally NOT set — the row is being requeued,
    not finished. Only the per-(customer, source) advisory lock the
    worker takes inside ``_run_one`` prevents a still-alive prior worker
    from clobbering the new claim; if the prior worker is genuinely
    dead, the lock is uncontended and the new claim runs cleanly.
    """
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            UPDATE wiki_synthesis_runs
               SET status = 'pending',
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
            "bootstrap_reclaim.stale_runs_requeued",
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
