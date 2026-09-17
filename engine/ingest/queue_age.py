"""Backlog age, sampled once a minute and published where someone will see it.

WHY THIS EXISTS
---------------
On 2026-09-16 the ingestion queue's median wait was 33 minutes and its p90 was
113. Nothing in the system said so. It was found by hand, days later, by
someone who went looking. Every existing signal was green the whole time: the
worker's liveness probe answers on a database ping, the drain-stall beacon only
notices when the loop stops entirely, and a drain that is running at a third of
its normal rate is exactly as alive as one that is keeping up.

WHY NOT `enqueued_at`
---------------------
Every transcript batch bumps `enqueued_at` (session_completer reads
MAX(enqueued_at) as an idle signal), so an actively-batching session looks
permanently young. Age measured that way reports zero on precisely the workload
that is filling the queue. `first_enqueued_at` is stamped once on insert.

WHY A SAMPLER AND NOT A QUERY ON /health
----------------------------------------
The health endpoint is polled by kubelet every few seconds and its failure
restarts the pod. Putting a scan-shaped query behind it means a slow database
turns "the queue is deep" into "kill the thing that drains the queue" -- the
lesson already written into this worker's own liveness probe, which was changed
to stop reporting a hung drain as healthy but deliberately still does not take
new work to answer. So: one query a minute, a snapshot in memory, and /health
reads the snapshot without touching the database.

WHY PROCESSING ROWS COUNT
-------------------------
A row that has been `processing` for twenty minutes is twenty minutes of
latency to whoever is waiting for it. Counting only `pending` would hide a
worker that claims rows promptly and then crawls, which is the exact failure
that started this.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import asdict, dataclass
from typing import Any

from engine.shared.config import get_settings
from engine.shared.db import get_pool
from engine.shared.logging import get_logger
from engine.shared.ops_alert import capture

log = get_logger(__name__)

#: One sample a minute. The thing being measured moves in minutes, and this is
#: a full-table aggregate on an indexed predicate -- cheap, but not free.
SAMPLE_INTERVAL_SECONDS = 60.0

_SAMPLE_SQL = """
    SELECT
        COUNT(*) FILTER (WHERE status = 'pending')                  AS pending,
        COUNT(*) FILTER (WHERE status = 'processing')               AS processing,
        COALESCE(
            EXTRACT(EPOCH FROM (NOW() - MIN(first_enqueued_at))), 0
        )::bigint                                                    AS oldest_age_seconds
    FROM ingestion_queue
    WHERE status IN ('pending', 'processing')
"""


@dataclass(frozen=True)
class QueueAge:
    """The last sample. All zeros is the honest starting state: it means
    "nothing waiting", which is also what an empty queue looks like."""

    pending: int = 0
    processing: int = 0
    oldest_age_seconds: int = 0

    def as_body(self) -> dict[str, Any]:
        return {f"queue_{k}": v for k, v in asdict(self).items()}


_latest = QueueAge()


def latest() -> QueueAge:
    """The most recent sample, for /health. Never queries."""
    return _latest


async def sample_once() -> QueueAge:
    global _latest
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(_SAMPLE_SQL)
    _latest = QueueAge(
        pending=row["pending"],
        processing=row["processing"],
        oldest_age_seconds=row["oldest_age_seconds"],
    )
    return _latest


class QueueAgeReporter:
    """Samples the backlog and ships it to PostHog once a minute."""

    def __init__(self, shutdown: asyncio.Event | None = None) -> None:
        self._shutdown = shutdown or asyncio.Event()

    def shutdown(self) -> None:
        self._shutdown.set()

    async def run(self) -> None:
        log.info("queue_age_reporter.start", interval=SAMPLE_INTERVAL_SECONDS)
        while not self._shutdown.is_set():
            try:
                age = await sample_once()
                # Local first and always: PostHog being unreachable must not
                # cost the operator the number, and this log is what a
                # `kubectl logs | grep` finds during an incident.
                log.info(
                    "queue_age",
                    pending=age.pending,
                    processing=age.processing,
                    oldest_age_seconds=age.oldest_age_seconds,
                )
                # Fire-and-forget by construction (see shared.ops_alert): a
                # sampler that can fail the worker is worse than no sampler.
                await asyncio.to_thread(
                    capture,
                    "ingestion_queue_age",
                    {
                        "pending": age.pending,
                        "processing": age.processing,
                        "oldest_age_seconds": age.oldest_age_seconds,
                        "environment": get_settings().environment,
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # Never let the measurement break the thing measured.
                log.exception("queue_age_reporter.sample_failed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=SAMPLE_INTERVAL_SECONDS
                )
        log.info("queue_age_reporter.stop")
