"""WikiReclaimLoop — periodic sweep for crash-stuck wiki queue rows.

Runs as a third asyncio task in both the wiki-worker and wiki-synthesis
fly apps. Each tick calls `persistence.reclaim_stuck_rows` which
resets stale 'triaging'/'synthesizing' rows back to their prior state
if attempts < cap, else terminal 'failed'. Mirrors the
`services/ingestion/worker.py:ReclaimLoop` shape.

Why two reclaim instances (one per fly app) — the underlying SQL is
state-aware (only touches 'triaging' or 'synthesizing'), so two
loops sweeping the same queue is idempotent. Could collapse to one
app if ops simplification becomes valuable; v1 keeps both for
symmetry with their respective worker classes.

Threshold sizing: 10 minutes covers Sonnet/Pro synthesize calls
(typically 30-90s, occasionally 3 min on long contexts) plus
Normalizer._persist's chunking + embedding overhead. Pebble's
parallel drain wraps in ~4 min; 10-min threshold leaves headroom
without needing a per-batch heartbeat task.
"""

from __future__ import annotations

import asyncio
import contextlib

from engine.shared.logging import get_logger
from kb.synthesis import persistence

log = get_logger(__name__)


# 10 minutes — see module docstring for sizing rationale.
WIKI_RECLAIM_THRESHOLD_SECONDS = 600

# 2 minutes — same cadence as the ingestion ReclaimLoop. Tradeoff is
# detection latency vs DB load; the underlying SQL is bounded by the
# heartbeat partial index.
WIKI_RECLAIM_INTERVAL_SECONDS = 120.0


class WikiReclaimLoop:
    """Periodically reclaim stuck wiki_synthesis_queue rows.

    Fenced on attempts so poison rows dead-letter at 'failed' instead
    of looping forever and burning LLM spend.
    """

    def __init__(
        self,
        *,
        threshold_seconds: int = WIKI_RECLAIM_THRESHOLD_SECONDS,
        interval_seconds: float = WIKI_RECLAIM_INTERVAL_SECONDS,
        max_attempts: int,
    ) -> None:
        self._threshold = threshold_seconds
        self._interval = interval_seconds
        self._max_attempts = max_attempts
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        log.info(
            "wiki_reclaim_loop.start",
            threshold_seconds=self._threshold,
            interval_seconds=self._interval,
            max_attempts=self._max_attempts,
        )
        # Brief initial delay so the loop doesn't fire mid-boot before
        # the workers have had a chance to claim their first batch.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._shutdown.wait(), timeout=self._interval)
        while not self._shutdown.is_set():
            try:
                retried, failed = await persistence.reclaim_stuck_rows(
                    threshold_seconds=self._threshold,
                    max_attempts=self._max_attempts,
                )
                if retried or failed:
                    log.warning(
                        "wiki_reclaim_loop.swept",
                        retried=retried,
                        failed=failed,
                    )
            except Exception:  # pragma: no cover — keep loop alive
                log.exception("wiki_reclaim_loop.tick_failed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._shutdown.wait(), timeout=self._interval)
        log.info("wiki_reclaim_loop.stop")

    def shutdown(self) -> None:
        self._shutdown.set()
