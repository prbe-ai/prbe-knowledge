"""Polling-tick scheduler.

The scheduler is the loop that drives every per-source poller forward.
One ``PollScheduler`` instance runs in the polling pod (chart role
added in PR C); it sleeps ``tick_interval_seconds`` between ticks. Each
tick:

  1. ``list_due_cursors`` returns the (customer, source, resource)
     rows whose ``polled_at`` is older than ``min_resource_age_seconds``
     ago, oldest first, up to ``batch_size``.
  2. For each row, look up the registered poller class for that source
     (via ``get_poller``). Skip the row if no poller is registered
     (some sources never poll — MANUAL_UPLOAD, CUSTOM_INGEST, etc.).
  3. Instantiate the poller, call ``poll(customer_id, resource_id,
     cursor_value)`` — the per-source code does the upstream API call.
  4. Apply the ``PollResult``: enqueue documents (deferred to PR E1+
     wiring), persist the next cursor (or stamp the error).
  5. Sleep until the next tick.

The scheduler is single-instance for now — one pod, one loop. If we
need to horizontally scale we'd add an advisory-lock-per-resource
claim (matching the inferred_edges_queue pattern) so two pods can
walk the cursor list without racing each other on the same row.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from services.ingestion.polling.base import (
    PollResult,
    get_poller,
    registered_sources,
)
from services.ingestion.polling.cursors import (
    CursorRow,
    advance_cursor,
    list_due_cursors,
    stamp_error,
)
from shared.constants import SourceSystem

logger = logging.getLogger(__name__)


# Callback shape for "do something with the documents the poller
# emitted". The scheduler passes both ``customer_id`` and ``source`` so
# the sink can dispatch into the right ingestion-queue row (the queue's
# uniqueness key includes ``source_system``). The production sink lives
# at :class:`services.ingestion.polling.sink.PollDocumentSink`; tests
# inject a recorder.
DocumentSink = Callable[
    [str, SourceSystem, list[dict[str, Any]]], Coroutine[Any, Any, None]
]


async def _default_sink(
    customer_id: str, source: SourceSystem, documents: list[dict[str, Any]]
) -> None:
    """No-op sink used when no production sink is wired (early-stage
    tests, environments where INGESTION_MODE!=poll, or anywhere a real
    queue is undesirable). The real sink is
    :class:`services.ingestion.polling.sink.PollDocumentSink`."""
    if documents:
        logger.info(
            "polling.sink default no-op: customer_id=%s source=%s documents=%d",
            customer_id,
            source.value,
            len(documents),
        )


class PollScheduler:
    """Single-instance tick loop driving every registered per-source poller.

    Construct with ``sink`` set to the actual document-ingestion callable
    in production; default is a no-op that logs the count (used in tests
    + when no per-source pollers are registered yet)."""

    def __init__(
        self,
        *,
        tick_interval_seconds: int = 60,
        min_resource_age_seconds: int = 300,
        batch_size: int = 50,
        sink: DocumentSink | None = None,
    ) -> None:
        self.tick_interval_seconds = tick_interval_seconds
        self.min_resource_age_seconds = min_resource_age_seconds
        self.batch_size = batch_size
        self._sink: DocumentSink = sink or _default_sink
        self._stopping = asyncio.Event()

    async def run_forever(self) -> None:
        """Run until ``stop()`` is called. Each iteration is one tick."""
        logger.info(
            "polling.scheduler.start tick=%ds min_age=%ds batch=%d sources=%s",
            self.tick_interval_seconds,
            self.min_resource_age_seconds,
            self.batch_size,
            sorted(s.value for s in registered_sources()),
        )
        try:
            while not self._stopping.is_set():
                try:
                    await self.tick_once()
                except Exception:
                    logger.exception("polling.scheduler.tick failed; continuing")
                # asyncio.wait_for + an Event lets stop() interrupt the sleep.
                # Timeout is the normal "next tick" path — suppress it.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._stopping.wait(), self.tick_interval_seconds
                    )
        finally:
            logger.info("polling.scheduler.stop")

    def stop(self) -> None:
        """Tell ``run_forever`` to exit at the next tick boundary."""
        self._stopping.set()

    async def tick_once(self) -> int:
        """One tick. Returns the number of cursor rows processed (so
        tests + ops dashboards can sanity-check throughput)."""
        rows = await list_due_cursors(
            min_age_seconds=self.min_resource_age_seconds,
            limit=self.batch_size,
        )
        if not rows:
            return 0
        for row in rows:
            await self._poll_one(row)
        return len(rows)

    async def _poll_one(self, row: CursorRow) -> None:
        """Drive one (customer, source, resource) cursor forward.
        Failures are stamped onto the row; they do not abort the tick."""
        poller_cls = get_poller(row.source)
        if poller_cls is None:
            # Some sources have an ingestion_cursors row from a previous
            # build but no live poller class — skip without re-stamping
            # polled_at so we don't busy-loop, but also don't error.
            logger.debug(
                "polling.skip no_poller customer_id=%s source=%s resource=%s",
                row.customer_id,
                row.source.value,
                row.resource_id,
            )
            return

        poller = poller_cls()
        try:
            result: PollResult = await poller.poll(
                customer_id=row.customer_id,
                resource_id=row.resource_id,
                cursor=row.cursor_value,
            )
        except Exception as exc:
            logger.exception(
                "polling.poll_raised customer_id=%s source=%s resource=%s",
                row.customer_id,
                row.source.value,
                row.resource_id,
            )
            await stamp_error(
                customer_id=row.customer_id,
                source=row.source,
                resource_id=row.resource_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return

        if result.error is not None:
            await stamp_error(
                customer_id=row.customer_id,
                source=row.source,
                resource_id=row.resource_id,
                error=result.error,
            )
            return

        if result.documents:
            try:
                await self._sink(row.customer_id, row.source, result.documents)
            except Exception as exc:
                logger.exception(
                    "polling.sink_raised customer_id=%s source=%s resource=%s docs=%d",
                    row.customer_id,
                    row.source.value,
                    row.resource_id,
                    len(result.documents),
                )
                await stamp_error(
                    customer_id=row.customer_id,
                    source=row.source,
                    resource_id=row.resource_id,
                    error=f"sink error: {type(exc).__name__}: {exc}",
                )
                return

        await advance_cursor(
            customer_id=row.customer_id,
            source=row.source,
            resource_id=row.resource_id,
            new_cursor_value=result.next_cursor,
        )
