"""Async worker: drains ingestion_queue and runs the normalizer per row.

Uses `SELECT ... FOR UPDATE SKIP LOCKED` so many workers can run concurrently
without stepping on each other. Heartbeat every QUEUE_HEARTBEAT_INTERVAL_SECONDS
so the reclaim cron can detect stuck rows.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime

import asyncpg

from services.ingestion.handlers.base import ConnectorContext, make_default_context
from services.ingestion.normalizer import Normalizer
from shared.constants import (
    QUEUE_HEARTBEAT_INTERVAL_SECONDS,
    IngestionEventStatus,
    QueueStatus,
    SourceSystem,
)
from shared.db import get_pool, init_pool
from shared.exceptions import (
    DuplicateEventIgnored,
    PrbeError,
    UnsupportedEventType,
)
from shared.logging import bind_trace, get_logger

log = get_logger(__name__)


class Worker:
    def __init__(self, ctx: ConnectorContext, max_attempts: int = 5) -> None:
        self._ctx = ctx
        self._normalizer = Normalizer(ctx)
        self._max_attempts = max_attempts
        self._shutdown = asyncio.Event()

    async def run(self, poll_interval: float = 1.0) -> None:
        log.info("worker.start", max_attempts=self._max_attempts)
        while not self._shutdown.is_set():
            claimed = await self._claim_one()
            if claimed is None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._shutdown.wait(), timeout=poll_interval
                    )
                continue
            await self._process(claimed)
        log.info("worker.stop")

    def shutdown(self) -> None:
        self._shutdown.set()

    # ---- queue ops ----------------------------------------------------------

    async def _claim_one(self) -> asyncpg.Record | None:
        """Atomically mark one pending row as processing and return it."""
        async with get_pool().acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                    SELECT queue_id, customer_id, source_system, source_event_id,
                           payload_s3_key, attempts
                    FROM ingestion_queue
                    WHERE status = $1
                    ORDER BY enqueued_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                    """,
                QueueStatus.PENDING.value,
            )
            if row is None:
                return None
            await conn.execute(
                """
                    UPDATE ingestion_queue
                    SET status = $1, started_at = NOW(), heartbeat_at = NOW(),
                        attempts = attempts + 1
                    WHERE queue_id = $2
                    """,
                QueueStatus.PROCESSING.value,
                row["queue_id"],
            )
            return row

    async def _process(self, row: asyncpg.Record) -> None:
        queue_id = row["queue_id"]
        customer_id = row["customer_id"]
        source = SourceSystem(row["source_system"])
        event_id = row["source_event_id"]
        attempts = row["attempts"] + 1

        bind_trace(f"queue-{queue_id}")
        heartbeat_task = asyncio.create_task(self._heartbeat(queue_id))
        try:
            outcome = await self._normalizer.process_queue_row(
                queue_id=queue_id,
                customer_id=customer_id,
                source_system=source,
                source_event_id=event_id,
                payload_s3_key=row["payload_s3_key"],
            )
            await self._mark_done(queue_id, customer_id, source, event_id, outcome)
        except DuplicateEventIgnored as exc:
            log.info("worker.skipped", queue_id=queue_id, reason=str(exc))
            await self._mark_skipped(queue_id, customer_id, source, event_id, str(exc))
        except UnsupportedEventType as exc:
            log.info("worker.unsupported", queue_id=queue_id, reason=str(exc))
            await self._mark_skipped(queue_id, customer_id, source, event_id, str(exc))
        except PrbeError as exc:
            transient = getattr(exc, "transient", False)
            await self._on_error(queue_id, attempts, str(exc), transient=transient)
        except Exception as exc:  # pragma: no cover — last-resort
            log.exception("worker.unhandled", queue_id=queue_id)
            await self._on_error(queue_id, attempts, repr(exc), transient=False)
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task

    async def _heartbeat(self, queue_id: int) -> None:
        while True:
            await asyncio.sleep(QUEUE_HEARTBEAT_INTERVAL_SECONDS)
            async with get_pool().acquire() as conn:
                await conn.execute(
                    "UPDATE ingestion_queue SET heartbeat_at = NOW() WHERE queue_id = $1",
                    queue_id,
                )

    async def _mark_done(
        self,
        queue_id: int,
        customer_id: str,
        source: SourceSystem,
        event_id: str,
        outcome,
    ) -> None:
        async with get_pool().acquire() as conn, conn.transaction():
            await conn.execute(
                """
                    UPDATE ingestion_queue
                    SET status = $1, completed_at = NOW(), error = NULL
                    WHERE queue_id = $2
                    """,
                QueueStatus.DONE.value,
                queue_id,
            )
            await conn.execute(
                """
                    INSERT INTO ingestion_events (
                        customer_id, source_system, event_type, source_event_id,
                        payload_s3_key, status, doc_ids_produced, processed_at,
                        normalizer_version
                    ) VALUES ($1, $2, 'webhook', $3,
                              (SELECT payload_s3_key FROM ingestion_queue WHERE queue_id=$4),
                              $5, $6, NOW(), 'v1')
                    ON CONFLICT (customer_id, source_system, source_event_id)
                    DO UPDATE SET status = EXCLUDED.status, processed_at = NOW(),
                                  doc_ids_produced = EXCLUDED.doc_ids_produced
                    """,
                customer_id,
                source.value,
                event_id,
                queue_id,
                IngestionEventStatus.PROCESSED.value,
                outcome.doc_ids,
            )

    async def _mark_skipped(
        self,
        queue_id: int,
        customer_id: str,
        source: SourceSystem,
        event_id: str,
        reason: str,
    ) -> None:
        async with get_pool().acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_queue
                SET status = $1, completed_at = NOW(), error = $2
                WHERE queue_id = $3
                """,
                QueueStatus.DONE.value,
                reason,
                queue_id,
            )
            await conn.execute(
                """
                INSERT INTO ingestion_events (
                    customer_id, source_system, event_type, source_event_id,
                    payload_s3_key, status, error, processed_at, normalizer_version
                ) VALUES ($1, $2, 'webhook', $3,
                          (SELECT payload_s3_key FROM ingestion_queue WHERE queue_id=$4),
                          $5, $6, NOW(), 'v1')
                ON CONFLICT (customer_id, source_system, source_event_id)
                DO UPDATE SET status = EXCLUDED.status, error = EXCLUDED.error,
                              processed_at = NOW()
                """,
                customer_id,
                source.value,
                event_id,
                queue_id,
                IngestionEventStatus.SKIPPED.value,
                reason,
            )

    async def _on_error(
        self, queue_id: int, attempts: int, error: str, *, transient: bool
    ) -> None:
        dead = (not transient) or attempts >= self._max_attempts
        log.warning(
            "worker.error",
            queue_id=queue_id,
            attempts=attempts,
            transient=transient,
            dead=dead,
            error=error,
        )
        async with get_pool().acquire() as conn:
            if dead:
                await conn.execute(
                    """
                    UPDATE ingestion_queue
                    SET status = $1, completed_at = NOW(), error = $2
                    WHERE queue_id = $3
                    """,
                    QueueStatus.DLQ.value,
                    error,
                    queue_id,
                )
            else:
                await conn.execute(
                    """
                    UPDATE ingestion_queue
                    SET status = $1, error = $2, heartbeat_at = NULL, started_at = NULL
                    WHERE queue_id = $3
                    """,
                    QueueStatus.PENDING.value,
                    error,
                    queue_id,
                )


async def run_worker_forever() -> None:
    """Entry point for `python -m services.ingestion.worker`."""
    from shared.config import get_settings
    from shared.logging import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    # Import handlers package so @register_connector decorators run.
    import services.ingestion.handlers  # noqa: F401

    ctx = make_default_context()
    worker = Worker(ctx, max_attempts=settings.worker_max_attempts)
    log.info(
        "worker.boot",
        environment=settings.environment,
        timestamp=datetime.now(UTC).isoformat(),
    )
    try:
        await worker.run(poll_interval=settings.worker_poll_interval_seconds)
    finally:
        await ctx.http.aclose()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_worker_forever())
