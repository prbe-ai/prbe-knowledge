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

from engine.ingest.handlers.base import ConnectorContext
from engine.ingest.normalizer import Normalizer
from engine.shared.constants import (
    QUEUE_HEARTBEAT_INTERVAL_SECONDS,
    QUEUE_RECLAIM_THRESHOLD_SECONDS,
    BackfillStatus,
    IngestionEventStatus,
    IngestionEventType,
    QueueStatus,
    SourceSystem,
)
from engine.shared.db import get_pool
from engine.shared.exceptions import (
    DuplicateEventIgnored,
    PrbeError,
    UnsupportedEventType,
)
from engine.shared.logging import bind_trace, get_logger
from engine.shared.storage import get_store

log = get_logger(__name__)


class Worker:
    def __init__(
        self,
        ctx: ConnectorContext,
        max_attempts: int = 5,
        concurrency: int = 1,
        per_customer_max_inflight: int = 10,
    ) -> None:
        self._ctx = ctx
        self._normalizer = Normalizer(ctx)
        self._max_attempts = max_attempts
        self._concurrency = max(1, concurrency)
        self._per_customer_max_inflight = max(1, per_customer_max_inflight)
        self._shutdown = asyncio.Event()

    async def run(self, poll_interval: float = 1.0) -> None:
        log.info(
            "worker.start",
            max_attempts=self._max_attempts,
            concurrency=self._concurrency,
            per_customer_max_inflight=self._per_customer_max_inflight,
        )
        # Loud boot signal when the connector registry is empty. engine/
        # never imports kb/ itself; the deploy wrapper (python -m
        # services.ingestion.worker -> kb.worker) imports kb.handlers. A
        # direct engine launch without a connector pack would otherwise
        # sit "healthy" while every claimed row DLQs with HandlerNotFound.
        from engine.ingest.handlers.registry import list_registered

        if not list_registered():
            log.error(
                "worker.boot.connector_registry_empty",
                hint=(
                    "no connectors are registered -- every queue row will "
                    "fail with HandlerNotFound. Launch via the deploy "
                    "wrapper (python -m services.ingestion.worker) or "
                    "import kb.handlers (or your connector pack) before "
                    "starting the worker."
                ),
            )
        # Each loop independently claims via FOR UPDATE SKIP LOCKED, so N
        # parallel loops in the same process safely share the queue.
        await asyncio.gather(*(self._claim_loop(poll_interval) for _ in range(self._concurrency)))
        log.info("worker.stop")

    async def _claim_loop(self, poll_interval: float) -> None:
        while not self._shutdown.is_set():
            claimed = await self._claim_one()
            if claimed is None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._shutdown.wait(), timeout=poll_interval)
                continue
            await self._process(claimed)

    def shutdown(self) -> None:
        self._shutdown.set()

    # ---- queue ops ----------------------------------------------------------

    async def _claim_one(self) -> asyncpg.Record | None:
        """Atomically mark one pending row as processing and return it.

        Returns: queue_id, customer_id, source_system, source_event_id,
        payload_s3_key (legacy back-compat), payload_s3_keys (the array
        of R2 paths coalesced into the row — for non-CC this is
        single-element), version (monotonic counter for the CAS commit),
        attempts.

        Higher `priority` claims first. Tier order at insert time
        (shared.source_registry, registered per connector): live(100) >
        claude_code(75) > backfill(50). One chatty CC user can't block
        github/slack/notion/linear/granola/sentry traffic.

        Coalescing collapses N batches of the same Claude Code session
        into one queue row (services/ingestion/main.py:_enqueue UPSERTs
        on (customer_id, source_system, session_id) and bumps version).
        So same-session serialization is structural — there's only ever
        one row per session — and the explicit NOT EXISTS clause that
        approximated this in PR #33 is now dead code, removed here.
        """
        async with get_pool().acquire() as conn, conn.transaction():
            # Per-customer in-flight cap: a customer with N rows already in
            # `processing` is excluded from this claim. Soft cap (snapshot
            # count, not a hard lock) — the goal is preventing one
            # workspace's install-time burst from monopolizing the fleet,
            # not strict enforcement. Two racing claim loops can both pass
            # the threshold and over-spill by 1; that's fine.
            row = await conn.fetchrow(
                """
                    WITH inflight AS (
                        SELECT customer_id, COUNT(*) AS cnt
                        FROM ingestion_queue
                        WHERE status = $2
                        GROUP BY customer_id
                    )
                    SELECT q.queue_id, q.customer_id, q.source_system, q.source_event_id,
                           q.payload_s3_key, q.payload_s3_keys, q.version, q.attempts
                    FROM ingestion_queue q
                    LEFT JOIN inflight i ON i.customer_id = q.customer_id
                    WHERE q.status = $1
                      AND COALESCE(i.cnt, 0) < $3
                    ORDER BY q.priority DESC, q.enqueued_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                    """,
                QueueStatus.PENDING.value,
                QueueStatus.PROCESSING.value,
                self._per_customer_max_inflight,
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
        # Coalesced rows have payload_s3_keys (array). Legacy rows from
        # before migration 0026 have payload_s3_key (single string) and
        # an empty array — the migration backfilled all existing rows so
        # the array should always be non-empty in practice. Belt and
        # suspenders: fall back to wrapping the single key.
        payload_s3_keys: list[str] = list(row["payload_s3_keys"] or [])
        if not payload_s3_keys and row["payload_s3_key"]:
            payload_s3_keys = [row["payload_s3_key"]]
        # The legacy single-key path is still used to populate
        # ingestion_events.payload_s3_key for audit. Use the first
        # (oldest) coalesced key — stable across re-claims.
        payload_s3_key = payload_s3_keys[0] if payload_s3_keys else ""
        # Captured version: the CAS guard for commit. If a new batch
        # lands during Phase A (embeds) and bumps the row's version,
        # _mark_done's WHERE version=$captured matches 0 rows and the
        # row stays 'pending' to be re-claimed with the extended array.
        captured_version: int = row["version"]
        attempts = row["attempts"] + 1

        bind_trace(f"queue-{queue_id}")
        heartbeat_task = asyncio.create_task(self._heartbeat(queue_id))
        try:
            outcome = await self._normalizer.process_queue_row(
                queue_id=queue_id,
                customer_id=customer_id,
                source_system=source,
                source_event_id=event_id,
                payload_s3_keys=payload_s3_keys,
            )
            if source == SourceSystem.MANUAL_UPLOAD:
                await self._cleanup_manual_upload_original(customer_id, event_id)
            await self._mark_done(
                queue_id,
                customer_id,
                source,
                event_id,
                payload_s3_key,
                outcome,
                captured_version,
            )
        except DuplicateEventIgnored as exc:
            log.info("worker.skipped", queue_id=queue_id, reason=str(exc))
            await self._mark_skipped(
                queue_id,
                customer_id,
                source,
                event_id,
                payload_s3_key,
                str(exc),
                captured_version,
            )
        except UnsupportedEventType as exc:
            log.info("worker.unsupported", queue_id=queue_id, reason=str(exc))
            await self._mark_skipped(
                queue_id,
                customer_id,
                source,
                event_id,
                payload_s3_key,
                str(exc),
                captured_version,
            )
        except PrbeError as exc:
            transient = getattr(exc, "transient", False)
            dead = await self._on_error(
                queue_id,
                attempts,
                str(exc),
                transient=transient,
                captured_version=captured_version,
            )
            if dead and source == SourceSystem.MANUAL_UPLOAD:
                with contextlib.suppress(Exception):
                    await self._mark_manual_upload_failed_ingest(customer_id, event_id, str(exc))
        except Exception as exc:  # pragma: no cover — last-resort
            # Unknown error: assume transient so we don't burn data on a single
            # network blip / OOM / unwrapped httpx error / asyncpg connection
            # drop. Deterministic bugs still DLQ after worker_max_attempts.
            # Anything that should permanently DLQ on first try must raise a
            # PrbeError subclass with `transient = False`.
            log.exception("worker.unhandled", queue_id=queue_id)
            error = repr(exc)
            dead = await self._on_error(
                queue_id,
                attempts,
                error,
                transient=True,
                captured_version=captured_version,
            )
            if dead and source == SourceSystem.MANUAL_UPLOAD:
                with contextlib.suppress(Exception):
                    await self._mark_manual_upload_failed_ingest(customer_id, event_id, error)
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
        payload_s3_key: str,
        outcome,
        captured_version: int,
    ) -> None:
        """CAS-commit the row to status='done'.

        The UPDATE has `WHERE version = $captured_version`. If a new batch
        landed during Phase A (UPSERT in services/ingestion/main.py:_enqueue
        bumps version), the row's version advanced and the WHERE matches
        0 rows. We log `worker.cas_retry` and leave the row at 'processing'
        — the heartbeat reclaim cron picks it up at the threshold and the
        worker re-runs Phase A on the now-extended payload_s3_keys array.
        Phase A is naturally idempotent: chunks dedupe by content_hash,
        so re-running only re-embeds genuinely new content.
        """
        async with get_pool().acquire() as conn, conn.transaction():
            done_row = await conn.fetchrow(
                """
                UPDATE ingestion_queue
                SET status = $1, completed_at = NOW(), error = NULL
                WHERE queue_id = $2 AND version = $3
                RETURNING queue_id
                """,
                QueueStatus.DONE.value,
                queue_id,
                captured_version,
            )
            if done_row is None:
                # New batch landed mid-Phase-A. Don't write ingestion_events
                # yet (the next attempt will), don't mark done. Reclaim cron
                # will pick this row up at heartbeat threshold and the worker
                # re-runs against the extended payload_s3_keys array.
                log.info(
                    "worker.cas_retry",
                    queue_id=queue_id,
                    captured_version=captured_version,
                    reason="new batch arrived during processing",
                )
                return
            await conn.execute(
                """
                INSERT INTO ingestion_events (
                    customer_id, source_system, event_type, source_event_id,
                    payload_s3_key, status, doc_ids_produced, processed_at,
                    normalizer_version
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), 'v1')
                ON CONFLICT (customer_id, source_system, source_event_id)
                DO UPDATE SET status = EXCLUDED.status, processed_at = NOW(),
                              doc_ids_produced = EXCLUDED.doc_ids_produced
                """,
                customer_id,
                source.value,
                _event_type_for_source(source),
                event_id,
                payload_s3_key,
                IngestionEventStatus.PROCESSED.value,
                outcome.doc_ids,
            )

    async def _mark_skipped(
        self,
        queue_id: int,
        customer_id: str,
        source: SourceSystem,
        event_id: str,
        payload_s3_key: str,
        reason: str,
        captured_version: int,
    ) -> None:
        """CAS-commit the row to status='done' with a skipped error reason.

        Same CAS guard as _mark_done — if a new batch arrived during
        processing, leave the row pending and let reclaim re-run.
        """
        async with get_pool().acquire() as conn, conn.transaction():
            done_row = await conn.fetchrow(
                """
                UPDATE ingestion_queue
                SET status = $1, completed_at = NOW(), error = $2
                WHERE queue_id = $3 AND version = $4
                RETURNING queue_id
                """,
                QueueStatus.DONE.value,
                reason,
                queue_id,
                captured_version,
            )
            if done_row is None:
                log.info(
                    "worker.cas_retry",
                    queue_id=queue_id,
                    captured_version=captured_version,
                    reason="new batch arrived during processing (skipped path)",
                )
                return
            await conn.execute(
                """
                INSERT INTO ingestion_events (
                    customer_id, source_system, event_type, source_event_id,
                    payload_s3_key, status, error, processed_at, normalizer_version
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), 'v1')
                ON CONFLICT (customer_id, source_system, source_event_id)
                DO UPDATE SET status = EXCLUDED.status, error = EXCLUDED.error,
                              processed_at = NOW()
                """,
                customer_id,
                source.value,
                _event_type_for_source(source),
                event_id,
                payload_s3_key,
                IngestionEventStatus.SKIPPED.value,
                reason,
            )

    async def _cleanup_manual_upload_original(
        self,
        customer_id: str,
        upload_id: str,
    ) -> None:
        """Delete staged original bytes after normalization succeeds.

        This runs before _mark_done. If storage deletion fails, the queue row
        remains retryable and the original object is not orphaned.
        """
        async with get_pool().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT staging_object_key, original_deleted_at
                FROM manual_uploads
                WHERE customer_id = $1 AND upload_id = $2
                """,
                customer_id,
                upload_id,
            )
        if row is None:
            log.warning(
                "manual_upload.cleanup_missing_row",
                customer=customer_id,
                upload_id=upload_id,
            )
            return

        staging_key = row["staging_object_key"]
        should_delete = staging_key and row["original_deleted_at"] is None
        if should_delete:
            store = get_store()
            await store.delete(await store.bucket_for(customer_id), staging_key)

        async with get_pool().acquire() as conn:
            await conn.execute(
                """
                UPDATE manual_uploads
                SET status = 'indexed',
                    indexed_at = COALESCE(indexed_at, NOW()),
                    original_deleted_at = COALESCE(original_deleted_at, NOW()),
                    updated_at = NOW(),
                    parse_error = NULL
                WHERE customer_id = $1 AND upload_id = $2
                """,
                customer_id,
                upload_id,
            )

    async def _mark_manual_upload_failed_ingest(
        self,
        customer_id: str,
        upload_id: str,
        error: str,
    ) -> None:
        async with get_pool().acquire() as conn:
            await conn.execute(
                """
                UPDATE manual_uploads
                SET status = 'failed_ingest',
                    parse_error = $3,
                    updated_at = NOW()
                WHERE customer_id = $1 AND upload_id = $2
                """,
                customer_id,
                upload_id,
                error[:4000],
            )

    async def _on_error(
        self,
        queue_id: int,
        attempts: int,
        error: str,
        *,
        transient: bool,
        captured_version: int,
    ) -> bool:
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
                result = await conn.execute(
                    """
                    UPDATE ingestion_queue
                    SET status = $1, completed_at = NOW(), error = $2
                    WHERE queue_id = $3 AND version = $4
                    """,
                    QueueStatus.DLQ.value,
                    error,
                    queue_id,
                    captured_version,
                )
            else:
                result = await conn.execute(
                    """
                    UPDATE ingestion_queue
                    SET status = $1, error = $2, heartbeat_at = NULL, started_at = NULL
                    WHERE queue_id = $3 AND version = $4
                    """,
                    QueueStatus.PENDING.value,
                    error,
                    queue_id,
                    captured_version,
                )
        if result == "UPDATE 0":
            log.info(
                "worker.cas_retry",
                queue_id=queue_id,
                captured_version=captured_version,
                reason="new batch arrived during processing (error path)",
            )
            return False
        return dead


def _event_type_for_source(source: SourceSystem) -> str:
    if source == SourceSystem.MANUAL_UPLOAD:
        return IngestionEventType.MANUAL.value
    return IngestionEventType.WEBHOOK.value


class ReclaimLoop:
    """Periodically resets stuck `processing` rows back to `pending`.

    A worker that crashes mid-`_process` (OOM, SIGKILL, host migration,
    deploy mid-row) leaves its claimed row at status='processing' with a
    stale heartbeat. Without reclaim that row is wedged forever, and
    because ingestion_queue's UNIQUE (customer_id, source_system,
    source_event_id) blocks redeliveries from the source platform's
    retry, the event is permanently lost.

    Runs in-process inside the worker so we don't need separate cron
    infra. Single worker machine ⇒ single reclaim loop ⇒ no race.

    The reclaim UPDATE is NOT fenced on `attempts` today — a long-running
    but still-alive worker (slow embed, OpenAI backoff) whose heartbeat
    lapses briefly will have its row reclaimed and re-claimed by the next
    poll. This is acceptable under single-machine deployment because the
    worker discards its in-flight work on the doubly-claimed row; if we
    scale out, add `AND attempts = $4` to the UPDATE plus an attempts
    bump on claim to make the fence airtight. See the warning in
    fly.worker.toml.
    """

    def __init__(
        self,
        threshold_seconds: int = QUEUE_RECLAIM_THRESHOLD_SECONDS,
        interval_seconds: float = 120.0,
        backfill_threshold_seconds: int = QUEUE_RECLAIM_THRESHOLD_SECONDS,
    ) -> None:
        self._threshold = threshold_seconds
        self._interval = interval_seconds
        self._backfill_threshold = backfill_threshold_seconds
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        log.info(
            "reclaim_loop.start",
            threshold_seconds=self._threshold,
            interval_seconds=self._interval,
        )
        # Sweep immediately on boot to catch rows stranded by a non-graceful
        # shutdown (OOM/SIGKILL); 5-min reclaim threshold > 30s heartbeat so a
        # freshly-claimed live row has ample margin and won't be stolen.
        while not self._shutdown.is_set():
            try:
                queue_n, backfill_n = await self._reclaim_once()
                if queue_n or backfill_n:
                    log.warning(
                        "reclaim_loop.reclaimed",
                        queue_count=queue_n,
                        backfill_count=backfill_n,
                    )
            except Exception:  # pragma: no cover — keep loop alive
                log.exception("reclaim_loop.tick_failed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._shutdown.wait(), timeout=self._interval)
        log.info("reclaim_loop.stop")

    async def _reclaim_once(self) -> tuple[int, int]:
        async with get_pool().acquire() as conn:
            rows = await conn.fetch(
                """
                UPDATE ingestion_queue
                SET status = $1,
                    heartbeat_at = NULL,
                    started_at = NULL,
                    error = COALESCE(error, '') || ' | reclaimed: heartbeat stale'
                WHERE status = $2
                  AND heartbeat_at IS NOT NULL
                  AND heartbeat_at < NOW() - make_interval(secs => $3)
                RETURNING queue_id, customer_id, source_system, attempts
                """,
                QueueStatus.PENDING.value,
                QueueStatus.PROCESSING.value,
                self._threshold,
            )
            # last_cursor and events_enqueued are intentionally preserved so
            # the next worker resumes exactly where the dead one stopped.
            backfill_rows = await conn.fetch(
                """
                UPDATE backfill_state
                   SET status = $1,
                       started_at = NULL,
                       heartbeat_at = NULL,
                       last_error = COALESCE(last_error, '') || ' | reclaimed: heartbeat stale'
                 WHERE status = $2
                   AND heartbeat_at IS NOT NULL
                   AND heartbeat_at < NOW() - make_interval(secs => $3)
                RETURNING customer_id, source_system
                """,
                BackfillStatus.PENDING.value,
                BackfillStatus.RUNNING.value,
                self._backfill_threshold,
            )
        for r in rows:
            log.warning(
                "queue.reclaimed",
                queue_id=r["queue_id"],
                customer=r["customer_id"],
                source=r["source_system"],
                attempts=r["attempts"],
            )
        for r in backfill_rows:
            log.warning(
                "backfill.reclaimed",
                customer_id=r["customer_id"],
                source_system=r["source_system"],
                stale_seconds_threshold=self._backfill_threshold,
            )
        return len(rows), len(backfill_rows)

    def shutdown(self) -> None:
        self._shutdown.set()


def _build_health_app():
    """FastAPI app exposing `/health` so Fly can probe liveness.

    Deliberately separate from the full ingestion app — this one runs
    alongside the drain loop in the worker process so Fly's health
    check has something to hit.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from engine.shared.db import health_check

    app = FastAPI(title="prbe-knowledge worker health", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> JSONResponse:
        db_ok = await health_check()
        body = {
            "status": "ok" if db_ok else "degraded",
            "db": db_ok,
            "time": datetime.now(UTC).isoformat(),
        }
        return JSONResponse(body, status_code=200 if db_ok else 503)

    return app


