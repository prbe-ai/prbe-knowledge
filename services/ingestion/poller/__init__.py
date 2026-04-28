"""Granola scheduler process — runs as Fly app `prbe-knowledge-poller`.

Tiny single-instance process whose only job is to re-enqueue Granola backfills
on a 5-minute cadence so steady-state polling continues after the initial sync
completes. Manual refreshes go through /admin/.../granola/refresh and don't
involve this process.

Why a separate Fly app instead of bundling into the worker (see plan-eng-review
A1 decision): operational isolation. The worker process owns the heavy
embedding + DB-write loop; this process owns the schedule. Single-instance
is required because there's no leader election — see the comment in
fly.poller.toml. Switch to a distributed lock (Redis/Postgres advisory) before
horizontally scaling.

Flow per tick:
    1. SELECT all customers with an active Granola token whose backfill_state
       is 'complete' or 'failed' AND last_progress_at < now - 5min
    2. For each: call re_enqueue_for_polling (preserves cursor watermark)
    3. NOTIFY granola_refresh so the worker's BackfillWorker wakes immediately

Backfill EXECUTION still happens in the worker process. This process never
calls Granola's API directly — it only flips backfill_state rows to pending.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime

from services.ingestion.backfill_runner import re_enqueue_for_polling
from shared.config import get_settings
from shared.constants import (
    GRANOLA_POLL_INTERVAL_SECONDS,
    GRANOLA_REFRESH_CHANNEL,
    BackfillStatus,
    IntegrationStatus,
    SourceSystem,
)
from shared.db import get_pool, init_pool, raw_conn
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)


class GranolaScheduler:
    """Wakes every GRANOLA_POLL_INTERVAL_SECONDS and re-enqueues stale backfills."""

    def __init__(self, interval_seconds: int = GRANOLA_POLL_INTERVAL_SECONDS) -> None:
        self._interval = interval_seconds
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        log.info("granola_scheduler.start", interval_seconds=self._interval)
        # Run a tick immediately at boot so a freshly-deployed poller doesn't
        # wait the full interval before its first sweep.
        await self._tick_once()
        while not self._shutdown.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self._interval
                )
            if self._shutdown.is_set():
                break
            await self._tick_once()
        log.info("granola_scheduler.stop")

    def shutdown(self) -> None:
        self._shutdown.set()

    async def _tick_once(self) -> None:
        try:
            customers = await self._fetch_due_customers()
        except Exception:
            log.exception("granola_scheduler.fetch_failed")
            return

        if not customers:
            log.debug("granola_scheduler.tick_idle")
            return

        for customer_id in customers:
            try:
                triggered = await re_enqueue_for_polling(
                    customer_id, SourceSystem.GRANOLA
                )
            except Exception:
                log.exception(
                    "granola_scheduler.enqueue_failed", customer=customer_id
                )
                continue

            if not triggered:
                # Row was already pending or running — skip the notify.
                continue

            try:
                async with get_pool().acquire() as conn:
                    await conn.execute(
                        "SELECT pg_notify($1, $2)",
                        GRANOLA_REFRESH_CHANNEL,
                        customer_id,
                    )
            except Exception:
                log.exception(
                    "granola_scheduler.notify_failed", customer=customer_id
                )

            log.info(
                "granola_scheduler.re_enqueued",
                customer=customer_id,
                tick_at=datetime.now(UTC).isoformat(),
            )

    async def _fetch_due_customers(self) -> list[str]:
        """Customers whose Granola backfill is complete/failed and gone stale.

        'pending' and 'running' rows are skipped — they're already in the
        queue, no point re-enqueueing. NULL backfill_state row means the
        initial backfill never ran (shouldn't happen post-connect since
        connect_granola_route enqueues one) — also skip.
        """
        async with raw_conn() as conn:
            rows = await conn.fetch(
                """
                SELECT t.customer_id
                FROM integration_tokens t
                JOIN backfill_state b
                  ON b.customer_id = t.customer_id
                 AND b.source_system = t.source_system
                WHERE t.source_system = $1
                  AND t.status = $2
                  AND b.status IN ($3, $4)
                  AND (
                    b.last_progress_at IS NULL
                    OR b.last_progress_at < NOW() - make_interval(secs => $5)
                  )
                ORDER BY b.last_progress_at NULLS FIRST
                """,
                SourceSystem.GRANOLA.value,
                IntegrationStatus.ACTIVE.value,
                BackfillStatus.COMPLETE.value,
                BackfillStatus.FAILED.value,
                self._interval,
            )
        return [r["customer_id"] for r in rows]


async def run_poller_forever() -> None:
    """Entry point for `python -m services.ingestion.poller`."""
    import os

    import uvicorn

    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    scheduler = GranolaScheduler()

    health_port = int(os.environ.get("POLLER_HEALTH_PORT", "8083"))
    health_config = uvicorn.Config(
        _build_health_app(),
        host="0.0.0.0",
        port=health_port,
        log_config=None,
        lifespan="off",
        access_log=False,
    )
    health_server = uvicorn.Server(health_config)

    log.info(
        "poller.boot",
        environment=settings.environment,
        health_port=health_port,
        interval_seconds=GRANOLA_POLL_INTERVAL_SECONDS,
    )
    await asyncio.gather(scheduler.run(), health_server.serve())


def _build_health_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from shared.db import health_check

    app = FastAPI(title="prbe-knowledge poller health", docs_url=None, redoc_url=None)

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


__all__ = ["GranolaScheduler", "run_poller_forever"]
