"""Entry point for the prbe-knowledge-wiki-bootstrap fly app.

Listens on ``WIKI_BOOTSTRAP_CHANNEL`` for trigger events. Each NOTIFY
payload carries ``{customer_id, sources?, wipe_first?, reason?}`` (JSON);
the listener parses it and calls ``BootstrapOrchestrator.bootstrap``.

Three concurrent asyncio tasks (mirrors ``synthesis_app.py`` /
``triage_app.py``):

  - ``BootstrapListener`` — LISTEN on WIKI_BOOTSTRAP_CHANNEL with full
    payload routing.
  - tiny health server (Fly probe).
  - signal handler to drain on SIGTERM.

Lane C ships with REGISTRY empty. NOTIFYs still parse + log + open run
rows (with no crawlers, every drain is a no-op). Lane D adds the GitHub
crawler at module-import time, which makes this app start producing
real bootstrap output.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import asyncpg
import httpx
import uvicorn

from services.synthesis.bootstrap_orchestrator import (
    BootstrapOrchestrator,
    parse_trigger_payload,
)
from shared.config import get_settings
from shared.constants import WIKI_BOOTSTRAP_CHANNEL
from shared.db import init_pool
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Listener — payload-aware, parses JSON, calls orchestrator directly.
# ---------------------------------------------------------------------------


class BootstrapListener:
    """LISTEN on ``WIKI_BOOTSTRAP_CHANNEL`` and dispatch each NOTIFY
    payload through the orchestrator.

    Mirrors the connection management of ``services.synthesis.listeners.
    NotifyListener`` (dedicated asyncpg conn, exponential reconnect,
    periodic SELECT 1). Differs in that it does not just set a wake
    event — it parses each payload and calls the orchestrator inline.
    """

    def __init__(
        self,
        *,
        dsn: str,
        orchestrator: BootstrapOrchestrator,
    ) -> None:
        self._dsn = dsn
        self._orchestrator = orchestrator
        self._shutdown = asyncio.Event()
        self._task_set: set[asyncio.Task[None]] = set()

    async def run(self) -> None:
        log.info("bootstrap_listener.start", channel=WIKI_BOOTSTRAP_CHANNEL)
        backoff = 1.0
        while not self._shutdown.is_set():
            try:
                conn = await asyncpg.connect(self._dsn)
            except (asyncpg.PostgresError, OSError) as exc:
                log.warning(
                    "bootstrap_listener.connect_failed",
                    error=str(exc),
                    backoff_seconds=backoff,
                )
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._shutdown.wait(), timeout=backoff)
                backoff = min(backoff * 2, 60.0)
                continue

            backoff = 1.0
            try:

                def _on_notify(_conn, _pid, _channel, payload) -> None:
                    log.info(
                        "bootstrap_listener.notified",
                        channel=_channel,
                        payload=payload,
                    )
                    # Dispatch off the listener thread so a long-running
                    # orchestrator drain doesn't block subsequent
                    # NOTIFYs on the same connection.
                    task = asyncio.create_task(self._dispatch(payload))
                    self._task_set.add(task)
                    task.add_done_callback(self._task_set.discard)

                await conn.add_listener(WIKI_BOOTSTRAP_CHANNEL, _on_notify)
                log.info("bootstrap_listener.ready")
                while not self._shutdown.is_set():
                    try:
                        await asyncio.wait_for(self._shutdown.wait(), timeout=30.0)
                    except TimeoutError:
                        try:
                            await conn.fetchval("SELECT 1")
                        except (asyncpg.PostgresError, OSError) as exc:
                            log.warning("bootstrap_listener.lost", error=str(exc))
                            break
            finally:
                with contextlib.suppress(Exception):
                    await conn.close()

        # Drain in-flight orchestrator tasks before declaring shutdown
        # complete. Each task has its own per-source DB rows opened, so
        # the right thing to do is let them finish vs. leave half-closed
        # runs.
        if self._task_set:
            log.info("bootstrap_listener.draining", in_flight=len(self._task_set))
            await asyncio.gather(*self._task_set, return_exceptions=True)
        log.info("bootstrap_listener.stop")

    async def _dispatch(self, payload: str | None) -> None:
        try:
            kwargs = parse_trigger_payload(payload or "{}")
        except Exception as exc:
            log.warning(
                "bootstrap_listener.payload_parse_failed",
                error=str(exc),
                payload=payload,
            )
            return
        customer_id = kwargs.pop("customer_id")
        if not customer_id:
            log.warning("bootstrap_listener.missing_customer_id", payload=payload)
            return
        try:
            await self._orchestrator.bootstrap(customer_id=customer_id, **kwargs)
        except Exception as exc:
            log.warning(
                "bootstrap_listener.orchestrator_crashed",
                customer=customer_id,
                error=str(exc),
                error_class=type(exc).__name__,
            )

    def shutdown(self) -> None:
        self._shutdown.set()


# ---------------------------------------------------------------------------
# Health server (Fly probe)
# ---------------------------------------------------------------------------


def _build_health_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from shared.db import health_check

    app = FastAPI(
        title="prbe-knowledge-wiki-bootstrap health",
        docs_url=None,
        redoc_url=None,
    )

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


@contextlib.asynccontextmanager
async def build_http_client() -> AsyncIterator[httpx.AsyncClient]:
    """Shared httpx client used by every crawler.

    Generous timeout because individual crawlers may pull large pages
    of data; per-call retries / rate-limit handling live inside each
    api_client wrapper.
    """
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=5.0)
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        yield client


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run_bootstrap_app_forever() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    # Import handlers package so any decorators run (mirror the other
    # wiki apps; bootstrap reuses the same Normalizer write path inside
    # the wiki tools).
    import services.ingestion.handlers  # noqa: F401

    # LISTEN must run on a direct (non-pooler) DSN. Neon's pgbouncer
    # transaction-pooler resets `LISTEN *` between txns, so a listener
    # holding a pooled conn never receives any NOTIFY.
    listener_dsn = settings.database_url_unpooled or settings.database_url

    health_port = int(os.environ.get("WORKER_HEALTH_PORT", "8082"))
    # IPv4 bind for Fly's `[[http_service.checks]]` IPv4 health probe.
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
        "bootstrap_app.boot",
        environment=settings.environment,
        health_port=health_port,
        timestamp=datetime.now(UTC).isoformat(),
    )

    async with build_http_client() as http:
        orchestrator = BootstrapOrchestrator(settings=settings, http=http)
        listener = BootstrapListener(dsn=listener_dsn, orchestrator=orchestrator)

        loop = asyncio.get_running_loop()
        gather_future: asyncio.Future | None = None  # type: ignore[type-arg]
        shutdown_started = False

        def handle_signal(signame: str) -> None:
            nonlocal shutdown_started
            if shutdown_started:
                return
            shutdown_started = True
            log.info("bootstrap_app.shutdown_signal", signal=signame)
            listener.shutdown()
            health_server.should_exit = True
            if gather_future is not None and not gather_future.done():
                gather_future.cancel()

        for signame in ("SIGTERM", "SIGINT"):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(getattr(signal, signame), handle_signal, signame)

        gather_future = asyncio.gather(
            listener.run(),
            health_server.serve(),
        )
        try:
            await gather_future
        except asyncio.CancelledError:
            log.info("bootstrap_app.shutdown_complete")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_bootstrap_app_forever())
