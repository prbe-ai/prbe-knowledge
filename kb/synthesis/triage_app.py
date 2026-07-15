"""Entry point for the prbe-knowledge-wiki-worker fly app.

Runs three concurrent asyncio tasks:
  - NotifyListener on `wiki_synthesize_pending` (sets wake_event).
  - TriageWorker.run (drains pending → triaged on wake / periodic timer).
  - tiny health server (so Fly's HTTP probe has something to hit).

Mirrors `engine.ingest.worker.run_worker_forever` shape — same
graceful-shutdown signal handling, same uvicorn health server.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from datetime import UTC, datetime

import uvicorn

from engine.shared.config import get_settings
from engine.shared.constants import WIKI_PENDING_CHANNEL, WIKI_SYNTHESIS_MAX_ATTEMPTS
from engine.shared.db import init_pool
from engine.shared.logging import configure_logging, get_logger
from kb.synthesis.listeners import NotifyListener
from kb.synthesis.reclaim import WikiReclaimLoop
from kb.synthesis.triage_worker import TriageWorker

log = get_logger(__name__)


def _build_health_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from engine.shared.db import health_check

    app = FastAPI(
        title="prbe-knowledge-wiki-worker health",
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


async def run_triage_app_forever() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    # Import handlers package so any decorators run (the triage worker
    # doesn't itself import handlers, but the persistence layer's
    # body-fetch path uses the same chunks join the connectors set up).
    import kb.handlers  # noqa: F401

    wake_event = asyncio.Event()
    # LISTEN must run on a direct (non-pooler) DSN. Neon's pgbouncer
    # transaction-pooler resets `LISTEN *` between txns, so a listener
    # holding a pooled conn never receives any NOTIFY. See config.py
    # `database_url_unpooled` for the rationale; falls back to
    # `database_url` for local dev / non-pooler deploys.
    listener_dsn = settings.database_url_unpooled or settings.database_url
    listener = NotifyListener(
        listener_dsn,
        WIKI_PENDING_CHANNEL,
        wake_event,
        log_prefix="triage_listener",
    )
    worker = TriageWorker(wake_event)
    reclaim_loop = WikiReclaimLoop(max_attempts=WIKI_SYNTHESIS_MAX_ATTEMPTS)

    health_port = int(os.environ.get("WORKER_HEALTH_PORT", "8082"))
    # Bind on 0.0.0.0 to match services/ingestion/worker.py — Fly's
    # `[[http_service.checks]]` health probe reaches the machine via
    # IPv4, and on this Docker base image binding `::` doesn't accept
    # IPv4 connections (kernel default `bindv6only=1` or runtime quirk).
    # Result was every health check getting ECONNREFUSED until the
    # 30-min hard limit hit and Fly SIGTERMed the machine. The wiki
    # workers don't need 6PN-internal HTTP ingress (they're driven by
    # pg_notify, not RPC), so IPv4-only is correct here.
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
        "triage_app.boot",
        environment=settings.environment,
        health_port=health_port,
        timestamp=datetime.now(UTC).isoformat(),
    )

    loop = asyncio.get_running_loop()
    gather_future: asyncio.Future | None = None  # type: ignore[type-arg]
    shutdown_started = False

    def handle_signal(signame: str) -> None:
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True
        log.info("triage_app.shutdown_signal", signal=signame)
        listener.shutdown()
        worker.shutdown()
        reclaim_loop.shutdown()
        health_server.should_exit = True
        if gather_future is not None and not gather_future.done():
            gather_future.cancel()

    for signame in ("SIGTERM", "SIGINT"):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(getattr(signal, signame), handle_signal, signame)

    gather_future = asyncio.gather(
        listener.run(),
        worker.run(),
        reclaim_loop.run(),
        health_server.serve(),
    )
    try:
        await gather_future
    except asyncio.CancelledError:
        log.info("triage_app.shutdown_complete")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_triage_app_forever())
