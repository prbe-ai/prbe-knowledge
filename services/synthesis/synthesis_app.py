"""Entry point for the prbe-knowledge-wiki-synthesis fly app.

Runs three concurrent asyncio tasks:
  - NotifyListener on `wiki_synthesize_triaged` (sets wake_event).
  - SynthesisWorker.run (drains triaged → done/verifier_rejected).
  - tiny health server.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from datetime import UTC, datetime

import uvicorn

from services.synthesis.listeners import NotifyListener
from services.synthesis.reclaim import WikiReclaimLoop
from services.synthesis.synthesis_worker import SynthesisWorker
from shared.config import get_settings
from shared.constants import WIKI_SYNTHESIS_MAX_ATTEMPTS, WIKI_TRIAGED_CHANNEL
from shared.db import init_pool
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)


def _build_health_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from shared.db import health_check

    app = FastAPI(
        title="prbe-knowledge-wiki-synthesis health",
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


async def run_synthesis_app_forever() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    import services.ingestion.handlers  # noqa: F401

    wake_event = asyncio.Event()
    listener = NotifyListener(
        settings.database_url,
        WIKI_TRIAGED_CHANNEL,
        wake_event,
        log_prefix="synthesis_listener",
    )
    worker = SynthesisWorker(wake_event)
    reclaim_loop = WikiReclaimLoop(max_attempts=WIKI_SYNTHESIS_MAX_ATTEMPTS)

    health_port = int(os.environ.get("WORKER_HEALTH_PORT", "8082"))
    # See triage_app.py for the rationale on 0.0.0.0 vs ::. Same fix.
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
        "synthesis_app.boot",
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
        log.info("synthesis_app.shutdown_signal", signal=signame)
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
        log.info("synthesis_app.shutdown_complete")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_synthesis_app_forever())
