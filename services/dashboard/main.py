"""Dashboard backend.

Lean FastAPI service that bridges Better Auth's Organization plugin to
prbe-knowledge data plane. Most team / member / invitation operations
flow directly from the Next.js client to Neon Auth via the
`@neondatabase/auth` SDK; this service handles the bits that only our
side knows about:

  * customers <-> organization linking
  * soft-delete + audit
  * Neon Auth webhook receiver (signature-verified)
  * prbe-knowledge data endpoints (integrations, ingestion, query)

Auth: every non-webhook route validates the Authorization: Bearer JWT
issued by Neon Auth (JWKS-verified). The active org and role are
resolved per request from neon_auth.member.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from services.dashboard import webhook
from services.dashboard.routers import teams
from shared.config import get_settings
from shared.db import health_check, init_pool
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    if not settings.neon_auth_base_url:
        log.warning("dashboard.boot.no_neon_auth_base_url")
    log.info("dashboard.boot", environment=settings.environment)
    yield


app = FastAPI(title="prbe-knowledge dashboard", lifespan=lifespan)
app.include_router(teams.router, prefix="/api")
app.include_router(webhook.router)


@app.get("/health")
async def health() -> JSONResponse:
    db_ok = await health_check()
    body = {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "neon_auth_configured": get_settings().neon_auth_base_url is not None,
        "time": datetime.now(UTC).isoformat(),
    }
    return JSONResponse(body, status_code=200 if db_ok else 503)
