"""Ingestion service — internal worker behind the prbe-backend gateway.

Public webhooks land at api.prbe.ai/webhooks/{source}. prbe-backend
verifies signatures, resolves the tenant via customer_source_mapping,
then POSTs to this service's `/webhooks/{source}` with X-Internal-Knowledge-Key
+ X-Prbe-Customer headers. We trust those headers and skip signature
re-verification — the gateway is the single signing-secret holder.

Per request:
  1. Validate X-Internal-Knowledge-Key.
  2. Read X-Prbe-Customer (no fallback).
  3. Dispatch to the right Connector.
  4. parse_webhook_event → None means ignore.
  5. Persist raw payload to R2.
  6. Insert ingestion_queue row (UNIQUE dedupes redeliveries).
  7. Return 200.

OAuth callbacks also land at the gateway. After verifying state, the
gateway POSTs to /api/oauth/{source}/exchange (admin-key gated) and we
do the per-source token exchange + storage. See admin_routes.py.
"""

from __future__ import annotations

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import orjson
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from services.ingestion.admin_routes import router as admin_router
from services.ingestion.backfill_routes import router as backfill_router
from services.ingestion.internal_devices import router as devices_router
from services.ingestion.handlers.base import make_default_context
from services.ingestion.handlers.registry import (
    build_connector,
    get_connector_class,
    list_registered,
)
from shared.config import get_settings
from shared.constants import SourceSystem
from shared.db import get_pool, health_check, init_pool
from shared.exceptions import (
    HandlerNotFound,
    InvalidWebhookPayload,
    PrbeError,
)
from shared.logging import bind_trace, configure_logging, get_logger
from shared.storage import get_store

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    # Trigger @register_connector decorators.
    import services.ingestion.handlers  # noqa: F401

    app.state.ctx = make_default_context()
    app.state.store = get_store()
    log.info(
        "ingestion.boot",
        environment=settings.environment,
        connectors=[s.value for s in list_registered()],
    )
    try:
        yield
    finally:
        await app.state.ctx.http.aclose()


app = FastAPI(title="prbe-knowledge ingestion", lifespan=lifespan)
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
app.include_router(backfill_router)
app.include_router(admin_router)
app.include_router(devices_router)


@app.get("/health")
async def health() -> JSONResponse:
    db_ok = await health_check()
    body = {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "connectors": [s.value for s in list_registered()],
        "time": datetime.now(UTC).isoformat(),
    }
    return JSONResponse(body, status_code=200 if db_ok else 503)


def _verify_internal_key(request: Request) -> None:
    expected = get_settings().internal_knowledge_api_key
    if expected is None or not expected.get_secret_value():
        raise HTTPException(
            status_code=503,
            detail="INTERNAL_KNOWLEDGE_API_KEY not configured",
        )
    presented = request.headers.get("x-internal-knowledge-key")
    if not presented or not hmac.compare_digest(
        presented, expected.get_secret_value()
    ):
        raise HTTPException(
            status_code=401, detail="missing or invalid X-Internal-Knowledge-Key"
        )


@app.post("/webhooks/{source}")
async def webhook(
    source: str,
    request: Request,
    x_trace_id: str | None = Header(default=None),
    x_prbe_customer: str | None = Header(default=None),
) -> JSONResponse:
    """Internal-only webhook endpoint. Called by prbe-backend gateway.

    Trusts X-Internal-Knowledge-Key + X-Prbe-Customer; does NOT verify the source
    platform's signature (gateway already did).
    """
    _verify_internal_key(request)
    if not x_prbe_customer:
        raise HTTPException(status_code=400, detail="missing X-Prbe-Customer")

    trace_id = x_trace_id or f"wh-{int(datetime.now().timestamp() * 1000)}"
    bind_trace(trace_id)

    try:
        source_enum = SourceSystem(source)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"unknown source '{source}'") from exc

    try:
        get_connector_class(source_enum)
    except HandlerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    raw_body = await request.body()
    connector = build_connector(source_enum, request.app.state.ctx)

    try:
        payload = orjson.loads(raw_body) if raw_body else {}
    except orjson.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc

    customer_id = x_prbe_customer
    try:
        parsed = connector.parse_webhook_event(
            customer_id, dict(request.headers), payload
        )
    except InvalidWebhookPayload as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if parsed is None:
        return JSONResponse({"status": "ignored", "trace_id": trace_id})

    envelope = orjson.dumps(
        {
            "_headers": {k: v for k, v in request.headers.items()},
            "payload": payload,
            "received_at": parsed.received_at.isoformat(),
            "trace_id": trace_id,
        }
    )
    store = request.app.state.store
    bucket = store.bucket_for(customer_id)
    key = _payload_key(source_enum, customer_id, parsed.source_event_id)

    try:
        await store.ensure_bucket(bucket)
        await store.put(bucket, key, envelope)
    except PrbeError as exc:
        log.error("ingestion.storage_put_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="storage unavailable") from exc

    inserted = await _enqueue(
        customer_id=customer_id,
        source=source_enum,
        source_event_id=parsed.source_event_id,
        payload_s3_key=key,
    )
    log.info(
        "ingestion.accepted",
        customer=customer_id,
        source=source,
        event_id=parsed.source_event_id,
        duplicate=not inserted,
    )
    return JSONResponse(
        {
            "status": "accepted" if inserted else "duplicate",
            "trace_id": trace_id,
            "source_event_id": parsed.source_event_id,
        }
    )


# ---- helpers ----------------------------------------------------------------


def _payload_key(source: SourceSystem, customer_id: str, event_id: str) -> str:
    now = datetime.now(UTC)
    safe_event = event_id.replace("/", "_")
    return (
        f"raw/{source.value}/{customer_id}/"
        f"{now.strftime('%Y/%m/%d')}/{safe_event}.json"
    )


async def _enqueue(
    customer_id: str,
    source: SourceSystem,
    source_event_id: str,
    payload_s3_key: str,
) -> bool:
    """Returns True if row was newly inserted, False if UNIQUE collision."""
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id, payload_s3_key)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (customer_id, source_system, source_event_id) DO NOTHING
            RETURNING queue_id
            """,
            customer_id,
            source.value,
            source_event_id,
            payload_s3_key,
        )
    return row is not None


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "services.ingestion.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
    )
