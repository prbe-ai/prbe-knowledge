"""Ingestion service: webhook fast path.

Per request:
    1. Look up customer_id from subdomain / header / path param.
    2. Dispatch to the right Connector via the registry.
    3. verify_signature → 401 on miss.
    4. parse_webhook_event → None means ignore (200 no-op).
    5. Put raw payload in R2.
    6. Insert ingestion_queue row (UNIQUE dedupes redeliveries).
    7. Return 200.

Kept thin on purpose. Anything beyond "accept + persist + enqueue" happens
in the worker, so transient failures in downstream systems never reject a
webhook from the source.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import orjson
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from services.ingestion.admin_routes import router as admin_router
from services.ingestion.backfill_routes import router as backfill_router
from services.ingestion.handlers.base import make_default_context
from services.ingestion.handlers.registry import (
    build_connector,
    get_connector_class,
    list_registered,
)
from services.ingestion.oauth import router as oauth_router
from shared.config import get_settings
from shared.constants import SourceSystem
from shared.customer_mapping import resolve_customer, single_customer_fallback
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
# Trust X-Forwarded-Proto / X-Forwarded-For from Fly's upstream so
# `request.url.scheme` is `https` and OAuth redirect_uri matches what we registered.
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
app.include_router(oauth_router)
app.include_router(backfill_router)
app.include_router(admin_router)


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


@app.post("/webhooks/{source}")
async def webhook(
    source: str,
    request: Request,
    x_trace_id: str | None = Header(default=None),
) -> JSONResponse:
    trace_id = x_trace_id or f"wh-{int(datetime.now().timestamp()*1000)}"
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

    if not connector.verify_signature(dict(request.headers), raw_body):
        raise HTTPException(status_code=401, detail="signature verification failed")

    try:
        payload = orjson.loads(raw_body) if raw_body else {}
    except orjson.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc

    # Resolve customer_id in this priority order:
    #   1. X-Prbe-Customer header (internal callers / tests / manual routing)
    #   2. Payload's external_id → customer_source_mapping lookup
    #   3. single_customer_fallback (dev / solo-tenant convenience)
    customer_id = request.headers.get("x-prbe-customer")
    if not customer_id:
        external_id = connector.extract_external_id_from_payload(
            dict(request.headers), payload
        )
        if external_id:
            customer_id = await resolve_customer(source_enum, external_id)
        if not customer_id:
            customer_id = await single_customer_fallback()
    if not customer_id:
        raise HTTPException(
            status_code=400,
            detail="could not resolve customer (no X-Prbe-Customer header, no mapping, multiple tenants exist)",
        )

    try:
        parsed = connector.parse_webhook_event(
            customer_id, dict(request.headers), payload
        )
    except InvalidWebhookPayload as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if parsed is None:
        return JSONResponse({"status": "ignored", "trace_id": trace_id})

    # Persist raw payload + headers for full replayability.
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
