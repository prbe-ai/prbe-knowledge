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

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import orjson
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

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

    # customer_id arrives as a header today. Future: resolve from OAuth state
    # or install-level signing key. Phase 0 is operator-provisioned.
    customer_id = request.headers.get("x-prbe-customer")
    if not customer_id:
        raise HTTPException(status_code=400, detail="missing X-Prbe-Customer header")

    connector = build_connector(source_enum, request.app.state.ctx)

    if not connector.verify_signature(dict(request.headers), raw_body):
        raise HTTPException(status_code=401, detail="signature verification failed")

    try:
        payload = orjson.loads(raw_body) if raw_body else {}
    except orjson.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc

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


# Keep asyncio import used in reload flows / background tasks.
_ = asyncio
