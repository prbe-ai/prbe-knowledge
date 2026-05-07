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

import contextlib
import hashlib
import hmac
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import orjson
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from services.ingestion.admin_routes import router as admin_router
from services.ingestion.backfill_routes import router as backfill_router
from services.ingestion.custom_ingest_routes import router as custom_ingest_router
from services.ingestion.handlers.base import make_default_context
from services.ingestion.handlers.registry import (
    build_connector,
    get_connector_class,
    list_registered,
)
from services.ingestion.internal_devices import router as devices_router
from services.ingestion.manual_uploads import (
    MAX_MANUAL_UPLOAD_BYTES,
    MAX_MANUAL_UPLOAD_FILES,
    ManualUploadParseError,
    parse_manual_upload,
    safe_filename,
)
from services.ingestion.slack_lifecycle import handle_slack_lifecycle_event
from services.ingestion.wiki_routes import router as wiki_router
from services.system_settings import get_ingestion_killswitch
from shared.config import get_settings
from shared.constants import (
    DEFAULT_INGESTION_PRIORITY,
    SOURCE_INGESTION_PRIORITY,
    SourceSystem,
)
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
app.include_router(wiki_router)
app.include_router(custom_ingest_router)


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


@app.get("/api/internal/ingestion-status")
async def ingestion_status(request: Request) -> JSONResponse:
    """Read the global ingestion killswitch.

    Called by prbe-backend's /agent-tap/ingestion-status proxy. Bypasses
    the cache so admin polling sees flips immediately. Auth via the same
    X-Internal-Knowledge-Key as the webhook endpoint.
    """
    _verify_internal_key(request)
    ks = await get_ingestion_killswitch(force_refresh=True)
    return JSONResponse({"enabled": ks.enabled, "reason": ks.reason})


@app.post("/api/manual-uploads")
async def create_manual_uploads(
    request: Request,
    files: list[UploadFile] = File(...),
    uploaded_by: str | None = Form(default=None),
    x_trace_id: str | None = Header(default=None),
    x_prbe_customer: str | None = Header(default=None),
) -> JSONResponse:
    """Accept dashboard manual uploads, stage originals, and enqueue extracted text."""
    _verify_internal_key(request)

    ks = await get_ingestion_killswitch()
    if not ks.enabled:
        raise HTTPException(
            status_code=503,
            detail={
                "reason": ks.reason or "ingestion paused",
                "retry_after_s": 300,
            },
            headers={"Retry-After": "300"},
        )

    if not x_prbe_customer:
        raise HTTPException(status_code=400, detail="missing X-Prbe-Customer")
    if not files:
        raise HTTPException(status_code=400, detail="at least one file is required")
    if len(files) > MAX_MANUAL_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"at most {MAX_MANUAL_UPLOAD_FILES} files can be uploaded at once",
        )

    customer_id = x_prbe_customer
    trace_id = x_trace_id or f"manual-{int(datetime.now().timestamp() * 1000)}"
    bind_trace(trace_id)

    store = request.app.state.store
    bucket = store.bucket_for(customer_id)
    try:
        await store.ensure_bucket(bucket)
    except PrbeError as exc:
        log.error("manual_upload.bucket_failed", customer=customer_id, error=str(exc))
        raise HTTPException(status_code=503, detail="storage unavailable") from exc

    uploads: list[dict[str, object]] = []
    for upload in files:
        uploaded_at = datetime.now(UTC)
        upload_id = f"manual-{uuid.uuid4().hex}"
        filename = safe_filename(upload.filename)
        content_type = upload.content_type or "application/octet-stream"
        file_size = _upload_file_size(upload)
        staging_key = _manual_staging_key(customer_id, upload_id, filename, uploaded_at)
        doc_id = f"manual_upload:{upload_id}"

        if file_size > MAX_MANUAL_UPLOAD_BYTES:
            uploads.append(
                await _record_rejected_manual_upload(
                    customer_id=customer_id,
                    upload_id=upload_id,
                    filename=filename,
                    content_type=content_type,
                    file_size_bytes=file_size,
                    file_sha256="",
                    uploaded_by=uploaded_by,
                    uploaded_at=uploaded_at,
                    parse_error=f"file exceeds {MAX_MANUAL_UPLOAD_BYTES} byte limit",
                )
            )
            continue

        body = await upload.read()
        file_sha256 = hashlib.sha256(body).hexdigest()

        try:
            await store.put(
                bucket,
                staging_key,
                body,
                content_type=content_type,
            )
        except PrbeError as exc:
            log.error("manual_upload.stage_failed", customer=customer_id, error=str(exc))
            raise HTTPException(status_code=503, detail="storage unavailable") from exc

        try:
            parsed = parse_manual_upload(filename, content_type, body)
        except ManualUploadParseError as exc:
            with contextlib.suppress(PrbeError):
                await store.delete(bucket, staging_key)
            uploads.append(
                await _record_rejected_manual_upload(
                    customer_id=customer_id,
                    upload_id=upload_id,
                    filename=filename,
                    content_type=content_type,
                    file_size_bytes=file_size,
                    file_sha256=file_sha256,
                    uploaded_by=uploaded_by,
                    uploaded_at=uploaded_at,
                    parse_error=str(exc),
                    original_deleted=True,
                )
            )
            continue

        payload = {
            "upload_id": upload_id,
            "filename": parsed.filename,
            "content_type": content_type,
            "file_size_bytes": file_size,
            "file_sha256": file_sha256,
            "uploaded_by": uploaded_by,
            "uploaded_at": uploaded_at.isoformat(),
            "original_object_key": staging_key,
            "extracted_text": parsed.text,
            "parse_engine": parsed.parse_engine,
            "doc_type": parsed.doc_type,
            "doc_id": doc_id,
        }
        safe_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in _SENSITIVE_HEADERS
        }
        envelope = orjson.dumps(
            {
                "_headers": safe_headers,
                "payload": payload,
                "received_at": uploaded_at.isoformat(),
                "trace_id": trace_id,
            }
        )
        payload_key = _payload_key(SourceSystem.MANUAL_UPLOAD, customer_id, upload_id)

        try:
            await store.put(bucket, payload_key, envelope)
            await _insert_manual_upload_row(
                customer_id=customer_id,
                upload_id=upload_id,
                filename=parsed.filename,
                content_type=content_type,
                file_size_bytes=file_size,
                file_sha256=file_sha256,
                staging_object_key=staging_key,
                payload_object_key=payload_key,
                uploaded_by=uploaded_by,
                uploaded_at=uploaded_at,
                status="queued",
                parse_engine=parsed.parse_engine,
                extracted_chars=len(parsed.text),
                doc_id=doc_id,
            )
            inserted = await _enqueue(
                customer_id=customer_id,
                source=SourceSystem.MANUAL_UPLOAD,
                source_event_id=upload_id,
                payload_s3_key=payload_key,
            )
        except Exception as exc:
            with contextlib.suppress(PrbeError):
                await store.delete(bucket, staging_key)
            with contextlib.suppress(PrbeError):
                await store.delete(bucket, payload_key)
            with contextlib.suppress(Exception):
                await _mark_manual_upload_enqueue_failed(
                    customer_id, upload_id, str(exc)
                )
            log.exception(
                "manual_upload.enqueue_failed",
                customer=customer_id,
                error=str(exc),
            )
            raise HTTPException(status_code=503, detail="manual upload enqueue failed") from exc

        uploads.append(
            {
                "upload_id": upload_id,
                "filename": parsed.filename,
                "content_type": content_type,
                "file_size_bytes": file_size,
                "file_sha256": file_sha256,
                "status": "queued" if inserted else "duplicate",
                "parse_engine": parsed.parse_engine,
                "parse_error": None,
                "extracted_chars": len(parsed.text),
                "doc_id": doc_id,
                "uploaded_by": uploaded_by,
                "uploaded_at": uploaded_at.isoformat(),
                "indexed_at": None,
                "original_deleted_at": None,
            }
        )

    return JSONResponse(
        {
            "trace_id": trace_id,
            "uploads": uploads,
            "accepted": sum(1 for u in uploads if u["status"] == "queued"),
            "failed": sum(1 for u in uploads if u["status"] == "failed_parse"),
        }
    )


@app.get("/api/manual-uploads")
async def list_manual_uploads(
    request: Request,
    x_prbe_customer: str | None = Header(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> JSONResponse:
    """List manual uploads for dashboard file explorer views."""
    _verify_internal_key(request)
    if not x_prbe_customer:
        raise HTTPException(status_code=400, detail="missing X-Prbe-Customer")

    async with get_pool().acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT mu.upload_id, mu.filename, mu.content_type, mu.file_size_bytes,
                   mu.file_sha256, mu.uploaded_by, mu.uploaded_at, mu.status,
                   mu.parse_engine, mu.parse_error, mu.extracted_chars, mu.doc_id,
                   mu.indexed_at, mu.original_deleted_at, mu.updated_at,
                   COALESCE((
                       SELECT COUNT(*)
                       FROM chunks c
                       WHERE c.customer_id = mu.customer_id
                         AND c.doc_id = mu.doc_id
                         AND c.valid_to IS NULL
                   ), 0)::INT AS chunk_count
            FROM manual_uploads mu
            WHERE mu.customer_id = $1
            ORDER BY mu.uploaded_at DESC
            LIMIT $2
            """,
            x_prbe_customer,
            limit,
        )
    return JSONResponse({"uploads": [_manual_upload_json(row) for row in rows]})


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

    # Global ingestion killswitch — short-circuit BEFORE doing any of the
    # heavy work (R2 write, queue insert). If an operator has flipped the
    # switch off (maintenance, runaway customer, panic stop), every plugin
    # webhook gets a 503 with Retry-After so well-behaved clients back off.
    # Cache lives in services/system_settings (30s TTL).
    ks = await get_ingestion_killswitch()
    if not ks.enabled:
        raise HTTPException(
            status_code=503,
            detail={
                "reason": ks.reason or "ingestion paused",
                "retry_after_s": 300,
            },
            headers={"Retry-After": "300"},
        )

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
    if source_enum == SourceSystem.SLACK:
        lifecycle = await handle_slack_lifecycle_event(
            request.app.state.ctx,
            customer_id,
            payload,
        )
        if lifecycle is not None:
            lifecycle["trace_id"] = trace_id
            return JSONResponse(lifecycle)

    try:
        parsed = connector.parse_webhook_event(
            customer_id, dict(request.headers), payload
        )
    except InvalidWebhookPayload as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if parsed is None:
        return JSONResponse({"status": "ignored", "trace_id": trace_id})

    # Headers are persisted with the raw payload for replayability. Strip
    # bearer/cookie/api-key headers before write — even though the gateway
    # is supposed to filter, a single misconfigured caller could land a
    # plaintext device-token in long-term R2 storage. Defense in depth.
    safe_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _SENSITIVE_HEADERS
    }
    envelope = orjson.dumps(
        {
            "_headers": safe_headers,
            "payload": payload,
            "received_at": parsed.received_at.isoformat(),
            "trace_id": trace_id,
        }
    )
    store = request.app.state.store
    bucket = store.bucket_for(customer_id)
    storage_id = _compose_storage_id(
        source_enum, parsed.source_event_id, parsed.parse_hint
    )
    key = _payload_key(source_enum, customer_id, storage_id)

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


# Headers we never want to land in R2 alongside the raw payload. Lower-case;
# matched against the lower-cased header name in the envelope-build step.
_SENSITIVE_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "cookie",
        "x-api-key",
        "x-internal-knowledge-key",
    }
)


# Sources that ingest in coalescing mode: queue's source_event_id is the
# bare session_id (so multiple batches collapse onto one queue row), and
# the R2 storage path needs a per-batch suffix so deliveries don't
# overwrite each other on object storage.
_COALESCING_AGENT_SOURCES: frozenset[SourceSystem] = frozenset(
    {SourceSystem.CLAUDE_CODE, SourceSystem.CODEX}
)


def _compose_storage_id(
    source: SourceSystem,
    source_event_id: str,
    parse_hint: object,
) -> str:
    """Compose the R2 storage namespace key.

    For agent-session sources (claude_code, codex) the queue source_event_id
    is the bare session_id (so the UPSERT can coalesce). The R2 path must
    still be unique per delivery — we suffix it with `:<batch_seq>` so
    each batch writes a distinct envelope and retries with the same
    batch_seq are idempotent (last-write-wins on identical content).

    Without the suffix, batches 1..N-1 of a multi-batch session silently
    overwrite each other in R2 before the worker reads them and only the
    final batch survives.

    Other sources use bare source_event_id — every event is its own queue
    row, so the source_event_id is already unique per delivery.
    """
    if source in _COALESCING_AGENT_SOURCES and isinstance(parse_hint, dict):
        batch_seq = parse_hint.get("batch_seq")
        if isinstance(batch_seq, int):
            return f"{source_event_id}:{batch_seq}"
    return source_event_id


def _payload_key(source: SourceSystem, customer_id: str, event_id: str) -> str:
    now = datetime.now(UTC)
    safe_event = event_id.replace("/", "_")
    return (
        f"raw/{source.value}/{customer_id}/"
        f"{now.strftime('%Y/%m/%d')}/{safe_event}.json"
    )


def _manual_staging_key(
    customer_id: str,
    upload_id: str,
    filename: str,
    uploaded_at: datetime,
) -> str:
    return (
        f"manual_uploads/staging/{customer_id}/"
        f"{uploaded_at.strftime('%Y/%m/%d')}/{upload_id}/{filename}"
    )


async def _insert_manual_upload_row(
    *,
    customer_id: str,
    upload_id: str,
    filename: str,
    content_type: str,
    file_size_bytes: int,
    file_sha256: str,
    staging_object_key: str | None,
    payload_object_key: str | None,
    uploaded_by: str | None,
    uploaded_at: datetime,
    status: str,
    parse_engine: str | None = None,
    parse_error: str | None = None,
    extracted_chars: int = 0,
    doc_id: str | None = None,
    original_deleted_at: datetime | None = None,
) -> None:
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO manual_uploads (
                upload_id, customer_id, filename, content_type, file_size_bytes,
                file_sha256, staging_object_key, payload_object_key, uploaded_by,
                uploaded_at, status, parse_engine, parse_error, extracted_chars,
                doc_id, original_deleted_at
            )
            VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10, $11, $12, $13, $14,
                $15, $16
            )
            """,
            upload_id,
            customer_id,
            filename,
            content_type,
            file_size_bytes,
            file_sha256,
            staging_object_key,
            payload_object_key,
            uploaded_by,
            uploaded_at,
            status,
            parse_engine,
            parse_error,
            extracted_chars,
            doc_id,
            original_deleted_at,
        )


async def _record_rejected_manual_upload(
    *,
    customer_id: str,
    upload_id: str,
    filename: str,
    content_type: str,
    file_size_bytes: int,
    file_sha256: str,
    uploaded_by: str | None,
    uploaded_at: datetime,
    parse_error: str,
    original_deleted: bool = False,
) -> dict[str, object]:
    original_deleted_at = datetime.now(UTC) if original_deleted else None
    await _insert_manual_upload_row(
        customer_id=customer_id,
        upload_id=upload_id,
        filename=filename,
        content_type=content_type,
        file_size_bytes=file_size_bytes,
        file_sha256=file_sha256,
        staging_object_key=None,
        payload_object_key=None,
        uploaded_by=uploaded_by,
        uploaded_at=uploaded_at,
        status="failed_parse",
        parse_error=parse_error,
        original_deleted_at=original_deleted_at,
    )
    return {
        "upload_id": upload_id,
        "filename": filename,
        "content_type": content_type,
        "file_size_bytes": file_size_bytes,
        "file_sha256": file_sha256,
        "status": "failed_parse",
        "parse_engine": None,
        "parse_error": parse_error,
        "extracted_chars": 0,
        "doc_id": None,
        "uploaded_by": uploaded_by,
        "uploaded_at": uploaded_at.isoformat(),
        "indexed_at": None,
        "original_deleted_at": _iso_or_none(original_deleted_at),
    }


def _manual_upload_json(row) -> dict[str, object]:
    return {
        "upload_id": row["upload_id"],
        "filename": row["filename"],
        "content_type": row["content_type"],
        "file_size_bytes": row["file_size_bytes"],
        "file_sha256": row["file_sha256"],
        "uploaded_by": row["uploaded_by"],
        "uploaded_at": _iso_or_none(row["uploaded_at"]),
        "status": row["status"],
        "parse_engine": row["parse_engine"],
        "parse_error": row["parse_error"],
        "extracted_chars": row["extracted_chars"],
        "chunk_count": row["chunk_count"],
        "doc_id": row["doc_id"],
        "indexed_at": _iso_or_none(row["indexed_at"]),
        "original_deleted_at": _iso_or_none(row["original_deleted_at"]),
        "updated_at": _iso_or_none(row["updated_at"]),
    }


def _upload_file_size(upload: UploadFile) -> int:
    try:
        upload.file.seek(0, os.SEEK_END)
        return upload.file.tell()
    finally:
        upload.file.seek(0)


async def _mark_manual_upload_enqueue_failed(
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
                original_deleted_at = COALESCE(original_deleted_at, NOW()),
                updated_at = NOW()
            WHERE customer_id = $1 AND upload_id = $2
            """,
            customer_id,
            upload_id,
            error[:4000],
        )


def _iso_or_none(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


async def _enqueue(
    customer_id: str,
    source: SourceSystem,
    source_event_id: str,
    payload_s3_key: str,
) -> bool:
    """Persist a webhook batch to the ingestion queue.

    Two paths:

    1. **agent sessions (claude_code, codex) — coalescing** — multiple
       batches for the same session collapse into a single queue row.
       UPSERT keyed on (customer_id, source_system,
       source_event_id=session_id) appends the new R2 key to
       `payload_s3_keys` and bumps `version`. The worker captures
       `version` at claim time and CAS-commits on it, so any batch landing
       during Phase A causes a clean re-claim with the extended array.
       See migration 0026 for column shape.

    2. **other connectors** — INSERT each event as its own row with
       `payload_s3_keys = ARRAY[key]`, `version = 0`. ON CONFLICT
       DO NOTHING dedupes redeliveries from the source platform.

    Returns True if a new row was created OR an existing agent-session row
    had its array extended; False only on the non-agent-session duplicate
    path (ON CONFLICT swallowed the insert).

    `payload_s3_key` is also written to the legacy column for back-compat
    until a follow-up PR drops it. Both columns hold the same first key
    on insert; CC UPSERTs leave the legacy column at whatever the
    earliest batch wrote (fine, nothing reads it once the deploy lands).
    """
    priority = SOURCE_INGESTION_PRIORITY.get(source, DEFAULT_INGESTION_PRIORITY)

    if source in (SourceSystem.CLAUDE_CODE, SourceSystem.CODEX):
        # Session-keyed UPSERT: append to array, bump version, refresh
        # status to 'pending' so the worker picks it up even if the row
        # was previously 'done' (session resumed after idle).
        async with get_pool().acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO ingestion_queue
                    (customer_id, source_system, source_event_id,
                     payload_s3_key, payload_s3_keys, status, priority,
                     version, enqueued_at)
                VALUES ($1, $2, $3, $4, ARRAY[$4], 'pending', $5, 1, NOW())
                ON CONFLICT (customer_id, source_system, source_event_id) DO UPDATE
                    SET payload_s3_keys = ingestion_queue.payload_s3_keys
                                          || EXCLUDED.payload_s3_keys,
                        status = 'pending',
                        version = ingestion_queue.version + 1,
                        completed_at = NULL,
                        error = NULL,
                        -- Bump enqueued_at to reflect most-recent activity so
                        -- session_completer's MAX(enqueued_at) tracks idle
                        -- correctly. Side effect: chatty sessions get pushed
                        -- to the back of the priority tier within CC, which
                        -- is intentional — quieter sessions drain first.
                        enqueued_at = NOW()
                RETURNING queue_id
                """,
                customer_id,
                source.value,
                source_event_id,
                payload_s3_key,
                priority,
            )
        # UPSERT always returns a queue_id, so this is True for both
        # first-batch (new row) and Nth-batch (extended array) cases.
        # Callers just want to know "did we accept the payload?" — we did.
        return row is not None

    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id,
                 payload_s3_key, payload_s3_keys, priority)
            VALUES ($1, $2, $3, $4, ARRAY[$4], $5)
            ON CONFLICT (customer_id, source_system, source_event_id) DO NOTHING
            RETURNING queue_id
            """,
            customer_id,
            source.value,
            source_event_id,
            payload_s3_key,
            priority,
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
