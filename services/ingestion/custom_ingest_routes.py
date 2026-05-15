"""Internal custom-ingest route used by prbe-backend.

The public bearer-token check lives in prbe-backend. This service only trusts
the backend's internal key and tenant header, then stores raw payloads and
queues one document per row for the existing normalizer pipeline.
"""

from __future__ import annotations

import hmac
from datetime import UTC, datetime
from typing import Any

import orjson
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from services.system_settings import get_ingestion_killswitch
from shared.config import get_settings
from shared.constants import (
    DEFAULT_INGESTION_PRIORITY,
    SOURCE_INGESTION_PRIORITY,
    SourceSystem,
)
from shared.custom_ingest import (
    CustomIngestDocument,
    CustomIngestEnvelope,
    document_content_hash,
    document_payload_key,
    json_size,
    source_event_id,
)
from shared.db import get_pool
from shared.exceptions import PrbeError
from shared.logging import bind_trace, get_logger
from shared.storage import get_store

router = APIRouter()
log = get_logger(__name__)


@router.post("/api/custom-ingest/documents")
async def custom_ingest_documents(
    request: Request,
    x_trace_id: str | None = Header(default=None),
    x_prbe_customer: str | None = Header(default=None),
) -> JSONResponse:
    _verify_internal_key(request)

    if not x_prbe_customer:
        raise HTTPException(status_code=400, detail="missing X-Prbe-Customer")

    settings = get_settings()
    _require_json_content_type(request)
    _validate_content_length(request, settings.custom_ingest_max_request_bytes)
    raw_body = await request.body()
    if len(raw_body) > settings.custom_ingest_max_request_bytes:
        raise HTTPException(status_code=413, detail="request body is too large")

    try:
        payload = orjson.loads(raw_body)
    except orjson.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON") from exc

    try:
        envelope = CustomIngestEnvelope.model_validate(payload)
        _validate_document_limits(envelope)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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

    customer_id = x_prbe_customer
    trace_id = x_trace_id or f"custom-ingest-{int(datetime.now().timestamp() * 1000)}"
    bind_trace(trace_id)

    store = getattr(request.app.state, "store", None) or get_store()
    bucket = await store.bucket_for(customer_id)
    try:
        await store.ensure_bucket(bucket)
    except PrbeError as exc:
        log.error("custom_ingest.bucket_failed", customer=customer_id, error=str(exc))
        raise HTTPException(status_code=503, detail="storage unavailable") from exc

    accepted = 0
    duplicates = 0
    for document in envelope.documents:
        content_hash = document_content_hash(envelope.source_key, document)
        payload_key = document_payload_key(
            customer_id=customer_id,
            source_key=envelope.source_key,
            document_id=document.id,
            content_hash=content_hash,
        )
        source_event = source_event_id(envelope, document, content_hash)
        raw_envelope = _raw_document_envelope(
            customer_id=customer_id,
            envelope=envelope,
            document=document,
            content_hash=content_hash,
            source_event=source_event,
            received_at=datetime.now(UTC),
        )

        try:
            await store.put(
                bucket,
                payload_key,
                orjson.dumps(raw_envelope, option=orjson.OPT_SORT_KEYS),
                content_type="application/json",
            )
        except PrbeError as exc:
            log.error(
                "custom_ingest.object_put_failed",
                customer=customer_id,
                source_key=envelope.source_key,
                document_id=document.id,
                error=str(exc),
            )
            raise HTTPException(status_code=503, detail="storage unavailable") from exc

        inserted = await _enqueue_custom_document(
            customer_id=customer_id,
            source_event_id=source_event,
            payload_s3_key=payload_key,
        )
        if inserted:
            accepted += 1
        else:
            duplicates += 1

    return JSONResponse(
        {
            "status": "accepted",
            "source_key": envelope.source_key,
            "accepted": accepted,
            "duplicates": duplicates,
            "errors": [],
        },
        status_code=202,
    )


def _verify_internal_key(request: Request) -> None:
    expected = get_settings().internal_knowledge_api_key
    if expected is None or not expected.get_secret_value():
        raise HTTPException(
            status_code=503,
            detail="INTERNAL_KNOWLEDGE_API_KEY not configured",
        )
    presented = request.headers.get("x-internal-knowledge-key")
    if not presented or not hmac.compare_digest(
        presented,
        expected.get_secret_value(),
    ):
        raise HTTPException(
            status_code=401,
            detail="missing or invalid X-Internal-Knowledge-Key",
        )


def _require_json_content_type(request: Request) -> None:
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")


def _validate_content_length(request: Request, max_bytes: int) -> None:
    header = request.headers.get("content-length")
    if not header:
        return
    try:
        content_length = int(header)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid Content-Length") from exc
    if content_length > max_bytes:
        raise HTTPException(status_code=413, detail="request body is too large")


def _validate_document_limits(envelope: CustomIngestEnvelope) -> None:
    settings = get_settings()
    for document in envelope.documents:
        if len(document.body.encode("utf-8")) > settings.custom_ingest_max_body_bytes:
            raise ValueError(
                f"document {document.id!r} body exceeds "
                f"{settings.custom_ingest_max_body_bytes} bytes"
            )
        if json_size(document.metadata) > settings.custom_ingest_max_metadata_bytes:
            raise ValueError(
                f"document {document.id!r} metadata exceeds "
                f"{settings.custom_ingest_max_metadata_bytes} bytes"
            )


def _raw_document_envelope(
    *,
    customer_id: str,
    envelope: CustomIngestEnvelope,
    document: CustomIngestDocument,
    content_hash: str,
    source_event: str,
    received_at: datetime,
) -> dict[str, Any]:
    return {
        "_headers": {},
        "payload": {
            "customer_id": customer_id,
            "source_key": envelope.source_key,
            "batch_id": envelope.batch_id,
            "received_at": received_at.isoformat(),
            "source_event_id": source_event,
            "content_hash": content_hash,
            "document": document.model_dump(mode="json"),
        },
    }


async def _enqueue_custom_document(
    *,
    customer_id: str,
    source_event_id: str,
    payload_s3_key: str,
) -> bool:
    priority = SOURCE_INGESTION_PRIORITY.get(
        SourceSystem.CUSTOM_INGEST,
        DEFAULT_INGESTION_PRIORITY,
    )
    # Intentionally NOT gated by services.ingestion.connectedness:
    # CUSTOM_INGEST uses BYO signed tokens (custom_ingest_tokens table,
    # validated upstream at the API boundary), not integration_tokens.
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
            SourceSystem.CUSTOM_INGEST.value,
            source_event_id,
            payload_s3_key,
            priority,
        )
    return row is not None
