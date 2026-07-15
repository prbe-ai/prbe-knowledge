"""Custom-ingest routes: push (upsert + delete) and enumeration.

Auth is dual-mode (see _resolve_custom_ingest_customer): hosted deployments
trust the gateway's X-Internal-Knowledge-Key + X-Prbe-Customer headers;
standalone (community) accepts the static KNOWLEDGE_API_TOKEN bearer scoped
to the single configured tenant.

POST /api/custom-ingest/documents  — batch upserts; a document entry with
    `"deleted": true` (body optional) tombstones the doc: the live version
    and its chunks are closed (valid_to), same semantics as connector
    deletes.
GET  /api/custom-ingest/documents  — keyset-paginated enumeration for
    consumer-side reconcilers: per document the caller's original id, the
    envelope content hash, deleted flag, updated_at, and stored metadata.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
from datetime import UTC, datetime
from typing import Any

import orjson
from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from engine.shared.config import get_settings
from engine.shared.constants import SourceSystem
from engine.shared.custom_ingest import (
    CustomIngestDocument,
    CustomIngestEnvelope,
    document_content_hash,
    document_payload_key,
    is_valid_source_key,
    json_size,
    source_event_id,
)
from engine.shared.db import get_pool, raw_conn, with_tenant
from engine.shared.exceptions import PrbeError
from engine.shared.logging import bind_trace, get_logger
from engine.shared.source_registry import ingestion_priority_for
from engine.shared.storage import get_store
from engine.system_settings import get_ingestion_killswitch

router = APIRouter()
log = get_logger(__name__)

# Enumeration page-size bounds. The reconciler walks the whole corpus in
# pages; 100 default / 500 cap keeps a page comfortably under response-size
# limits even with fat per-document metadata.
ENUMERATION_DEFAULT_LIMIT = 100
ENUMERATION_MAX_LIMIT = 500


@router.post("/api/custom-ingest/documents")
async def custom_ingest_documents(
    request: Request,
    x_trace_id: str | None = Header(default=None),
    x_prbe_customer: str | None = Header(default=None),
) -> JSONResponse:
    customer_id = await _resolve_custom_ingest_customer(request, x_prbe_customer)

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


@router.get("/api/custom-ingest/documents")
async def list_custom_ingest_documents(
    request: Request,
    source_key: str = Query(min_length=1, max_length=128),
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=ENUMERATION_DEFAULT_LIMIT, ge=1, le=ENUMERATION_MAX_LIMIT),
    x_prbe_customer: str | None = Header(default=None),
) -> JSONResponse:
    """Enumerate the live custom-ingest documents under one source_key.

    Built for the consumer-side reconciler (research-os index_outbox
    reconciliation): the caller diffs `content_hash` against its own source
    of truth and re-pushes / tombstones drift. Tombstoned documents are
    included with `deleted: true` until a later upsert revives them.

    Response shape:
        {
          "documents": [
            {"id": "<caller's original document id>",
             "content_hash": "<envelope content hash>" | null,   # null = pre-hash-stamping ingest
             "deleted": bool,
             "updated_at": "<ISO 8601>",
             "metadata": {...}},                                 # stored metadata (caller keys + engine keys)
            ...
          ],
          "next_cursor": "<opaque>" | null
        }

    Keyset pagination: results are ordered by internal doc id; pass
    `next_cursor` back verbatim to fetch the next page. Same auth as the
    POST route (hosted internal-key headers or standalone bearer).
    """
    customer_id = await _resolve_custom_ingest_customer(request, x_prbe_customer)

    key = source_key.strip()
    if not is_valid_source_key(key):
        raise HTTPException(status_code=422, detail="invalid source_key")

    after_doc_id = _decode_enumeration_cursor(cursor)

    # Live version per doc_id = the single row with valid_to IS NULL
    # (bitemporal invariant); tombstones keep a live row with deleted_at
    # set. Tenant scoping via the RLS GUC (documents are RLS-protected).
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT doc_id, metadata, deleted_at, updated_at
            FROM documents
            WHERE customer_id = $1
              AND source_system = $2
              AND valid_to IS NULL
              AND metadata->>'source_key' = $3
              AND ($4::text IS NULL OR doc_id > $4)
            ORDER BY doc_id
            LIMIT $5
            """,
            customer_id,
            SourceSystem.CUSTOM_INGEST.value,
            key,
            after_doc_id,
            limit + 1,
        )

    has_more = len(rows) > limit
    rows = rows[:limit]

    doc_id_prefix = f"custom_ingest:{customer_id}:{key}:"
    documents: list[dict[str, Any]] = []
    for row in rows:
        metadata = _coerce_metadata(row["metadata"])
        original_id = metadata.get("custom_document_id") or row["doc_id"].removeprefix(
            doc_id_prefix
        )
        updated_at = row["updated_at"]
        documents.append(
            {
                "id": original_id,
                "content_hash": metadata.get("content_hash"),
                "deleted": row["deleted_at"] is not None,
                "updated_at": updated_at.isoformat() if updated_at else None,
                "metadata": metadata,
            }
        )

    next_cursor = _encode_enumeration_cursor(rows[-1]["doc_id"]) if has_more else None
    return JSONResponse({"documents": documents, "next_cursor": next_cursor})


def _encode_enumeration_cursor(doc_id: str) -> str:
    return base64.urlsafe_b64encode(doc_id.encode("utf-8")).decode("ascii")


def _decode_enumeration_cursor(cursor: str | None) -> str | None:
    if cursor is None or not cursor.strip():
        return None
    try:
        return base64.urlsafe_b64decode(cursor.strip().encode("ascii")).decode("utf-8")
    except (binascii.Error, UnicodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid cursor") from exc


def _coerce_metadata(value: object) -> dict[str, Any]:
    """documents.metadata is JSONB; asyncpg hands it back as a str."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = orjson.loads(value)
        except orjson.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


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


async def _resolve_custom_ingest_customer(
    request: Request, x_prbe_customer: str | None
) -> str:
    """Authorize the request and resolve the target tenant (dual-mode).

    HOSTED: when INTERNAL_KNOWLEDGE_API_KEY is configured, trust the gateway's
    X-Internal-Knowledge-Key + X-Prbe-Customer exactly as before.
    STANDALONE (community): no internal key — accept the static KNOWLEDGE_API_TOKEN
    bearer and scope to DEFAULT_CUSTOMER_ID (same seeded-hash path as /query).
    """
    settings = get_settings()
    gateway_mode = bool(
        settings.internal_knowledge_api_key
        and settings.internal_knowledge_api_key.get_secret_value()
    )
    if gateway_mode:
        _verify_internal_key(request)
        if not x_prbe_customer:
            raise HTTPException(status_code=400, detail="missing X-Prbe-Customer")
        return x_prbe_customer
    return await _resolve_bearer_customer(request)


async def _resolve_bearer_customer(request: Request) -> str:
    """Resolve customer_id from an Authorization: Bearer token (standalone mode).

    Matched against customers.api_key_hash; in single-tenant mode the default
    customer is seeded with sha256(KNOWLEDGE_API_TOKEN) on boot, so a valid token
    resolves to DEFAULT_CUSTOMER_ID. Mirrors services/retrieval/auth.py.
    """
    authorization = request.headers.get("authorization")
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="missing bearer token or X-Internal-Knowledge-Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=401,
            detail="invalid authorization scheme",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token_hash = hashlib.sha256(token.strip().encode()).hexdigest()
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT customer_id FROM customers WHERE api_key_hash = $1",
            token_hash,
        )
    if row is None:
        raise HTTPException(
            status_code=401,
            detail="invalid api key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return str(row["customer_id"])


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
    priority = ingestion_priority_for(SourceSystem.CUSTOM_INGEST.value)
    # Intentionally NOT gated by engine.ingest.connectedness:
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
