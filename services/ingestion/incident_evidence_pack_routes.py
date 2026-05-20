"""POST / GET ``/api/incident-evidence-packs`` routes.

The orchestrator's Pass 1 produces an ``EvidencePack`` per (resolved,
approved) incident and POSTs it here for caching on the
``incident_investigations.evidence_pack`` jsonb column. Pass 2's
postmortem author reads the cache back via GET on the same path on
re-runs so authoring is deterministic w.r.t. the gathered evidence.

Idempotency: duplicate POSTs land on the same row. The pre-read /
UPDATE split (rather than UPDATE ... RETURNING) gives correct
duplicate detection: ``had_pack`` is the pre-state, not the post-
state. Without the split a fresh write would report duplicate=true
because the row "had a pack" by the time RETURNING ran.
"""
from __future__ import annotations

import orjson
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from shared.db import with_tenant
from shared.logging import get_logger
from shared.schemas.evidence_pack import (
    EvidencePackWritebackRequest,
    EvidencePackWritebackResponse,
)

log = get_logger(__name__)

router = APIRouter(
    prefix="/api/incident-evidence-packs",
    tags=["incident-evidence-packs"],
)


def _verify_key(request: Request) -> None:
    # Local import avoids the circular import with main.py — same
    # pattern as investigation_review_routes / writeback_routes.
    from services.ingestion.main import _verify_internal_key
    _verify_internal_key(request)


@router.post("", response_model=EvidencePackWritebackResponse)
async def writeback(
    payload: EvidencePackWritebackRequest,
    request: Request,
) -> EvidencePackWritebackResponse:
    _verify_key(request)

    # Pre-read: capture the pre-write state so the response's
    # ``duplicate`` field reflects whether this is a redelivery.
    # We need the row's existence first (404 vs 200), then whether
    # ``evidence_pack`` was already populated (duplicate flag).
    async with with_tenant(payload.customer_id) as conn:
        existing = await conn.fetchrow(
            "SELECT (evidence_pack IS NOT NULL) AS had_pack "
            "FROM incident_investigations "
            "WHERE customer_id = $1 AND incident_doc_id = $2",
            payload.customer_id, payload.incident_doc_id,
        )
        if existing is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no incident_investigations row for "
                    f"{payload.incident_doc_id}"
                ),
            )
        had_pack = bool(existing["had_pack"])

        # Pydantic model_dump_json + ::jsonb cast is the canonical
        # pattern for writing a jsonb column from a pydantic model:
        # it preserves enum string values and handles datetime ISO
        # formatting without a custom orjson default.
        await conn.execute(
            "UPDATE incident_investigations "
            "SET evidence_pack = $3::jsonb, updated_at = now() "
            "WHERE customer_id = $1 AND incident_doc_id = $2",
            payload.customer_id,
            payload.incident_doc_id,
            payload.evidence_pack.model_dump_json(),
        )

    log.info(
        "evidence_pack.written",
        customer_id=payload.customer_id,
        incident_doc_id=payload.incident_doc_id,
        duplicate=had_pack,
        mode=payload.evidence_pack.mode,
    )

    return EvidencePackWritebackResponse(
        incident_doc_id=payload.incident_doc_id,
        duplicate=had_pack,
    )


@router.get("")
async def get_pack(
    request: Request,
    customer_id: str = Query(..., min_length=1),
    incident_doc_id: str = Query(..., min_length=1),
) -> JSONResponse:
    """Read back the cached EvidencePack.

    Returns the raw jsonb body so the orchestrator's Plan B
    ``get_evidence_pack`` client can deserialize directly into its
    ``EvidencePack`` model. Returns 404 when either the row is missing
    OR ``evidence_pack`` is NULL — both states mean "no cached pack
    available for this incident" from the caller's POV.
    """
    _verify_key(request)

    # asyncpg returns jsonb as a Python string (bytes-encoded JSON
    # text) when no custom codec is registered; we want the body to
    # land in the response untouched. Reading the column as text via
    # the ::text cast skips any decode/encode round-trip and lets
    # FastAPI's JSONResponse handle the framing.
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            "SELECT evidence_pack::text AS pack "
            "FROM incident_investigations "
            "WHERE customer_id = $1 AND incident_doc_id = $2",
            customer_id, incident_doc_id,
        )
    if row is None or row["pack"] is None:
        raise HTTPException(
            status_code=404,
            detail=f"no evidence_pack for {incident_doc_id}",
        )
    # ``row["pack"]`` is the jsonb body as a JSON string already.
    # Parse once so FastAPI's JSONResponse can serialize it back; the
    # round-trip is acceptable for what is at most a few KB of cached
    # evidence, and keeps the response Content-Type / framing standard.
    return JSONResponse(content=orjson.loads(row["pack"]))
