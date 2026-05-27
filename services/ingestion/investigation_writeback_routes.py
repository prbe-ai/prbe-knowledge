"""POST /api/incident-investigations writeback route.

Persists an InvestigationReport as a typed Document with
source_system=PAGERDUTY|INCIDENT_IO and doc_type=INCIDENT_INVESTIGATION,
then upserts the per-incident review-state row. Bypasses the ingestion
queue — the agent has already produced the final shape — but routes the
doc through the same chunker + embedder + SQL writes (via
Normalizer.persist_single_document) as the standard normalize path so
retrieval-parity with other documents is automatic.

Idempotency: keyed on (customer_id, doc_id). A redelivered POST for the
same (customer_id, source_event_id, version) lands on the same doc_id
and returns `duplicate=true` without re-persisting.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from fastapi import APIRouter, Request

from services.ingestion.chunker import count_tokens
from services.ingestion.investigation_state import (
    get_detail,
    upsert_pending_review,
)
from shared.constants import (
    DocClass,
    DocType,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.db import with_tenant
from shared.investigation_schemas import (
    InvestigationWritebackRequest,
    InvestigationWritebackResponse,
)
from shared.models import ACLPrincipal, ACLSnapshot, Document

router = APIRouter(
    prefix="/api/incident-investigations",
    tags=["incident-investigations"],
)


def _doc_id(payload: InvestigationWritebackRequest) -> str:
    # Strip the source prefix from the incident_doc_id to get the
    # provider-side incident id, then build the investigation-doc id
    # with our own prefix. Stable across all writes for the same
    # (incident, version).
    incident_source_id = (
        payload.incident_doc_id.split(":")[-1]
        if ":" in payload.incident_doc_id
        else payload.incident_doc_id
    )
    prefix = "pd" if payload.source_system == "pagerduty" else "iio"
    return f"{prefix}:investigation:{incident_source_id}:v{payload.version}"


def _content_hash(payload: InvestigationWritebackRequest) -> str:
    return hashlib.sha256(
        f"{payload.source_event_id}|{payload.version}|{payload.body_markdown}".encode()
    ).hexdigest()


async def _existing_doc_id(customer_id: str, doc_id: str) -> bool:
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM documents WHERE doc_id = $1 "
            "AND customer_id = $2 AND valid_to IS NULL",
            doc_id, customer_id,
        )
    return row is not None


@router.post("", response_model=InvestigationWritebackResponse)
async def writeback(
    payload: InvestigationWritebackRequest,
    request: Request,
) -> InvestigationWritebackResponse:
    # Trust boundary — same as every other internal route on this app.
    from services.ingestion.main import _verify_internal_key
    _verify_internal_key(request)

    source_system = (
        SourceSystem.PAGERDUTY
        if payload.source_system == "pagerduty"
        else SourceSystem.INCIDENT_IO
    )
    doc_id = _doc_id(payload)

    # Idempotency: redelivered POSTs land on the same doc_id; return early
    # without re-running the chunker/embedder.
    if await _existing_doc_id(payload.customer_id, doc_id):
        detail = await get_detail(payload.customer_id, payload.incident_doc_id)
        # If the state row exists, use its current state. If it doesn't
        # (race window before upsert_pending_review ran on the first POST),
        # fall back to pending_review — the row will exist by next read.
        current_state = detail.state if detail else "pending_review"
        return InvestigationWritebackResponse(
            report_doc_id=doc_id,
            state=current_state,
            duplicate=True,
        )

    now = datetime.now(UTC)
    acl = ACLSnapshot(
        principals=[
            ACLPrincipal(
                principal_type=PrincipalType.WORKSPACE,
                principal_id=payload.customer_id,
                permission=Permission.READ,
            ),
        ],
        captured_at=now,
    )
    doc = Document(
        doc_id=doc_id,
        customer_id=payload.customer_id,
        source_system=source_system,
        source_id=doc_id,
        source_url="",
        doc_class=DocClass.AGENT_ARTIFACT,
        doc_type=DocType.INCIDENT_INVESTIGATION,
        content_type="text/markdown",
        content_hash=_content_hash(payload),
        title=payload.title,
        body=payload.body_markdown,
        body_preview=payload.body_markdown[:280],
        body_size_bytes=len(payload.body_markdown.encode("utf-8")),
        body_token_count=count_tokens(payload.body_markdown),
        parent_doc_id=payload.incident_doc_id,
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        acl=acl,
        metadata={
            "version": payload.version,
            "mode": payload.mode,
            "evidence": [e.model_dump() for e in payload.evidence],
            "narrative": payload.narrative,
            "triage": (
                payload.triage.model_dump() if payload.triage is not None else None
            ),
            "tool_trace_run_id": payload.tool_trace_run_id,
            "prior_report_doc_id": payload.prior_report_doc_id,
            "reviewer_feedback": payload.reviewer_feedback,
            "incident_doc_id": payload.incident_doc_id,
            "source_event_id": payload.source_event_id,
        },
    )

    normalizer = request.app.state.normalizer
    await normalizer.persist_single_document(payload.customer_id, doc)

    detail = await upsert_pending_review(
        customer_id=payload.customer_id,
        incident_doc_id=payload.incident_doc_id,
        report_doc_id=doc_id,
        version=payload.version,
        mode=payload.mode,
    )

    return InvestigationWritebackResponse(
        report_doc_id=doc_id,
        state=detail.state,
        duplicate=False,
    )
