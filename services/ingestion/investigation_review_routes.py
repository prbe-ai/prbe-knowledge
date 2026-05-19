"""Review endpoints for incident investigations.

GET  /api/incident-investigations
GET  /api/incident-investigations/{incident_doc_id}
POST /api/incident-investigations/{incident_doc_id}/approve
POST /api/incident-investigations/{incident_doc_id}/reject

All endpoints are X-Internal-Knowledge-Key gated; the dashboard BFF
(Plan 5) proxies dashboard requests through and supplies `customer_id`
from the session JWT.

Reject does NOT yet re-dispatch to orchestrator — Plan 4 wires the
re-run hook later. For now reject only records the rejection in
`incident_investigations` and returns the new state.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from services.ingestion.investigation_state import (
    get_detail,
    list_for_customer,
    mark_approved,
    mark_rejected,
)
from shared.exceptions import InvestigationNotFound
from shared.investigation_schemas import (
    ApproveRequest,
    InvestigationDetail,
    InvestigationListItem,
    InvestigationState,
    RejectRequest,
)

router = APIRouter(
    prefix="/api/incident-investigations",
    tags=["incident-investigations"],
)


def _verify_key(request: Request) -> None:
    # Local import avoids circular import with main.py — same pattern as
    # the writeback route module.
    from services.ingestion.main import _verify_internal_key
    _verify_internal_key(request)


@router.get("", response_model=list[InvestigationListItem])
async def list_investigations(
    request: Request,
    customer_id: str = Query(..., min_length=1),
    state: InvestigationState | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[InvestigationListItem]:
    _verify_key(request)
    return await list_for_customer(
        customer_id, state=state, limit=limit, offset=offset,
    )


@router.get("/{incident_doc_id:path}", response_model=InvestigationDetail)
async def detail(
    incident_doc_id: str,
    request: Request,
    customer_id: str = Query(..., min_length=1),
) -> InvestigationDetail:
    _verify_key(request)
    d = await get_detail(customer_id, incident_doc_id)
    if d is None:
        raise HTTPException(status_code=404, detail="investigation_not_found")
    return d


@router.post(
    "/{incident_doc_id:path}/approve",
    response_model=InvestigationDetail,
)
async def approve(
    incident_doc_id: str,
    body: ApproveRequest,
    request: Request,
    customer_id: str = Query(..., min_length=1),
) -> InvestigationDetail:
    _verify_key(request)
    try:
        return await mark_approved(
            customer_id=customer_id,
            incident_doc_id=incident_doc_id,
            reviewer_id=body.reviewer_id,
        )
    except InvestigationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/{incident_doc_id:path}/reject",
    response_model=InvestigationDetail,
)
async def reject(
    incident_doc_id: str,
    body: RejectRequest,
    request: Request,
    customer_id: str = Query(..., min_length=1),
) -> InvestigationDetail:
    _verify_key(request)
    try:
        return await mark_rejected(
            customer_id=customer_id,
            incident_doc_id=incident_doc_id,
            reviewer_id=body.reviewer_id,
            feedback=body.feedback,
        )
    except InvestigationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
