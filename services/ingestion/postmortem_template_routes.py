"""Per-customer postmortem template routes.

Three endpoints:
- GET    /api/customer-postmortem-templates/{customer_id}             override row | null
- GET    /api/customer-postmortem-templates/{customer_id}/effective   resolved template
- PUT    /api/customer-postmortem-templates/{customer_id}             upsert override

Used by:
- The dashboard BFF (Plan C) to render the "Customize postmortem
  template" page and accept inline / doc_ref overrides from admins.
- The orchestrator's postmortem author (Plan B) to fetch the effective
  template body to render the draft against.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from services.post_approval.template_resolver import (
    get_effective_template,
    get_override,
    upsert_override,
)
from shared.logging import get_logger
from shared.schemas.postmortem_template import (
    TemplateEffectiveResponse,
    TemplateRow,
    TemplateUpsertRequest,
)

log = get_logger(__name__)

router = APIRouter(
    prefix="/api/customer-postmortem-templates",
    tags=["postmortem-templates"],
)


def _verify_key(request: Request) -> None:
    from services.ingestion.main import _verify_internal_key
    _verify_internal_key(request)


@router.get("/{customer_id}", response_model=TemplateRow | None)
async def get(
    customer_id: str,
    request: Request,
) -> TemplateRow | None:
    """Return the override row for this customer, or null when no
    override exists (caller falls back to the default template).
    """
    _verify_key(request)
    return await get_override(customer_id)


@router.get(
    "/{customer_id}/effective",
    response_model=TemplateEffectiveResponse,
)
async def effective(
    customer_id: str,
    request: Request,
) -> TemplateEffectiveResponse:
    """Resolve the actual template body the agent will use.

    Falls through to the bundled default when no override is set or
    when a doc_ref override points at an unresolvable doc. See
    ``services/post_approval/template_resolver.py::get_effective_template``
    for the resolution order.
    """
    _verify_key(request)
    return await get_effective_template(customer_id)


@router.put("/{customer_id}", response_model=TemplateRow)
async def put(
    customer_id: str,
    req: TemplateUpsertRequest,
    request: Request,
) -> TemplateRow:
    """Insert or update the customer's template override.

    The path ``customer_id`` and body ``customer_id`` must match — a
    mismatch is a 400. ``upsert_override`` validates that doc_ref mode
    points at a readable approved doc; a ValueError there surfaces as
    422 with the resolver's error message.
    """
    _verify_key(request)
    if req.customer_id != customer_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"path customer_id '{customer_id}' does not match "
                f"body customer_id '{req.customer_id}'"
            ),
        )
    try:
        row = await upsert_override(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    log.info(
        "postmortem.template_updated",
        customer_id=customer_id,
        mode=row.mode,
        has_ref_doc=row.ref_doc_id is not None,
    )

    return row


__all__ = ["router"]
