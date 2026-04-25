"""Team / customer bridge endpoints.

These endpoints sit between Better Auth's organization plugin and our
prbe-knowledge data plane. The split:

  * Better Auth (via the Next.js SDK):
      organization.create / update / delete (admin)
      member.invite / accept / role / remove / leave
      Read paths: listMembers / listInvitations / getActiveMember

  * This service:
      POST /teams                — bridge: after authClient.organization.create,
                                   call this to create the linked customers row
      DELETE /teams/{org_id}     — soft-delete the customer (keeps the org;
                                   the user can also delete the org afterwards
                                   in Better Auth, which will then fail
                                   because of ON DELETE RESTRICT — they must
                                   wait for the offline reaper)
      GET /me                    — return the resolved session

  * Out of scope here (handled by Better Auth client SDK directly):
      everything else.

This is the FORM of the boundary. Per-tenant prbe-knowledge endpoints
(integrations, ingestion stats, query) live alongside in routers/data.py
and reuse `require_session` + `require_role` for auth.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from services.dashboard.auth import (
    Session,
    require_role,
    require_session,
)
from shared import audit
from shared.db import raw_conn
from shared.logging import get_logger
from shared.provisioning import (
    CustomerAlreadyExists,
    CustomerNotFound,
    OrganizationAlreadyClaimed,
    create_customer_for_organization,
    get_customer_by_organization,
    soft_delete_customer,
)
from shared.storage import get_store

log = get_logger(__name__)

router = APIRouter(tags=["teams"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CreateTeamRequest(BaseModel):
    organization_id: str = Field(..., description="UUID from neon_auth.organization.id")
    customer_id: str = Field(
        ...,
        description="Caller-chosen tenant identifier. Must be unique across all "
        "customers; typically derived from the organization slug. "
        "Immutable — propagates to per-tenant R2 bucket names.",
        min_length=3,
        max_length=64,
    )
    display_name: str = Field(..., min_length=1, max_length=128)


class CreateTeamResponse(BaseModel):
    customer_id: str
    organization_id: str
    display_name: str
    api_key: str  # shown once; not recoverable afterwards
    bucket: str


class MeResponse(BaseModel):
    user_id: str
    email: str
    organization_id: str | None
    customer_id: str | None
    role: str | None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/me", response_model=MeResponse)
async def me(session: Session = Depends(require_session)) -> MeResponse:
    return MeResponse(
        user_id=session.user_id,
        email=session.email,
        organization_id=session.organization_id,
        customer_id=session.customer_id,
        role=session.role,
    )


@router.post("/teams", response_model=CreateTeamResponse, status_code=201)
async def create_team(
    body: CreateTeamRequest,
    session: Session = Depends(require_session),
) -> CreateTeamResponse:
    """Bridge: create the customers row linked to a Better Auth organization.

    Caller flow (frontend):
      1. authClient.organization.create({ name, slug, ... })  -> returns org
      2. POST /api/teams { organization_id, customer_id, display_name }

    The caller MUST be a member of the org with role=owner. We resolve
    that from session.role on the active org. If the user just created
    the org, Better Auth has already inserted them as owner, so this
    check passes naturally.
    """
    # 1. Caller must be the owner of the organization they're claiming.
    if session.organization_id != body.organization_id:
        raise HTTPException(
            status_code=403,
            detail="caller is not a member of the requested organization",
        )
    if session.role != "owner":
        raise HTTPException(
            status_code=403,
            detail="only the organization owner can claim a customer record",
        )

    # 2. Fast-fail if the org already has a customer (idempotency check;
    #    the underlying partial-unique index also enforces this).
    existing = await get_customer_by_organization(body.organization_id)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail="organization already has a customer",
        )

    # 3. Create the customer row + R2 bucket. Atomic at the DB layer; bucket
    #    creation is best-effort and idempotent.
    try:
        api_key = await create_customer_for_organization(
            customer_id=body.customer_id,
            organization_id=body.organization_id,
            display_name=body.display_name,
        )
    except CustomerAlreadyExists as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OrganizationAlreadyClaimed as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    store = get_store()
    bucket = store.bucket_for(body.customer_id)
    await store.ensure_bucket(bucket)

    # 4. Audit log (separate transaction is fine — the customer row commits
    #    above. Best-effort: a failed audit log row should not 500 the API.)
    try:
        async with raw_conn() as conn:
            await audit.record(
                conn,
                customer_id=body.customer_id,
                actor_id=session.user_id,
                action=audit.AuditAction.CUSTOMER_LINKED_TO_ORG,
                resource_type="customer",
                resource_id=body.customer_id,
                details={
                    "organization_id": body.organization_id,
                    "display_name": body.display_name,
                },
            )
    except Exception as exc:  # pragma: no cover — log + swallow
        log.warning(
            "dashboard.teams.audit_log_failed",
            customer=body.customer_id,
            error=str(exc),
        )

    return CreateTeamResponse(
        customer_id=body.customer_id,
        organization_id=body.organization_id,
        display_name=body.display_name,
        api_key=api_key,
        bucket=bucket,
    )


@router.delete("/teams/{organization_id}", status_code=204)
async def delete_team(
    organization_id: str,
    session: Session = Depends(require_role("owner")),
) -> None:
    """Soft-delete the customer linked to this organization.

    Better Auth's organization is left in place; the dashboard hides it
    from the UI because there is no active customer to back it. The
    offline reaper later hard-deletes both the customer (cascading
    per-tenant data) and the organization.
    """
    if session.organization_id != organization_id:
        raise HTTPException(
            status_code=403,
            detail="caller is not a member of the requested organization",
        )

    customer = await get_customer_by_organization(organization_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="no customer for this organization")

    try:
        await soft_delete_customer(customer["customer_id"])
    except CustomerNotFound as exc:  # pragma: no cover (we just looked it up)
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        async with raw_conn() as conn:
            await audit.record(
                conn,
                customer_id=customer["customer_id"],
                actor_id=session.user_id,
                action=audit.AuditAction.CUSTOMER_SOFT_DELETED,
                resource_type="customer",
                resource_id=customer["customer_id"],
                details={"organization_id": organization_id},
            )
    except Exception as exc:  # pragma: no cover
        log.warning(
            "dashboard.teams.audit_log_failed",
            customer=customer["customer_id"],
            error=str(exc),
        )
