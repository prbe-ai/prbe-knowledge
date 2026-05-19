"""Review surface for wiki artifacts.

Endpoints:
- GET  /api/wiki-artifacts                      list pending/approved/rejected
- GET  /api/wiki-artifacts/{artifact_doc_id}    full detail + version lineage
- POST /api/wiki-artifacts/{artifact_doc_id}/approve   atomic visibility flip
- POST /api/wiki-artifacts/{artifact_doc_id}/reject    + orchestrator re-dispatch

The writeback route (services/ingestion/wiki_artifact_writeback_routes.py)
owns POST "" on the same prefix — read/write split across two route
modules keeps each focused and mirrors the investigation-route split
(writeback + review) shipped in Plan 4.

Approve is genuinely atomic: documents.visibility, chunks.visibility,
and wiki_review_queue.state all flip in one ``async with conn.transaction()``
block under the tenant GUC. A failure mid-block rolls back all three.

Reject is durable independent of orchestrator re-dispatch: the state
flip persists even if the orchestrator POST exhausts retries; the
metadata.re_dispatch_failed flag lets the dashboard surface the
stuck row for manual recovery.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

from services.post_approval.wiki_review_state import (
    get_detail,
    list_for_customer,
    mark_rejected,
)
from shared.config import get_settings
from shared.db import with_tenant
from shared.logging import get_logger
from shared.schemas.wiki_artifact import (
    ApproveRequest,
    ArtifactKind,
    ArtifactState,
    RejectRequest,
    WikiArtifactDetail,
    WikiArtifactListItem,
)

log = get_logger(__name__)

# Re-dispatch retry budget mirrors services/post_approval/dispatch.py —
# same backoff schedule (1s/3s/9s) so ops sees consistent latency
# behavior across all orchestrator HTTP fanout paths.
_DISPATCH_TIMEOUT_S = 30
_MAX_RETRIES = 3
_BACKOFF_BASE_S = 1.0


router = APIRouter(
    prefix="/api/wiki-artifacts",
    tags=["wiki-artifacts"],
)


def _verify_key(request: Request) -> None:
    from services.ingestion.main import _verify_internal_key
    _verify_internal_key(request)


# ---- list / detail -------------------------------------------------------


@router.get("", response_model=list[WikiArtifactListItem])
async def list_artifacts(
    request: Request,
    customer_id: str = Query(..., min_length=1),
    state: ArtifactState | None = Query(default=None),
    artifact_kind: ArtifactKind | None = Query(default=None),
    incident_doc_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[WikiArtifactListItem]:
    _verify_key(request)
    return await list_for_customer(
        customer_id,
        state=state,
        artifact_kind=artifact_kind,
        incident_doc_id=incident_doc_id,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{artifact_doc_id:path}",
    response_model=WikiArtifactDetail,
)
async def detail(
    artifact_doc_id: str,
    request: Request,
    customer_id: str = Query(..., min_length=1),
) -> WikiArtifactDetail:
    _verify_key(request)
    d = await get_detail(customer_id, artifact_doc_id)
    if d is None:
        raise HTTPException(status_code=404, detail="wiki_artifact_not_found")
    return d


# ---- approve -------------------------------------------------------------


@router.post(
    "/{artifact_doc_id:path}/approve",
    response_model=WikiArtifactDetail,
)
async def approve(
    artifact_doc_id: str,
    body: ApproveRequest,
    request: Request,
    customer_id: str = Query(..., min_length=1),
) -> WikiArtifactDetail:
    """Approve flips documents.visibility, chunks.visibility, and
    wiki_review_queue.state to ``approved`` in one transaction.

    Idempotent on already-approved. Returns 409 when the row is
    already in the terminal ``rejected`` state — we don't silently
    resurrect a rejected draft.
    """
    _verify_key(request)

    existing = await get_detail(customer_id, artifact_doc_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="wiki_artifact_not_found")
    if existing.state == "approved":
        return existing  # idempotent
    if existing.state == "rejected":
        raise HTTPException(
            status_code=409,
            detail="cannot approve a rejected artifact",
        )

    # Three writes under one transaction. with_tenant already opens
    # an asyncpg transaction (sets the tenant GUC under that tx); the
    # nested `async with conn.transaction()` runs as a SAVEPOINT,
    # which rolls back the three writes together on any failure
    # without dropping the outer tenant binding.
    async with with_tenant(customer_id) as conn, conn.transaction():
        await conn.execute(
            "UPDATE documents "
            "SET visibility = 'approved', updated_at = now() "
            "WHERE customer_id = $1 AND doc_id = $2",
            customer_id, artifact_doc_id,
        )
        await conn.execute(
            "UPDATE chunks "
            "SET visibility = 'approved' "
            "WHERE customer_id = $1 AND doc_id = $2",
            customer_id, artifact_doc_id,
        )
        await conn.execute(
            "UPDATE wiki_review_queue "
            "SET state = 'approved', "
            "    reviewer_id = $3, "
            "    reviewed_at = now(), "
            "    updated_at = now() "
            "WHERE customer_id = $1 AND artifact_doc_id = $2",
            customer_id, artifact_doc_id, body.reviewer_id,
        )

    log.info(
        "wiki_artifact.review.approved",
        customer_id=customer_id,
        artifact_doc_id=artifact_doc_id,
        reviewer_id=body.reviewer_id,
    )

    refreshed = await get_detail(customer_id, artifact_doc_id)
    assert refreshed is not None
    return refreshed


# ---- reject + re-dispatch -----------------------------------------------


async def _post_rerun_dispatch(payload: dict[str, Any]) -> bool:
    """POST a re-run dispatch to the orchestrator with bounded retry.

    Mirrors services/post_approval/dispatch.py::_post_dispatch (same
    timeout / backoff schedule / 4xx-no-retry semantics / header set).
    Duplicated here rather than imported to keep the cross-module
    coupling minimal — the function is small, and the reject path's
    payload shape differs from the post-approval payload (we send
    reviewer_feedback + prior_artifact_doc_id; that path sends
    approved_at + resolved_at).
    """
    settings = get_settings()
    base_url = settings.orchestrator_base_url.rstrip("/")
    if not base_url:
        log.error(
            "wiki_artifact.re_dispatch_no_orchestrator_url",
            payload_keys=list(payload.keys()),
        )
        return False
    url = f"{base_url}/internal/post-approval-actions"
    headers = {
        "x-internal-backend-key":
            settings.internal_backend_api_key.get_secret_value(),
        # Orchestrator's route uses Depends(require_customer_id) and 400s
        # without this header even though the body carries customer_id.
        "x-prbe-customer": payload["customer_id"],
        "content-type": "application/json",
    }

    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=_DISPATCH_TIMEOUT_S) as client:
                resp = await client.post(url, headers=headers, json=payload)
            if 200 <= resp.status_code < 300:
                return True
            if resp.status_code >= 500:
                log.warning(
                    "wiki_artifact.re_dispatch_retry",
                    status=resp.status_code,
                    attempt=attempt + 1,
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_BACKOFF_BASE_S * (3 ** attempt))
                continue
            log.error(
                "wiki_artifact.re_dispatch_4xx",
                status=resp.status_code,
                body=resp.text[:200],
            )
            return False
        except httpx.HTTPError as exc:
            log.warning(
                "wiki_artifact.re_dispatch_http_error",
                error=str(exc),
                error_class=type(exc).__name__,
                attempt=attempt + 1,
            )
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BACKOFF_BASE_S * (3 ** attempt))
            continue
        except Exception as exc:
            log.exception(
                "wiki_artifact.re_dispatch_unexpected_error",
                error_class=type(exc).__name__,
                attempt=attempt + 1,
            )
            return False

    log.error("wiki_artifact.re_dispatch_failed_exhausted")
    return False


async def _stamp_re_dispatch_failed(
    customer_id: str, artifact_doc_id: str,
) -> None:
    """Stamp the queue row's metadata.re_dispatch_failed=true.

    Surfaces in the dashboard's "stuck artifacts" view so an operator
    can manually re-trigger the rerun. The state flip itself is
    durable — only the re-dispatch flag is added.
    """
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            "UPDATE wiki_review_queue "
            "SET metadata = metadata "
            "    || jsonb_build_object('re_dispatch_failed', true), "
            "    updated_at = now() "
            "WHERE customer_id = $1 AND artifact_doc_id = $2",
            customer_id, artifact_doc_id,
        )


@router.post(
    "/{artifact_doc_id:path}/reject",
    response_model=WikiArtifactDetail,
)
async def reject(
    artifact_doc_id: str,
    body: RejectRequest,
    request: Request,
    customer_id: str = Query(..., min_length=1),
) -> WikiArtifactDetail:
    """Reject flips state to ``rejected`` and persists feedback, then
    fires a re-run dispatch to the orchestrator with the feedback so
    the agent can author a v2 draft.

    Critical invariant: the state flip is durable even if the orchestrator
    re-dispatch exhausts retries. The dashboard would otherwise show a
    perpetually pending_review row whose reviewer thought they rejected
    it. The re_dispatch_failed metadata flag lets ops re-trigger.
    """
    _verify_key(request)

    existing = await get_detail(customer_id, artifact_doc_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="wiki_artifact_not_found")
    if existing.state == "approved":
        raise HTTPException(
            status_code=409,
            detail="cannot reject an approved artifact",
        )
    if existing.state == "rejected":
        # Idempotent: already rejected. Return existing detail rather
        # than re-firing the orchestrator (which would double-dispatch
        # the re-run). The first reject's feedback is canonical.
        return existing

    try:
        await mark_rejected(
            customer_id=customer_id,
            artifact_doc_id=artifact_doc_id,
            reviewer_id=body.reviewer_id,
            feedback=body.feedback,
        )
    except LookupError as exc:
        # Race: another caller deleted the row between get_detail and
        # mark_rejected. Surface as 404 — consistent with the missing-
        # row branch above.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        # Race: another concurrent reject landed first (this caller's
        # mark_rejected raised because the row is now in terminal state).
        # The winning caller is responsible for the orchestrator re-run
        # dispatch; this loser must NOT double-fire — refetch the
        # post-state detail and return it WITHOUT calling
        # ``_post_rerun_dispatch``.
        log.info(
            "wiki_artifact.reject_race",
            customer_id=customer_id,
            artifact_doc_id=artifact_doc_id,
            error=str(exc),
        )
        refreshed = await get_detail(customer_id, artifact_doc_id)
        if refreshed is None:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return refreshed

    re_dispatch_payload = {
        "customer_id": customer_id,
        "incident_doc_id": existing.incident_doc_id,
        "prior_artifact_doc_id": artifact_doc_id,
        "artifact_kind": existing.artifact_kind,
        "reviewer_feedback": body.feedback,
    }
    ok = await _post_rerun_dispatch(re_dispatch_payload)
    if not ok:
        await _stamp_re_dispatch_failed(customer_id, artifact_doc_id)
        log.error(
            "wiki_artifact.review.re_dispatch_failed",
            customer_id=customer_id,
            artifact_doc_id=artifact_doc_id,
        )

    log.info(
        "wiki_artifact.review.rejected",
        customer_id=customer_id,
        artifact_doc_id=artifact_doc_id,
        reviewer_id=body.reviewer_id,
        feedback_length=len(body.feedback),
        re_dispatch_ok=ok,
    )

    refreshed = await get_detail(customer_id, artifact_doc_id)
    assert refreshed is not None
    return refreshed


__all__ = ["router"]
