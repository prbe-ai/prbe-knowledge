"""Entity merge suggestions — internal API.

Reads + decisions on the `entity_merge_suggestions` table populated by
the AutoMergeAnalyzer for medium/low-confidence verdicts (high-confidence
verdicts fire merges directly via the existing /api/entity-clusters/merge
endpoint; they don't appear here unless POST_WRITE_EXECUTE=false at the
time the verdict landed).

Auth: X-Internal-Knowledge-Key (same gate as entity_clusters_routes.py).
The dashboard BFF in prbe-backend proxies these endpoints.

Endpoints:

  GET    /api/entity-merge-suggestions
         List pending suggestions for the caller's customer_id, ordered
         newest-first. Optional confidence filter.

  POST   /api/entity-merge-suggestions/{suggestion_id}/approve
         Fire the merge for this suggestion via the existing
         merge_cluster transaction, then stamp the suggestion as
         'applied'. Idempotent: a suggestion whose alias is already
         merged into a cluster gets flipped to 'dismissed' instead.

  POST   /api/entity-merge-suggestions/{suggestion_id}/dismiss
         Mark suggestion as 'dismissed' without firing a merge.
"""

from __future__ import annotations

import hmac
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from engine.ingest.entity_clusters_routes import (
    MergeRequest,
    merge_cluster,
)
from engine.shared.config import get_settings
from engine.shared.db import with_tenant

log = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/entity-merge-suggestions", tags=["internal-api"]
)


SYSTEM_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# Auth (identical to entity_clusters_routes)
# ---------------------------------------------------------------------------


def _require_internal_key(
    x_internal_knowledge_key: str | None = Header(default=None),
) -> None:
    expected = get_settings().internal_knowledge_api_key
    if expected is None or not expected.get_secret_value():
        raise HTTPException(
            status_code=503,
            detail="disabled — set INTERNAL_KNOWLEDGE_API_KEY",
        )
    if not x_internal_knowledge_key or not hmac.compare_digest(
        x_internal_knowledge_key, expected.get_secret_value()
    ):
        raise HTTPException(status_code=401, detail="invalid internal key")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Suggestion(BaseModel):
    suggestion_id: uuid.UUID
    label: str
    primary_canonical_id: str
    candidate_canonical_id: str
    confidence: str
    rationale: str | None
    llm_model: str
    status: str
    created_at: datetime
    decided_at: datetime | None


class ListSuggestionsResponse(BaseModel):
    suggestions: list[Suggestion]


class DecisionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suggestion_id: uuid.UUID
    status: str  # 'applied' | 'dismissed'
    merge_id: uuid.UUID | None = None


# ---------------------------------------------------------------------------
# GET /api/entity-merge-suggestions
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=ListSuggestionsResponse,
    dependencies=[Depends(_require_internal_key)],
)
async def list_suggestions(
    customer_id: str = Query(..., min_length=1, max_length=128),
    confidence: str | None = Query(default=None),
    status: str = Query(default="pending"),
    limit: int = Query(default=50, ge=1, le=200),
) -> ListSuggestionsResponse:
    """List entity merge suggestions for a customer.

    Defaults: pending status, all confidence levels, newest first, 50 max.
    """
    if confidence is not None and confidence not in {"high", "medium", "low"}:
        raise HTTPException(status_code=400, detail="invalid confidence filter")
    if status not in {"pending", "approved", "dismissed", "applied"}:
        raise HTTPException(status_code=400, detail="invalid status filter")

    async with with_tenant(customer_id) as conn:
        if confidence:
            rows = await conn.fetch(
                """
                SELECT suggestion_id, label, primary_canonical_id,
                       candidate_canonical_id, confidence, rationale,
                       llm_model, status, created_at, decided_at
                FROM entity_merge_suggestions
                WHERE status = $1 AND confidence = $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                status, confidence, limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT suggestion_id, label, primary_canonical_id,
                       candidate_canonical_id, confidence, rationale,
                       llm_model, status, created_at, decided_at
                FROM entity_merge_suggestions
                WHERE status = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                status, limit,
            )
    return ListSuggestionsResponse(
        suggestions=[
            Suggestion(
                suggestion_id=r["suggestion_id"],
                label=r["label"],
                primary_canonical_id=r["primary_canonical_id"],
                candidate_canonical_id=r["candidate_canonical_id"],
                confidence=r["confidence"],
                rationale=r["rationale"],
                llm_model=r["llm_model"],
                status=r["status"],
                created_at=r["created_at"],
                decided_at=r["decided_at"],
            )
            for r in rows
        ]
    )


# ---------------------------------------------------------------------------
# POST /api/entity-merge-suggestions/{id}/approve
# ---------------------------------------------------------------------------


class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., min_length=1, max_length=128)
    performed_by_user_id: uuid.UUID = Field(
        default_factory=lambda: SYSTEM_USER_ID,
        description=(
            "UUID of the human approving. Defaults to the nil UUID so the "
            "audit trail still records the approval; callers SHOULD pass "
            "the real user when available."
        ),
    )


@router.post(
    "/{suggestion_id}/approve",
    response_model=DecisionResponse,
    dependencies=[Depends(_require_internal_key)],
)
async def approve_suggestion(
    suggestion_id: uuid.UUID,
    body: ApproveRequest,
) -> DecisionResponse:
    """Fire the merge for this suggestion + mark it applied."""
    async with with_tenant(body.customer_id) as conn:
        row = await conn.fetchrow(
            """
            SELECT label, primary_canonical_id, candidate_canonical_id, rationale, status
            FROM entity_merge_suggestions
            WHERE suggestion_id = $1
            """,
            suggestion_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="suggestion not found")
    if row["status"] != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"suggestion already {row['status']}",
        )

    try:
        resp = await merge_cluster(
            MergeRequest(
                customer_id=body.customer_id,
                performed_by_user_id=body.performed_by_user_id,
                label=row["label"],
                primary_canonical_id=row["primary_canonical_id"],
                alias_canonical_ids=[row["candidate_canonical_id"]],
                reason=f"approved suggestion {suggestion_id}: {(row['rationale'] or '')[:160]}",
            )
        )
    except HTTPException as e:
        # If the alias is already in a cluster, flip to dismissed rather
        # than 409ing — the merge effect is already in place.
        if e.status_code in (404, 409):
            async with with_tenant(body.customer_id) as conn:
                await conn.execute(
                    "UPDATE entity_merge_suggestions "
                    "SET status='dismissed', decided_at=NOW(), decided_by_user_id=$2 "
                    "WHERE suggestion_id=$1",
                    suggestion_id, body.performed_by_user_id,
                )
            return DecisionResponse(
                suggestion_id=suggestion_id, status="dismissed", merge_id=None
            )
        raise

    async with with_tenant(body.customer_id) as conn:
        await conn.execute(
            "UPDATE entity_merge_suggestions "
            "SET status='applied', decided_at=NOW(), decided_by_user_id=$2 "
            "WHERE suggestion_id=$1",
            suggestion_id, body.performed_by_user_id,
        )
    return DecisionResponse(
        suggestion_id=suggestion_id, status="applied", merge_id=resp.merge_id
    )


# ---------------------------------------------------------------------------
# POST /api/entity-merge-suggestions/{id}/dismiss
# ---------------------------------------------------------------------------


class DismissRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., min_length=1, max_length=128)
    performed_by_user_id: uuid.UUID = Field(default_factory=lambda: SYSTEM_USER_ID)


@router.post(
    "/{suggestion_id}/dismiss",
    response_model=DecisionResponse,
    dependencies=[Depends(_require_internal_key)],
)
async def dismiss_suggestion(
    suggestion_id: uuid.UUID,
    body: DismissRequest,
) -> DecisionResponse:
    async with with_tenant(body.customer_id) as conn:
        result = await conn.execute(
            "UPDATE entity_merge_suggestions "
            "SET status='dismissed', decided_at=NOW(), decided_by_user_id=$2 "
            "WHERE suggestion_id=$1 AND status='pending'",
            suggestion_id, body.performed_by_user_id,
        )
    if result == "UPDATE 0":
        raise HTTPException(
            status_code=404, detail="suggestion not found or not pending"
        )
    return DecisionResponse(
        suggestion_id=suggestion_id, status="dismissed", merge_id=None
    )


# ---------------------------------------------------------------------------
# POST /api/entity-merge-suggestions/approve-all
# ---------------------------------------------------------------------------


class ApproveAllRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., min_length=1, max_length=128)
    performed_by_user_id: uuid.UUID = Field(default_factory=lambda: SYSTEM_USER_ID)
    confidence: str | None = Field(
        default=None,
        description=(
            "Optional filter: only approve suggestions at this confidence "
            "level ('high'/'medium'/'low'). When unset, approves every "
            "pending suggestion."
        ),
    )
    max_to_approve: int = Field(
        default=200,
        ge=1,
        le=500,
        description="Safety cap so a stuck UI doesn't fire thousands.",
    )


class ApproveAllResult(BaseModel):
    suggestion_id: uuid.UUID
    outcome: str  # 'applied' | 'dismissed_already_merged' | 'error'
    merge_id: uuid.UUID | None = None
    error: str | None = None


class ApproveAllResponse(BaseModel):
    total_considered: int
    approved: int
    dismissed_already_merged: int
    errors: int
    results: list[ApproveAllResult]


@router.post(
    "/approve-all",
    response_model=ApproveAllResponse,
    dependencies=[Depends(_require_internal_key)],
)
async def approve_all_suggestions(body: ApproveAllRequest) -> ApproveAllResponse:
    """Bulk-approve pending suggestions. Sequential — one merge txn per row.

    Failure modes per row are recorded individually so a single bad merge
    doesn't abort the batch. Already-merged aliases (409 on the inner
    merge_cluster call) get flipped to 'dismissed' rather than counted as
    errors. Returns a per-row result list for the UI to render.
    """
    if body.confidence is not None and body.confidence not in {"high", "medium", "low"}:
        raise HTTPException(status_code=400, detail="invalid confidence filter")

    async with with_tenant(body.customer_id) as conn:
        if body.confidence:
            rows = await conn.fetch(
                """
                SELECT suggestion_id, label, primary_canonical_id,
                       candidate_canonical_id, rationale
                FROM entity_merge_suggestions
                WHERE status = 'pending' AND confidence = $1
                ORDER BY created_at ASC
                LIMIT $2
                """,
                body.confidence, body.max_to_approve,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT suggestion_id, label, primary_canonical_id,
                       candidate_canonical_id, rationale
                FROM entity_merge_suggestions
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT $1
                """,
                body.max_to_approve,
            )

    results: list[ApproveAllResult] = []
    approved = 0
    dismissed = 0
    errors = 0

    for r in rows:
        sid: uuid.UUID = r["suggestion_id"]
        try:
            resp = await merge_cluster(
                MergeRequest(
                    customer_id=body.customer_id,
                    performed_by_user_id=body.performed_by_user_id,
                    label=r["label"],
                    primary_canonical_id=r["primary_canonical_id"],
                    alias_canonical_ids=[r["candidate_canonical_id"]],
                    reason=f"approve-all: {(r['rationale'] or '')[:160]}",
                )
            )
            async with with_tenant(body.customer_id) as conn:
                await conn.execute(
                    "UPDATE entity_merge_suggestions SET status='applied', "
                    "decided_at=NOW(), decided_by_user_id=$2 WHERE suggestion_id=$1",
                    sid, body.performed_by_user_id,
                )
            approved += 1
            results.append(
                ApproveAllResult(
                    suggestion_id=sid, outcome="applied", merge_id=resp.merge_id
                )
            )
        except HTTPException as e:
            # 404 (alias node already deleted) or 409 (alias already in cluster)
            # → the merge effect is already in place. Dismiss the row.
            if e.status_code in (404, 409):
                async with with_tenant(body.customer_id) as conn:
                    await conn.execute(
                        "UPDATE entity_merge_suggestions SET status='dismissed', "
                        "decided_at=NOW(), decided_by_user_id=$2 WHERE suggestion_id=$1",
                        sid, body.performed_by_user_id,
                    )
                dismissed += 1
                results.append(
                    ApproveAllResult(
                        suggestion_id=sid, outcome="dismissed_already_merged"
                    )
                )
            else:
                errors += 1
                results.append(
                    ApproveAllResult(
                        suggestion_id=sid,
                        outcome="error",
                        error=f"{e.status_code}: {e.detail}",
                    )
                )
        except Exception as e:
            log.exception(
                "approve_all.merge_failed",
                extra={"suggestion_id": str(sid)},
            )
            errors += 1
            results.append(
                ApproveAllResult(
                    suggestion_id=sid, outcome="error", error=repr(e)[:240]
                )
            )

    return ApproveAllResponse(
        total_considered=len(rows),
        approved=approved,
        dismissed_already_merged=dismissed,
        errors=errors,
        results=results,
    )
