"""Read-only candidates endpoint for the staff dashboard's triage view.

One handler — ``GET /kg/candidates`` — returns the authenticated tenant's
``kg_candidates`` rows, ordered by ``created_at DESC``, with a status
filter (default ``pending``) and a bounded ``limit`` (default 100, max
500). Backs the staff dashboard's Task 25 triage surface.

Schema notes (see migration ``0032_kg_candidates`` for the full shape):

  * ``status`` is a CHECK-constrained four-state enum
    (``pending|accepted|rejected|merged``); we mirror that as a Literal
    in the query model so FastAPI returns 422 for unknown values.
  * ``notes_embedding`` (vector(1536)) is intentionally NOT returned —
    it is large (per-row 6KB-ish) and not useful for the UI; the staff
    dashboard renders ``payload`` and ``payload_hash`` instead.
  * RLS scoping is handled by ``with_tenant(customer_id)``; the
    explicit ``WHERE customer_id = $1`` is defense-in-depth.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from services.retrieval.auth import authenticate_query
from shared.db import with_tenant

router = APIRouter()


# Mirrors the CHECK constraint on kg_candidates.status. FastAPI uses this
# to validate the ?status= query param and 422 unknown values.
CandidateStatus = Literal["pending", "accepted", "rejected", "merged"]


class CandidateItem(BaseModel):
    """One row in the ``GET /kg/candidates`` response.

    Shape mirrors ``kg_candidates`` minus ``notes_embedding`` (large; not
    useful for the UI). ``candidate_id`` is serialized as a string
    (uuid-as-text) so the dashboard doesn't need a UUID parser.
    """

    candidate_id: str
    payload_hash: str
    payload: dict[str, Any]
    status: str
    repeat_count: int
    created_at: datetime
    resolved_at: datetime | None


class CandidateListResponse(BaseModel):
    """Envelope for ``GET /kg/candidates``."""

    items: list[CandidateItem]
    total_returned: int


def _decode_jsonb(value: Any) -> Any:
    """Normalize asyncpg's JSONB return value to a Python object.

    Same rationale as ``services.kg.api.read._decode_jsonb`` — asyncpg
    returns JSONB as either ``str``/``bytes`` or ``dict`` depending on
    whether a JSONB codec is registered. Handle both.
    """
    if isinstance(value, (str, bytes, bytearray)):
        return json.loads(value)
    return value


@router.get("/candidates", response_model=CandidateListResponse)
async def list_candidates(
    customer_id: str = Depends(authenticate_query),
    status: CandidateStatus = Query(default="pending"),
    limit: int = Query(default=100, ge=1, le=500),
) -> CandidateListResponse:
    """Return the most recent candidates for the authenticated tenant.

    Ordered by ``created_at DESC`` so the dashboard sees the freshest
    triage items first. ``notes_embedding`` is deliberately omitted —
    see module docstring.
    """
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT candidate_id,
                   payload_hash,
                   payload,
                   status,
                   repeat_count,
                   created_at,
                   resolved_at
              FROM kg_candidates
             WHERE customer_id = $1
               AND status      = $2
             ORDER BY created_at DESC
             LIMIT $3
            """,
            customer_id,
            status,
            limit,
        )

    items = [
        CandidateItem(
            candidate_id=str(r["candidate_id"]),
            payload_hash=r["payload_hash"],
            payload=_decode_jsonb(r["payload"]),
            status=r["status"],
            repeat_count=r["repeat_count"],
            created_at=r["created_at"],
            resolved_at=r["resolved_at"],
        )
        for r in rows
    ]
    return CandidateListResponse(items=items, total_returned=len(items))
