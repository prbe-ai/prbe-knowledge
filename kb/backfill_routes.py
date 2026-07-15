"""Backfill status endpoint — internal API.

    GET /backfill/status
    GET /backfill/status?source=<slack|linear|...>

Gated by X-Internal-Knowledge-Key (caller is the prbe-backend gateway);
the tenant comes from the X-Prbe-Customer header the gateway sets —
never from the client. Returns per-source backfill progress (status,
events enqueued, cursor, last heartbeat). Useful for a dashboard
onboarding progress bar.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from services.ingestion.admin_routes import verify_internal_knowledge_key
from shared.constants import SourceSystem
from shared.db import with_tenant

router = APIRouter(prefix="/backfill", tags=["backfill"])


def _require_customer(
    x_prbe_customer: str | None = Header(default=None, alias="X-Prbe-Customer"),
) -> str:
    if not x_prbe_customer:
        raise HTTPException(status_code=400, detail="missing X-Prbe-Customer")
    return x_prbe_customer


@router.get("/status", dependencies=[Depends(verify_internal_knowledge_key)])
async def backfill_status(
    customer_id: str = Depends(_require_customer),
    source: str | None = Query(default=None),
) -> dict:
    if source is not None:
        try:
            SourceSystem(source)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown source '{source}'") from exc

    async with with_tenant(customer_id) as conn:
        if source:
            rows = await conn.fetch(
                """
                SELECT source_system, status, events_enqueued, last_error,
                       started_at, heartbeat_at, last_progress_at, completed_at
                FROM backfill_state
                WHERE customer_id = $1 AND source_system = $2
                """,
                customer_id,
                source,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT source_system, status, events_enqueued, last_error,
                       started_at, heartbeat_at, last_progress_at, completed_at
                FROM backfill_state
                WHERE customer_id = $1
                ORDER BY source_system
                """,
                customer_id,
            )

    return {
        "customer_id": customer_id,
        "sources": [
            {
                "source": r["source_system"],
                "status": r["status"],
                "events_enqueued": r["events_enqueued"],
                "last_error": r["last_error"],
                "started_at": r["started_at"].isoformat() if r["started_at"] else None,
                "heartbeat_at": r["heartbeat_at"].isoformat() if r["heartbeat_at"] else None,
                "last_progress_at": (
                    r["last_progress_at"].isoformat() if r["last_progress_at"] else None
                ),
                "completed_at": (
                    r["completed_at"].isoformat() if r["completed_at"] else None
                ),
            }
            for r in rows
        ],
    }
