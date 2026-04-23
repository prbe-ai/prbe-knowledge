"""Backfill status endpoint.

    GET /backfill/status?customer_id=<id>
    GET /backfill/status?customer_id=<id>&source=<slack|linear|...>

Returns per-source backfill progress (status, events enqueued, cursor,
last heartbeat). Useful for a dashboard onboarding progress bar.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from shared.constants import SourceSystem
from shared.db import raw_conn

router = APIRouter(prefix="/backfill", tags=["backfill"])


@router.get("/status")
async def backfill_status(
    customer_id: str = Query(...),
    source: str | None = Query(default=None),
) -> dict:
    if source is not None:
        try:
            SourceSystem(source)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown source '{source}'") from exc

    async with raw_conn() as conn:
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
