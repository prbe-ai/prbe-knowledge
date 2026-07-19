"""Per-source ingestion statistics — internal API.

    GET /api/stats/ingestion

Gated by X-Internal-Knowledge-Key; the tenant comes from the X-Prbe-Customer
header the gateway sets — never from the client. Returns per-source live
document/chunk counts, queue depth, and last-ingested timestamps, plus the
backfill_state rows and index-wide totals. Generic aggregates only — no
source-specific logic; consumers (a dashboard ingestion/health page) render
them however they like.

"Live" follows the bitemporal model: document versions and chunks with
valid_to IS NULL, documents additionally not soft-deleted.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException

from engine.shared.db import with_tenant
from kb.admin_routes import verify_internal_knowledge_key

router = APIRouter(prefix="/api/stats", tags=["stats"])


def _require_customer(
    x_prbe_customer: str | None = Header(default=None, alias="X-Prbe-Customer"),
) -> str:
    if not x_prbe_customer:
        raise HTTPException(status_code=400, detail="missing X-Prbe-Customer")
    return x_prbe_customer


@router.get("/ingestion", dependencies=[Depends(verify_internal_knowledge_key)])
async def ingestion_stats(customer_id: str = Depends(_require_customer)) -> dict:
    async with with_tenant(customer_id) as conn:
        doc_rows = await conn.fetch(
            """
            SELECT source_system,
                   COUNT(*)          AS docs,
                   MAX(ingested_at)  AS last_ingested_at
            FROM documents
            WHERE customer_id = $1 AND valid_to IS NULL AND deleted_at IS NULL
            GROUP BY source_system
            """,
            customer_id,
        )
        chunk_rows = await conn.fetch(
            """
            SELECT d.source_system, COUNT(*) AS chunks
            FROM chunks c
            JOIN documents d
              ON d.customer_id = c.customer_id
             AND d.doc_id      = c.doc_id
             AND d.valid_to   IS NULL
             AND d.deleted_at IS NULL
            WHERE c.customer_id = $1 AND c.valid_to IS NULL
            GROUP BY d.source_system
            """,
            customer_id,
        )
        queue_rows = await conn.fetch(
            """
            SELECT source_system, status, COUNT(*) AS n
            FROM ingestion_queue
            WHERE customer_id = $1 AND status IN ('pending', 'processing', 'dlq')
            GROUP BY source_system, status
            """,
            customer_id,
        )
        backfill_rows = await conn.fetch(
            """
            SELECT source_system, status, events_enqueued, last_error,
                   started_at, last_progress_at, completed_at
            FROM backfill_state
            WHERE customer_id = $1
            ORDER BY source_system
            """,
            customer_id,
        )

    sources: dict[str, dict] = {}

    def entry(source: str) -> dict:
        return sources.setdefault(
            source,
            {
                "source": source,
                "docs": 0,
                "chunks": 0,
                "pending": 0,
                "processing": 0,
                "dlq": 0,
                "last_ingested_at": None,
            },
        )

    for r in doc_rows:
        e = entry(r["source_system"])
        e["docs"] = r["docs"]
        e["last_ingested_at"] = (
            r["last_ingested_at"].isoformat() if r["last_ingested_at"] else None
        )
    for r in chunk_rows:
        entry(r["source_system"])["chunks"] = r["chunks"]
    for r in queue_rows:
        entry(r["source_system"])[r["status"]] = r["n"]

    per_source = sorted(sources.values(), key=lambda e: (-e["docs"], e["source"]))
    return {
        "customer_id": customer_id,
        "totals": {
            "docs": sum(e["docs"] for e in per_source),
            "chunks": sum(e["chunks"] for e in per_source),
        },
        "sources": per_source,
        "backfills": [
            {
                "source": r["source_system"],
                "status": r["status"],
                "events_enqueued": r["events_enqueued"],
                "last_error": r["last_error"],
                "started_at": r["started_at"].isoformat() if r["started_at"] else None,
                "last_progress_at": (
                    r["last_progress_at"].isoformat() if r["last_progress_at"] else None
                ),
                "completed_at": (
                    r["completed_at"].isoformat() if r["completed_at"] else None
                ),
            }
            for r in backfill_rows
        ],
    }
