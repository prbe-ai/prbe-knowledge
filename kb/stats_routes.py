"""Per-source ingestion statistics — internal API.

    GET /api/stats/ingestion
    GET /api/stats/ingestion/{source}/devices

Gated by X-Internal-Knowledge-Key; the tenant comes from the X-Prbe-Customer
header the gateway sets — never from the client. Returns per-source live
document/chunk counts, queue depth, and last-ingested timestamps, plus the
backfill_state rows and index-wide totals. The device route is limited to
device-paired sources and keeps queue counts on their parent source row.

"Live" follows the bitemporal model: document versions and chunks with
valid_to IS NULL, documents additionally not soft-deleted.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException

from engine.shared.constants import SourceSystem
from engine.shared.db import with_tenant
from kb.admin_routes import verify_internal_knowledge_key

router = APIRouter(prefix="/api/stats", tags=["stats"])

_DEVICE_PAIRED_SOURCES = frozenset(
    {SourceSystem.CLAUDE_CODE.value, SourceSystem.CODEX.value}
)


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


@router.get(
    "/ingestion/{source}/devices",
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def ingestion_device_stats(
    source: str,
    customer_id: str = Depends(_require_customer),
) -> dict[str, Any]:
    if source not in _DEVICE_PAIRED_SOURCES:
        raise HTTPException(
            status_code=404,
            detail=f"source does not support per-device stats: {source}",
        )

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            WITH live_docs AS (
                SELECT DISTINCT ON (d.customer_id, d.doc_id)
                       d.customer_id,
                       d.doc_id,
                       d.version,
                       d.parent_doc_id,
                       d.metadata,
                       d.ingested_at
                FROM documents d
                WHERE d.customer_id = $1
                  AND d.source_system = $2
                  AND d.valid_to IS NULL
                  AND d.deleted_at IS NULL
                ORDER BY d.customer_id, d.doc_id, d.version DESC
            ),
            attributed_docs AS (
                SELECT d.customer_id,
                       d.doc_id,
                       d.ingested_at,
                       COALESCE(
                           NULLIF(BTRIM(d.metadata->>'device_id'), ''),
                           NULLIF(BTRIM(parent.metadata->>'device_id'), '')
                       ) AS device_id
                FROM live_docs d
                LEFT JOIN live_docs parent
                  ON parent.doc_id = COALESCE(
                      NULLIF(BTRIM(d.parent_doc_id), ''),
                      NULLIF(BTRIM(d.metadata->>'parent_doc_id'), '')
                  )
            )
            SELECT d.device_id,
                   COUNT(DISTINCT d.doc_id) AS docs,
                   COUNT(c.chunk_id)        AS chunks,
                   MAX(d.ingested_at)       AS last_ingested_at
            FROM attributed_docs d
            LEFT JOIN chunks c
              ON c.customer_id = d.customer_id
             AND c.doc_id      = d.doc_id
             AND c.valid_to   IS NULL
            WHERE d.device_id IS NOT NULL
            GROUP BY d.device_id
            ORDER BY last_ingested_at DESC, device_id
            LIMIT 10
            """,
            customer_id,
            source,
        )

    return {
        "customer_id": customer_id,
        "source": source,
        "devices": [
            {
                "device_id": row["device_id"],
                "docs": row["docs"],
                "chunks": row["chunks"],
                "last_ingested_at": (
                    row["last_ingested_at"].isoformat()
                    if row["last_ingested_at"]
                    else None
                ),
            }
            for row in rows
        ],
    }
