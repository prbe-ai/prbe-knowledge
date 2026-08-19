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

The aggregate route is CACHED (30s, per tenant) and fans its four queries out
CONCURRENTLY. Both are there because it backs a header that used to take 8.2s
cold; see `_STATS_CACHE_TTL_S` and `_collect_ingestion_stats` for what each one
costs. `?refresh=true` bypasses the cache for the dashboard's Refresh button.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from engine.shared.constants import SourceSystem
from engine.shared.db import with_tenant
from kb.admin_routes import verify_internal_knowledge_key

router = APIRouter(prefix="/api/stats", tags=["stats"])

_DEVICE_PAIRED_SOURCES = frozenset(
    {SourceSystem.CLAUDE_CODE.value, SourceSystem.CODEX.value}
)

#: These counts change on ingestion, not per request, and the dashboard refetches
#: them on every mount of /knowledge. 30s is long enough that a page reload is
#: free and short enough that watching a backfill still feels live. Same shape as
#: engine/system_settings/store.py, including the double-check under the lock so
#: a thundering herd costs one query and not N.
_STATS_CACHE_TTL_S = 30.0

_DOC_SQL = """
    SELECT source_system,
           COUNT(*)          AS docs,
           MAX(ingested_at)  AS last_ingested_at
    FROM documents
    WHERE customer_id = $1 AND valid_to IS NULL AND deleted_at IS NULL
    GROUP BY source_system
"""

#: The join to `documents` is NOT redundant with a source_system lookup: a chunk
#: can be live while its document is superseded or soft-deleted (~3% of live
#: chunks, measured across every tenant on the research plane), and those must
#: not be counted. Migration 0111 indexes both sides so this is a hash join
#: between two index-only scans. `(customer_id, doc_id)` is unique among live
#: non-deleted documents, so COUNT(*) cannot be inflated by fan-out.
_CHUNK_SQL = """
    SELECT d.source_system, COUNT(*) AS chunks
    FROM chunks c
    JOIN documents d
      ON d.customer_id = c.customer_id
     AND d.doc_id      = c.doc_id
     AND d.valid_to   IS NULL
     AND d.deleted_at IS NULL
    WHERE c.customer_id = $1 AND c.valid_to IS NULL
    GROUP BY d.source_system
"""

_QUEUE_SQL = """
    SELECT source_system, status, COUNT(*) AS n
    FROM ingestion_queue
    WHERE customer_id = $1 AND status IN ('pending', 'processing', 'dlq')
    GROUP BY source_system, status
"""

_BACKFILL_SQL = """
    SELECT source_system, status, events_enqueued, last_error,
           started_at, last_progress_at, completed_at
    FROM backfill_state
    WHERE customer_id = $1
    ORDER BY source_system
"""


@dataclass(frozen=True)
class _CachedStats:
    payload: dict[str, Any]
    fetched_at: float


_stats_cache: dict[str, _CachedStats] = {}

#: A lock PER TENANT, not one for the cache.
#:
#: One global lock would make every cold collection serialise across tenants
#: while being held for the whole DB round trip -- and Starlette does not cancel
#: a handler when the client goes away, so research-os giving up at its 10s relay
#: timeout would NOT release it. One slow tenant (a plan regression, an
#: unvacuumed `chunks`, plain DB load) would hold the lock for up to the 300s
#: statement timeout and 502 every OTHER tenant's /knowledge header behind it.
#: Before any of this existed a slow tenant only ever slowed itself, and that
#: property is worth keeping.
_stats_cache_locks: dict[str, asyncio.Lock] = {}


def _lock_for(customer_id: str) -> asyncio.Lock:
    lock = _stats_cache_locks.get(customer_id)
    if lock is None:
        lock = _stats_cache_locks[customer_id] = asyncio.Lock()
    return lock


def reset_stats_cache() -> None:
    """Drop every cached tenant. For tests; nothing in production calls it."""
    _stats_cache.clear()
    _stats_cache_locks.clear()


def _cache_get(customer_id: str, now: float) -> dict[str, Any] | None:
    cached = _stats_cache.get(customer_id)
    if cached is not None and (now - cached.fetched_at) < _STATS_CACHE_TTL_S:
        return cached.payload
    return None


def _cache_put(customer_id: str, payload: dict[str, Any], now: float) -> None:
    # Prune on write rather than on a timer: the dict is keyed by tenant, and on
    # the managed plane that is unbounded. Anything past its TTL is dead weight
    # whether or not it is ever asked for again.
    for key, cached in list(_stats_cache.items()):
        if (now - cached.fetched_at) >= _STATS_CACHE_TTL_S:
            del _stats_cache[key]
    _stats_cache[customer_id] = _CachedStats(payload=payload, fetched_at=now)

    # Same reasoning for the lock table, which is otherwise a second unbounded
    # per-tenant dict. Only drop a lock that is neither held nor backing a live
    # entry: dropping a held one would hand the next caller a different lock and
    # quietly undo the single-flight.
    for key, lock in list(_stats_cache_locks.items()):
        if key not in _stats_cache and not lock.locked():
            del _stats_cache_locks[key]


async def _fetch_rows(sql: str, customer_id: str) -> list[Any]:
    """One query, one pooled connection, tenant GUC bound.

    A connection per query rather than one shared across all four: asyncpg
    serialises statements on a single connection, so `asyncio.gather` over one
    connection would be sequential with extra steps.
    """
    async with with_tenant(customer_id) as conn:
        return await conn.fetch(sql, customer_id)


def _require_customer(
    x_prbe_customer: str | None = Header(default=None, alias="X-Prbe-Customer"),
) -> str:
    if not x_prbe_customer:
        raise HTTPException(status_code=400, detail="missing X-Prbe-Customer")
    return x_prbe_customer


async def _collect_ingestion_stats(customer_id: str) -> dict[str, Any]:
    """Run the four aggregates concurrently and assemble the payload.

    Four connections, four transactions -- so the results are no longer the
    single consistent snapshot one shared transaction gave. That is a deliberate
    trade and it is safe here: these are display counters, they go stale the
    instant they are read, and the cache in front of this makes any
    within-request skew invisible. Do not copy the pattern anywhere a caller
    could act on the relationship BETWEEN the four.
    """
    doc_rows, chunk_rows, queue_rows, backfill_rows = await asyncio.gather(
        _fetch_rows(_DOC_SQL, customer_id),
        _fetch_rows(_CHUNK_SQL, customer_id),
        _fetch_rows(_QUEUE_SQL, customer_id),
        _fetch_rows(_BACKFILL_SQL, customer_id),
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


@router.get("/ingestion", dependencies=[Depends(verify_internal_knowledge_key)])
async def ingestion_stats(
    customer_id: str = Depends(_require_customer),
    refresh: bool = Query(
        default=False,
        description="Bypass the 30s cache. The dashboard's Refresh button sets it.",
    ),
) -> dict:
    now = time.monotonic()
    if not refresh:
        cached = _cache_get(customer_id, now)
        if cached is not None:
            return cached

    async with _lock_for(customer_id):
        # Double-check inside the lock: N concurrent first-loads of /knowledge
        # cost one set of queries, not N. A forced refresh still re-reads -- it
        # is an explicit user action, and the button rate-limits it.
        if not refresh:
            cached = _cache_get(customer_id, time.monotonic())
            if cached is not None:
                return cached
        payload = await _collect_ingestion_stats(customer_id)
        _cache_put(customer_id, payload, time.monotonic())
        return payload


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
