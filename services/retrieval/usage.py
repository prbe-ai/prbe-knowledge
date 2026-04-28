"""usage_events: write path + read endpoints.

Three concerns live in this module:

  1. UsageEvent dataclass + write_usage_event() — used by the post-response
     middleware to persist one row per /retrieve, /query, /sources call.
  2. event_type_for() / parse_window() — small pure helpers shared between
     the middleware and the read endpoints.
  3. usage_router — three GET endpoints (/usage/feed, /usage/stats,
     /usage/search) consumed by the dashboard's live-feed page.

All reads run through `with_tenant(customer_id)` so the RLS policy on
usage_events stops cross-tenant leakage at the DB level even if a
handler bug forgets to scope its WHERE clause. All writes also use
with_tenant() — but the write path additionally swallows every exception
(DB down, RLS misfire, JSON encode error) and logs a warning rather than
500'ing the user-visible request that scheduled it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from services.retrieval.auth import authenticate_query
from shared.db import with_tenant
from shared.logging import get_logger
from shared.models import (
    UsageEventOut,
    UsageFeedResponse,
    UsageStatsResponse,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# event_type labels.
#
# Stable wire-level strings — the dashboard pivots on these and lane B/C/D
# consumes them. Don't rename without a coordinated change.
# ---------------------------------------------------------------------------
EVENT_TYPE_RETRIEVE = "knowledge.retrieve"
EVENT_TYPE_QUERY = "knowledge.query"
EVENT_TYPE_GET_SOURCE = "knowledge.get_source"
EVENT_TYPE_UNKNOWN = "knowledge.unknown"

# Caller-kind values stored in usage_events.caller_kind. The X-Caller-Kind
# header is free-form on the wire (we never reject for unknown values),
# but these are the canonical labels the dashboard groups on.
CALLER_KIND_UNKNOWN = "unknown"
KNOWN_CALLER_KINDS = frozenset({"mcp", "dashboard", "orchestrator", "external"})

# Status values.
STATUS_OK = "ok"
STATUS_ERROR = "error"

# Defense against summary abuse: cap stored text at ~10KB. Real queries
# are ~tens of bytes; anything larger is either a bug or a probe.
SUMMARY_MAX_BYTES = 10 * 1024


@dataclass
class UsageEvent:
    """In-flight shape passed from middleware to write_usage_event."""

    customer_id: str
    caller_kind: str
    event_type: str
    endpoint: str
    status: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    caller_subject: str | None = None
    request_id: str | None = None
    summary: str | None = None
    error_class: str | None = None
    latency_ms: int | None = None
    result_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def event_type_for(endpoint: str, body: BaseModel | None = None) -> str:
    """Map an endpoint path to the wire-stable event_type label.

    `body` is accepted for forward-compat (a future caller may want body
    type-based dispatch) but is unused today — the path alone determines
    the label.
    """
    if endpoint.startswith("/retrieve"):
        return EVENT_TYPE_RETRIEVE
    if endpoint.startswith("/query"):
        return EVENT_TYPE_QUERY
    if endpoint.startswith("/sources"):
        return EVENT_TYPE_GET_SOURCE
    return EVENT_TYPE_UNKNOWN


# Window literals accepted by every /usage/* endpoint. Values are
# documented on each endpoint's `window` query param.
_WINDOWS: dict[str, timedelta | None] = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
    "all": None,
}


def parse_window(window: str) -> datetime | None:
    """Return the lower-bound timestamp for a window string.

    None means "no lower bound" (the 'all' window). Raises HTTPException
    400 on unknown literals so the dashboard sees a structured error.
    """
    if window not in _WINDOWS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown window {window!r}; allowed: {sorted(_WINDOWS)}",
        )
    delta = _WINDOWS[window]
    if delta is None:
        return None
    return datetime.now(UTC) - delta


def _truncate_summary(summary: str | None) -> str | None:
    """Cap summary at SUMMARY_MAX_BYTES bytes. Defends against a caller
    that POSTs a megabyte-long query — the row is for human eyeballs +
    FTS, neither of which benefits from unbounded text."""
    if summary is None:
        return None
    encoded = summary.encode("utf-8")
    if len(encoded) <= SUMMARY_MAX_BYTES:
        return summary
    # Truncate on a byte boundary, then decode tolerantly so a chopped
    # multi-byte sequence at the cut doesn't raise.
    return encoded[:SUMMARY_MAX_BYTES].decode("utf-8", errors="ignore")


async def write_usage_event(event: UsageEvent) -> None:
    """Persist one usage_events row. Never raises.

    Wrapped in a broad try/except: this runs in a post-response background
    task, and the user-visible request has already returned 200 by the
    time we get here. Anything that goes wrong (DB pool exhausted, RLS
    misfire, JSON encode error) gets logged at WARNING and dropped.
    """
    try:
        summary = _truncate_summary(event.summary)
        metadata_json = json.dumps(event.metadata or {})
        async with with_tenant(event.customer_id) as conn:
            await conn.execute(
                """
                INSERT INTO usage_events (
                    customer_id, occurred_at, caller_kind, caller_subject,
                    event_type, request_id, endpoint, summary, status,
                    error_class, latency_ms, result_count, metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6::uuid, $7, $8, $9,
                    $10, $11, $12, $13::jsonb
                )
                """,
                event.customer_id,
                event.occurred_at,
                event.caller_kind,
                event.caller_subject,
                event.event_type,
                event.request_id,
                event.endpoint,
                summary,
                event.status,
                event.error_class,
                event.latency_ms,
                event.result_count,
                metadata_json,
            )
    except Exception as exc:
        log.warning(
            "usage_events.write_failed",
            customer_id=event.customer_id,
            endpoint=event.endpoint,
            error=str(exc),
            error_class=type(exc).__name__,
        )


# ---------------------------------------------------------------------------
# Read endpoints — auth-gated via authenticate_query, RLS-isolated by
# with_tenant. Mounted on the FastAPI app as `usage_router`.
# ---------------------------------------------------------------------------

usage_router = APIRouter()


def _row_to_event_out(row: Any) -> UsageEventOut:
    """Map an asyncpg Record to the public Pydantic model.

    JSONB metadata can come back as a Python dict (asyncpg's JSON codec)
    OR as a JSON string (when no codec is registered) — handle both so
    we don't depend on pool-startup codec configuration.
    """
    metadata = row["metadata"]
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (TypeError, ValueError):
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return UsageEventOut(
        event_id=str(row["event_id"]),
        occurred_at=row["occurred_at"],
        caller_kind=row["caller_kind"],
        caller_subject=row["caller_subject"],
        event_type=row["event_type"],
        request_id=str(row["request_id"]) if row["request_id"] is not None else None,
        endpoint=row["endpoint"],
        summary=row["summary"],
        status=row["status"],
        error_class=row["error_class"],
        latency_ms=row["latency_ms"],
        result_count=row["result_count"],
        metadata=metadata,
    )


@usage_router.get("/usage/feed", response_model=UsageFeedResponse)
async def usage_feed(
    window: str = Query(default="24h", description="One of 24h, 7d, 30d, 90d, all"),
    caller_kind: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=100),
    customer_id: str = Depends(authenticate_query),
) -> UsageFeedResponse:
    """Top-N most recent events for the authenticated tenant within a window."""
    since = parse_window(window)

    # Build the SQL with optional predicates appended. asyncpg uses $N
    # positional placeholders, so we track an index as we go.
    sql_parts = ["SELECT * FROM usage_events WHERE customer_id = $1"]
    params: list[Any] = [customer_id]

    if since is not None:
        params.append(since)
        sql_parts.append(f"AND occurred_at >= ${len(params)}")
    if caller_kind is not None:
        params.append(caller_kind)
        sql_parts.append(f"AND caller_kind = ${len(params)}")
    if event_type is not None:
        params.append(event_type)
        sql_parts.append(f"AND event_type = ${len(params)}")

    params.append(limit)
    sql_parts.append(f"ORDER BY occurred_at DESC LIMIT ${len(params)}")
    sql = "\n".join(sql_parts)

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(sql, *params)

    events = [_row_to_event_out(r) for r in rows]
    return UsageFeedResponse(events=events, window=window, count=len(events))


@usage_router.get("/usage/stats", response_model=UsageStatsResponse)
async def usage_stats(
    window: str = Query(default="24h", description="One of 24h, 7d, 30d, 90d, all"),
    customer_id: str = Depends(authenticate_query),
) -> UsageStatsResponse:
    """Aggregate counts + latency percentiles over the window.

    Single SQL pass: percentile_cont over status='ok' rows, COUNTs by
    group, plus a status='error' count. Returns zeros instead of None
    when the window contains no rows.
    """
    since = parse_window(window)

    # Single set of params reused across all three aggregate queries.
    # `since IS NULL OR occurred_at >= since` collapses to "no lower bound"
    # at plan time when since is NULL, so we don't need conditional SQL.
    params: list[Any] = [customer_id, since]

    overall_sql = """
        SELECT
            COUNT(*)::int AS total,
            COUNT(*) FILTER (WHERE status = 'error')::int AS error_count_,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms)
                FILTER (WHERE status = 'ok' AND latency_ms IS NOT NULL) AS p50,
            percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
                FILTER (WHERE status = 'ok' AND latency_ms IS NOT NULL) AS p95
        FROM usage_events
        WHERE customer_id = $1
          AND ($2::timestamptz IS NULL OR occurred_at >= $2)
    """
    by_caller_sql = """
        SELECT caller_kind, COUNT(*)::int AS n
        FROM usage_events
        WHERE customer_id = $1
          AND ($2::timestamptz IS NULL OR occurred_at >= $2)
        GROUP BY caller_kind
    """
    by_type_sql = """
        SELECT event_type, COUNT(*)::int AS n
        FROM usage_events
        WHERE customer_id = $1
          AND ($2::timestamptz IS NULL OR occurred_at >= $2)
        GROUP BY event_type
    """

    async with with_tenant(customer_id) as conn:
        overall = await conn.fetchrow(overall_sql, *params)
        caller_rows = await conn.fetch(by_caller_sql, *params)
        type_rows = await conn.fetch(by_type_sql, *params)

    if overall is None:
        # Defensive — fetchrow on an aggregate query always returns a row.
        return UsageStatsResponse(
            total=0,
            by_caller_kind={},
            by_event_type={},
            latency_p50_ms=None,
            latency_p95_ms=None,
            error_count=0,
            window=window,
        )

    p50 = overall["p50"]
    p95 = overall["p95"]
    return UsageStatsResponse(
        total=int(overall["total"] or 0),
        by_caller_kind={r["caller_kind"]: int(r["n"]) for r in caller_rows},
        by_event_type={r["event_type"]: int(r["n"]) for r in type_rows},
        latency_p50_ms=int(p50) if p50 is not None else None,
        latency_p95_ms=int(p95) if p95 is not None else None,
        error_count=int(overall["error_count_"] or 0),
        window=window,
    )


@usage_router.get("/usage/search", response_model=UsageFeedResponse)
async def usage_search(
    q: str = Query(..., min_length=1, description="Free-text query over summary"),
    since: datetime | None = Query(default=None),
    until: datetime | None = Query(default=None),
    caller_kind: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    customer_id: str = Depends(authenticate_query),
) -> UsageFeedResponse:
    """FTS over usage_events.summary.

    Uses plainto_tsquery (NOT to_tsquery) so caller-supplied text never
    parses as tsquery operators — '&', '|', '!', '<->', adjacent parens,
    etc. are treated as literal characters. plainto_tsquery is the right
    primitive for "user typed something into a search box."
    """
    sql_parts = [
        "SELECT * FROM usage_events",
        "WHERE customer_id = $1",
        "AND to_tsvector('simple', coalesce(summary, '')) @@ plainto_tsquery('simple', $2)",
    ]
    params: list[Any] = [customer_id, q]

    if since is not None:
        params.append(since)
        sql_parts.append(f"AND occurred_at >= ${len(params)}")
    if until is not None:
        params.append(until)
        sql_parts.append(f"AND occurred_at <= ${len(params)}")
    if caller_kind is not None:
        params.append(caller_kind)
        sql_parts.append(f"AND caller_kind = ${len(params)}")
    if event_type is not None:
        params.append(event_type)
        sql_parts.append(f"AND event_type = ${len(params)}")

    params.append(limit)
    sql_parts.append(f"ORDER BY occurred_at DESC LIMIT ${len(params)}")
    sql = "\n".join(sql_parts)

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(sql, *params)

    events = [_row_to_event_out(r) for r in rows]
    # `window` is informational on this shape; search uses since/until so
    # we report a stable label rather than synthesizing one.
    return UsageFeedResponse(events=events, window="search", count=len(events))


__all__ = [
    "CALLER_KIND_UNKNOWN",
    "EVENT_TYPE_GET_SOURCE",
    "EVENT_TYPE_QUERY",
    "EVENT_TYPE_RETRIEVE",
    "EVENT_TYPE_UNKNOWN",
    "KNOWN_CALLER_KINDS",
    "STATUS_ERROR",
    "STATUS_OK",
    "SUMMARY_MAX_BYTES",
    "UsageEvent",
    "event_type_for",
    "parse_window",
    "usage_router",
    "write_usage_event",
]
