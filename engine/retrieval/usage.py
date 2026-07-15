"""usage_events: write path + read endpoints.

Three concerns live in this module:

  1. UsageEvent dataclass + write_usage_event() — used by the post-response
     middleware to persist one row per /retrieve, /query, /sources, or
     /source-view call.
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

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from engine.retrieval.auth import authenticate_query
from engine.shared.db import with_tenant
from engine.shared.logging import get_logger
from engine.shared.metrics import counter
from engine.shared.models import (
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
    # Outbox counters drained by the data-plane telemetry uploader
    # (occurred_at + uploaded_at + counters form the outbox shape — see
    # migration 0065_usage_events_outbox). Empty for now; a separate task
    # threads real LLM token counts in from the call sites. `uploaded_at`
    # is left NULL on insert (= "needs flushing") by omitting it.
    counters: dict[str, Any] = field(default_factory=dict)


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
    if endpoint.startswith("/sources") or endpoint.startswith("/source-view"):
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
        counters_json = json.dumps(event.counters or {})
        async with with_tenant(event.customer_id) as conn:
            await conn.execute(
                # uploaded_at is intentionally omitted -> stays NULL ("needs
                # flushing" for the data-plane telemetry uploader's drain
                # query). counters defaults to {} when no token counts were
                # threaded in.
                """
                INSERT INTO usage_events (
                    customer_id, occurred_at, caller_kind, caller_subject,
                    event_type, request_id, endpoint, summary, status,
                    error_class, latency_ms, result_count, metadata, counters
                ) VALUES (
                    $1, $2, $3, $4, $5, $6::uuid, $7, $8, $9,
                    $10, $11, $12, $13::jsonb, $14::jsonb
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
                counters_json,
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
# query_traces: full request/response payload log per retrieval call.
#
# Sister table to usage_events. Where usage_events stores thin metrics
# (latency, status, caller_kind), query_traces stores the parsed request
# and response so we can evaluate retrieval effectiveness over time.
#
# Same write contract as write_usage_event: post-response BackgroundTask,
# RLS-scoped via with_tenant(), swallows ALL exceptions including
# CancelledError (Python 3.11+ doesn't let except Exception catch it).
# ---------------------------------------------------------------------------

# Cap the serialized response payload at 256KB. Real responses are
# ~10-100KB; anything larger is a bug, a pathological top_k, or a 1MB
# Linear comment that the chunker shouldn't have produced. We still
# record the row so the audit trail stays complete — just substitute a
# small marker for the response and flip response_truncated=true.
RESPONSE_MAX_BYTES = 256 * 1024

# Pre-serialization gate: skip the model_dump entirely if the response
# carries an obvious-pathology number of chunks. Bounds CPU before we
# pay for full JSON serialization. Tuned against typical top_k=10-50
# with ~1-2KB chunks ≈ 100KB; 200 chunks comfortably exceeds the cap.
MAX_CHUNK_COUNT_BEFORE_TRUNCATE = 200

# Schema version stamped on every row. Bump when the request/response
# JSONB shape changes in a way that breaks dashboard SQL — old rows
# stay readable indefinitely, consumers filter by version.
#
# v2 (2026-05-21): AnswerResponse inherits RetrieveResponse, so /query
# response JSONB now carries 7 additional top-level keys (related_entities,
# related_entities_error, gatherer_notes, query_root_doc_id, aggregations,
# router_hit_cache, applied_min_confidence) and chunks now carry
# why_relevant + chunk-level matched_via. Pre-v2 rows lack these.
QUERY_TRACE_SCHEMA_VERSION = 2


@dataclass
class QueryTrace:
    """In-flight shape passed from middleware to write_query_trace."""

    customer_id: str
    request_id: str
    event_type: str
    request_payload: BaseModel | dict[str, Any] | None
    response_payload: BaseModel | dict[str, Any] | None
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    error_class: str | None = None
    error_message: str | None = None
    # Router Intelligence v1 columns (Task 6)
    grounding_bundle: dict[str, Any] | None = None
    router_raw: dict[str, Any] | None = None
    intents_count: int | None = None
    intent_dispatch: list[dict[str, Any]] | None = None
    cache_tokens: dict[str, Any] | None = None
    router_model: str | None = None
    failure_recovered: bool = False
    # Pointer to the per-turn R2 transcript (migration 0079). Nullable:
    # sampled-out rows + R2-write-failure rows still get the summary
    # row, just without the blob pointer. Lookup-by-request_id via
    # idx_query_traces_request_id is sufficient; no new index needed.
    trace_blob_key: str | None = None


def _payload_to_jsonable(payload: BaseModel | dict[str, Any] | None) -> dict[str, Any]:
    """Coerce a request or response payload to a JSON-serializable dict.

    Pydantic models go through model_dump(mode='json') so datetimes,
    enums, and other custom types serialize cleanly. Plain dicts pass
    through unchanged. None becomes empty dict (the column is NOT NULL).
    """
    if payload is None:
        return {}
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    return dict(payload)


def _truncated_marker(size_bytes: int, chunk_count: int | None) -> dict[str, Any]:
    """Stub object stored in `response` when the real payload exceeds the cap.

    Pairs with `response_truncated=true` on the row; consumers that want
    to differentiate stub-vs-real must check the boolean column, not the
    JSONB shape (a real response could legitimately contain a `_truncated`
    key — the column avoids that existential ambiguity)."""
    marker: dict[str, Any] = {"size_bytes": size_bytes}
    if chunk_count is not None:
        marker["chunk_count"] = chunk_count
    return marker


def _total_chunk_count(response: BaseModel | dict[str, Any] | None) -> int | None:
    """Sum chunks across all Document results in a polymorphic response.

    Returns None when the response has no `results` attribute (e.g.
    SourceResponse uses a flat `chunk_count` int). Entity results
    contribute zero -- they carry no body chunks. The doc-grouped
    structure means a single Document can contribute many chunks; we
    sum them so the truncation gate triggers on TOTAL chunks, not on
    distinct documents.
    """
    results = getattr(response, "results", None)
    if results is None:
        return None
    total = 0
    for r in results:
        chunks = getattr(r, "chunks", None)
        if chunks is not None:
            total += len(chunks)
    return total


def _build_response_payload(
    response: BaseModel | dict[str, Any] | None,
) -> tuple[dict[str, Any], int, bool]:
    """Return (payload_dict, size_bytes, truncated).

    Three-stage gate:
      1. Pre-check total chunk count across all Document results to
         short-circuit pathological responses before we pay for
         serialization.
      2. Serialize, measure size.
      3. If serialized > RESPONSE_MAX_BYTES, replace with truncation marker.

    The chunk-count pre-check sums `chunks` across `RetrieveResponse.results`
    / `AnswerResponse.results` (polymorphic Document/Entity). SourceResponse
    uses `chunk_count` instead and returns reassembled `content`, which is
    bounded by the chunker.
    """
    chunk_count = _total_chunk_count(response)
    if chunk_count is not None and chunk_count > MAX_CHUNK_COUNT_BEFORE_TRUNCATE:
        marker = _truncated_marker(size_bytes=0, chunk_count=chunk_count)
        # size_bytes=0 here means "not measured" — the gate fired before
        # serialization. Distinguishable from a real 0 only via the
        # response_truncated boolean.
        marker["size_bytes"] = len(json.dumps(marker).encode("utf-8"))
        return marker, marker["size_bytes"], True

    payload = _payload_to_jsonable(response)
    serialized = json.dumps(payload).encode("utf-8")
    size_bytes = len(serialized)
    if size_bytes > RESPONSE_MAX_BYTES:
        marker = _truncated_marker(size_bytes=size_bytes, chunk_count=chunk_count)
        return marker, size_bytes, True
    return payload, size_bytes, False


async def write_query_trace(trace: QueryTrace) -> None:
    """Persist one query_traces row. Never raises.

    Wrapped in `except (Exception, asyncio.CancelledError)`: this runs
    in a post-response BackgroundTask, and the user-visible request has
    already returned 200 (or its error) by the time we get here. If the
    server is shutting down or the client disconnected, asyncio cancels
    the task — `except Exception` alone would NOT catch that in Python
    3.11+ since CancelledError became BaseException-rooted. Catching
    both keeps the behavior identical: log a warning, drop the row,
    let the request stream out cleanly.

    On error responses (handler raised), `request_payload` is still
    populated by the handler if it stashed before raising; `response_payload`
    is None and we record `{error_class, error_message}` instead.
    """
    try:
        request_dict = _payload_to_jsonable(trace.request_payload)

        if trace.error_class is not None:
            response_dict: dict[str, Any] = {
                "error_class": trace.error_class,
                "error_message": (trace.error_message or "")[: SUMMARY_MAX_BYTES],
            }
            response_size_bytes = len(json.dumps(response_dict).encode("utf-8"))
            response_truncated = False
        else:
            response_dict, response_size_bytes, response_truncated = (
                _build_response_payload(trace.response_payload)
            )

        request_json = json.dumps(request_dict)
        response_json = json.dumps(response_dict)

        grounding_bundle_json = (
            json.dumps(trace.grounding_bundle) if trace.grounding_bundle is not None else None
        )
        router_raw_json = (
            json.dumps(trace.router_raw) if trace.router_raw is not None else None
        )
        intent_dispatch_json = (
            json.dumps(trace.intent_dispatch) if trace.intent_dispatch is not None else None
        )
        cache_tokens_json = (
            json.dumps(trace.cache_tokens) if trace.cache_tokens is not None else None
        )

        async with with_tenant(trace.customer_id) as conn:
            await conn.execute(
                """
                INSERT INTO query_traces (
                    customer_id, occurred_at, request_id, event_type,
                    schema_version, request, response,
                    response_size_bytes, response_truncated,
                    grounding_bundle, router_raw, intents_count,
                    intent_dispatch, cache_tokens, router_model, failure_recovered,
                    trace_blob_key
                ) VALUES (
                    $1, $2, $3::uuid, $4, $5, $6::jsonb, $7::jsonb, $8, $9,
                    $10::jsonb, $11::jsonb, $12,
                    $13::jsonb, $14::jsonb, $15, $16,
                    $17
                )
                """,
                trace.customer_id,
                trace.occurred_at,
                trace.request_id,
                trace.event_type,
                QUERY_TRACE_SCHEMA_VERSION,
                request_json,
                response_json,
                response_size_bytes,
                response_truncated,
                grounding_bundle_json,
                router_raw_json,
                trace.intents_count,
                intent_dispatch_json,
                cache_tokens_json,
                trace.router_model,
                trace.failure_recovered,
                trace.trace_blob_key,
            )
    except (Exception, asyncio.CancelledError) as exc:
        counter("query_traces.write_failed", 1, event_type=trace.event_type)
        log.warning(
            "query_traces.write_failed",
            customer_id=trace.customer_id,
            event_type=trace.event_type,
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
    "MAX_CHUNK_COUNT_BEFORE_TRUNCATE",
    "QUERY_TRACE_SCHEMA_VERSION",
    "RESPONSE_MAX_BYTES",
    "STATUS_ERROR",
    "STATUS_OK",
    "SUMMARY_MAX_BYTES",
    "QueryTrace",
    "UsageEvent",
    "event_type_for",
    "parse_window",
    "usage_router",
    "write_query_trace",
    "write_usage_event",
]
