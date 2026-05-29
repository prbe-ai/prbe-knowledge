"""Retrieval service — FastAPI app + endpoints.

Endpoints exposed:
    POST /retrieve         raw chunks pipeline (vector + BM25 + graph fusion)
    POST /query            /retrieve + LLM synthesis (cited answer)
    POST /query/stream     /query as SSE stream
    POST /graph/explore    knowledge-graph viz: default mode (top-N by degree)
                           or anchor mode (tiered BFS centered on a node)
    POST /graph/search     prefix typeahead for the /graph/explore anchor picker
    GET  /source-view/...  hydrated document/chunk view
    GET  /sources/...      raw document fetch
    GET  /health           liveness + DB ping
    /usage/...             read endpoints for usage_events (mounted via router)

The pipeline itself lives in:
    pipeline.py         dispatcher (mode=list vs mode=search)
    list_pipeline.py    deterministic SQL window/aggregate path
    search_pipeline.py  vector + BM25 + graph + RRF + dedup + ACL path
    graph_explore.py    /graph/explore + /graph/search SQL-only paths
    auth.py             auth resolution helpers
    helpers.py          shared utilities (entity filter, embedding fetch)
    retrievers/         retriever implementations

This module is intentionally thin — endpoints, lifespan, and synthesis
glue only.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from services.retrieval.auth import authenticate_query
from services.retrieval.graph_explore import (
    EXPLORE_CONFIDENCES,
    EXPLORE_EDGE_TYPES,
    ExploreFilters,
    anchor_exists,
    anchor_graph_query,
    default_graph_query,
    graph_search_query,
)
from services.retrieval.middleware import UsageLoggingMiddleware
from services.retrieval.pipeline import (
    run_retrieval,
    run_router_phase,
    run_search_phase,
)
from services.retrieval.synthesis import (
    StreamDelta,
    StreamFinal,
    SynthesisError,
    flatten_documents_for_synthesis,
    synthesize,
    synthesize_stream,
)
from services.retrieval.usage import usage_router
from shared.community import ensure_default_customer
from shared.config import get_settings
from shared.constants import (
    DEFAULT_SYNTHESIS_MODEL,
    GRAPH_SEARCH_DEFAULT_LIMIT,
    GRAPH_SEARCH_MAX_LIMIT,
    SourceSystem,
)
from shared.db import health_check, init_pool, with_tenant
from shared.logging import configure_logging, get_logger
from shared.models import (
    AnswerRequest,
    AnswerResponse,
    QueryRequest,
    RetrieveResponse,
    SourceResponse,
    SourceViewResponse,
    SourceViewSection,
)

log = get_logger(__name__)

_SOURCE_VIEW_MODES = frozenset({"preview", "search", "grep", "range", "chunk", "tail", "full"})
_SOURCE_VIEW_DEFAULT_LIMIT_LINES = 80
_SOURCE_VIEW_MAX_LIMIT_LINES = 100
_SOURCE_VIEW_DEFAULT_MAX_BYTES = 12_000
_SOURCE_VIEW_MAX_BYTES = 20_000
_SOURCE_VIEW_MAX_CONTEXT_LINES = 20
_SOURCE_VIEW_MAX_MATCHES = 50
# OOM-defense ceiling for mode="full". Real-world docs are <1MB; 100MB
# exists only to keep one pathological doc from killing the worker.
# Per-request peak memory is ~4-5x doc size through reassembly +
# serialization, so a 100MB doc reaches ~500MB peak.
_SOURCE_VIEW_MAX_FULL_BYTES = 100_000_000


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    await ensure_default_customer()  # no-op unless DEFAULT_CUSTOMER_ID set
    log.info("retrieval.boot", environment=settings.environment)
    yield


app = FastAPI(title="prbe-knowledge retrieval", lifespan=lifespan)
# UsageLoggingMiddleware records one row per /retrieve, /query, /sources
# call into the usage_events table from a post-response BackgroundTask.
# Skips /health and /usage/* by design — see middleware.py for rationale.
app.add_middleware(UsageLoggingMiddleware)


@app.get("/health")
async def health() -> JSONResponse:
    db_ok = await health_check()
    body = {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "time": datetime.now(UTC).isoformat(),
    }
    return JSONResponse(body, status_code=200 if db_ok else 503)


def _log_query_handled(
    *,
    endpoint: str,
    req_query: str,
    # AnswerResponse is a RetrieveResponse subclass, so the base type covers
    # both /retrieve and /query callers without an explicit union.
    resp: RetrieveResponse,
    total_ms: float,
    stage_ms: dict[str, float],
    extra: dict[str, object] | None = None,
) -> None:
    """Structured per-query log. Captures everything an operator needs to
    debug a misroute or empty result without re-running the query."""
    payload: dict[str, object] = {
        "trace_id": resp.trace_id,
        "endpoint": endpoint,
        "query": req_query,
        "applied_mode": resp.applied_mode,
        "applied_doc_types": resp.applied_doc_types,
        "applied_temporal_mode": (resp.applied_temporal or {}).get("mode"),
        "applied_sort_field": (resp.applied_sort or {}).get("field"),
        "extracted_entity_types": [e.get("entity_type") for e in resp.extracted_entities],
        "results_count": len(resp.results),
        "aggregation_present": resp.aggregation is not None,
        "total_candidates": resp.total_candidates,
        "total_ms": total_ms,
        "stage_ms": stage_ms,
    }
    if extra:
        payload.update(extra)
    log.info("query.handled", extra=payload)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    req: QueryRequest,
    request: Request,
    customer_id: str = Depends(authenticate_query),
) -> RetrieveResponse:
    """Raw-chunks retrieval. Branches on Haiku-emitted mode:
    list → SQL window/aggregate; search → vector + BM25 + graph fusion.

    Agents wanting full fidelity (code generation, verification queries,
    iterative re-retrieval) call this. Linear-style enrichment or
    human-readable answers go to /query.
    """
    # Stash for UsageLoggingMiddleware. customer_id scopes the usage row;
    # usage_summary is the FTS-searchable text; result_count is set after
    # retrieval runs so the middleware can record it.
    # usage_request_payload + usage_response_payload feed the query_traces
    # write — reference-only assignments, the model_dump happens later
    # inside the post-response BackgroundTask.
    request.state.customer_id = customer_id
    request.state.usage_summary = req.query
    request.state.usage_request_payload = req
    t_total = time.perf_counter()
    resp = await run_retrieval(req, customer_id, request=request)
    request.state.result_count = len(resp.results)
    request.state.usage_response_payload = resp
    total_ms = (time.perf_counter() - t_total) * 1000
    _log_query_handled(
        endpoint="/retrieve",
        req_query=req.query,
        resp=resp,
        total_ms=total_ms,
        stage_ms=resp.timing_ms,
    )
    return resp


@app.post("/query", response_model=AnswerResponse)
async def query(
    req: AnswerRequest,
    request: Request,
    customer_id: str = Depends(authenticate_query),
) -> AnswerResponse:
    """Retrieve + synthesize. Calls retrieval internally, then runs an LLM
    over the resulting chunks to produce a cited answer.

    Pick the model via `model: "<provider>/<model>"`. Defaults to
    anthropic/claude-sonnet-4-6. Allowed models live in
    shared.constants.SYNTHESIS_MODELS — Sonnet and Haiku only.
    """
    # Stash for UsageLoggingMiddleware (see /retrieve for shape).
    request.state.customer_id = customer_id
    request.state.usage_summary = req.query
    request.state.usage_request_payload = req
    t_total = time.perf_counter()
    base_req = QueryRequest(**req.model_dump(exclude={"model", "max_tokens"}))
    rresp = await run_retrieval(base_req, customer_id, request=request)
    request.state.result_count = len(rresp.results)

    model = req.model or DEFAULT_SYNTHESIS_MODEL
    # Flatten Document.chunks into a flat list the synthesizer cites by
    # 1-indexed position. Entity results have no content and are skipped.
    syn_chunks = flatten_documents_for_synthesis(rresp.results)

    t_syn = time.perf_counter()
    try:
        result = await synthesize(req.query, syn_chunks, model=model, max_tokens=req.max_tokens)
    except SynthesisError as exc:
        log.warning("query.synthesis_failed", model=model, error=str(exc))
        raise HTTPException(status_code=502, detail=f"synthesis failed: {exc}") from exc
    timing = dict(rresp.timing_ms)
    timing["synthesis_ms"] = (time.perf_counter() - t_syn) * 1000

    # AnswerResponse inherits every retrieval field from RetrieveResponse,
    # so we splat rresp once and add the four synthesis-specific fields.
    # Override timing_ms to carry synthesis_ms; the rest passes through.
    answer = AnswerResponse(
        **{**rresp.model_dump(), "timing_ms": timing},
        answer=result.answer,
        citations=result.citations,
        insufficient_context=result.insufficient_context,
        model=result.model,
    )
    request.state.usage_response_payload = answer
    total_ms = (time.perf_counter() - t_total) * 1000
    _log_query_handled(
        endpoint="/query",
        req_query=req.query,
        resp=answer,
        total_ms=total_ms,
        stage_ms=timing,
        extra={"model": result.model, "insufficient_context": answer.insufficient_context},
    )
    return answer


def _sse(event: str, data: dict[str, object]) -> bytes:
    """Format one Server-Sent Event frame.

    Each chunk is `event: <name>\\ndata: <json>\\n\\n`. JSON is single-line
    (no embedded newlines) so the SSE parser treats it as one `data` field.
    """
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode()


@app.post("/query/stream")
async def query_stream(
    req: AnswerRequest,
    request: Request,
    customer_id: str = Depends(authenticate_query),
) -> StreamingResponse:
    """Streaming /query: emits SSE events as each phase finishes.

    Event sequence (happy path):
        step:refining     → router (Haiku) is running
        entities          → router done; entities + temporal/sort/mode resolved
        step:searching    → vec/bm25/graph/list pipeline starts
        results           → retrieval done; polymorphic results +
                            applied_* fields ready. Replaces the old
                            `chunks` event (PR feat/polymorphic-search-results).
        step:synthesizing → answer LLM starts
        delta(*)          → 1+ text chunks of the answer
        done              → final {answer, citations, insufficient_context,
                            model, timing_ms, trace_id}

    On error, a single `error` event with {detail} replaces the remainder
    of the stream. The HTTP status is 200 either way (status comes through
    the SSE channel) — that's the convention SSE-aware clients expect.
    """
    request.state.customer_id = customer_id
    request.state.usage_summary = req.query
    # Stash request payload now; response payload is built and stashed at
    # the end of the SSE generator (after the 'done' event), so query_traces
    # captures both sides for streaming queries — same shape as /query.
    # Starlette runs response.background AFTER the generator exits, which
    # is when the BackgroundTask reads request.state.
    request.state.usage_request_payload = req
    t_total = time.perf_counter()
    base_req = QueryRequest(**req.model_dump(exclude={"model", "max_tokens"}))
    model = req.model or DEFAULT_SYNTHESIS_MODEL

    async def _gen() -> AsyncIterator[bytes]:
        try:
            yield _sse("step", {"step": "refining"})
            phase = await run_router_phase(base_req, customer_id, request=request)
            # Build union of entities across all intents. Each entity gets an
            # `intent_idx` so consumers can attribute it to its source intent.
            all_entities: list[dict] = []
            for idx, ri in enumerate(phase.resolved_intents):
                for e in ri.extracted_entities:
                    all_entities.append({**e, "intent_idx": idx})

            per_intent_meta = [
                {
                    "intent_idx": idx,
                    "mode": ri.dispatch_mode,
                    "doc_types": ri.doc_types,
                    "applied_temporal": ri.temporal_meta,
                    "applied_sort": ri.sort_meta,
                }
                for idx, ri in enumerate(phase.resolved_intents)
            ]

            # Back-compat keys from first intent (kept at top level for
            # consumers that depended on the old single-intent shape).
            _first = phase.resolved_intents[0] if phase.resolved_intents else None
            yield _sse(
                "entities",
                {
                    "extracted_entities": all_entities,
                    "intents_count": len(phase.resolved_intents),
                    "per_intent": per_intent_meta,
                    "trace_id": phase.trace_id,
                    # Back-compat: first-intent values at top level.
                    "applied_temporal": _first.temporal_meta if _first else None,
                    "applied_sort": _first.sort_meta if _first else None,
                    "applied_doc_types": _first.doc_types if _first else None,
                    "applied_mode": _first.dispatch_mode if _first else "search",
                },
            )

            yield _sse("step", {"step": "searching"})
            rresp = await run_search_phase(base_req, customer_id, phase, request=request)
            request.state.result_count = len(rresp.results)
            yield _sse(
                "results",
                {
                    # Pydantic v2 dumps discriminated unions cleanly with
                    # `mode='json'` -- `node_type` survives, datetimes
                    # serialize to ISO strings, and Entity vs Document
                    # variants are distinguishable on the wire.
                    "results": [r.model_dump(mode="json") for r in rresp.results],
                    "total_candidates": rresp.total_candidates,
                    "confidence_breakdown": rresp.confidence_breakdown,
                    "applied_entity_filter": rresp.applied_entity_filter,
                    "applied_mode": rresp.applied_mode,
                    "applied_doc_types": rresp.applied_doc_types,
                    "aggregation": rresp.aggregation,
                    "related_entities": (
                        [e.model_dump(mode="json") for e in rresp.related_entities]
                        if rresp.related_entities is not None
                        else None
                    ),
                    "related_entities_error": rresp.related_entities_error,
                },
            )

            yield _sse("step", {"step": "synthesizing"})
            # Flatten the polymorphic results into a flat synthesis-chunk
            # list. Entity results are skipped (no body content to cite).
            syn_chunks = flatten_documents_for_synthesis(rresp.results)
            t_syn = time.perf_counter()
            final: StreamFinal | None = None
            async for evt in synthesize_stream(
                req.query, syn_chunks, model=model, max_tokens=req.max_tokens
            ):
                if isinstance(evt, StreamDelta):
                    yield _sse("delta", {"text": evt.text})
                else:
                    final = evt

            timing = dict(rresp.timing_ms)
            timing["synthesis_ms"] = (time.perf_counter() - t_syn) * 1000
            timing["total_ms"] = (time.perf_counter() - t_total) * 1000

            assert final is not None  # synthesize_stream always yields a StreamFinal
            yield _sse(
                "done",
                {
                    "answer": final.answer,
                    "citations": final.citations,
                    "insufficient_context": final.insufficient_context,
                    "model": final.model,
                    "timing_ms": timing,
                    "trace_id": phase.trace_id,
                },
            )

            # Stash a synthetic AnswerResponse on request.state so query_traces
            # captures the streamed response in the same shape /query writes.
            # Mirrors lines 190-207 above. Without this, the trace would land
            # with response={} and consumers couldn't tell streaming rows apart
            # from a real /query with zero results. Closes PR #64's documented
            # known-limitation for usage_events as a bonus (result_count is
            # already set on line 286).
            # AnswerResponse inherits every retrieval field from
            # RetrieveResponse — splat rresp once, override timing_ms.
            # Wrapped in its own try so a stash failure (Pydantic re-validation
            # edge case, etc.) doesn't surface as an `error` SSE frame AFTER
            # the client already saw `done`. The trace is observability —
            # losing one is acceptable; misleading the client is not.
            try:
                request.state.usage_response_payload = AnswerResponse(
                    **{**rresp.model_dump(), "timing_ms": timing},
                    answer=final.answer,
                    citations=final.citations,
                    insufficient_context=final.insufficient_context,
                    model=final.model,
                )
            except Exception:
                log.exception(
                    "query.stream_trace_stash_failed",
                    extra={"trace_id": phase.trace_id, "query": req.query},
                )

            log.info(
                "query.handled",
                extra={
                    "trace_id": phase.trace_id,
                    "endpoint": "/query/stream",
                    "query": req.query,
                    "applied_mode": rresp.applied_mode,
                    "applied_doc_types": rresp.applied_doc_types,
                    "results_count": len(rresp.results),
                    "total_candidates": rresp.total_candidates,
                    "total_ms": timing["total_ms"],
                    "stage_ms": timing,
                    "model": final.model,
                    "insufficient_context": final.insufficient_context,
                },
            )
        except HTTPException as exc:
            yield _sse("error", {"detail": str(exc.detail), "status": exc.status_code})
        except SynthesisError as exc:
            yield _sse("error", {"detail": f"synthesis failed: {exc}", "status": 502})
        except Exception as exc:
            log.exception("query.stream_failed", extra={"query": req.query})
            yield _sse("error", {"detail": str(exc), "status": 500})

    # `text/event-stream` + `Cache-Control: no-cache` are the SSE-required
    # headers. `X-Accel-Buffering: no` disables nginx-style proxy buffering
    # so deltas reach the browser immediately rather than being batched.
    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# /graph/explore + /graph/search: knowledge-graph visualization endpoints.
#
# Powers the dashboard graph page. Default mode renders the top-N nodes by
# degree; anchor mode renders a tiered BFS centered on a single node.
# Both run SQL-only queries against graph_nodes / graph_edges (no LLM hop)
# inside with_tenant() so RLS enforces tenant isolation.
# ---------------------------------------------------------------------------


class GraphExploreFilters(BaseModel):
    """Filters applied to the /graph/explore query.

    Subset of EdgeType / confidence / SourceSystem enums. The validator
    rejects unknown enum values with a 422 (versus silently dropping
    them, which would mask a frontend typo).
    """

    edge_types: list[str] | None = None
    confidences: list[str] | None = None
    source_systems: list[str] | None = None
    since: datetime | None = None

    @model_validator(mode="after")
    def _validate_enum_values(self) -> GraphExploreFilters:
        # Each list must be a non-empty subset of the matching allowlist.
        # An empty list is treated as "no filter" -- coerce to None so
        # downstream SQL builds skip the WHERE clause entirely.
        if self.edge_types is not None:
            unknown = sorted(set(self.edge_types) - EXPLORE_EDGE_TYPES)
            if unknown:
                raise ValueError(f"unknown edge_types: {unknown}")
            if not self.edge_types:
                self.edge_types = None
        if self.confidences is not None:
            unknown = sorted(set(self.confidences) - EXPLORE_CONFIDENCES)
            if unknown:
                raise ValueError(f"unknown confidences: {unknown}")
            if not self.confidences:
                self.confidences = None
        if self.source_systems is not None:
            valid = {s.value for s in SourceSystem}
            unknown = sorted(set(self.source_systems) - valid)
            if unknown:
                raise ValueError(f"unknown source_systems: {unknown}")
            if not self.source_systems:
                self.source_systems = None
        return self

    def to_dataclass(self) -> ExploreFilters:
        return ExploreFilters(
            edge_types=self.edge_types,
            confidences=self.confidences,
            source_systems=self.source_systems,
            since=self.since,
        )


class GraphExploreRequest(BaseModel):
    """Body for POST /graph/explore.

    `mode='default'` ignores anchor_node_id; `mode='anchor'` requires it
    (validator enforces this -- a missing anchor in anchor mode is a 422,
    not a 404).
    """

    mode: Literal["default", "anchor"]
    anchor_node_id: str | None = None
    filters: GraphExploreFilters | None = None

    @model_validator(mode="after")
    def _validate_mode_anchor(self) -> GraphExploreRequest:
        if self.mode == "anchor" and not self.anchor_node_id:
            raise ValueError("anchor_node_id is required when mode='anchor'")
        return self


class GraphExploreNode(BaseModel):
    id: str
    label: str
    title: str | None = None
    source_system: str | None = None
    community_id: int | None = None
    degree: int


class GraphExploreEdge(BaseModel):
    """Vendor-neutral graph-viz field naming: source / target (not from_/to_).

    `why` is capped to GRAPH_EXPLORE_WHY_MAX_CHARS at serialization time
    in graph_explore._truncate_why -- the cap lives there so the
    truncation is enforced regardless of which call path produced the
    edge row.
    """

    source: str
    target: str
    edge_type: str
    confidence: str
    why: str | None = None


class GraphExploreResponse(BaseModel):
    nodes: list[GraphExploreNode]
    edges: list[GraphExploreEdge]
    truncated: bool
    total_nodes_available: int
    total_edges_available: int


class GraphSearchRequest(BaseModel):
    q: str
    limit: int = Field(default=GRAPH_SEARCH_DEFAULT_LIMIT, ge=1, le=GRAPH_SEARCH_MAX_LIMIT)


class GraphSearchMatch(BaseModel):
    id: str
    label: str
    title: str | None = None
    source_system: str | None = None
    degree: int


class GraphSearchResponse(BaseModel):
    matches: list[GraphSearchMatch]


async def _resolve_anchor_alias(*, customer_id: str, anchor_canonical_id: str) -> str:
    """Translate a user-typed canonical_id through entity_aliases.

    If the input is an alias of a merged cluster, returns the primary's
    canonical_id. Otherwise returns the input unchanged (the lookup
    returns 0 rows for both unmerged nodes and primaries).

    Label-less by design — the anchor endpoint doesn't carry label
    context, and ``anchor_exists`` matches across labels too. The
    LIMIT 1 guards against the unlikely case where the same canonical_id
    is an alias under two different labels.
    """
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            SELECT primary_canonical_id
            FROM entity_aliases
            WHERE customer_id = $1
              AND alias_canonical_id = $2
            LIMIT 1
            """,
            customer_id, anchor_canonical_id,
        )
    return row["primary_canonical_id"] if row else anchor_canonical_id


@app.post("/graph/explore", response_model=GraphExploreResponse)
async def graph_explore(
    req: GraphExploreRequest,
    request: Request,
    customer_id: str = Depends(authenticate_query),
) -> GraphExploreResponse:
    """Knowledge-graph visualization query.

    Two modes:
      default  - top GRAPH_EXPLORE_NODE_CAP nodes by graph_nodes.degree DESC,
                 plus 1-hop edges among the selected set.
      anchor   - tiered BFS centered on `anchor_node_id`. Hop 1 caps at
                 GRAPH_EXPLORE_HOP1_CAP, Hop 2 fills up to NODE_CAP.

    The endpoint is intentionally not logged via UsageLoggingMiddleware
    (no usage_summary set) -- graph viz is a UI-only path, not a search
    intent worth recording. If we later want graph-page audit trails,
    set request.state.usage_summary here.
    """
    request.state.customer_id = customer_id

    if req.mode == "anchor":
        # Cheap RLS-filtered existence check before the expensive BFS.
        # Translates a missing-anchor case to 404 (rather than returning
        # 200 with empty nodes/edges, which the frontend can't
        # distinguish from "exists but no edges").
        # `anchor_node_id` is guaranteed non-None by the request
        # validator above; assert for type narrowing.
        assert req.anchor_node_id is not None

        # Phase 2: translate alias canonical_id to the cluster's primary
        # before the existence check. Without this, anchors typed as an
        # alias (e.g. mahit@prbe.ai post-merge) return 404 because their
        # graph_nodes row was hard-deleted at merge time.
        anchor_canonical_id = await _resolve_anchor_alias(
            customer_id=customer_id,
            anchor_canonical_id=req.anchor_node_id,
        )

        if not await anchor_exists(
            customer_id=customer_id, anchor_canonical_id=anchor_canonical_id
        ):
            raise HTTPException(status_code=404, detail="anchor_node_id not found")
        result = await anchor_graph_query(
            customer_id=customer_id,
            anchor_canonical_id=anchor_canonical_id,
            filters=req.filters.to_dataclass() if req.filters else None,
        )
    else:
        result = await default_graph_query(
            customer_id=customer_id,
            filters=req.filters.to_dataclass() if req.filters else None,
        )

    truncated = (
        len(result.nodes) < result.total_nodes_available
        or len(result.edges) < result.total_edges_available
    )
    return GraphExploreResponse(
        nodes=[
            GraphExploreNode(
                id=n.id,
                label=n.label,
                title=n.title,
                source_system=n.source_system,
                community_id=n.community_id,
                degree=n.degree,
            )
            for n in result.nodes
        ],
        edges=[
            GraphExploreEdge(
                source=e.source,
                target=e.target,
                edge_type=e.edge_type,
                confidence=e.confidence,
                why=e.why,
            )
            for e in result.edges
        ],
        truncated=truncated,
        total_nodes_available=result.total_nodes_available,
        total_edges_available=result.total_edges_available,
    )


@app.post("/graph/search", response_model=GraphSearchResponse)
async def graph_search(
    req: GraphSearchRequest,
    request: Request,
    customer_id: str = Depends(authenticate_query),
) -> GraphSearchResponse:
    """Prefix typeahead for the /graph/explore anchor picker.

    Matches `q` (lowercased, prefix only -- no leading wildcard so the
    LOWER()-functional indexes are usable) against canonical_id,
    properties->>'name', and properties->>'title'. Ordered by degree
    DESC so high-signal entities surface first.
    """
    request.state.customer_id = customer_id
    hits = await graph_search_query(
        customer_id=customer_id, q=req.q, limit=req.limit
    )
    return GraphSearchResponse(
        matches=[
            GraphSearchMatch(
                id=h.id,
                label=h.label,
                title=h.title,
                source_system=h.source_system,
                degree=h.degree,
            )
            for h in hits
        ]
    )


def _cap_bytes(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


def _bounded_lines(
    lines: list[str],
    *,
    start_line: int,
    limit_lines: int,
    max_bytes: int,
) -> tuple[str, int | None, int | None, str | None, bool]:
    total_lines = len(lines)
    start = max(start_line, 1)
    if total_lines == 0 or start > total_lines:
        return "", None, None, None, False

    end = min(total_lines, start + limit_lines - 1)
    content, byte_truncated = _cap_bytes("\n".join(lines[start - 1 : end]), max_bytes)
    next_cursor = str(end + 1) if end < total_lines else None
    return content, start, end, next_cursor, byte_truncated or end < total_lines


def _chunk_line_offsets(chunk_rows: list[object]) -> dict[int, tuple[int, int]]:
    offsets: dict[int, tuple[int, int]] = {}
    current_line = 1
    for row in chunk_rows:
        line_count = len(str(row["content"]).splitlines()) or 1  # type: ignore[index]
        chunk_index = int(row["chunk_index"])  # type: ignore[index]
        offsets[chunk_index] = (current_line, current_line + line_count - 1)
        # Full source reassembly joins chunks with "\n\n", which creates one
        # blank separator line between adjacent chunk bodies.
        current_line += line_count + 1
    return offsets


def _parse_cursor(cursor: str | None) -> int | None:
    if cursor is None:
        return None
    try:
        value = int(cursor)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="cursor must be a line number") from exc
    if value < 1:
        raise HTTPException(status_code=400, detail="cursor must be a positive line number")
    return value


def _source_view_response(
    *,
    doc: object,
    chunk_rows: list[object],
    mode: str,
    content: str,
    sections: list[SourceViewSection],
    line_start: int | None,
    line_end: int | None,
    total_lines: int,
    next_cursor: str | None,
    truncated: bool,
    max_bytes: int,
    limit_lines: int,
) -> SourceViewResponse:
    return SourceViewResponse(
        doc_id=doc["doc_id"],  # type: ignore[index]
        doc_version=doc["version"],  # type: ignore[index]
        source_system=SourceSystem(doc["source_system"]),  # type: ignore[index]
        source_url=doc["source_url"],  # type: ignore[index]
        title=doc["title"],  # type: ignore[index]
        content=content,
        author_id=doc["author_id"],  # type: ignore[index]
        mode=mode,  # type: ignore[arg-type]
        sections=sections,
        line_start=line_start,
        line_end=line_end,
        total_lines=total_lines,
        next_cursor=next_cursor,
        truncated=truncated,
        chunk_count=len(chunk_rows),
        body_size_bytes=doc["body_size_bytes"] or 0,  # type: ignore[index]
        max_bytes=max_bytes,
        limit_lines=limit_lines,
    )


async def _load_source_doc_and_chunks(
    *,
    customer_id: str,
    doc_id: str,
    version: int | None,
    include_drafts: bool = False,
) -> tuple[object, list[object]]:
    """Direct doc-by-id fetch + its chunks.

    `include_drafts` defaults to False — drafts are invisible to API-key
    callers (Plan A Component 6). Reviewer-scoped BFF flips this to True
    after role-checking ``wiki_reviewer``.
    """
    # Default branch hides ``visibility='draft'`` rows. Sibling of the
    # existing ``valid_to IS NULL`` filter — matches the style used in
    # the retriever chokepoints.
    doc_visibility_filter = "" if include_drafts else "AND visibility = 'approved'"
    chunk_visibility_filter = "" if include_drafts else "AND visibility = 'approved'"
    async with with_tenant(customer_id) as conn:
        if version is not None:
            doc = await conn.fetchrow(
                f"""
                SELECT doc_id, version, source_system, source_id, source_url,
                       title, body_size_bytes, author_id, metadata, entities,
                       created_at, updated_at, ingested_at, deleted_at
                FROM documents
                WHERE customer_id = $1 AND doc_id = $2 AND version = $3
                  {doc_visibility_filter}
                """,
                customer_id,
                doc_id,
                version,
            )
        else:
            doc = await conn.fetchrow(
                f"""
                SELECT doc_id, version, source_system, source_id, source_url,
                       title, body_size_bytes, author_id, metadata, entities,
                       created_at, updated_at, ingested_at, deleted_at
                FROM documents
                WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL
                  {doc_visibility_filter}
                ORDER BY version DESC
                LIMIT 1
                """,
                customer_id,
                doc_id,
            )
        if doc is None:
            raise HTTPException(status_code=404, detail=f"document not found: {doc_id}")

        chunk_rows = await conn.fetch(
            f"""
            SELECT content, chunk_index
            FROM chunks
            WHERE customer_id = $1
              AND doc_id = $2
              AND valid_to IS NULL
              AND kind = 'content'
              AND $3 BETWEEN first_seen_version AND last_seen_version
              {chunk_visibility_filter}
            ORDER BY chunk_index
            """,
            customer_id,
            doc_id,
            doc["version"],
        )
    return doc, list(chunk_rows)


def _rank_source_chunks(chunk_rows: list[object], query: str) -> list[tuple[object, float]]:
    terms = [term.casefold() for term in query.split() if term.strip()]
    if not terms:
        raise HTTPException(status_code=400, detail="query is required for search mode")

    ranked: list[tuple[object, float]] = []
    for row in chunk_rows:
        content = str(row["content"]).casefold()  # type: ignore[index]
        score = sum(content.count(term) for term in terms)
        if score:
            ranked.append((row, float(score)))
    return sorted(ranked, key=lambda item: (-item[1], int(item[0]["chunk_index"])))  # type: ignore[index]


def _source_search_view(
    *,
    doc: object,
    chunk_rows: list[object],
    query: str,
    total_lines: int,
    limit_lines: int,
    max_bytes: int,
    max_matches: int,
) -> SourceViewResponse:
    offsets = _chunk_line_offsets(chunk_rows)
    ranked = _rank_source_chunks(chunk_rows, query)
    sections: list[SourceViewSection] = []
    parts: list[str] = []
    remaining_lines = limit_lines
    truncated = len(ranked) > max_matches

    for row, score in ranked[:max_matches]:
        if remaining_lines <= 0:
            truncated = True
            break
        chunk_index = int(row["chunk_index"])  # type: ignore[index]
        chunk_lines = str(row["content"]).splitlines()  # type: ignore[index]
        take = min(len(chunk_lines), remaining_lines)
        if take <= 0:
            continue
        chunk_start, _ = offsets[chunk_index]
        sections.append(
            SourceViewSection(
                chunk_index=chunk_index,
                line_start=chunk_start,
                line_end=chunk_start + take - 1,
                score=score,
            )
        )
        parts.append("\n".join(chunk_lines[:take]))
        remaining_lines -= take
        if take < len(chunk_lines):
            truncated = True
            break

    content, byte_truncated = _cap_bytes("\n\n".join(parts), max_bytes)
    truncated = truncated or byte_truncated
    return _source_view_response(
        doc=doc,
        chunk_rows=chunk_rows,
        mode="search",
        content=content,
        sections=sections,
        line_start=sections[0].line_start if sections else None,
        line_end=sections[-1].line_end if sections else None,
        total_lines=total_lines,
        next_cursor=None,
        truncated=truncated,
        max_bytes=max_bytes,
        limit_lines=limit_lines,
    )


def _source_grep_view(
    *,
    doc: object,
    chunk_rows: list[object],
    pattern: str,
    full_lines: list[str],
    limit_lines: int,
    max_bytes: int,
    context_lines: int,
    max_matches: int,
) -> SourceViewResponse:
    if not pattern:
        raise HTTPException(status_code=400, detail="pattern is required for grep mode")

    needle = pattern.casefold()
    windows: list[tuple[int, int]] = []
    total_lines = len(full_lines)
    for line_number, line in enumerate(full_lines, start=1):
        if needle not in line.casefold():
            continue
        start = max(1, line_number - context_lines)
        end = min(total_lines, line_number + context_lines)
        if windows and start <= windows[-1][1] + 1:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
        if len(windows) >= max_matches:
            break

    sections: list[SourceViewSection] = []
    parts: list[str] = []
    remaining_lines = limit_lines
    truncated = len(windows) >= max_matches
    for start, end in windows:
        if remaining_lines <= 0:
            truncated = True
            break
        take_end = min(end, start + remaining_lines - 1)
        sections.append(SourceViewSection(line_start=start, line_end=take_end))
        parts.append("\n".join(full_lines[start - 1 : take_end]))
        remaining_lines -= take_end - start + 1
        if take_end < end:
            truncated = True
            break

    content, byte_truncated = _cap_bytes("\n\n---\n\n".join(parts), max_bytes)
    truncated = truncated or byte_truncated
    return _source_view_response(
        doc=doc,
        chunk_rows=chunk_rows,
        mode="grep",
        content=content,
        sections=sections,
        line_start=sections[0].line_start if sections else None,
        line_end=sections[-1].line_end if sections else None,
        total_lines=total_lines,
        next_cursor=None,
        truncated=truncated,
        max_bytes=max_bytes,
        limit_lines=limit_lines,
    )


@app.get("/source-view/{doc_id:path}", response_model=SourceViewResponse)
async def get_source_view(
    doc_id: str,
    request: Request,
    customer_id: str = Depends(authenticate_query),
    version: int | None = Query(default=None),
    mode: str = Query(default="preview"),
    query: str | None = Query(default=None),
    pattern: str | None = Query(default=None),
    start_line: int | None = Query(default=None, ge=1),
    limit_lines: int = Query(default=_SOURCE_VIEW_DEFAULT_LIMIT_LINES, ge=1),
    chunk_index: int | None = Query(default=None, ge=0),
    context_lines: int = Query(default=3, ge=0),
    max_matches: int = Query(default=20, ge=1),
    cursor: str | None = Query(default=None),
    max_bytes: int = Query(default=_SOURCE_VIEW_DEFAULT_MAX_BYTES, ge=1),
) -> SourceViewResponse:
    """Bounded source content for MCP/agent drill-down.

    The full dashboard/debug endpoint remains `/sources/{doc_id}`. This
    endpoint is deliberately bounded so agents can inspect source context
    without pulling an entire Slack thread, PR, or generated session log
    into model context.
    """
    if mode not in _SOURCE_VIEW_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown mode {mode!r}; allowed: {sorted(_SOURCE_VIEW_MODES)}",
        )

    limit_lines = min(limit_lines, _SOURCE_VIEW_MAX_LIMIT_LINES)
    max_bytes = min(max_bytes, _SOURCE_VIEW_MAX_BYTES)
    context_lines = min(context_lines, _SOURCE_VIEW_MAX_CONTEXT_LINES)
    max_matches = min(max_matches, _SOURCE_VIEW_MAX_MATCHES)

    request.state.customer_id = customer_id
    request.state.usage_summary = doc_id
    request.state.usage_request_payload = {
        "doc_id": doc_id,
        "version": version,
        "mode": mode,
        "query": query,
        "pattern": pattern,
        "start_line": start_line,
        "limit_lines": limit_lines,
        "chunk_index": chunk_index,
        "context_lines": context_lines,
        "max_matches": max_matches,
        "cursor": cursor,
        "max_bytes": max_bytes,
    }

    doc, chunk_rows = await _load_source_doc_and_chunks(
        customer_id=customer_id,
        doc_id=doc_id,
        version=version,
    )
    full_content = "\n\n".join(str(c["content"]) for c in chunk_rows)  # type: ignore[index]
    full_lines = full_content.splitlines()
    total_lines = len(full_lines)
    sections: list[SourceViewSection] = []

    if mode == "search":
        if query is None:
            raise HTTPException(status_code=400, detail="query is required for search mode")
        source_response = _source_search_view(
            doc=doc,
            chunk_rows=chunk_rows,
            query=query,
            total_lines=total_lines,
            limit_lines=limit_lines,
            max_bytes=max_bytes,
            max_matches=max_matches,
        )
    elif mode == "grep":
        if pattern is None:
            raise HTTPException(status_code=400, detail="pattern is required for grep mode")
        source_response = _source_grep_view(
            doc=doc,
            chunk_rows=chunk_rows,
            pattern=pattern,
            full_lines=full_lines,
            limit_lines=limit_lines,
            max_bytes=max_bytes,
            context_lines=context_lines,
            max_matches=max_matches,
        )
    elif mode == "full":
        # Explicit opt-in for whole-document retrieval. limit_lines /
        # max_bytes / start_line params are ignored — the only ceiling
        # is the OOM-defense cap. cursor is honored so a doc that
        # exceeds the cap (rare) can still be paginated.
        start = _parse_cursor(cursor) or 1
        content, line_start, line_end, next_cursor, truncated = _bounded_lines(
            full_lines,
            start_line=start,
            limit_lines=max(total_lines, 1),
            max_bytes=_SOURCE_VIEW_MAX_FULL_BYTES,
        )
        if line_start is not None and line_end is not None:
            sections.append(SourceViewSection(line_start=line_start, line_end=line_end))
        source_response = _source_view_response(
            doc=doc,
            chunk_rows=chunk_rows,
            mode=mode,
            content=content,
            sections=sections,
            line_start=line_start,
            line_end=line_end,
            total_lines=total_lines,
            next_cursor=next_cursor,
            truncated=truncated,
            max_bytes=_SOURCE_VIEW_MAX_FULL_BYTES,
            limit_lines=total_lines,
        )
    elif mode == "chunk":
        if chunk_index is None:
            raise HTTPException(status_code=400, detail="chunk_index is required for chunk mode")
        chunk = next(
            (row for row in chunk_rows if int(row["chunk_index"]) == chunk_index),  # type: ignore[index]
            None,
        )
        if chunk is None:
            raise HTTPException(status_code=404, detail=f"chunk not found: {chunk_index}")
        offsets = _chunk_line_offsets(chunk_rows)
        chunk_start, _ = offsets[chunk_index]
        chunk_lines = str(chunk["content"]).splitlines()  # type: ignore[index]
        content, local_start, local_end, next_cursor, truncated = _bounded_lines(
            chunk_lines,
            start_line=1,
            limit_lines=limit_lines,
            max_bytes=max_bytes,
        )
        line_start = chunk_start + local_start - 1 if local_start is not None else None
        line_end = chunk_start + local_end - 1 if local_end is not None else None
        if line_start is not None and line_end is not None:
            sections.append(
                SourceViewSection(
                    chunk_index=chunk_index,
                    line_start=line_start,
                    line_end=line_end,
                )
            )
        source_response = _source_view_response(
            doc=doc,
            chunk_rows=chunk_rows,
            mode=mode,
            content=content,
            sections=sections,
            line_start=line_start,
            line_end=line_end,
            total_lines=total_lines,
            next_cursor=next_cursor,
            truncated=truncated,
            max_bytes=max_bytes,
            limit_lines=limit_lines,
        )
    else:
        if mode == "tail":
            start = max(1, total_lines - limit_lines + 1)
        elif mode == "range":
            start = _parse_cursor(cursor) or start_line or 1
        else:
            start = 1

        content, line_start, line_end, next_cursor, truncated = _bounded_lines(
            full_lines,
            start_line=start,
            limit_lines=limit_lines,
            max_bytes=max_bytes,
        )
        if mode == "tail" and start > 1:
            truncated = True
            next_cursor = None
        if line_start is not None and line_end is not None:
            sections.append(SourceViewSection(line_start=line_start, line_end=line_end))
        source_response = _source_view_response(
            doc=doc,
            chunk_rows=chunk_rows,
            mode=mode,
            content=content,
            sections=sections,
            line_start=line_start,
            line_end=line_end,
            total_lines=total_lines,
            next_cursor=next_cursor,
            truncated=truncated,
            max_bytes=max_bytes,
            limit_lines=limit_lines,
        )

    request.state.result_count = len(source_response.sections)
    request.state.usage_response_payload = source_response
    return source_response


@app.get("/sources/{doc_id:path}", response_model=SourceResponse)
async def get_source(
    doc_id: str,
    request: Request,
    customer_id: str = Depends(authenticate_query),
    version: int | None = Query(
        default=None,
        description="Specific document version. Defaults to the live version.",
    ),
) -> SourceResponse:
    """Full source content for one document, reassembled from its chunks.

    `doc_id` is the colon-delimited identifier emitted by each connector
    (e.g. `slack:T123:C456:1234567890.123456` or `github:owner/repo:pr:42`).
    Path uses `:path` so colons aren't URL-decoded out from under us.
    """
    # Stash for UsageLoggingMiddleware. The summary for /sources is the
    # doc_id itself — usable enough to FTS for "show me every time agent
    # X pulled github:foo/bar:pr:42".
    request.state.customer_id = customer_id
    request.state.usage_summary = doc_id
    # query_traces request payload: GET has no body, so the doc_id +
    # version are the entire request shape.
    request.state.usage_request_payload = {"doc_id": doc_id, "version": version}
    # /sources is an API-key surface; never bypass the approved-only
    # filter. Reviewer surfaces (Plan C BFF) route through a separate
    # endpoint that calls _load_source_doc_and_chunks(include_drafts=True).
    async with with_tenant(customer_id) as conn:
        if version is not None:
            doc = await conn.fetchrow(
                """
                SELECT doc_id, version, source_system, source_id, source_url,
                       title, body_size_bytes, author_id, metadata, entities,
                       created_at, updated_at, ingested_at, deleted_at
                FROM documents
                WHERE customer_id = $1 AND doc_id = $2 AND version = $3
                  AND visibility = 'approved'
                """,
                customer_id,
                doc_id,
                version,
            )
        else:
            doc = await conn.fetchrow(
                """
                SELECT doc_id, version, source_system, source_id, source_url,
                       title, body_size_bytes, author_id, metadata, entities,
                       created_at, updated_at, ingested_at, deleted_at
                FROM documents
                WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL
                  AND visibility = 'approved'
                ORDER BY version DESC
                LIMIT 1
                """,
                customer_id,
                doc_id,
            )
        if doc is None:
            raise HTTPException(status_code=404, detail=f"document not found: {doc_id}")

        chunk_rows = await conn.fetch(
            """
            SELECT content, chunk_index
            FROM chunks
            WHERE customer_id = $1
              AND doc_id = $2
              AND valid_to IS NULL
              AND kind = 'content'
              AND $3 BETWEEN first_seen_version AND last_seen_version
              AND visibility = 'approved'
            ORDER BY chunk_index
            """,
            customer_id,
            doc_id,
            doc["version"],
        )

    full_content = "\n\n".join(c["content"] for c in chunk_rows)
    metadata = doc["metadata"] if isinstance(doc["metadata"], dict) else {}
    entities = doc["entities"] if isinstance(doc["entities"], list) else []
    if isinstance(doc["metadata"], str):
        try:
            metadata = json.loads(doc["metadata"])
        except (TypeError, ValueError):
            metadata = {}
    if isinstance(doc["entities"], str):
        try:
            entities = json.loads(doc["entities"])
        except (TypeError, ValueError):
            entities = []

    request.state.result_count = len(chunk_rows)
    source_response = SourceResponse(
        doc_id=doc["doc_id"],
        doc_version=doc["version"],
        source_system=SourceSystem(doc["source_system"]),
        source_id=doc["source_id"],
        source_url=doc["source_url"],
        title=doc["title"],
        content=full_content,
        author_id=doc["author_id"],
        chunk_count=len(chunk_rows),
        body_size_bytes=doc["body_size_bytes"] or 0,
        metadata=metadata,
        entities=entities,
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        ingested_at=doc["ingested_at"],
        deleted_at=doc["deleted_at"],
    )
    request.state.usage_response_payload = source_response
    return source_response


# Mount the usage_events read endpoints (/usage/feed, /usage/stats,
# /usage/search). The router carries its own auth via authenticate_query
# and is excluded from UsageLoggingMiddleware so reads don't recursively
# log themselves.
app.include_router(usage_router)


__all__ = [
    "app",
    "authenticate_query",
]


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "services.retrieval.main:app",
        host="0.0.0.0",
        port=8081,
        reload=False,
    )
