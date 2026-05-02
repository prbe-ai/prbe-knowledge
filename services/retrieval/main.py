"""Retrieval service — FastAPI app + endpoints.

The pipeline itself lives in:
    pipeline.py         dispatcher (mode=list vs mode=search)
    list_pipeline.py    deterministic SQL window/aggregate path
    search_pipeline.py  vector + BM25 + graph + RRF + dedup + ACL path
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

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from services.retrieval.auth import authenticate_query
from services.retrieval.middleware import UsageLoggingMiddleware
from services.retrieval.pipeline import (
    run_retrieval,
    run_router_phase,
    run_search_phase,
)
from services.retrieval.synthesis import (
    StreamDelta,
    StreamFinal,
    SynthesisChunk,
    SynthesisError,
    synthesize,
    synthesize_stream,
)
from services.retrieval.usage import usage_router
from shared.config import get_settings
from shared.constants import DEFAULT_SYNTHESIS_MODEL, SourceSystem
from shared.db import health_check, init_pool, with_tenant
from shared.logging import configure_logging, get_logger
from shared.models import (
    AnswerRequest,
    AnswerResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
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
    resp: QueryResponse | AnswerResponse,
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
        "chunks_count": len(resp.chunks),
        "aggregation_present": resp.aggregation is not None,
        "total_candidates": resp.total_candidates,
        "total_ms": total_ms,
        "stage_ms": stage_ms,
    }
    if extra:
        payload.update(extra)
    log.info("query.handled", extra=payload)


@app.post("/retrieve", response_model=QueryResponse)
async def retrieve(
    req: QueryRequest,
    request: Request,
    customer_id: str = Depends(authenticate_query),
) -> QueryResponse:
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
    resp = await run_retrieval(req, customer_id)
    request.state.result_count = len(resp.chunks)
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
    rresp = await run_retrieval(base_req, customer_id)
    request.state.result_count = len(rresp.chunks)

    model = req.model or DEFAULT_SYNTHESIS_MODEL
    syn_chunks = [
        SynthesisChunk(
            chunk_id=c.chunk_id,
            title=c.title,
            content=c.content,
            source_system=c.source_system.value,
            source_url=c.source_url,
            updated_at=c.updated_at.isoformat(),
        )
        for c in rresp.chunks
    ]

    t_syn = time.perf_counter()
    try:
        result = await synthesize(req.query, syn_chunks, model=model, max_tokens=req.max_tokens)
    except SynthesisError as exc:
        log.warning("query.synthesis_failed", model=model, error=str(exc))
        raise HTTPException(status_code=502, detail=f"synthesis failed: {exc}") from exc
    timing = dict(rresp.timing_ms)
    timing["synthesis_ms"] = (time.perf_counter() - t_syn) * 1000

    answer = AnswerResponse(
        query=req.query,
        answer=result.answer,
        citations=result.citations,
        insufficient_context=result.insufficient_context,
        model=result.model,
        chunks=rresp.chunks,
        total_candidates=rresp.total_candidates,
        applied_temporal=rresp.applied_temporal,
        applied_sort=rresp.applied_sort,
        applied_entity_filter=rresp.applied_entity_filter,
        applied_mode=rresp.applied_mode,
        applied_doc_types=rresp.applied_doc_types,
        extracted_entities=rresp.extracted_entities,
        aggregation=rresp.aggregation,
        timing_ms=timing,
        trace_id=rresp.trace_id,
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
        chunks            → retrieval done; chunks + applied_* fields ready
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
            phase = await run_router_phase(base_req, customer_id)
            yield _sse(
                "entities",
                {
                    "extracted_entities": phase.extracted_entities,
                    "applied_temporal": phase.temporal_meta,
                    "applied_sort": phase.sort_meta,
                    "applied_doc_types": phase.doc_types,
                    "applied_mode": phase.dispatch_mode,
                    "trace_id": phase.trace_id,
                },
            )

            yield _sse("step", {"step": "searching"})
            rresp = await run_search_phase(base_req, customer_id, phase)
            request.state.result_count = len(rresp.chunks)
            yield _sse(
                "chunks",
                {
                    "chunks": [c.model_dump(mode="json") for c in rresp.chunks],
                    "total_candidates": rresp.total_candidates,
                    "applied_entity_filter": rresp.applied_entity_filter,
                    "applied_mode": rresp.applied_mode,
                    "applied_doc_types": rresp.applied_doc_types,
                    "aggregation": rresp.aggregation,
                },
            )

            yield _sse("step", {"step": "synthesizing"})
            syn_chunks = [
                SynthesisChunk(
                    chunk_id=c.chunk_id,
                    title=c.title,
                    content=c.content,
                    source_system=c.source_system.value,
                    source_url=c.source_url,
                    updated_at=c.updated_at.isoformat(),
                )
                for c in rresp.chunks
            ]
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
            request.state.usage_response_payload = AnswerResponse(
                query=req.query,
                answer=final.answer,
                citations=final.citations,
                insufficient_context=final.insufficient_context,
                model=final.model,
                chunks=rresp.chunks,
                total_candidates=rresp.total_candidates,
                applied_temporal=rresp.applied_temporal,
                applied_sort=rresp.applied_sort,
                applied_entity_filter=rresp.applied_entity_filter,
                applied_mode=rresp.applied_mode,
                applied_doc_types=rresp.applied_doc_types,
                extracted_entities=rresp.extracted_entities,
                aggregation=rresp.aggregation,
                timing_ms=timing,
                trace_id=rresp.trace_id,
            )

            log.info(
                "query.handled",
                extra={
                    "trace_id": phase.trace_id,
                    "endpoint": "/query/stream",
                    "query": req.query,
                    "applied_mode": rresp.applied_mode,
                    "applied_doc_types": rresp.applied_doc_types,
                    "chunks_count": len(rresp.chunks),
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
    async with with_tenant(customer_id) as conn:
        if version is not None:
            doc = await conn.fetchrow(
                """
                SELECT doc_id, version, source_system, source_id, source_url,
                       title, body_size_bytes, author_id, metadata, entities,
                       created_at, updated_at, ingested_at, deleted_at
                FROM documents
                WHERE customer_id = $1 AND doc_id = $2 AND version = $3
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
              AND $3 BETWEEN first_seen_version AND last_seen_version
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
