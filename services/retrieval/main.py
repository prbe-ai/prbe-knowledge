"""Retrieval service — full Tier 5 query pipeline.

    /query → router (Haiku + cache)
           → parallel: vector / BM25 / graph
           → RRF fusion (doc-level collapse)
           → cosine dedup
           → ACL filter (pass-through in Phase 0)
           → QueryResponse with per-stage timing
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from services.retrieval.acl import filter_by_acl
from services.retrieval.dedup import dedupe
from services.retrieval.fusion import fuse
from services.retrieval.retrievers.bm25 import bm25_search
from services.retrieval.retrievers.graph import graph_search
from services.retrieval.retrievers.vector import vector_search
from services.retrieval.router import route_query
from services.retrieval.synthesis import (
    SynthesisChunk,
    SynthesisError,
    synthesize,
)
from services.retrieval.temporal import resolve_temporal
from shared.config import get_settings
from shared.constants import DEFAULT_SYNTHESIS_MODEL, SourceSystem
from shared.db import health_check, init_pool, raw_conn, with_tenant
from shared.logging import configure_logging, get_logger
from shared.models import (
    AnswerRequest,
    AnswerResponse,
    QueryChunk,
    QueryRequest,
    QueryResponse,
    TemporalMode,
    TemporalSpec,
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


@app.get("/health")
async def health() -> JSONResponse:
    db_ok = await health_check()
    body = {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "time": datetime.now(UTC).isoformat(),
    }
    return JSONResponse(body, status_code=200 if db_ok else 503)


_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


class _AuthResult:
    __slots__ = ("auth_present", "customer_id")

    def __init__(self, customer_id: str | None, auth_present: bool) -> None:
        self.customer_id = customer_id
        self.auth_present = auth_present


async def _resolve_customer_from_bearer(token: str) -> str:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT customer_id FROM customers WHERE api_key_hash = $1",
            token_hash,
        )
    if row is None:
        raise HTTPException(
            status_code=401,
            detail="invalid api key",
            headers=_UNAUTHORIZED_HEADERS,
        )
    return row["customer_id"]


async def authenticate_query(request: Request) -> _AuthResult:
    """Derive customer_id from Authorization: Bearer <api_key>.

    Dev-only bypass: environment=local + missing header lets the handler
    fall back to a customer_id in the request body (for smoke tests and
    local curl). Production environments always require the header.
    """
    authorization = request.headers.get("authorization")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(
                status_code=401,
                detail="invalid authorization scheme",
                headers=_UNAUTHORIZED_HEADERS,
            )
        resolved = await _resolve_customer_from_bearer(token.strip())
        return _AuthResult(customer_id=resolved, auth_present=True)

    if get_settings().is_local:
        return _AuthResult(customer_id=None, auth_present=False)

    raise HTTPException(
        status_code=401,
        detail="missing bearer token",
        headers=_UNAUTHORIZED_HEADERS,
    )


@app.post("/retrieve", response_model=QueryResponse)
async def retrieve(
    req: QueryRequest,
    auth: _AuthResult = Depends(authenticate_query),
) -> QueryResponse:
    """Raw-chunks retrieval. Vector + BM25 + graph fusion, returns ranked
    chunks with metadata. No LLM synthesis — that's /query's job.

    Agents and tools wanting full fidelity (code generation, verification
    queries, iterative re-retrieval) call this. Linear-style enrichment
    or human-readable answers go to /query.
    """
    return await _run_retrieval(req, auth)


async def _run_retrieval(
    req: QueryRequest,
    auth: _AuthResult,
) -> QueryResponse:
    if auth.auth_present:
        # Header is authoritative; a mismatching body value is almost certainly
        # a caller bug (or a cross-tenant probe) — refuse rather than silently
        # shadowing the header.
        if req.customer_id and req.customer_id != auth.customer_id:
            raise HTTPException(
                status_code=400,
                detail="customer_id in body does not match authenticated tenant",
            )
        customer_id = auth.customer_id
    else:
        # Local dev bypass path.
        customer_id = req.customer_id
        if not customer_id:
            raise HTTPException(
                status_code=401,
                detail="missing bearer token",
                headers=_UNAUTHORIZED_HEADERS,
            )
    assert customer_id is not None  # for type-checker, guaranteed above
    req = req.model_copy(update={"customer_id": customer_id})
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")

    trace_id = req.trace_id or f"q-{int(datetime.now().timestamp()*1000)}"
    timing: dict[str, float] = {}
    sources = [s.value for s in req.sources] if req.sources else None

    t_router = time.perf_counter()
    routed = await route_query(req.customer_id, req.query)
    timing["router_ms"] = (time.perf_counter() - t_router) * 1000

    # Resolve temporal: caller's explicit (non-default) TemporalSpec wins.
    # Otherwise use Haiku-extracted symbolic temporal. Otherwise fall back
    # to the default LATEST. extraction_failed surfaces unresolvable
    # event anchors ("since the auth refactor") — agent decides what to do.
    now = datetime.now(UTC)
    inferred_spec, extraction_error = resolve_temporal(routed.temporal, now=now)
    caller_explicit = (
        req.temporal.mode != TemporalMode.LATEST
        or req.temporal.since is not None
        or req.temporal.until is not None
        or req.temporal.as_of is not None
    )
    if caller_explicit:
        spec = req.temporal
        temporal_source = "caller"
    elif inferred_spec is not None:
        spec = inferred_spec
        temporal_source = "inferred"
    elif extraction_error is not None:
        spec = TemporalSpec()  # default LATEST
        temporal_source = "extraction_failed"
    else:
        spec = TemporalSpec()
        temporal_source = "default"

    # When sort intent ("oldest", "newest", "first") fires, fetch a wider net
    # so the truly oldest/newest matches don't get dropped by relevance ranking
    # before sort runs.
    pool_multiplier = 5 if routed.sort else 2

    # Run retrievers in parallel. Each retriever runs the raw query; BM25 also
    # runs each router expansion (disjoint score space under RRF so it's fine).
    queries = [req.query, *routed.expansions]

    async def _vec_runner() -> list:
        return await vector_search(
            req.customer_id,
            req.query,
            top_k=req.top_k * pool_multiplier,
            sources=sources,
            temporal=spec,
        )

    async def _bm25_runner() -> list:
        hits_by_chunk: dict = {}
        for q in queries:
            for hit in await bm25_search(
                req.customer_id,
                q,
                top_k=req.top_k * pool_multiplier,
                sources=sources,
                temporal=spec,
            ):
                prior = hits_by_chunk.get(hit.chunk_id)
                if prior is None or hit.score > prior.score:
                    hits_by_chunk[hit.chunk_id] = hit
        return sorted(hits_by_chunk.values(), key=lambda h: h.score, reverse=True)

    async def _graph_runner() -> list:
        if not routed.entities:
            return []
        return await graph_search(
            req.customer_id,
            [(e.entity_type, e.canonical_id) for e in routed.entities],
            temporal=spec,
        )

    t_retrieve = time.perf_counter()
    vec_hits, bm25_hits, graph_hits = await asyncio.gather(
        _vec_runner(), _bm25_runner(), _graph_runner()
    )
    timing["vector_ms"] = (time.perf_counter() - t_retrieve) * 1000

    t_fuse = time.perf_counter()
    fused = fuse(
        {"vector": vec_hits, "bm25": bm25_hits, "graph": graph_hits},
        top_k=req.top_k * pool_multiplier,
        recency_half_life_days=req.recency_half_life_days,
        now=now,
        sort=routed.sort,
    )
    timing["fusion_ms"] = (time.perf_counter() - t_fuse) * 1000

    # Optional entity filter — kills vector-similarity false positives on
    # entity-anchored queries ("whats going on with klavis" otherwise matches
    # any "whats going on" Slack message). Only fires when the toggle is on
    # AND the router extracted at least one high-confidence entity.
    applied_entity_filter: dict[str, object] | None = None
    if req.entity_must_match:
        pre_count = len(fused)
        fused, applied_entity_filter = _apply_entity_filter(
            fused, routed.entities, threshold=req.entity_match_threshold
        )
        applied_entity_filter["candidates_before"] = pre_count
        applied_entity_filter["candidates_after"] = len(fused)

    # Dedup uses the vector hits' embeddings where available (the only retriever
    # that carries them). BM25/graph-only hits fall through the dedup filter.
    t_dedup = time.perf_counter()
    embeddings = await _embeddings_for_chunks(req.customer_id, [h.chunk_id for h in fused])
    deduped = dedupe(fused, embeddings)
    timing["dedup_ms"] = (time.perf_counter() - t_dedup) * 1000

    t_acl = time.perf_counter()
    filtered = await filter_by_acl(req.customer_id, req.requesting_user_id, deduped)
    timing["acl_ms"] = (time.perf_counter() - t_acl) * 1000

    top = filtered[: req.top_k]
    chunks = [
        QueryChunk(
            chunk_id=h.chunk_id,
            doc_id=h.doc_id,
            doc_version=h.doc_version,
            source_system=SourceSystem(h.source_system),
            source_url=h.source_url,
            title=h.title,
            content=h.content,
            created_at=h.created_at,
            updated_at=h.updated_at,
            score=h.score,
            rank=i + 1,
            retriever_scores=h.retriever_scores,
        )
        for i, h in enumerate(top)
    ]

    raw_phrase = (routed.temporal or {}).get("raw_phrase") if routed.temporal else None
    applied_temporal: dict[str, object] = {
        "mode": spec.mode.value,
        "source": temporal_source,
        "raw_phrase": raw_phrase,
        "error": extraction_error,
    }
    if spec.mode == TemporalMode.CHANGED_BETWEEN:
        applied_temporal["since"] = spec.since.isoformat() if spec.since else None
        applied_temporal["until"] = spec.until.isoformat() if spec.until else None
        applied_temporal["basis"] = spec.time_basis
    elif spec.mode == TemporalMode.AS_OF:
        applied_temporal["as_of"] = spec.as_of.isoformat() if spec.as_of else None
    # LATEST + ALL omit since/until/basis — they don't apply to those modes.

    applied_sort: dict[str, object] | None = None
    if routed.sort:
        applied_sort = {
            "field": routed.sort.get("field"),
            "direction": routed.sort.get("direction"),
            "trigger_phrase": routed.sort.get("trigger_phrase"),
            "source": "inferred",
        }

    extracted_entities = [
        {
            "entity_type": e.entity_type,
            "canonical_id": e.canonical_id,
            "display_name": e.display_name,
            "confidence": e.confidence,
        }
        for e in routed.entities
    ]

    return QueryResponse(
        query=req.query,
        chunks=chunks,
        total_candidates=len(fused),
        router_hit_cache=False,
        applied_temporal=applied_temporal,
        applied_sort=applied_sort,
        applied_entity_filter=applied_entity_filter,
        extracted_entities=extracted_entities,
        timing_ms=timing,
        trace_id=trace_id,
    )


@app.post("/query", response_model=AnswerResponse)
async def query(
    req: AnswerRequest,
    auth: _AuthResult = Depends(authenticate_query),
) -> AnswerResponse:
    """Retrieve + synthesize. Calls /retrieve internally, then runs an LLM
    over the resulting chunks to produce a cited answer.

    Pick the model via `model: "<provider>/<model>"`. Defaults to
    anthropic/claude-sonnet-4-6. Allowed models live in
    shared.constants.SYNTHESIS_MODELS — Anthropic, OpenAI, Google.
    """
    base_req = QueryRequest(**req.model_dump(exclude={"model", "max_tokens"}))
    rresp = await _run_retrieval(base_req, auth)

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
        result = await synthesize(
            req.query, syn_chunks, model=model, max_tokens=req.max_tokens
        )
    except SynthesisError as exc:
        log.warning("query.synthesis_failed", model=model, error=str(exc))
        raise HTTPException(status_code=502, detail=f"synthesis failed: {exc}") from exc
    timing = dict(rresp.timing_ms)
    timing["synthesis_ms"] = (time.perf_counter() - t_syn) * 1000

    return AnswerResponse(
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
        extracted_entities=rresp.extracted_entities,
        timing_ms=timing,
        trace_id=rresp.trace_id,
    )


def _apply_entity_filter(
    fused: list, entities: list, threshold: float
) -> tuple[list, dict[str, object]]:
    """Drop fused chunks whose content/title doesn't textually contain any
    extracted entity meeting the confidence threshold.

    Pure function on top of the fused list. Returns (filtered_hits, info).
    `info` reports which entity strings were used as needles + a `skipped`
    flag if no qualifying entities were available (the filter is a no-op
    in that case so callers can surface that to the user).
    """
    qualifying = [e for e in entities if e.confidence >= threshold]
    info: dict[str, object] = {"enabled": True, "threshold": threshold}
    if not qualifying:
        info["skipped"] = (
            f"no entities with confidence >= {threshold:.2f} extracted"
        )
        info["needles"] = []
        return fused, info

    needles: list[str] = []
    seen: set[str] = set()
    for e in qualifying:
        for token in (e.canonical_id, e.display_name):
            if not token:
                continue
            tok = token.lower().strip()
            if tok and tok not in seen:
                seen.add(tok)
                needles.append(tok)
    info["needles"] = needles

    matched: list = []
    for hit in fused:
        haystack = ((hit.content or "") + " " + (hit.title or "")).lower()
        if any(n in haystack for n in needles):
            matched.append(hit)
    return matched, info


async def _embeddings_for_chunks(
    customer_id: str, chunk_ids: list[str]
) -> dict[str, list[float]]:
    if not chunk_ids:
        return {}
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT chunk_id, embedding::text AS emb
            FROM chunks
            WHERE customer_id = $1 AND chunk_id = ANY($2::text[])
            """,
            customer_id,
            chunk_ids,
        )
    out: dict[str, list[float]] = {}
    for r in rows:
        raw = r["emb"].strip("[]")
        out[r["chunk_id"]] = [float(x) for x in raw.split(",")] if raw else []
    return out


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "services.retrieval.main:app",
        host="0.0.0.0",
        port=8081,
        reload=False,
    )
