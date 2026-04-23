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
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from services.retrieval.acl import filter_by_acl
from services.retrieval.dedup import dedupe
from services.retrieval.fusion import fuse
from services.retrieval.retrievers.bm25 import bm25_search
from services.retrieval.retrievers.graph import graph_search
from services.retrieval.retrievers.vector import vector_search
from services.retrieval.router import route_query
from shared.config import get_settings
from shared.constants import SourceSystem
from shared.db import health_check, init_pool, with_tenant
from shared.logging import configure_logging, get_logger
from shared.models import QueryChunk, QueryRequest, QueryResponse

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


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")

    trace_id = req.trace_id or f"q-{int(datetime.now().timestamp()*1000)}"
    timing: dict[str, float] = {}
    sources = [s.value for s in req.sources] if req.sources else None

    t_router = time.perf_counter()
    routed = await route_query(req.customer_id, req.query)
    timing["router_ms"] = (time.perf_counter() - t_router) * 1000

    # Run retrievers in parallel. Each retriever runs the raw query; BM25 also
    # runs each router expansion (disjoint score space under RRF so it's fine).
    queries = [req.query, *routed.expansions]

    async def _vec_runner() -> list:
        return await vector_search(
            req.customer_id,
            req.query,
            top_k=req.top_k * 2,
            sources=sources,
            temporal=req.temporal,
        )

    async def _bm25_runner() -> list:
        hits_by_chunk: dict = {}
        for q in queries:
            for hit in await bm25_search(
                req.customer_id,
                q,
                top_k=req.top_k * 2,
                sources=sources,
                temporal=req.temporal,
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
            temporal=req.temporal,
        )

    t_retrieve = time.perf_counter()
    vec_hits, bm25_hits, graph_hits = await asyncio.gather(
        _vec_runner(), _bm25_runner(), _graph_runner()
    )
    timing["vector_ms"] = (time.perf_counter() - t_retrieve) * 1000

    t_fuse = time.perf_counter()
    fused = fuse(
        {"vector": vec_hits, "bm25": bm25_hits, "graph": graph_hits},
        top_k=req.top_k * 2,
    )
    timing["fusion_ms"] = (time.perf_counter() - t_fuse) * 1000

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
            score=h.score,
            rank=i + 1,
            retriever_scores=h.retriever_scores,
        )
        for i, h in enumerate(top)
    ]

    return QueryResponse(
        query=req.query,
        chunks=chunks,
        total_candidates=len(fused),
        router_hit_cache=routed.hit_cache,
        timing_ms=timing,
        trace_id=trace_id,
    )


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
