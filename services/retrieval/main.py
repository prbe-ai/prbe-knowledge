"""Retrieval service — minimal /query.

Phase 0 Tier 3: vector-only, no router / BM25 / graph / RRF fusion / dedup.
Those land in Tier 5. Kept stand-alone so the smoke test (Tier 3 gate) can
exercise the ingestion → query round-trip.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from services.retrieval.retrievers.vector import vector_search
from shared.config import get_settings
from shared.constants import SourceSystem
from shared.db import health_check, init_pool
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

    t0 = time.perf_counter()
    hits = await vector_search(
        customer_id=req.customer_id,
        query_text=req.query,
        top_k=req.top_k,
        sources=[s.value for s in req.sources] if req.sources else None,
    )
    timing["vector_ms"] = (time.perf_counter() - t0) * 1000

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
            retriever_scores={"vector": h.score},
        )
        for i, h in enumerate(hits)
    ]

    return QueryResponse(
        query=req.query,
        chunks=chunks,
        total_candidates=len(hits),
        router_hit_cache=False,
        timing_ms=timing,
        trace_id=trace_id,
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "services.retrieval.main:app",
        host="0.0.0.0",
        port=8081,
        reload=False,
    )
