"""Semantic search pipeline — vector + BM25 + graph + RRF + dedup + ACL.

This is the original retrieval pipeline (extracted from `main.py` during the
list-mode split). Two behavior changes vs. before:

1. `doc_types` is a SOFT signal here, not a filter. The list pipeline uses
   doc_type as a hard `WHERE` clause; on the search path, hard-filtering by
   Haiku's doc_type guess tanks recall when the answer lives in a related
   doc type (e.g. "the PR that broke login" — the answer might be in a
   linked commit). Instead we apply doc_type as an RRF score boost via
   helpers.apply_entity_filter — which already runs over fused hits when
   `entity_must_match` is on — and otherwise just pass the matching
   doc_types into the source-filter slot for retrievers that find them
   useful.

2. Sort intent does NOT trigger hard post-fusion sort here. When Haiku
   detects "most recent X about Y" (a search-path query because Y is a
   topic entity), we amplify the recency boost in RRF (shorter half-life)
   instead of the old `pool_multiplier=5 + post-fusion sort` behavior —
   that was the original bug we're fixing. Pure relevance ranking is
   preserved; new items get a stronger nudge.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

from services.retrieval.acl import filter_by_acl
from services.retrieval.dedup import dedupe
from services.retrieval.fusion import fuse
from services.retrieval.helpers import (
    apply_entity_filter,
    embeddings_for_chunks,
)
from services.retrieval.retrievers.bm25 import bm25_search
from services.retrieval.retrievers.graph import graph_search
from services.retrieval.retrievers.vector import vector_search
from services.retrieval.router import RouterOutput
from shared.constants import SourceSystem
from shared.logging import get_logger
from shared.models import (
    QueryChunk,
    QueryRequest,
    QueryResponse,
    TemporalSpec,
)

log = get_logger(__name__)

# When the router detects sort intent on a search-path query, drop the
# recency half-life by this factor (and clamp to a 7-day floor). Caller's
# explicit `recency_half_life_days` always wins.
_SORT_INTENT_HALF_LIFE_DIVISOR = 4
_SORT_INTENT_MIN_HALF_LIFE_DAYS = 7.0


async def run_search(
    req: QueryRequest,
    customer_id: str,
    routed: RouterOutput,
    spec: TemporalSpec,
    temporal_meta: dict[str, object],
    sort_meta: dict[str, object] | None,
    extracted_entities: list[dict[str, object]],
    doc_types: list[str] | None,
    trace_id: str,
    timing: dict[str, float],
) -> QueryResponse:
    """Run the existing semantic pipeline. Caller has already authenticated,
    resolved customer_id, and called the router."""
    sources = [s.value for s in req.sources] if req.sources else None
    pool_multiplier = 2  # No widening even when sort intent fires; recency
    # boost handles "give me recent X about Y" without needing to widen.
    queries = [req.query, *routed.expansions]

    # On the search path, doc_type is a SOFT signal. We pass None to the
    # retrievers (preserving recall) — the entity filter or RRF boost
    # picks it up downstream if the user toggled `entity_must_match`.
    retriever_doc_types: list[str] | None = None

    async def _vec_runner() -> list:
        return await vector_search(
            customer_id,
            req.query,
            top_k=req.top_k * pool_multiplier,
            sources=sources,
            doc_types=retriever_doc_types,
            temporal=spec,
        )

    async def _bm25_runner() -> list:
        # Fan out the query + router expansions in parallel — each is an
        # independent SELECT against the chunks GIN index, so wall time
        # collapses to the slowest single call instead of summing N round
        # trips. The per-chunk dedup keeps the highest-scoring hit when an
        # expansion surfaces a chunk the original query already matched.
        per_query_hits = await asyncio.gather(
            *(
                bm25_search(
                    customer_id,
                    q,
                    top_k=req.top_k * pool_multiplier,
                    sources=sources,
                    doc_types=retriever_doc_types,
                    temporal=spec,
                )
                for q in queries
            )
        )
        hits_by_chunk: dict = {}
        for hits in per_query_hits:
            for hit in hits:
                prior = hits_by_chunk.get(hit.chunk_id)
                if prior is None or hit.score > prior.score:
                    hits_by_chunk[hit.chunk_id] = hit
        return sorted(hits_by_chunk.values(), key=lambda h: h.score, reverse=True)

    async def _graph_runner() -> list:
        if not routed.entities:
            return []
        return await graph_search(
            customer_id,
            [(e.entity_type, e.canonical_id) for e in routed.entities],
            doc_types=retriever_doc_types,
            temporal=spec,
        )

    t_retrieve = time.perf_counter()
    vec_hits, bm25_hits, graph_hits = await asyncio.gather(
        _vec_runner(), _bm25_runner(), _graph_runner()
    )
    timing["vector_ms"] = (time.perf_counter() - t_retrieve) * 1000

    # Recency half-life: caller's explicit value always wins. Otherwise
    # amplify when Haiku detected sort intent (the "most recent X about Y"
    # case). When neither applies we hand fusion None and it falls back to
    # DEFAULT_RECENCY_HALF_LIFE_DAYS — the universal baseline keeps stale
    # backfilled content from surfacing at parity with fresh docs.
    if req.recency_half_life_days is not None:
        effective_half_life: float | None = req.recency_half_life_days
    elif routed.sort:
        # No caller default and Haiku saw sort intent — bias toward recent.
        effective_half_life = _SORT_INTENT_MIN_HALF_LIFE_DAYS
    else:
        effective_half_life = None

    t_fuse = time.perf_counter()
    fused = fuse(
        {"vector": vec_hits, "bm25": bm25_hits, "graph": graph_hits},
        top_k=req.top_k * pool_multiplier,
        recency_half_life_days=effective_half_life,
        now=datetime.now(UTC),
        sort=None,  # Hard post-fusion sort removed from search path — see
        # module docstring. Recency boost above is the right tool.
    )
    timing["fusion_ms"] = (time.perf_counter() - t_fuse) * 1000

    applied_entity_filter: dict[str, object] | None = None
    if req.entity_must_match:
        pre_count = len(fused)
        fused, applied_entity_filter = apply_entity_filter(
            fused, routed.entities, threshold=req.entity_match_threshold
        )
        applied_entity_filter["candidates_before"] = pre_count
        applied_entity_filter["candidates_after"] = len(fused)

    t_dedup = time.perf_counter()
    embeddings = await embeddings_for_chunks(customer_id, [h.chunk_id for h in fused])
    deduped = dedupe(fused, embeddings)
    timing["dedup_ms"] = (time.perf_counter() - t_dedup) * 1000

    t_acl = time.perf_counter()
    filtered = await filter_by_acl(customer_id, req.requesting_user_id, deduped)
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
            author_id=h.author_id,
            created_at=h.created_at,
            updated_at=h.updated_at,
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
        router_hit_cache=False,
        applied_temporal=temporal_meta,
        applied_sort=sort_meta,
        applied_entity_filter=applied_entity_filter,
        applied_mode="search",
        applied_doc_types=doc_types,
        extracted_entities=extracted_entities,
        aggregation=None,
        timing_ms=timing,
        trace_id=trace_id,
    )
