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
from services.retrieval.retrievers.bm25 import BM25Hit, bm25_search
from services.retrieval.retrievers.graph import (
    CODE_GRAPH_LABELS,
    GraphHit,
    graph_search,
)
from services.retrieval.retrievers.related_entities import (
    build_exclude_node_keys,
    walk_result_doc_neighbors,
)
from services.retrieval.retrievers.vector import vector_search
from services.retrieval.router import RouterOutput
from services.retrieval.temporal import build_predicate
from shared.constants import SourceSystem
from shared.db import with_tenant
from shared.logging import get_logger
from shared.models import (
    QueryBundle,
    QueryChunk,
    QueryRequest,
    QueryResponse,
    RelatedEntity,
    TemporalSpec,
    normalize_author_id,
)

log = get_logger(__name__)

# When the router detects sort intent on a search-path query, drop the
# recency half-life by this factor (and clamp to a 7-day floor). Caller's
# explicit `recency_half_life_days` always wins.
_SORT_INTENT_HALF_LIFE_DIVISOR = 4
_SORT_INTENT_MIN_HALF_LIFE_DAYS = 7.0
_CODING_AGENT_SOURCES = {
    SourceSystem.CLAUDE_CODE.value,
    SourceSystem.CODEX.value,
}


async def _content_fallbacks_for_metadata_only_agent_hits(
    customer_id: str,
    ranked_lists: dict[str, list],
    spec: TemporalSpec,
) -> list[BM25Hit]:
    """Return displayable content chunks for coding-agent metadata-only hits.

    Metadata chunks carry searchable titles, names, emails, hostnames, and
    source URLs. Fusion must not return that synthetic text directly, but for
    coding-agent sessions the metadata is often the only place a user/person
    name appears. When a Codex/Claude Code doc matches only via metadata, fetch
    the first real content chunk so fusion can return the transcript with the
    metadata score.
    """
    metadata_doc_ids: set[str] = set()
    content_doc_ids: set[str] = set()

    for hits in ranked_lists.values():
        for hit in hits:
            kind = getattr(hit, "kind", "content")
            if kind == "metadata" and hit.source_system in _CODING_AGENT_SOURCES:
                metadata_doc_ids.add(hit.doc_id)
            elif kind != "metadata":
                content_doc_ids.add(hit.doc_id)

    doc_ids = sorted(metadata_doc_ids - content_doc_ids)
    if not doc_ids:
        return []

    async with with_tenant(customer_id) as conn:
        params: list = [customer_id, doc_ids]
        pred = build_predicate(
            spec, doc_alias="d", chunk_alias="c", next_param_index=len(params) + 1
        )
        params.extend(pred.params)
        rows = await conn.fetch(
            f"""
            SELECT DISTINCT ON (c.doc_id)
                   c.chunk_id,
                   c.doc_id,
                   d.version AS doc_version,
                   d.source_system,
                   d.source_url,
                   d.title,
                   d.author_id,
                   c.content,
                   d.created_at,
                   d.updated_at
            FROM chunks c
            JOIN documents d
              ON c.doc_id = d.doc_id
             AND d.customer_id = c.customer_id
             AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
            WHERE c.customer_id = $1
              AND c.doc_id = ANY($2::text[])
              AND COALESCE(c.kind, 'content') = 'content'
              {pred.chunk_sql}
              {pred.doc_sql}
            ORDER BY c.doc_id, c.chunk_index ASC
            """,
            *params,
        )

    return [
        BM25Hit(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            doc_version=r["doc_version"],
            source_system=r["source_system"],
            source_url=r["source_url"],
            title=r["title"],
            content=r["content"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            score=0.0,
            author_id=normalize_author_id(r["author_id"]),
            kind="content_fallback",
        )
        for r in rows
    ]


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
            min_confidence=req.min_confidence,
        )

    t_retrieve = time.perf_counter()
    vec_hits, bm25_hits, graph_hits = await asyncio.gather(
        _vec_runner(), _bm25_runner(), _graph_runner()
    )
    ranked_lists = {"vector": vec_hits, "bm25": bm25_hits, "graph": graph_hits}
    metadata_content_fallbacks = await _content_fallbacks_for_metadata_only_agent_hits(
        customer_id, ranked_lists, spec
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
        {
            **ranked_lists,
            "metadata_content_fallback": metadata_content_fallbacks,
        },
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

    related: list[RelatedEntity] | None = None
    related_error: str | None = None
    if req.top_k_related > 0:
        # Fuzzy exclusion (codex-P2): gate routed entities by
        # entity_match_threshold so low-confidence router misfires don't
        # suppress real entities, and emit normalized variants
        # (lowered + namespace-stripped, canonical_id + display_name) so
        # `Service:prbe-backend` (router) excludes `Service:prbe-ai/prbe-backend`
        # (graph) without needing exact canonical_id alignment.
        exclude_keys = build_exclude_node_keys(
            routed.entities,
            entity_match_threshold=req.entity_match_threshold,
        )
        # Dedupe doc_id, keep best (lowest) rank per doc -- multiple chunks
        # per doc otherwise inflate doc_rank inputs to the SQL.
        best_rank: dict[str, int] = {}
        for i, h in enumerate(top, start=1):
            best_rank.setdefault(h.doc_id, i)
        ranked_docs = sorted(best_rank.items(), key=lambda kv: kv[1])
        t_related = time.perf_counter()
        try:
            related = await walk_result_doc_neighbors(
                customer_id,
                ranked_result_docs=ranked_docs,
                exclude_node_keys=exclude_keys,
                min_confidence=req.min_confidence,
                top_n=req.top_k_related,
            )
        except Exception as exc:
            # Enrichment field -- never break host search response. Log +
            # surface error on the response so MCP/LLM can distinguish
            # "broken" from "no neighbors" (codex-B4). related stays None;
            # error name flows back via the dedicated response field, NOT
            # via timing_ms (which dashboards read as durations).
            log.warning(
                "related_entities walk failed", exc_info=exc, trace_id=trace_id
            )
            related = None
            related_error = type(exc).__name__
        timing["related_entities_ms"] = (time.perf_counter() - t_related) * 1000

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

    bundles = _build_bundles(graph_hits) if graph_hits else None

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
        applied_min_confidence=req.min_confidence,
        extracted_entities=extracted_entities,
        aggregation=None,
        timing_ms=timing,
        trace_id=trace_id,
        bundles=bundles,
        related_entities=related,
        related_entities_error=related_error,
    )


def _build_bundles(graph_hits: list[GraphHit]) -> list[QueryBundle] | None:
    """Group graph_hits by seed entity, when the seed is a code-graph node.

    Builds a `QueryBundle` per (via_entity, via_label) pair where via_label
    is one of the code-graph labels (Function/Method/Class/Module/Symbol).
    Returns None when no code-graph entities seeded the search — keeps the
    response field unset so dashboard NL consumers don't see an empty list.
    """
    by_seed: dict[tuple[str, str], dict] = {}
    for hit in graph_hits:
        if not hit.via_label or hit.via_label not in CODE_GRAPH_LABELS:
            continue
        key = (hit.via_entity, hit.via_label)
        slot = by_seed.setdefault(
            key,
            {
                "chunk_ids": [],
                "confidence_breakdown": {"EXTRACTED": 0, "INFERRED": 0, "AMBIGUOUS": 0},
            },
        )
        slot["chunk_ids"].append(hit.chunk_id)
        tier = hit.confidence or "EXTRACTED"
        slot["confidence_breakdown"][tier] = slot["confidence_breakdown"].get(tier, 0) + 1

    if not by_seed:
        return None

    return [
        QueryBundle(
            seed_entity=seed,
            seed_label=label,
            related_chunk_ids=slot["chunk_ids"],
            confidence_breakdown=slot["confidence_breakdown"],
        )
        for (seed, label), slot in by_seed.items()
    ]
