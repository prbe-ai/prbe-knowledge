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
from services.retrieval.fusion import FusedChunk, FusedDocument, fuse
from services.retrieval.helpers import (
    apply_entity_filter,
    embeddings_for_chunks,
)
from services.retrieval.retrievers.bm25 import BM25Hit, bm25_search
from services.retrieval.retrievers.graph import graph_search
from services.retrieval.retrievers.id_lookup import id_lookup_search, is_lookup_candidate
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
    GraphEvidence,
    QueryChunk,
    QueryDocument,
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


def _inject_id_lookup_hits(
    fused: list[FusedDocument], id_hits: list
) -> list[FusedDocument]:
    """Append synthetic FusedDocuments for any id_lookup-matched doc that
    didn't survive `fuse()`'s top_k cap.

    Why this exists: at MCP defaults (top_k=5), the fused pool is
    5 * pool_multiplier = 10. The fusion pipeline applies a per-source
    multiplier (claude_code/codex = 0.5x) and recency decay (7-day
    half-life), which routinely buries an exact-id match below 10 — so a
    pure pin-after-fusion would have nothing to pin. Injecting these docs
    here puts them into the dedupe/ACL/pin path with the rest, and the
    pin pass downstream guarantees they end up at rank 1.

    Synthetic docs carry `score=1.0` and one chunk with
    `retriever_scores={"id_lookup": 1.0}` so response telemetry still
    reflects which retriever surfaced them.
    """
    if not id_hits:
        return fused
    fused_doc_ids = {f.doc_id for f in fused}
    extras: list[FusedDocument] = []
    for h in id_hits:
        if h.doc_id in fused_doc_ids:
            continue
        fused_doc_ids.add(h.doc_id)
        extras.append(
            FusedDocument(
                doc_id=h.doc_id,
                doc_version=h.doc_version,
                source_system=h.source_system,
                source_url=h.source_url,
                title=h.title,
                created_at=h.created_at,
                updated_at=h.updated_at,
                score=1.0,
                author_id=h.author_id,
                retriever_scores={"id_lookup": 1.0},
                chunks=[
                    FusedChunk(
                        chunk_id=h.chunk_id,
                        content=h.content,
                        score=1.0,
                        retriever_scores={"id_lookup": 1.0},
                        rank_in_doc=1,
                    )
                ],
            )
        )
    return fused + extras


def _pin_id_lookup_matches(
    fused: list[FusedDocument], id_hits: list
) -> list[FusedDocument]:
    """Float docs whose source_id was an exact match by `id_lookup_search`
    to the top of the fused list, preserving id_lookup's original order.

    Why: a UUID-precise query like "session 3c325e11-..." should land the
    matching doc at rank 1. Fusion's per-source multiplier (claude_code /
    codex are 0.5x) and short half-life (7 days) can demote a session
    below code_graph BM25 hits even when `id_lookup` nailed an exact
    match — observed live: target session at rank 36 with id_lookup score
    1.0 still buried under code_graph chunks at BM25 ranks 1-35. This
    pin guarantees exact-id matches sit on top regardless of the discount
    factors, without changing the ranking of unmatched docs below.

    Pairs with `_inject_id_lookup_hits`: injection ensures the doc is in
    the candidate pool when top_k is small (MCP default 5); pin then
    floats it to position 0.
    """
    if not id_hits:
        return fused
    seen: set[str] = set()
    id_order: list[str] = []
    for h in id_hits:
        if h.doc_id not in seen:
            seen.add(h.doc_id)
            id_order.append(h.doc_id)
    by_doc = {f.doc_id: f for f in fused}
    pinned = [by_doc[did] for did in id_order if did in by_doc]
    rest = [f for f in fused if f.doc_id not in seen]
    return pinned + rest


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

    async def _id_lookup_runner() -> list:
        # Pin docs whose source_id/doc_id matches a router-extracted stable
        # identifier (UUID, ticket code, PR ref). Vector and BM25 both miss
        # exact-id queries — see retrievers/id_lookup.py for the rationale.
        ids = [
            e.canonical_id for e in routed.entities if is_lookup_candidate(e.canonical_id)
        ]
        if not ids:
            return []
        return await id_lookup_search(customer_id, ids, temporal=spec)

    t_retrieve = time.perf_counter()
    vec_hits, bm25_hits, graph_hits, id_hits = await asyncio.gather(
        _vec_runner(), _bm25_runner(), _graph_runner(), _id_lookup_runner()
    )
    ranked_lists = {
        "vector": vec_hits,
        "bm25": bm25_hits,
        "graph": graph_hits,
        "id_lookup": id_hits,
    }
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

    # Inject id_lookup-matched docs that didn't survive fuse()'s top_k cap
    # so they reach dedupe/ACL/pin even at MCP defaults (top_k=5 → pool 10).
    # Without this, the source-multiplier-discounted exact-id match can land
    # at rank 36+ pre-cap and never enter the candidate set the pin sees.
    fused = _inject_id_lookup_hits(fused, id_hits)

    applied_entity_filter: dict[str, object] | None = None
    if req.entity_must_match:
        pre_count = len(fused)
        fused, applied_entity_filter = apply_entity_filter(
            fused, routed.entities, threshold=req.entity_match_threshold
        )
        applied_entity_filter["candidates_before"] = pre_count
        applied_entity_filter["candidates_after"] = len(fused)

    t_dedup = time.perf_counter()
    # Dedup keys on the doc's representative chunk_id (best chunk by RRF).
    # Two docs whose top chunks are near-duplicate embeddings collapse to
    # the higher-ranked doc — same semantics as before, just at doc level.
    embeddings = await embeddings_for_chunks(
        customer_id, [d.chunk_id for d in fused if d.chunk_id]
    )
    deduped = dedupe(fused, embeddings)
    timing["dedup_ms"] = (time.perf_counter() - t_dedup) * 1000

    t_acl = time.perf_counter()
    filtered = await filter_by_acl(customer_id, req.requesting_user_id, deduped)
    timing["acl_ms"] = (time.perf_counter() - t_acl) * 1000

    # Pin exact-id matches on top of the ACL-filtered list so an extracted
    # canonical_id (UUID, ticket code, PR ref) always wins rank 1 — the
    # fusion source-multiplier + recency decay otherwise demote sessions
    # below code_graph noise. Must run AFTER ACL so we don't surface a
    # doc the requesting user can't see.
    filtered = _pin_id_lookup_matches(filtered, id_hits)

    top: list[FusedDocument] = filtered[: req.top_k]

    related: list[RelatedEntity] | None = None
    related_error: str | None = None
    if req.top_k_related > 0:
        exclude_keys = build_exclude_node_keys(
            routed.entities,
            entity_match_threshold=req.entity_match_threshold,
        )
        ranked_docs = [(d.doc_id, i) for i, d in enumerate(top, start=1)]
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

    # Build (chunk_id -> list[GraphEvidence]) from the raw graph_hits list.
    # One chunk reached via N seeds carries N entries — preserve the M:N
    # relationship the response contract requires.
    graph_evidence_by_chunk: dict[str, list[GraphEvidence]] = {}
    seen_evidence_keys: dict[str, set[tuple]] = {}
    for gh in graph_hits:
        edge_type = gh.edge_type or ""
        confidence = gh.confidence or "EXTRACTED"
        via_entity = gh.via_entity
        key = (edge_type, confidence, via_entity, gh.via_label)
        if not edge_type:
            continue
        seen = seen_evidence_keys.setdefault(gh.chunk_id, set())
        if key in seen:
            continue
        seen.add(key)
        graph_evidence_by_chunk.setdefault(gh.chunk_id, []).append(
            GraphEvidence(
                edge_type=edge_type,
                confidence=confidence,
                via_entity=via_entity,
                reason=None,
            )
        )

    documents = [
        QueryDocument(
            doc_id=d.doc_id,
            doc_version=d.doc_version,
            source_system=SourceSystem(d.source_system),
            source_url=d.source_url,
            title=d.title,
            author_id=d.author_id,
            created_at=d.created_at,
            updated_at=d.updated_at,
            score=d.score,
            rank=i + 1,
            chunk_count=len(d.chunks),
            retriever_scores=d.retriever_scores,
            chunks=[
                QueryChunk(
                    chunk_id=c.chunk_id,
                    score=c.score,
                    rank_in_doc=c.rank_in_doc,
                    content=c.content,
                    retriever_scores=c.retriever_scores,
                    graph_evidence=graph_evidence_by_chunk.get(c.chunk_id, []),
                )
                for c in d.chunks
            ],
        )
        for i, d in enumerate(top)
    ]

    confidence_breakdown = {"EXTRACTED": 0, "INFERRED": 0, "AMBIGUOUS": 0}
    for doc in documents:
        for chunk in doc.chunks:
            for ev in chunk.graph_evidence:
                tier = ev.confidence
                confidence_breakdown[tier] = confidence_breakdown.get(tier, 0) + 1

    return QueryResponse(
        query=req.query,
        documents=documents,
        total_candidates=len(fused),
        router_hit_cache=False,
        confidence_breakdown=confidence_breakdown,
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
        related_entities=related,
        related_entities_error=related_error,
    )
