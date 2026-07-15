"""Reciprocal Rank Fusion — combines ranked lists from vector/BM25/graph retrievers.

RRF formula:   score(chunk) = Σ retriever 1 / (k + rank_in_retriever)

k=60 is standard per Cormack et al. 2009.

Doc-grouped fusion + kind-aware scoring
───────────────────────────────────────
Two kinds of chunks compete for ranking:

  - kind='content' — body text. What an agent actually consumes.
  - kind='metadata' — synthetic per-document key:value text generated at
                      ingestion (title, repo, author, source URL).
                      Searchable but never returned to the agent.

We fuse at the chunk level, then collapse per doc into a `FusedDocument`.
`top_k` is the GLOBAL chunk budget: we sort all content chunks by RRF and
keep the top `top_k`. A doc that has any surviving content chunk produces
one FusedDocument carrying just those survivors; doc count is naturally
<= top_k. Metadata chunks and content_fallback chunks DO NOT count
against the budget — they're never returned to the agent — but metadata
RRF still folds into the doc score as a booster.

Doc score:
    score = max(content_chunk_rrfs)
          + alpha * sum(other_content_chunk_rrfs)
          + sum(metadata_rrfs)

Where alpha = RRF_BREADTH_ALPHA. `max + alpha*sum_of_others` keeps
"best chunk wins ties" while still rewarding docs whose multiple
chunks all matched.

A doc whose only candidate-pool entry is its metadata chunk (no content
chunk surfaced from any retriever) is dropped unless the caller supplies a
`content_fallback` hit for the same doc.

Recency decay is always-on: every doc is multiplied by
exp(-ln2 * age_days / half_life). Half-life resolution order:

  1. Per-source override from the source registry (each connector
     registers its half_life_days in shared.source_registry).
  2. Caller-supplied `recency_half_life_days` if not None.
  3. DEFAULT_RECENCY_HALF_LIFE_DAYS.

All chunks of a doc share `updated_at`, so decay shifts between-doc ranking
only.

`sort` (deterministic time sort) is supported but the search pipeline no
longer uses it — sort intent on the search path becomes amplified
recency boost via half_life.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from shared.constants import (
    DEFAULT_RECENCY_HALF_LIFE_DAYS,
    DIRECTED_RETRIEVAL_WEIGHT,
    RRF_BREADTH_ALPHA,
    RRF_K,
)
from shared.source_registry import half_life_days_for, score_multiplier_for


@dataclass(slots=True)
class FusedChunk:
    chunk_id: str
    content: str
    score: float  # post-RRF chunk-level score (no source-mult / decay)
    retriever_scores: dict[str, float] = field(default_factory=dict)
    rank_in_doc: int = 0


@dataclass(slots=True)
class FusedDocument:
    doc_id: str
    doc_version: int
    source_system: str
    source_url: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    score: float  # doc-level (post-multiplier + decay)
    author_id: str | None = None
    retriever_scores: dict[str, float] = field(default_factory=dict)
    chunks: list[FusedChunk] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        # The doc's representative chunk_id (best chunk). Used by dedup
        # (keys on chunk embeddings) and id_lookup pin (was already
        # doc-keyed). Empty string if the doc somehow has no chunks.
        return self.chunks[0].chunk_id if self.chunks else ""

    @property
    def content(self) -> str:
        # Joined content across the doc's chunks. Used by the post-fusion
        # entity_filter — passing any chunk's text qualifies the doc.
        return "\n".join(c.content for c in self.chunks)


_LN2 = math.log(2)


# Multiplier cap applied to graph hits' RRF when discovery=True. Graph
# surprise scores are bounded [0.5, 4.0] (see services/retrieval/surprise.py).
# Uncapped, a top graph hit at rank 1 with surprise=4.0 would contribute
# (1/61) * 4 = 0.066 -- bigger than ANY vector hit (max 1/61 = 0.0164),
# which would over-dominate fusion. Cap at 2.0 lets the most-surprising
# graph hit compete with strong vector hits without flattening them: top
# graph contribution is at most 0.033 vs vector's 0.016, ~2x lift.
DISCOVERY_GRAPH_MULTIPLIER_CAP = 2.0


def _apply_source_decay(
    score: float,
    source_system: str,
    updated_at: datetime,
    baseline_half_life: float,
    ref_now: datetime,
) -> float:
    """Per-source multiplier + recency decay, applied at the doc level.

    Multiplier first so a brand-new claude_code doc still gets demoted;
    otherwise zero-decay at age=0 would bypass it. Half-life resolves via
    per-source override > caller-supplied baseline > universal default.
    """
    multiplier = score_multiplier_for(source_system)
    if multiplier != 1.0:
        score *= multiplier
    half_life = half_life_days_for(source_system, baseline_half_life)
    age_days = (ref_now - updated_at).total_seconds() / 86400.0
    if age_days >= 0:
        score *= math.exp(-_LN2 * age_days / half_life)
    return score


def fuse(
    ranked_lists: dict[str, list[Any]],
    top_k: int = 50,
    k: int = RRF_K,
    recency_half_life_days: float | None = None,
    now: datetime | None = None,
    sort: dict[str, str] | None = None,
    discovery: bool = False,
    directed_hits: list[Any] | None = None,
) -> list[FusedDocument]:
    """Combine ranked lists from multiple retrievers into doc-grouped output.

    `ranked_lists` is `{"vector": [VectorHit, ...], "bm25": [...], "graph": [...]}`.
    Each hit object must expose: chunk_id, doc_id, doc_version,
    source_system, source_url, title, content, created_at, updated_at,
    score, kind. `author_id` is optional and propagated when present.

    `directed_hits` is the optional list of DirectedHit objects from the
    directed-vector retriever. Directed signal contributes a doc-level
    booster (analogous to metadata_score_for_doc) — it does NOT inject
    its own chunk into the candidate pool. A doc that is ONLY surfaced
    via directed (no content chunk from any other retriever) is dropped:
    we never return a directed-only doc as a result. In practice wiki
    pages always carry content chunks so directed-only-misses don't
    happen; the design is to amplify ranking, not to be a sole source.

    Returns FusedDocuments whose chunk count totals at most `top_k`
    content chunks across all docs (the global chunk-budget cap). Each
    doc's `chunks` list contains the surviving content chunks for that
    doc, sorted by RRF descending, with `rank_in_doc` assigned. Doc
    count is naturally <= top_k. Metadata-only docs that pick up a
    `content_fallback` chunk also surface; the fallback chunk does NOT
    count against the budget (it's not part of any retriever's hit list,
    just a hydrated displayable body).
    """
    per_chunk_rrf: dict[str, float] = defaultdict(float)
    per_chunk_breakdown: dict[str, dict[str, float]] = defaultdict(dict)
    per_chunk_meta: dict[str, Any] = {}

    for retriever_name, hits in ranked_lists.items():
        for rank, hit in enumerate(hits, start=1):
            kind = getattr(hit, "kind", "content")
            if kind == "content_fallback":
                # A fallback only provides displayable content for a doc
                # that surfaced via metadata. It should not earn its own
                # RRF score or arbitrary rank-based boost.
                per_chunk_meta.setdefault(hit.chunk_id, hit)
                continue
            rrf = 1.0 / (k + rank)
            # Discovery mode: amplify graph contribution by surprise score
            # (capped) so cross-source / cross-community / inferred-edge
            # neighbors actually compete with vector/BM25 dominance. Only
            # graph CONTENT chunks carry meaningful surprise; metadata
            # chunks are synthetic key:value text and shouldn't be
            # amplified even if a future change has graph return them.
            # vector/BM25 hit.score is a similarity/relevance number on a
            # different scale and would need normalisation to participate
            # -- out of scope for v1. The multiplier compounds with
            # same-chunk vector/BM25 RRF: a chunk that's both surprising
            # AND semantically relevant gets both contributions.
            if discovery and retriever_name == "graph" and kind != "metadata":
                # hit.score may be None on legacy / synthetic paths -- default
                # to 1.0 (neutral) instead of crashing on `None * float`.
                surprise = hit.score if hit.score is not None else 1.0
                rrf *= min(surprise, DISCOVERY_GRAPH_MULTIPLIER_CAP)
            per_chunk_rrf[hit.chunk_id] += rrf
            # hit.score may be None on legacy/synthetic paths; coerce
            # to 0.0 in telemetry rather than crash on float(None).
            per_chunk_breakdown[hit.chunk_id][retriever_name] = (
                float(hit.score) if hit.score is not None else 0.0
            )
            per_chunk_meta[hit.chunk_id] = hit

    # Per-doc accounting:
    #   content_chunks_for_doc[doc_id] = list of (chunk_id, rrf_score) for content chunks
    #   metadata_score_for_doc[doc_id] = sum of metadata-chunk RRF scores
    #   fallback_content_for_doc[doc_id] = chunk_id of fallback content chunk (if any)
    content_chunks_for_doc: dict[str, list[tuple[str, float]]] = defaultdict(list)
    metadata_score_for_doc: dict[str, float] = defaultdict(float)
    metadata_breakdown_for_doc: dict[str, dict[str, float]] = defaultdict(dict)
    fallback_content_for_doc: dict[str, str] = {}

    for chunk_id, hit in per_chunk_meta.items():
        if getattr(hit, "kind", "content") == "content_fallback":
            fallback_content_for_doc.setdefault(hit.doc_id, chunk_id)

    for chunk_id, rrf_score in per_chunk_rrf.items():
        hit = per_chunk_meta[chunk_id]
        doc_id = hit.doc_id
        kind = getattr(hit, "kind", "content")

        if kind == "metadata":
            metadata_score_for_doc[doc_id] += rrf_score
            for retriever_name, score in per_chunk_breakdown[chunk_id].items():
                metadata_breakdown_for_doc[doc_id][f"metadata_{retriever_name}"] = score
            continue

        content_chunks_for_doc[doc_id].append((chunk_id, rrf_score))

    # Global chunk-budget cap: sort all content chunks by RRF desc, keep
    # the top `top_k`. Doc-grouped wire format is preserved (we still
    # build one FusedDocument per surviving doc), but doc count is now
    # naturally bounded by surviving-chunk count rather than by a doc-level
    # slice at the end. Restoring this pre-cf87b66 semantic keeps the
    # response payload bounded — the prior "max docs, every chunk kept"
    # behavior produced 12-16 chunks per response (worst case 32) vs the
    # historical ~5, doubling production P50 latency end-to-end.
    #
    # Tie-break on chunk_id asc keeps the cap deterministic when many
    # chunks share an RRF (common at the boundary). Cross-doc ordering is
    # otherwise irrelevant here — final doc ranking happens via doc.score
    # below.
    all_content: list[tuple[str, str, float]] = [
        (doc_id, chunk_id, rrf)
        for doc_id, chunks in content_chunks_for_doc.items()
        for chunk_id, rrf in chunks
    ]
    all_content.sort(key=lambda t: (-t[2], t[1]))
    surviving = all_content[:top_k]
    surviving_by_doc: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for doc_id, chunk_id, rrf in surviving:
        surviving_by_doc[doc_id].append((chunk_id, rrf))
    content_chunks_for_doc = surviving_by_doc

    # Directed-vector contributions: convert rank-in-list to RRF so the
    # signal lives on the same scale as vector/bm25/graph contributions
    # (rank-1 ≈ 1/61 ≈ 0.016, not raw cosine similarity ≈ 0.8). Without
    # this conversion, DIRECTED_RETRIEVAL_WEIGHT=1.0 would dominate the
    # ranking by ~50x — every doc with any directed hit would outrank
    # every doc without. The retriever guarantees DISTINCT ON (doc_id)
    # and orders by distance ASC, so list position == similarity rank;
    # first occurrence per doc wins (RRF convention).
    #
    # A directed hit for a doc that has no content chunk in the pool is
    # silently lost — fusion only builds FusedDocuments that have at
    # least one content chunk (or a content_fallback). Same rule as
    # graph/metadata signals: ranking boosters don't bring in a doc on
    # their own; only body-text retrievers do.
    directed_score_for_doc: dict[str, float] = defaultdict(float)
    if directed_hits:
        seen_doc_ids: set[str] = set()
        for rank, d in enumerate(directed_hits, start=1):
            if d.doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(d.doc_id)
            directed_score_for_doc[d.doc_id] = 1.0 / (k + rank)

    # Build FusedDocuments. Drop docs with NO content chunk in the candidate
    # pool unless a content_fallback exists.
    docs: dict[str, FusedDocument] = {}
    ref_now = now or datetime.now(UTC)
    baseline_half_life = (
        recency_half_life_days
        if recency_half_life_days is not None
        else DEFAULT_RECENCY_HALF_LIFE_DAYS
    )

    def _build_doc(
        doc_id: str,
        ranked_chunks: list[tuple[str, float]],
        metadata_score: float,
    ) -> FusedDocument:
        # Sort content chunks within doc by RRF desc; assign rank_in_doc.
        ranked_chunks.sort(key=lambda t: -t[1])
        chunks: list[FusedChunk] = []
        for i, (chunk_id, rrf_score) in enumerate(ranked_chunks, start=1):
            hit = per_chunk_meta[chunk_id]
            chunks.append(
                FusedChunk(
                    chunk_id=chunk_id,
                    content=hit.content,
                    score=rrf_score,
                    retriever_scores=dict(per_chunk_breakdown[chunk_id]),
                    rank_in_doc=i,
                )
            )

        rrfs = [s for _, s in ranked_chunks]
        if rrfs:
            best = rrfs[0]
            other_sum = sum(rrfs[1:])
            doc_score = best + RRF_BREADTH_ALPHA * other_sum + metadata_score
        else:
            doc_score = metadata_score

        # Directed-vector booster: doc-level signal contributed by per-doc
        # trigger phrases. Adds AFTER metadata_score so it stacks. Scaled
        # by DIRECTED_RETRIEVAL_WEIGHT (0.0 disables); see services/
        # retrieval/retrievers/directed.py for the source signal.
        directed_score = directed_score_for_doc.get(doc_id, 0.0)
        if directed_score > 0:
            doc_score += DIRECTED_RETRIEVAL_WEIGHT * directed_score

        # All chunks of a doc share source_system + updated_at, so the
        # first (highest-RRF) chunk's hit anchors the per-doc decay.
        anchor_chunk_id = ranked_chunks[0][0] if ranked_chunks else (
            fallback_content_for_doc[doc_id]
        )
        anchor_hit = per_chunk_meta[anchor_chunk_id]
        doc_score = _apply_source_decay(
            doc_score,
            anchor_hit.source_system,
            anchor_hit.updated_at,
            baseline_half_life,
            ref_now,
        )

        # Doc-level retriever_scores: aggregate of best chunk's breakdown +
        # any metadata-chunk contribution (visible on the doc, not duplicated
        # per chunk).
        doc_retriever_scores: dict[str, float] = {}
        if ranked_chunks:
            doc_retriever_scores.update(per_chunk_breakdown[ranked_chunks[0][0]])
        doc_retriever_scores.update(metadata_breakdown_for_doc.get(doc_id, {}))
        if directed_score > 0:
            doc_retriever_scores["directed"] = directed_score

        return FusedDocument(
            doc_id=doc_id,
            doc_version=anchor_hit.doc_version,
            source_system=anchor_hit.source_system,
            source_url=anchor_hit.source_url,
            title=anchor_hit.title,
            created_at=anchor_hit.created_at,
            updated_at=anchor_hit.updated_at,
            score=doc_score,
            author_id=getattr(anchor_hit, "author_id", None),
            retriever_scores=doc_retriever_scores,
            chunks=chunks,
        )

    for doc_id, ranked_chunks in content_chunks_for_doc.items():
        docs[doc_id] = _build_doc(
            doc_id,
            ranked_chunks,
            metadata_score_for_doc.get(doc_id, 0.0),
        )

    # Metadata-only docs with a fallback: synthesize one synthetic chunk so
    # the response carries real content, not synthetic key:value text.
    for doc_id, metadata_score in metadata_score_for_doc.items():
        if doc_id in docs:
            continue
        fallback_chunk_id = fallback_content_for_doc.get(doc_id)
        if fallback_chunk_id is None:
            continue
        fallback_hit = per_chunk_meta[fallback_chunk_id]
        chunks = [
            FusedChunk(
                chunk_id=fallback_chunk_id,
                content=fallback_hit.content,
                score=0.0,
                retriever_scores={},
                rank_in_doc=1,
            )
        ]
        doc_score = _apply_source_decay(
            metadata_score,
            fallback_hit.source_system,
            fallback_hit.updated_at,
            baseline_half_life,
            ref_now,
        )
        docs[doc_id] = FusedDocument(
            doc_id=doc_id,
            doc_version=fallback_hit.doc_version,
            source_system=fallback_hit.source_system,
            source_url=fallback_hit.source_url,
            title=fallback_hit.title,
            created_at=fallback_hit.created_at,
            updated_at=fallback_hit.updated_at,
            score=doc_score,
            author_id=getattr(fallback_hit, "author_id", None),
            retriever_scores=dict(metadata_breakdown_for_doc.get(doc_id, {})),
            chunks=chunks,
        )

    fused = list(docs.values())

    if sort:
        field_name = sort.get("field", "updated_at")
        direction = sort.get("direction", "desc")
        sign = -1 if direction == "desc" else 1
        if field_name == "created_at":
            fused.sort(key=lambda d: (sign * d.created_at.timestamp(), d.doc_id))
        else:
            fused.sort(key=lambda d: (sign * d.updated_at.timestamp(), d.doc_id))
    else:
        fused.sort(key=lambda d: (-d.score, -d.updated_at.timestamp(), d.doc_id))
    # No final doc-level slice: top_k is the chunk budget, applied above
    # via the surviving-chunk cap. Doc count is naturally <= top_k.
    return fused
