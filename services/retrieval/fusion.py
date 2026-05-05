"""Reciprocal Rank Fusion — combines ranked lists from vector/BM25/graph retrievers.

RRF formula:   score(doc) = Σ retriever 1 / (k + rank_in_retriever)

k=60 is standard per Cormack et al. 2009.

Doc-level collapse + kind-aware scoring
───────────────────────────────────────
Two kinds of chunks compete for ranking:

  - kind='content' — body text. Always what an agent actually wants to
                     see in the response.
  - kind='metadata' — synthetic per-document key:value text generated at
                      ingestion (title, repo, author, source URL).
                      Searchable but never returned to the agent.

We fuse at the chunk level (so a doc that surfaces via either or both
kinds gets the right combined signal), then collapse per doc into the
best CONTENT chunk for that doc, with the doc's metadata-chunk score
folded into the per-doc score as a booster.

A doc whose only candidate-pool entry is its metadata chunk (no content
chunk surfaced from any retriever) is dropped unless the caller supplies a
`content_fallback` hit for the same doc. Fallback hits let metadata select
the doc while preserving the response contract: agents still see only real
content chunks, never synthetic key:value metadata text.

Recency decay is always-on: every doc is multiplied by
exp(-ln2 * age_days / half_life). Half-life resolution order:

  1. Per-source override in SOURCE_HALF_LIFE_DAYS (e.g. claude_code/codex
     at 7d for noisy transcript sources).
  2. Caller-supplied `recency_half_life_days` if not None.
  3. DEFAULT_RECENCY_HALF_LIFE_DAYS — universal baseline so backfilled
     tenants don't surface stale year-old content at parity with fresh docs.

All chunks of a doc share `updated_at`, so decay shifts between-doc ranking
only.

`sort` (deterministic time sort) is supported but the search pipeline no
longer uses it — sort intent on the search path becomes amplified
recency boost via half_life. Kept for callers that want explicit control.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from shared.constants import (
    DEFAULT_RECENCY_HALF_LIFE_DAYS,
    RRF_K,
    SOURCE_HALF_LIFE_DAYS,
    SOURCE_SCORE_MULTIPLIERS,
)


@dataclass(slots=True)
class FusedHit:
    chunk_id: str
    doc_id: str
    doc_version: int
    source_system: str
    source_url: str
    title: str | None
    content: str
    created_at: datetime
    updated_at: datetime
    score: float
    author_id: str | None = None
    retriever_scores: dict[str, float] = field(default_factory=dict)
    # Always 'content' on FusedHit — fusion's contract is "synthetic
    # metadata text never escapes." Caller sees only body chunks.
    kind: str = "content"


def fuse(
    ranked_lists: dict[str, list[Any]],
    top_k: int = 50,
    k: int = RRF_K,
    recency_half_life_days: float | None = None,
    now: datetime | None = None,
    sort: dict[str, str] | None = None,
) -> list[FusedHit]:
    """Combine ranked lists from multiple retrievers.

    `ranked_lists` is `{"vector": [VectorHit, ...], "bm25": [...], "graph": [...]}`.
    Each hit object must expose: chunk_id, doc_id, doc_version,
    source_system, source_url, title, content, created_at, updated_at,
    score, kind. `author_id` is optional and propagated when present.

    Returns up to top_k FusedHits, each one a CONTENT chunk (metadata
    chunks contribute scoring signal but never appear in the response).
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
            per_chunk_rrf[hit.chunk_id] += rrf
            per_chunk_breakdown[hit.chunk_id][retriever_name] = float(hit.score)
            per_chunk_meta[hit.chunk_id] = hit

    # Per-doc accounting:
    #   best_content[doc_id] = chunk_id of highest-RRF content chunk
    #   content_score[doc_id] = the RRF score of that content chunk
    #   metadata_score[doc_id] = sum of metadata-chunk RRF scores for that doc
    #     (typically one per doc, but defensive against future cardinality)
    best_content_for_doc: dict[str, str] = {}
    content_score_for_doc: dict[str, float] = {}
    fallback_content_for_doc: dict[str, str] = {}
    metadata_score_for_doc: dict[str, float] = defaultdict(float)
    metadata_breakdown_for_doc: dict[str, dict[str, float]] = defaultdict(dict)

    for chunk_id, hit in per_chunk_meta.items():
        if getattr(hit, "kind", "content") == "content_fallback":
            fallback_content_for_doc.setdefault(hit.doc_id, chunk_id)

    for chunk_id, rrf_score in per_chunk_rrf.items():
        hit = per_chunk_meta[chunk_id]
        doc_id = hit.doc_id
        kind = getattr(hit, "kind", "content")

        if kind == "metadata":
            metadata_score_for_doc[doc_id] += rrf_score
            # Capture metadata's per-retriever signal for response telemetry —
            # merged into the surviving content chunk's breakdown so callers
            # can see "the metadata chunk also matched".
            for retriever_name, score in per_chunk_breakdown[chunk_id].items():
                metadata_breakdown_for_doc[doc_id][f"metadata_{retriever_name}"] = score
            continue

        # Content chunk — track the highest-scoring one per doc.
        prior_score = content_score_for_doc.get(doc_id)
        if prior_score is None or rrf_score > prior_score:
            best_content_for_doc[doc_id] = chunk_id
            content_score_for_doc[doc_id] = rrf_score

    # Combine: doc score = best content RRF + metadata RRF.
    # Drop docs with NO content chunk in the candidate pool.
    combined_for_doc: dict[str, float] = {}
    for doc_id, content_score in content_score_for_doc.items():
        combined_for_doc[doc_id] = content_score + metadata_score_for_doc.get(doc_id, 0.0)
    for doc_id, metadata_score in metadata_score_for_doc.items():
        if doc_id in combined_for_doc:
            continue
        fallback_chunk_id = fallback_content_for_doc.get(doc_id)
        if fallback_chunk_id is None:
            continue
        best_content_for_doc[doc_id] = fallback_chunk_id
        content_score_for_doc[doc_id] = 0.0
        combined_for_doc[doc_id] = metadata_score

    # Per-source-system score multiplier (Change A) and recency decay
    # (Change C). Multiplier first so a brand-new claude_code doc still gets
    # demoted; otherwise zero-decay at age=0 would bypass it.
    #
    # Half-life resolution: per-source override > caller global > universal
    # baseline. The baseline is always-on so backfilled tenants don't surface
    # 8-12 month old docs ranked equally with fresh ones.
    ref_now = now or datetime.now(UTC)
    ln2 = math.log(2)
    baseline_half_life = (
        recency_half_life_days
        if recency_half_life_days is not None
        else DEFAULT_RECENCY_HALF_LIFE_DAYS
    )
    for doc_id, combined in list(combined_for_doc.items()):
        chunk_id = best_content_for_doc[doc_id]
        hit = per_chunk_meta[chunk_id]
        source_system = hit.source_system

        multiplier = SOURCE_SCORE_MULTIPLIERS.get(source_system, 1.0)
        if multiplier != 1.0:
            combined *= multiplier

        half_life = SOURCE_HALF_LIFE_DAYS.get(source_system, baseline_half_life)
        age_days = (ref_now - hit.updated_at).total_seconds() / 86400.0
        if age_days >= 0:
            combined *= math.exp(-ln2 * age_days / half_life)

        combined_for_doc[doc_id] = combined

    fused: list[FusedHit] = []
    for doc_id, combined in combined_for_doc.items():
        chunk_id = best_content_for_doc[doc_id]
        hit = per_chunk_meta[chunk_id]
        retriever_scores = dict(per_chunk_breakdown[chunk_id])
        # Fold any metadata-chunk contribution into the breakdown for visibility.
        retriever_scores.update(metadata_breakdown_for_doc.get(doc_id, {}))
        fused.append(
            FusedHit(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_version=hit.doc_version,
                source_system=hit.source_system,
                source_url=hit.source_url,
                title=hit.title,
                content=hit.content,
                created_at=hit.created_at,
                updated_at=hit.updated_at,
                score=combined,
                author_id=getattr(hit, "author_id", None),
                retriever_scores=retriever_scores,
                kind="content",
            )
        )

    if sort:
        # Caller asked for a deterministic time sort. Score still gated which
        # docs made it here; this just orders the survivors.
        field_name = sort.get("field", "updated_at")
        direction = sort.get("direction", "desc")
        sign = -1 if direction == "desc" else 1
        if field_name == "created_at":
            fused.sort(key=lambda h: (sign * h.created_at.timestamp(), h.chunk_id))
        else:
            fused.sort(key=lambda h: (sign * h.updated_at.timestamp(), h.chunk_id))
    else:
        fused.sort(key=lambda h: (-h.score, -h.updated_at.timestamp(), h.chunk_id))
    return fused[:top_k]
