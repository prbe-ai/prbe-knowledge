"""Reciprocal Rank Fusion — combines ranked lists from vector/BM25/graph retrievers.

RRF formula:   score(doc) = Σ retriever 1 / (k + rank_in_retriever)

k=60 is standard per Cormack et al. 2009. Doc-level collapse happens after
per-retriever ranks are computed — if the same doc surfaces multiple chunks
across retrievers, we keep the chunk with the strongest combined signal.

Optional `recency_half_life_days` multiplies each fused chunk's score by
exp(-ln2 * age_days / half_life). All chunks of the same doc share an
`updated_at`, so decay only affects between-doc ranking — within-doc
chunk selection is unaffected.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from shared.constants import RRF_K


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
    retriever_scores: dict[str, float] = field(default_factory=dict)


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
    Each hit object must expose attributes: chunk_id, doc_id, doc_version,
    source_system, source_url, title, content, created_at, updated_at, score.

    `recency_half_life_days` is optional. When set, multiplies each chunk's
    fused RRF score by exp(-ln2 * age_days / half_life), with `now` as the
    reference (defaults to datetime.now(UTC)). Future timestamps (clock
    skew) skip the decay so they're not penalized.

    `sort` is optional. When set, replaces the default relevance sort with a
    deterministic time sort: `{"field": "created_at"|"updated_at",
    "direction": "asc"|"desc"}`. RRF score still drives which chunks make
    the candidate pool; sort only reorders the surviving hits.

    Returns up to top_k fused hits.
    """
    per_chunk_score: dict[str, float] = defaultdict(float)
    per_chunk_breakdown: dict[str, dict[str, float]] = defaultdict(dict)
    per_chunk_meta: dict[str, Any] = {}

    for retriever_name, hits in ranked_lists.items():
        for rank, hit in enumerate(hits, start=1):
            rrf = 1.0 / (k + rank)
            per_chunk_score[hit.chunk_id] += rrf
            per_chunk_breakdown[hit.chunk_id][retriever_name] = float(hit.score)
            per_chunk_meta[hit.chunk_id] = hit  # last writer wins for meta (fine)

    if recency_half_life_days is not None:
        # Decay multiplies all chunks of the same doc by the same factor
        # (they share updated_at), so this only shifts between-doc ranking.
        ref_now = now or datetime.now(UTC)
        ln2 = math.log(2)
        for chunk_id, hit in per_chunk_meta.items():
            age_days = (ref_now - hit.updated_at).total_seconds() / 86400.0
            if age_days < 0:
                continue
            per_chunk_score[chunk_id] *= math.exp(-ln2 * age_days / recency_half_life_days)

    # Collapse: keep one chunk per doc_id — the chunk with the highest combined score.
    best_per_doc: dict[str, str] = {}
    for chunk_id, score in per_chunk_score.items():
        hit = per_chunk_meta[chunk_id]
        doc_id = hit.doc_id
        if doc_id not in best_per_doc:
            best_per_doc[doc_id] = chunk_id
            continue
        current = per_chunk_score[best_per_doc[doc_id]]
        if score > current:
            best_per_doc[doc_id] = chunk_id

    fused: list[FusedHit] = []
    for chunk_id in best_per_doc.values():
        hit = per_chunk_meta[chunk_id]
        fused.append(
            FusedHit(
                chunk_id=chunk_id,
                doc_id=hit.doc_id,
                doc_version=hit.doc_version,
                source_system=hit.source_system,
                source_url=hit.source_url,
                title=hit.title,
                content=hit.content,
                created_at=hit.created_at,
                updated_at=hit.updated_at,
                score=per_chunk_score[chunk_id],
                retriever_scores=per_chunk_breakdown[chunk_id],
            )
        )

    if sort:
        # Caller asked for a deterministic time sort. Score still gated which
        # chunks made it here; this just orders the survivors.
        field_name = sort.get("field", "updated_at")
        direction = sort.get("direction", "desc")
        sign = -1 if direction == "desc" else 1
        if field_name == "created_at":
            fused.sort(key=lambda h: (sign * h.created_at.timestamp(), h.chunk_id))
        else:
            fused.sort(key=lambda h: (sign * h.updated_at.timestamp(), h.chunk_id))
    else:
        # Default: highest score, then most recent updated_at as tie-breaker.
        fused.sort(key=lambda h: (-h.score, -h.updated_at.timestamp(), h.chunk_id))
    return fused[:top_k]
