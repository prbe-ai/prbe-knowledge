"""Reciprocal Rank Fusion — combines ranked lists from vector/BM25/graph retrievers.

RRF formula:   score(doc) = Σ retriever 1 / (k + rank_in_retriever)

k=60 is standard per Cormack et al. 2009. Doc-level collapse happens after
per-retriever ranks are computed — if the same doc surfaces multiple chunks
across retrievers, we keep the chunk with the strongest combined signal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
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
    score: float
    retriever_scores: dict[str, float] = field(default_factory=dict)


def fuse(
    ranked_lists: dict[str, list[Any]],
    top_k: int = 50,
    k: int = RRF_K,
) -> list[FusedHit]:
    """Combine ranked lists from multiple retrievers.

    `ranked_lists` is `{"vector": [VectorHit, ...], "bm25": [...], "graph": [...]}`.
    Each hit object must expose attributes: chunk_id, doc_id, doc_version,
    source_system, source_url, title, content, score.

    Returns up to top_k fused hits, sorted by combined RRF score desc.
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
                score=per_chunk_score[chunk_id],
                retriever_scores=per_chunk_breakdown[chunk_id],
            )
        )

    fused.sort(key=lambda h: h.score, reverse=True)
    return fused[:top_k]
