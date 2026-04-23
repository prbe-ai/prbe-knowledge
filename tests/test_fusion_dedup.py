"""Unit tests for fusion + dedup — no DB needed."""

from __future__ import annotations

from dataclasses import dataclass

from services.retrieval.dedup import cosine, dedupe
from services.retrieval.fusion import fuse


@dataclass
class FakeHit:
    chunk_id: str
    doc_id: str
    doc_version: int = 1
    source_system: str = "slack"
    source_url: str = "u"
    title: str | None = "t"
    content: str = "c"
    score: float = 0.0


def test_fuse_combines_ranked_lists() -> None:
    vec = [FakeHit(chunk_id="c1", doc_id="d1", score=0.9),
           FakeHit(chunk_id="c2", doc_id="d2", score=0.8)]
    bm25 = [FakeHit(chunk_id="c2", doc_id="d2", score=0.7),
            FakeHit(chunk_id="c3", doc_id="d3", score=0.6)]

    fused = fuse({"vector": vec, "bm25": bm25}, top_k=10)
    # c2 appears in both → highest combined RRF score
    assert fused[0].chunk_id == "c2"
    assert fused[0].retriever_scores == {"vector": 0.8, "bm25": 0.7}


def test_fuse_doc_collapse() -> None:
    # Two chunks from the same doc — only the higher-combined one survives.
    vec = [
        FakeHit(chunk_id="c1a", doc_id="dup", score=0.1),
        FakeHit(chunk_id="c1b", doc_id="dup", score=0.9),
    ]
    bm25 = [FakeHit(chunk_id="c1b", doc_id="dup", score=0.5)]
    fused = fuse({"vector": vec, "bm25": bm25}, top_k=10)
    docs = {h.doc_id for h in fused}
    assert docs == {"dup"}


def test_cosine_identity() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_dedupe_drops_near_duplicate() -> None:
    hits = [
        FakeHit(chunk_id="c1", doc_id="d1"),
        FakeHit(chunk_id="c2", doc_id="d2"),
    ]
    embeddings = {"c1": [1.0, 0.0, 0.0], "c2": [0.999, 0.0447, 0.0]}  # cosine ≈ 0.999
    out = dedupe(hits, embeddings, threshold=0.95)
    assert len(out) == 1
    assert out[0].chunk_id == "c1"


def test_dedupe_keeps_dissimilar() -> None:
    hits = [
        FakeHit(chunk_id="c1", doc_id="d1"),
        FakeHit(chunk_id="c2", doc_id="d2"),
    ]
    embeddings = {"c1": [1.0, 0.0, 0.0], "c2": [0.0, 1.0, 0.0]}
    out = dedupe(hits, embeddings, threshold=0.95)
    assert len(out) == 2
