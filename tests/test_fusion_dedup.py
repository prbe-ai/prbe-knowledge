"""Unit tests for fusion + dedup — no DB needed."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from services.retrieval.dedup import cosine, dedupe
from services.retrieval.fusion import fuse

_NOW = datetime(2026, 4, 24, tzinfo=UTC)


@dataclass
class FakeHit:
    chunk_id: str
    doc_id: str
    doc_version: int = 1
    source_system: str = "slack"
    source_url: str = "u"
    title: str | None = "t"
    content: str = "c"
    created_at: datetime = field(default_factory=lambda: _NOW)
    updated_at: datetime = field(default_factory=lambda: _NOW)
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


# ---------------------------------------------------------------------------
# Recency decay
# ---------------------------------------------------------------------------


def test_fusion_recency_decay_halves_at_half_life() -> None:
    """At age == half_life_days, score is multiplied by exactly 0.5."""
    fresh = FakeHit(chunk_id="cf", doc_id="df", updated_at=_NOW)
    week_old = FakeHit(
        chunk_id="co", doc_id="do", updated_at=_NOW - timedelta(days=7)
    )
    # Each in its own retriever so both rank 1 → identical pre-decay RRF score.
    fused = fuse(
        {"vector": [fresh], "bm25": [week_old]},
        top_k=10,
        recency_half_life_days=7,
        now=_NOW,
    )
    by_chunk = {h.chunk_id: h.score for h in fused}
    # Older chunk's score is exactly half the fresh one.
    assert abs(by_chunk["co"] / by_chunk["cf"] - 0.5) < 1e-9


def test_fusion_recency_decay_skips_future_timestamps() -> None:
    """Clock-skew safe: future updated_at gets no decay penalty."""
    fresh = FakeHit(chunk_id="now", doc_id="now-doc", updated_at=_NOW)
    future = FakeHit(
        chunk_id="future", doc_id="future-doc", updated_at=_NOW + timedelta(days=10)
    )
    fused = fuse(
        {"vector": [fresh, future]},
        top_k=10,
        recency_half_life_days=7,
        now=_NOW,
    )
    by_chunk = {h.chunk_id: h.score for h in fused}
    # Future timestamp keeps its full RRF score (rank 2 = 1/(60+2)).
    assert by_chunk["future"] == 1.0 / 62


def test_fusion_no_decay_when_half_life_unset() -> None:
    week_old = FakeHit(
        chunk_id="old", doc_id="d", updated_at=_NOW - timedelta(days=365)
    )
    fused = fuse({"vector": [week_old]}, top_k=10, now=_NOW)
    # No decay → raw RRF score (rank 1 = 1/(60+1)).
    assert abs(fused[0].score - 1.0 / 61) < 1e-12


# ---------------------------------------------------------------------------
# Tie-break
# ---------------------------------------------------------------------------


def test_fusion_tie_breaks_on_updated_at_desc() -> None:
    """Identical RRF score → later updated_at ranks first."""
    older = FakeHit(
        chunk_id="aaa", doc_id="d-old", updated_at=_NOW - timedelta(days=30)
    )
    newer = FakeHit(chunk_id="zzz", doc_id="d-new", updated_at=_NOW)
    # Both rank 1 in their retriever → identical pre-sort score.
    fused = fuse(
        {"vector": [older], "bm25": [newer]},
        top_k=10,
    )
    assert fused[0].chunk_id == "zzz"  # later updated_at wins
    assert fused[1].chunk_id == "aaa"


def test_fusion_tertiary_tie_breaks_on_chunk_id_asc() -> None:
    """Equal score AND equal updated_at → chunk_id asc wins."""
    a = FakeHit(chunk_id="aaa", doc_id="d-a", updated_at=_NOW)
    b = FakeHit(chunk_id="bbb", doc_id="d-b", updated_at=_NOW)
    fused = fuse(
        {"vector": [a], "bm25": [b]},
        top_k=10,
    )
    assert fused[0].chunk_id == "aaa"
    assert fused[1].chunk_id == "bbb"
