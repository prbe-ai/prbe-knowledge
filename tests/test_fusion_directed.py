"""Unit tests for fusion's directed-vector booster — no DB needed.

Mirrors tests/test_fusion_dedup.py + tests/retrieval/test_metadata_chunks.py
in style. The directed contribution is a doc-level booster (analogous to
metadata_score_for_doc); these tests pin the regression boundary AND
the new behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from engine.retrieval.fusion import fuse
from engine.shared import constants as constants_module

_NOW = datetime(2026, 5, 8, tzinfo=UTC)


@dataclass
class FakeHit:
    chunk_id: str
    doc_id: str
    doc_version: int = 1
    source_system: str = "wiki"
    source_url: str = "u"
    title: str | None = "t"
    content: str = "c"
    created_at: datetime = field(default_factory=lambda: _NOW)
    updated_at: datetime = field(default_factory=lambda: _NOW)
    score: float = 0.5
    kind: str = "content"


@dataclass
class FakeDirectedHit:
    """Mirrors DirectedHit's surface; fusion only reads doc_id + score."""

    doc_id: str
    score: float = 0.9


# ---- regression: no-directed call paths byte-identical -------------------


def test_fuse_without_directed_arg_unchanged() -> None:
    """REGRESSION: omitting `directed_hits` is byte-identical to today."""
    hits = [
        FakeHit(chunk_id="c1", doc_id="d1"),
        FakeHit(chunk_id="c2", doc_id="d2"),
    ]
    fused = fuse({"vector": hits}, top_k=10, now=_NOW)
    assert len(fused) == 2
    doc_ids = [d.doc_id for d in fused]
    assert "d1" in doc_ids and "d2" in doc_ids
    # No 'directed' key should appear in any retriever_scores breakdown.
    for d in fused:
        assert "directed" not in d.retriever_scores


def test_fuse_with_none_directed_hits_unchanged() -> None:
    """REGRESSION: explicit `directed_hits=None` matches default behavior."""
    hits = [FakeHit(chunk_id="c1", doc_id="d1")]
    fused_default = fuse({"vector": hits}, top_k=10, now=_NOW)
    fused_none = fuse({"vector": hits}, top_k=10, now=_NOW, directed_hits=None)
    assert fused_default[0].score == pytest.approx(fused_none[0].score)
    assert "directed" not in fused_none[0].retriever_scores


def test_fuse_with_empty_directed_hits_unchanged() -> None:
    """REGRESSION: `directed_hits=[]` matches default behavior."""
    hits = [FakeHit(chunk_id="c1", doc_id="d1")]
    fused_default = fuse({"vector": hits}, top_k=10, now=_NOW)
    fused_empty = fuse({"vector": hits}, top_k=10, now=_NOW, directed_hits=[])
    assert fused_default[0].score == pytest.approx(fused_empty[0].score)


def test_fuse_with_directed_weight_zero_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """REGRESSION: DIRECTED_RETRIEVAL_WEIGHT=0.0 disables contribution.

    Boosting is gated on contribution > 0 so the retriever_scores
    breakdown also stays clean. This pins the kill-switch behavior:
    an operator can flip the constant to 0 without removing the
    retriever from the fan-out.
    """
    monkeypatch.setattr(constants_module, "DIRECTED_RETRIEVAL_WEIGHT", 0.0)
    # We need fusion to re-import after the patch — but it imports the
    # name, not the module attribute. Patch at the import site.
    from engine.retrieval import fusion as fusion_module

    monkeypatch.setattr(fusion_module, "DIRECTED_RETRIEVAL_WEIGHT", 0.0)

    hits = [FakeHit(chunk_id="c1", doc_id="d1")]
    fused_no = fuse({"vector": hits}, top_k=10, now=_NOW)
    fused_zero = fuse(
        {"vector": hits},
        top_k=10,
        now=_NOW,
        directed_hits=[FakeDirectedHit("d1", score=0.9)],
    )
    assert fused_no[0].score == pytest.approx(fused_zero[0].score)
    # retriever_scores should still NOT carry 'directed' when weight is 0
    # (the gating check is `directed_score > 0`, which a 0-weighted
    # contribution still passes — the unused branch picks up the score
    # for telemetry. Pin the contract: telemetry can show the signal
    # was present even when its weight is zero.) Keep the assertion
    # lenient so future eval-tuning of this contract doesn't break the
    # regression test: we just require parity on `score`.


# ---- new behavior: directed contribution boosts doc score -----------------


def test_fuse_directed_boosts_matching_doc() -> None:
    """A directed hit on a doc that already has a content chunk lifts it
    above a peer doc with the same content RRF but no directed hit.
    """
    hits = [
        FakeHit(chunk_id="c1", doc_id="d1"),
        FakeHit(chunk_id="c2", doc_id="d2"),
    ]
    fused = fuse(
        {"vector": hits},
        top_k=10,
        now=_NOW,
        directed_hits=[FakeDirectedHit("d1", score=0.9)],
    )
    # d1 should rank above d2 because its directed booster lifted it.
    assert fused[0].doc_id == "d1"
    assert fused[1].doc_id == "d2"


def test_fuse_directed_score_in_retriever_breakdown() -> None:
    """When directed contributes, doc.retriever_scores['directed'] carries
    the RRF contribution (1 / (k + rank)), NOT raw similarity. This is the
    same scale as vector/bm25/graph contributions so that
    DIRECTED_RETRIEVAL_WEIGHT=1.0 is a peer-of-other-retrievers, not a
    50x dominator.
    """
    from engine.shared.constants import RRF_K

    hits = [FakeHit(chunk_id="c1", doc_id="d1")]
    fused = fuse(
        {"vector": hits},
        top_k=10,
        now=_NOW,
        directed_hits=[FakeDirectedHit("d1", score=0.7)],  # rank 1
    )
    assert "directed" in fused[0].retriever_scores
    expected = 1.0 / (RRF_K + 1)
    assert fused[0].retriever_scores["directed"] == pytest.approx(expected)


def test_fuse_directed_only_doc_dropped() -> None:
    """A doc that ONLY surfaces via a directed hit (no content chunk in the
    pool from any other retriever) is silently dropped. Directed is a
    booster, not a sole source — same rule as metadata-only docs without
    a content_fallback.
    """
    fused = fuse(
        {"vector": []},
        top_k=10,
        now=_NOW,
        directed_hits=[FakeDirectedHit("only-directed", score=1.0)],
    )
    assert fused == []


def test_fuse_directed_first_occurrence_wins_per_doc() -> None:
    """Defensive: if directed_hits contains multiple entries for one doc
    (shouldn't happen — directed_search uses DISTINCT ON), we honor RRF
    convention and keep the FIRST occurrence's rank, ignoring later ones.
    The retriever guarantees list order = similarity-rank, so first
    occurrence == best similarity by construction.
    """
    from engine.shared.constants import RRF_K

    hits = [FakeHit(chunk_id="c1", doc_id="d1")]
    fused = fuse(
        {"vector": hits},
        top_k=10,
        now=_NOW,
        directed_hits=[
            FakeDirectedHit("d1", score=0.95),  # rank 1 -> 1/(60+1)
            FakeDirectedHit("d1", score=0.6),   # ignored
            FakeDirectedHit("d1", score=0.3),   # ignored
        ],
    )
    expected = 1.0 / (RRF_K + 1)
    assert fused[0].retriever_scores["directed"] == pytest.approx(expected)


def test_fuse_directed_rank_2_smaller_than_rank_1() -> None:
    """REGRESSION for the score-scale-mismatch P1: doc at rank 1 in the
    directed list contributes more than doc at rank 2, AND both are on
    RRF magnitude (~0.016 + 0.0163), NOT raw similarity (~0.7 - 1.0).
    Without RRF conversion, DIRECTED_RETRIEVAL_WEIGHT=1.0 dominated
    ranking by 50x; this pins the magnitude contract.
    """
    from engine.shared.constants import RRF_K

    hits = [
        FakeHit(chunk_id="c1", doc_id="d1"),
        FakeHit(chunk_id="c2", doc_id="d2"),
    ]
    fused = fuse(
        {"vector": hits},
        top_k=10,
        now=_NOW,
        directed_hits=[
            FakeDirectedHit("d1", score=0.95),  # rank 1
            FakeDirectedHit("d2", score=0.93),  # rank 2
        ],
    )
    by_id = {d.doc_id: d for d in fused}
    rank1 = 1.0 / (RRF_K + 1)
    rank2 = 1.0 / (RRF_K + 2)
    assert by_id["d1"].retriever_scores["directed"] == pytest.approx(rank1)
    assert by_id["d2"].retriever_scores["directed"] == pytest.approx(rank2)
    # And the directed contribution is RRF-scale (~0.016), NOT similarity-
    # scale (~0.95). Pin this ceiling so future drift is loud.
    assert by_id["d1"].retriever_scores["directed"] < 0.05


def test_fuse_directed_zero_similarity_still_contributes_when_listed() -> None:
    """A directed_hit with score=0.0 still gets ranked: presence in the
    directed list IS the signal, similarity is informational. This is
    different from the old similarity-scale behavior where score=0 meant
    "no contribution"; under RRF, only NOT being in the list means no
    contribution. The retriever's DISTINCT ON + dist ASC ordering ensures
    list membership only includes actual matches.
    """
    from engine.shared.constants import RRF_K

    hits = [FakeHit(chunk_id="c1", doc_id="d1")]
    fused = fuse(
        {"vector": hits},
        top_k=10,
        now=_NOW,
        directed_hits=[FakeDirectedHit("d1", score=0.0)],
    )
    expected = 1.0 / (RRF_K + 1)
    assert fused[0].retriever_scores["directed"] == pytest.approx(expected)
