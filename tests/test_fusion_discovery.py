"""Unit tests for discovery mode in fusion.

Discovery mode amplifies graph hits' RRF contribution by their surprise
score (capped at DISCOVERY_GRAPH_MULTIPLIER_CAP). Vector and BM25 hits
are unaffected because their `score` field is similarity/relevance on a
different scale and would need normalisation to participate.

The tests below pin:
- discovery=False is identical to today's behaviour (regression gate)
- discovery=True multiplies graph RRF by min(score, cap)
- the cap actually fires when score > cap
- vector/BM25 RRF stays flat regardless of discovery
- a chunk in BOTH graph and vector pools gets graph slice multiplied,
  vector slice unchanged (the multiplier compounds, doesn't replace)
- metadata-kind chunks are unaffected
- doc-aggregation behaviour: a doc with many graph chunks can outrank a
  doc with one very-surprising graph chunk (architectural pin from
  /plan-eng-review finding 1B; not a bug, but documented)
- defensive default when hit.score is None (legacy path safety)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from engine.retrieval.fusion import DISCOVERY_GRAPH_MULTIPLIER_CAP, fuse
from engine.shared.constants import RRF_K

_NOW = datetime(2026, 5, 8, tzinfo=UTC)
_RRF_RANK1 = 1.0 / (RRF_K + 1)  # ~0.01639


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
    score: float = 1.0
    kind: str = "content"


def _graph_chunks_for_doc(fused_docs, doc_id: str):
    """Return the FusedChunk list for the given doc_id, or None."""
    for d in fused_docs:
        if d.doc_id == doc_id:
            return d.chunks
    return None


# ---------------------------------------------------------------------------
# Regression: discovery=False is unchanged
# ---------------------------------------------------------------------------


def test_discovery_false_unchanged_from_default() -> None:
    """Regression gate: discovery=False produces identical fusion to today.

    Two graph hits with high surprise scores (2.0 and 4.0). With
    discovery=False, both contribute the SAME flat 1/(k+rank) RRF
    regardless of their score field.
    """
    graph = [
        FakeHit(chunk_id="c1", doc_id="d1", score=4.0),
        FakeHit(chunk_id="c2", doc_id="d2", score=2.0),
    ]

    fused_off = fuse({"graph": graph}, top_k=10, discovery=False)
    # Both docs present, both with the same RRF (rank 1 vs rank 2).
    by_doc = {d.doc_id: d for d in fused_off}
    assert by_doc["d1"].chunks[0].score == 1.0 / (RRF_K + 1)
    assert by_doc["d2"].chunks[0].score == 1.0 / (RRF_K + 2)


def test_discovery_default_is_false() -> None:
    """fuse() called without discovery kwarg behaves as discovery=False.

    Defensive: if a future caller forgets to forward the param, fusion
    falls back to safe focus-mode behaviour, never accidentally amplifies.
    """
    graph = [FakeHit(chunk_id="c1", doc_id="d1", score=4.0)]
    fused_default = fuse({"graph": graph}, top_k=10)
    fused_explicit = fuse({"graph": graph}, top_k=10, discovery=False)
    assert fused_default[0].chunks[0].score == fused_explicit[0].chunks[0].score


# ---------------------------------------------------------------------------
# Discovery=True amplifies graph RRF
# ---------------------------------------------------------------------------


def test_discovery_true_amplifies_graph_rrf_by_surprise() -> None:
    """Graph hit at rank 1 with surprise=1.5: RRF = (1/61) * 1.5."""
    graph = [FakeHit(chunk_id="c1", doc_id="d1", score=1.5)]
    fused = fuse({"graph": graph}, top_k=10, discovery=True)
    expected = (1.0 / (RRF_K + 1)) * 1.5
    assert fused[0].chunks[0].score == expected


def test_discovery_true_caps_at_cap_constant() -> None:
    """A hit with surprise=4.0 caps to DISCOVERY_GRAPH_MULTIPLIER_CAP=2.0."""
    assert DISCOVERY_GRAPH_MULTIPLIER_CAP == 2.0  # if this changes, update the test
    graph = [FakeHit(chunk_id="c1", doc_id="d1", score=4.0)]
    fused = fuse({"graph": graph}, top_k=10, discovery=True)
    expected = (1.0 / (RRF_K + 1)) * DISCOVERY_GRAPH_MULTIPLIER_CAP
    assert fused[0].chunks[0].score == expected
    # NOT score * 4.0 -- that would be 0.0656 (over-domination).
    assert fused[0].chunks[0].score < (1.0 / (RRF_K + 1)) * 4.0


def test_discovery_true_multiplier_below_one_demotes() -> None:
    """surprise<1.0 demotes the hit BELOW its rank-driven RRF.

    Pre-fix design: surprise score is in [0.5, 4.0]; the bottom of the
    range exists to demote AMBIGUOUS-without-bonus edges. Discovery
    mode must respect that.
    """
    graph = [FakeHit(chunk_id="c1", doc_id="d1", score=0.5)]
    fused = fuse({"graph": graph}, top_k=10, discovery=True)
    expected = (1.0 / (RRF_K + 1)) * 0.5
    assert fused[0].chunks[0].score == expected
    assert fused[0].chunks[0].score < 1.0 / (RRF_K + 1)


# ---------------------------------------------------------------------------
# Other retrievers unaffected
# ---------------------------------------------------------------------------


def test_discovery_does_not_amplify_vector_or_bm25() -> None:
    """Vector and BM25 hits get flat RRF regardless of discovery flag.

    Their `score` field is similarity/relevance on a different scale;
    using it as an RRF multiplier would need per-retriever normalisation
    (deliberately out of scope for v1).
    """
    vec = [FakeHit(chunk_id="vc", doc_id="dv", score=0.8)]
    bm25 = [FakeHit(chunk_id="bc", doc_id="db", score=10.0)]

    fused_off = fuse({"vector": vec, "bm25": bm25}, top_k=10, discovery=False)
    fused_on = fuse({"vector": vec, "bm25": bm25}, top_k=10, discovery=True)

    by_off = {d.doc_id: d for d in fused_off}
    by_on = {d.doc_id: d for d in fused_on}
    # Identical chunk-level scores for vector/BM25 across discovery modes.
    assert by_off["dv"].chunks[0].score == by_on["dv"].chunks[0].score
    assert by_off["db"].chunks[0].score == by_on["db"].chunks[0].score


# ---------------------------------------------------------------------------
# Multiplier compounds, doesn't replace
# ---------------------------------------------------------------------------


def test_discovery_compounds_when_chunk_in_multiple_pools() -> None:
    """A chunk in BOTH graph (rank 1, surprise=2) AND vector (rank 1) gets
    graph slice multiplied, vector slice unchanged.

    Discovery on:
      total RRF = (1/61)*2.0  + 1/61
                = 2*(1/61) + 1/61
                = 3 * (1/61)

    Discovery off:
      total RRF = (1/61) + (1/61)
                = 2 * (1/61)

    The chunk gets a strictly higher fusion score with discovery on,
    matching the intent: if a chunk is BOTH a surprising bridge AND
    semantically relevant, surface it more.
    """
    chunk_id = "shared_chunk"
    graph = [FakeHit(chunk_id=chunk_id, doc_id="d", score=2.0)]
    vec = [FakeHit(chunk_id=chunk_id, doc_id="d", score=0.9)]

    fused_off = fuse({"graph": graph, "vector": vec}, top_k=10, discovery=False)
    fused_on = fuse({"graph": graph, "vector": vec}, top_k=10, discovery=True)

    rrf_off = fused_off[0].chunks[0].score
    rrf_on = fused_on[0].chunks[0].score

    # off: 2 * (1/61).
    assert abs(rrf_off - 2 * (1.0 / (RRF_K + 1))) < 1e-12
    # on: graph slice doubled, vector slice unchanged.
    assert abs(rrf_on - 3 * (1.0 / (RRF_K + 1))) < 1e-12
    assert rrf_on > rrf_off


# ---------------------------------------------------------------------------
# Metadata chunks are unaffected
# ---------------------------------------------------------------------------


def test_discovery_does_not_amplify_metadata_chunks() -> None:
    """metadata-kind chunks contribute flat RRF even under discovery=True.

    Metadata chunks are synthetic key:value text from ingestion; they
    don't carry surprise-score semantics. Multiplying them would distort
    fusion's metadata booster pathway.
    """
    chunk_id = "meta_c"
    graph_meta = [FakeHit(chunk_id=chunk_id, doc_id="d", score=4.0, kind="metadata")]
    # Add a content chunk so the doc isn't dropped (metadata-only docs
    # need a content_fallback to survive).
    content = [FakeHit(chunk_id="content_c", doc_id="d", score=1.0)]

    fused_off = fuse({"graph": graph_meta, "vector": content}, top_k=10, discovery=False)
    fused_on = fuse({"graph": graph_meta, "vector": content}, top_k=10, discovery=True)

    # Doc score includes metadata RRF folded in; with metadata-multiplier
    # disabled the doc score should be IDENTICAL across modes (modulo
    # float jitter from datetime.now() inside _apply_source_decay).
    # We compare doc.score directly because metadata chunks do not
    # appear in the FusedDocument.chunks list -- they fold into the
    # doc-level score booster.
    assert fused_off[0].score == pytest.approx(fused_on[0].score, rel=1e-9)


# ---------------------------------------------------------------------------
# Doc-aggregation amplification (architectural pin)
# ---------------------------------------------------------------------------


def test_discovery_doc_aggregation_many_chunks_can_beat_one_surprising() -> None:
    """Pin /plan-eng-review finding 1B: a doc with many moderately-
    surprising graph chunks can outrank a doc with one very-surprising
    chunk.

    This is INTENTIONAL behaviour of fusion's doc score formula:
        doc_score = best_chunk_rrf + alpha * sum(other_chunks_rrf) + metadata
    A doc with many surprising connections IS more relevant in
    expectation. But the trade-off is real and must not silently
    flip in a future fusion change.
    """
    # Doc A: 1 graph chunk, very surprising (surprise=2.0 -> hits cap).
    doc_a_chunks = [FakeHit(chunk_id="a1", doc_id="docA", score=2.0)]
    # Doc B: 5 graph chunks, moderately surprising (surprise=1.5 each).
    doc_b_chunks = [
        FakeHit(chunk_id=f"b{i}", doc_id="docB", score=1.5) for i in range(1, 6)
    ]
    graph = doc_a_chunks + doc_b_chunks  # docA at rank 1, docB rank 2..6

    fused = fuse({"graph": graph}, top_k=10, discovery=True)
    by_doc = {d.doc_id: d for d in fused}

    # Doc B has 5 chunks contributing -> aggregated doc score outranks
    # Doc A's single chunk despite A's higher per-chunk multiplier.
    assert by_doc["docB"].score > by_doc["docA"].score
    # And Doc B should rank first overall.
    assert fused[0].doc_id == "docB"


# ---------------------------------------------------------------------------
# Defensive: hit.score=None
# ---------------------------------------------------------------------------


def test_discovery_safe_when_hit_score_is_none() -> None:
    """If a graph hit somehow has score=None (legacy/synthetic path),
    discovery mode must not crash on `None * float`.

    Defaults to multiplier=1.0 (neutral) so the hit contributes baseline
    RRF, same as discovery=False would.
    """
    graph = [FakeHit(chunk_id="c1", doc_id="d1", score=None)]  # type: ignore[arg-type]
    # Should not raise.
    fused = fuse({"graph": graph}, top_k=10, discovery=True)
    expected = 1.0 / (RRF_K + 1)
    assert fused[0].chunks[0].score == expected
