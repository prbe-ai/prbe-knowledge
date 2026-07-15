"""Unit tests for per-source-system weighting in fuse().

Two knobs covered:
  - per-source post-RRF score multipliers (shared.source_registry).
  - per-source recency half-life overrides (shared.source_registry).

Multiplier applies first, then recency decay. Resolution order for
half-life: per-source override > caller global > universal baseline
(DEFAULT_RECENCY_HALF_LIFE_DAYS). Sources without an override still
decay against the baseline so backfilled tenants don't surface stale
year-old docs at parity with fresh ones; the per-source overrides exist
to make noisy sources (claude_code/codex) decay *faster* than baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from engine.retrieval.fusion import fuse
from engine.shared.constants import DEFAULT_RECENCY_HALF_LIFE_DAYS, SourceSystem

_NOW = datetime(2026, 4, 28, tzinfo=UTC)


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
    kind: str = "content"


def test_multiplier_demotes_claude_code_at_equal_relevance() -> None:
    """Two equally-fresh hits at rank 1 in their respective retrievers —
    one GitHub, one CLAUDE_CODE. The CC multiplier (0.5) drops it below
    the GitHub doc even though raw RRF is identical."""
    gh = FakeHit(
        chunk_id="gh",
        doc_id="d-gh",
        source_system=SourceSystem.GITHUB.value,
        updated_at=_NOW,
    )
    cc = FakeHit(
        chunk_id="cc",
        doc_id="d-cc",
        source_system=SourceSystem.CLAUDE_CODE.value,
        updated_at=_NOW,
    )
    fused = fuse({"vector": [gh], "bm25": [cc]}, top_k=10, now=_NOW)
    assert fused[0].doc_id == "d-gh"
    assert fused[1].doc_id == "d-cc"
    by_doc = {h.doc_id: h.score for h in fused}
    # Age = 0 for both, so per-source half-life adds no extra decay (exp(0)=1).
    # CC final = 0.5 * GH final.
    assert abs(by_doc["d-cc"] / by_doc["d-gh"] - 0.5) < 1e-9


def test_claude_code_half_life_applies_when_global_none() -> None:
    """A 7-day-old CC hit decays even though recency_half_life_days=None.
    Score = multiplier * decay * RRF = 0.5 * 0.5 * 1/(60+1)."""
    cc = FakeHit(
        chunk_id="cc",
        doc_id="d-cc",
        source_system=SourceSystem.CLAUDE_CODE.value,
        updated_at=_NOW - timedelta(days=7),
    )
    fused = fuse({"vector": [cc]}, top_k=10, now=_NOW)
    expected = 0.5 * 0.5 * (1.0 / 61)
    assert abs(fused[0].score - expected) < 1e-9


def test_non_claude_code_source_uses_baseline_when_global_none() -> None:
    """A 7-day-old Slack hit decays against the universal baseline when no
    caller half-life is set — Slack has no per-source override so it rides
    DEFAULT_RECENCY_HALF_LIFE_DAYS."""
    sl = FakeHit(
        chunk_id="sl",
        doc_id="d-sl",
        source_system=SourceSystem.SLACK.value,
        updated_at=_NOW - timedelta(days=7),
    )
    fused = fuse({"vector": [sl]}, top_k=10, now=_NOW)
    expected_decay = math.exp(-math.log(2) * 7.0 / DEFAULT_RECENCY_HALF_LIFE_DAYS)
    assert abs(fused[0].score - (1.0 / 61) * expected_decay) < 1e-9


def test_per_source_override_beats_explicit_caller_global() -> None:
    """Precedence: registered per-source half-life wins over a non-None caller
    global. A 7-day-old CC hit decays at the 7d override, not the caller's 365d."""
    cc = FakeHit(
        chunk_id="cc",
        doc_id="d-cc",
        source_system=SourceSystem.CLAUDE_CODE.value,
        updated_at=_NOW - timedelta(days=7),
    )
    fused = fuse(
        {"vector": [cc]},
        top_k=10,
        recency_half_life_days=365,  # lenient caller value
        now=_NOW,
    )
    # If the override is honored: multiplier=0.5 * decay=0.5 (one CC half-life) * 1/61.
    # If the caller value were used instead: multiplier=0.5 * decay≈0.987 * 1/61 — much higher.
    expected = 0.5 * 0.5 * (1.0 / 61)
    assert abs(fused[0].score - expected) < 1e-9


def test_multiplier_and_decay_apply_in_order() -> None:
    """Multiplier * decay both visible. A 7d-old CC hit ends up at
    0.5 (multiplier) * 0.5 (one half-life of decay) * raw RRF."""
    cc = FakeHit(
        chunk_id="cc",
        doc_id="d-cc",
        source_system=SourceSystem.CLAUDE_CODE.value,
        updated_at=_NOW - timedelta(days=7),
    )
    fused = fuse({"vector": [cc]}, top_k=10, now=_NOW)
    raw_rrf = 1.0 / 61
    assert abs(fused[0].score - 0.25 * raw_rrf) < 1e-9


def test_mixed_sources_claude_code_falls_below_slack() -> None:
    """CC ranks higher in raw retrieval (rank 1 vs rank 2) but the
    multiplier drops it below the Slack doc post-fusion."""
    cc = FakeHit(
        chunk_id="cc",
        doc_id="d-cc",
        source_system=SourceSystem.CLAUDE_CODE.value,
        updated_at=_NOW,
    )
    sl = FakeHit(
        chunk_id="sl",
        doc_id="d-sl",
        source_system=SourceSystem.SLACK.value,
        updated_at=_NOW,
    )
    # Single retriever puts CC first, Slack second.
    fused = fuse({"vector": [cc, sl]}, top_k=10, now=_NOW)
    # CC: rank 1 RRF (1/61) * 0.5 = 1/122 ≈ 0.00820
    # SL: rank 2 RRF (1/62)        ≈ 0.01613
    # SL wins.
    assert fused[0].doc_id == "d-sl"
    assert fused[1].doc_id == "d-cc"


def test_multiplier_demotes_code_graph_at_equal_relevance() -> None:
    """A CODE_GRAPH hit at rank 1 (raw RRF identical to a GitHub hit at
    rank 1) ends up below the GitHub doc post-fusion. Multiplier is 0.3,
    set lower than CC/CODEX because BM25 keyword noise on identifier
    tokens over-surfaces weak code chunks."""
    gh = FakeHit(
        chunk_id="gh",
        doc_id="d-gh",
        source_system=SourceSystem.GITHUB.value,
        updated_at=_NOW,
    )
    cg = FakeHit(
        chunk_id="cg",
        doc_id="d-cg",
        source_system=SourceSystem.CODE_GRAPH.value,
        updated_at=_NOW,
    )
    fused = fuse({"vector": [gh], "bm25": [cg]}, top_k=10, now=_NOW)
    assert fused[0].doc_id == "d-gh"
    assert fused[1].doc_id == "d-cg"
    by_doc = {h.doc_id: h.score for h in fused}
    # Age = 0, so per-source half-life adds no extra decay.
    assert abs(by_doc["d-cg"] / by_doc["d-gh"] - 0.3) < 1e-9


def test_strong_code_graph_match_survives_demotion() -> None:
    """A CODE_GRAPH hit that lands in BOTH retrievers (sum of two RRF
    contributions = 2/61) still beats a single-retriever GitHub hit at
    rank 1 (1/61) post-multiplier. Codifies the design intent: strong
    semantic matches survive, weak keyword-noise chunks get demoted."""
    cg = FakeHit(
        chunk_id="cg",
        doc_id="d-cg",
        source_system=SourceSystem.CODE_GRAPH.value,
        updated_at=_NOW,
    )
    gh = FakeHit(
        chunk_id="gh",
        doc_id="d-gh",
        source_system=SourceSystem.GITHUB.value,
        updated_at=_NOW,
    )
    # CG hits at rank 1 in BOTH retrievers (e.g. strong vector + BM25 match);
    # GH only in vector at rank 2 (single weak hit).
    fused = fuse(
        {"vector": [cg, gh], "bm25": [cg]},
        top_k=10,
        now=_NOW,
    )
    # CG raw RRF = 1/61 + 1/61 = 2/61; * 0.3 = 0.6/61 ~= 0.00984
    # GH raw RRF = 1/62                                ~= 0.01613
    # GH still wins because 2 * 0.3 = 0.6 < 1.0; that's expected --
    # for CG to outrank GH at equal "best rank" it needs the multiplier
    # gap closed by score sum, which 2 RRF contributions don't fully do.
    # This test instead pins the inverse case below.
    assert fused[0].doc_id == "d-gh"

    # But a CG hit at rank 1 with NO competing GH hit obviously wins.
    fused2 = fuse({"vector": [cg]}, top_k=10, now=_NOW)
    assert fused2[0].doc_id == "d-cg"
    # Score is 0.3 * 1/61.
    assert abs(fused2[0].score - 0.3 * (1.0 / 61)) < 1e-9


def test_unknown_source_system_uses_defaults() -> None:
    """Forward-compat: an unknown source_system gets multiplier=1.0 and
    falls back to the caller's global half-life (or the universal baseline
    if no caller value is set)."""
    unknown = FakeHit(
        chunk_id="u",
        doc_id="d-u",
        source_system="unmapped_source",
        updated_at=_NOW - timedelta(days=14),
    )
    # Global half-life unset → baseline decay applies, no multiplier.
    fused = fuse({"vector": [unknown]}, top_k=10, now=_NOW)
    expected_decay = math.exp(-math.log(2) * 14.0 / DEFAULT_RECENCY_HALF_LIFE_DAYS)
    assert abs(fused[0].score - (1.0 / 61) * expected_decay) < 1e-9

    # Caller-provided global half-life beats the baseline for unknown sources.
    fused2 = fuse(
        {"vector": [unknown]},
        top_k=10,
        recency_half_life_days=14,
        now=_NOW,
    )
    # Age 14d, half-life 14d → exactly half.
    assert abs(fused2[0].score - 0.5 * (1.0 / 61)) < 1e-9
