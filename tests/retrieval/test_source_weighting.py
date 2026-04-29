"""Unit tests for per-source-system weighting in fuse().

Two knobs covered:
  - SOURCE_SCORE_MULTIPLIERS — post-RRF score multiplier per source.
  - SOURCE_HALF_LIFE_DAYS — per-source recency half-life override.

Multiplier applies first, then recency decay. Per-source half-life wins
over the caller's global half_life_days, and applies even when the
global is None — the whole point is that claude_code decays in days
without forcing other sources to decay too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from services.retrieval.fusion import fuse
from shared.constants import SourceSystem

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


def test_non_claude_code_source_no_decay_when_global_none() -> None:
    """A 7-day-old Slack hit gets full RRF score when no global half-life
    is set — proves the per-source dict isolates CC from other sources."""
    sl = FakeHit(
        chunk_id="sl",
        doc_id="d-sl",
        source_system=SourceSystem.SLACK.value,
        updated_at=_NOW - timedelta(days=7),
    )
    fused = fuse({"vector": [sl]}, top_k=10, now=_NOW)
    assert abs(fused[0].score - 1.0 / 61) < 1e-9


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


def test_unknown_source_system_uses_defaults() -> None:
    """Forward-compat: an unknown source_system gets multiplier=1.0 and
    falls back to the caller's global half-life (or no decay if None)."""
    unknown = FakeHit(
        chunk_id="u",
        doc_id="d-u",
        source_system="unmapped_source",
        updated_at=_NOW - timedelta(days=14),
    )
    # Global half-life unset → no decay, no multiplier → raw RRF.
    fused = fuse({"vector": [unknown]}, top_k=10, now=_NOW)
    assert abs(fused[0].score - 1.0 / 61) < 1e-9

    # Caller-provided global half-life applies to unknown sources.
    fused2 = fuse(
        {"vector": [unknown]},
        top_k=10,
        recency_half_life_days=14,
        now=_NOW,
    )
    # Age 14d, half-life 14d → exactly half.
    assert abs(fused2[0].score - 0.5 * (1.0 / 61)) < 1e-9
