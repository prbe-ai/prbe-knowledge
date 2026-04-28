"""Unit tests for the entity-must-match filter — pure function, no DB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from services.retrieval.helpers import apply_entity_filter as _apply_entity_filter
from services.retrieval.router import RouterEntity

_NOW = datetime(2026, 4, 24, tzinfo=UTC)


@dataclass
class FakeFused:
    chunk_id: str
    doc_id: str = "d"
    doc_version: int = 1
    source_system: str = "slack"
    source_url: str = "u"
    title: str | None = None
    content: str = ""
    created_at: datetime = _NOW
    updated_at: datetime = _NOW
    score: float = 0.0


def _entity(canonical_id: str, display_name: str, conf: float = 0.85) -> RouterEntity:
    return RouterEntity(
        entity_type="service",
        canonical_id=canonical_id,
        display_name=display_name,
        confidence=conf,
    )


def test_filter_drops_chunks_without_entity() -> None:
    hits = [
        FakeFused(chunk_id="bad", content="hello whats going on lads"),
        FakeFused(chunk_id="good", content="add Klavis integrations to MCP"),
    ]
    entities = [_entity("klavis", "Klavis", conf=0.9)]
    out, info = _apply_entity_filter(hits, entities, threshold=0.7)
    assert [h.chunk_id for h in out] == ["good"]
    assert info["enabled"] is True
    assert info["threshold"] == 0.7
    assert "klavis" in info["needles"]


def test_filter_matches_on_title_too() -> None:
    hits = [
        FakeFused(
            chunk_id="title-match",
            content="execution works end-to-end",  # no entity in content
            title="Migrate tool execution to MCP + add Klavis integrations",
        ),
    ]
    out, _ = _apply_entity_filter(hits, [_entity("klavis", "Klavis")], threshold=0.7)
    assert len(out) == 1


def test_filter_case_insensitive() -> None:
    hits = [FakeFused(chunk_id="c", content="KLAVIS shipped today")]
    out, _ = _apply_entity_filter(hits, [_entity("klavis", "klavis")], threshold=0.7)
    assert len(out) == 1


def test_filter_skips_when_no_entities_above_threshold() -> None:
    """Threshold=0.9 + low-confidence entities → filter is a no-op."""
    hits = [
        FakeFused(chunk_id="anything", content="completely unrelated text"),
    ]
    entities = [_entity("klavis", "Klavis", conf=0.4)]  # below threshold
    out, info = _apply_entity_filter(hits, entities, threshold=0.7)
    assert len(out) == 1  # no filtering happened
    assert "skipped" in info
    assert info["needles"] == []


def test_filter_threshold_lower_includes_more_entities() -> None:
    """At threshold=0.5, a 0.6-confidence entity qualifies."""
    hits = [
        FakeFused(chunk_id="match", content="we shipped klavis"),
        FakeFused(chunk_id="miss", content="something else"),
    ]
    entities = [_entity("klavis", "Klavis", conf=0.6)]
    out, info = _apply_entity_filter(hits, entities, threshold=0.5)
    assert [h.chunk_id for h in out] == ["match"]
    assert info["threshold"] == 0.5
    # Same entities at threshold=0.7 → no qualifying entities, no filter.
    out_strict, info_strict = _apply_entity_filter(hits, entities, threshold=0.7)
    assert len(out_strict) == 2  # filter no-op
    assert "skipped" in info_strict


def test_filter_uses_both_canonical_id_and_display_name() -> None:
    """display_name and canonical_id are both checked as needles."""
    hits = [
        FakeFused(chunk_id="canonical-only", content="reference to klavis-mcp here"),
        FakeFused(chunk_id="display-only", content="see The Klavis Project for details"),
    ]
    entities = [_entity("klavis-mcp", "The Klavis Project", conf=0.9)]
    out, _ = _apply_entity_filter(hits, entities, threshold=0.7)
    assert {h.chunk_id for h in out} == {"canonical-only", "display-only"}


def test_filter_dedupes_needles_when_canonical_equals_display() -> None:
    out, info = _apply_entity_filter([], [_entity("klavis", "klavis", conf=0.9)], threshold=0.7)
    assert info["needles"] == ["klavis"]  # single needle, not duplicated
    assert out == []


def test_filter_passes_chunks_by_source_system_for_platform_needles() -> None:
    """'what happened in slack recently?' should keep Slack messages even
    though their text doesn't contain the word 'slack'."""
    slack_msg = FakeFused(
        chunk_id="slack-1",
        content="hey team, deploying at 3pm",
        source_system="slack",
    )
    github_pr = FakeFused(
        chunk_id="gh-1",
        content="fix slack integration retry logic",
        source_system="github",
    )
    notion_page = FakeFused(
        chunk_id="notion-1",
        content="Q1 planning notes",
        source_system="notion",
    )
    out, info = _apply_entity_filter(
        [slack_msg, github_pr, notion_page],
        [
            RouterEntity(
                entity_type="channel",
                canonical_id="slack",
                display_name="Slack",
                confidence=0.85,
            )
        ],
        threshold=0.7,
    )
    # Slack message passes via source_system; GitHub PR passes via text match;
    # Notion page is dropped (no text or source match).
    kept = {h.chunk_id for h in out}
    assert kept == {"slack-1", "gh-1"}
    assert "slack" in info["source_needles"]


def test_filter_source_match_only_for_known_platforms() -> None:
    """A needle that happens to match a chunk's source_system value but
    isn't one of our known sources doesn't trigger the source-match branch.
    """
    chunk = FakeFused(chunk_id="x", content="totally unrelated", source_system="custom")
    out, _ = _apply_entity_filter(
        [chunk],
        [_entity("custom", "Custom", conf=0.9)],
        threshold=0.7,
    )
    # "custom" is not a known SourceSystem, so source_system match doesn't
    # rescue this chunk; text doesn't contain "custom" either → dropped.
    assert out == []
