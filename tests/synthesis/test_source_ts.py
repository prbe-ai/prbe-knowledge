"""Unit tests for `extract_source_ts(doc)`.

Covers each connector's metadata key dispatch and the
`documents.created_at` fallback for malformed / missing timestamps.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from services.synthesis.source_ts import extract_source_ts
from shared.constants import SourceSystem


def _doc(
    source_system: SourceSystem,
    metadata: dict | None = None,
    *,
    created_at: datetime | None = None,
) -> SimpleNamespace:
    """Build a minimal doc-shaped object for the extractor."""
    return SimpleNamespace(
        source_system=source_system,
        metadata=metadata or {},
        created_at=created_at or datetime(2026, 1, 1, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Per-source extraction
# ---------------------------------------------------------------------------


def test_slack_ts_string_to_datetime() -> None:
    doc = _doc(SourceSystem.SLACK, {"ts": "1717000000.123456"})
    out = extract_source_ts(doc)
    assert out == datetime.fromtimestamp(1717000000.123456, tz=UTC)


def test_github_created_at_to_datetime() -> None:
    doc = _doc(SourceSystem.GITHUB, {"created_at": "2026-05-04T09:00:00Z"})
    out = extract_source_ts(doc)
    assert out == datetime(2026, 5, 4, 9, 0, 0, tzinfo=UTC)


def test_linear_updated_at_to_datetime() -> None:
    doc = _doc(SourceSystem.LINEAR, {"updatedAt": "2026-05-04T10:30:00Z"})
    out = extract_source_ts(doc)
    assert out == datetime(2026, 5, 4, 10, 30, 0, tzinfo=UTC)


def test_granola_started_at_to_datetime() -> None:
    doc = _doc(SourceSystem.GRANOLA, {"startedAt": "2026-05-04T14:00:00+00:00"})
    out = extract_source_ts(doc)
    assert out == datetime(2026, 5, 4, 14, 0, 0, tzinfo=UTC)


def test_notion_last_edited_time_to_datetime() -> None:
    doc = _doc(SourceSystem.NOTION, {"last_edited_time": "2026-05-04T16:45:00Z"})
    out = extract_source_ts(doc)
    assert out == datetime(2026, 5, 4, 16, 45, 0, tzinfo=UTC)


def test_claude_code_falls_back_to_documents_created_at() -> None:
    """claude_code transcripts don't expose a stable per-message ts in
    metadata at the doc level; the connector parses sessions independently
    so we use documents.created_at as the agent-visible timestamp."""
    created = datetime(2026, 5, 4, 9, 30, 0, tzinfo=UTC)
    doc = _doc(SourceSystem.CLAUDE_CODE, {}, created_at=created)
    out = extract_source_ts(doc)
    assert out == created


def test_malformed_metadata_falls_back_to_created_at() -> None:
    """A malformed timestamp string in the metadata should fall back to
    documents.created_at rather than crash the queue insert. Slack ts of
    'banana' has no float coercion."""
    created = datetime(2026, 5, 4, 9, 30, 0, tzinfo=UTC)
    doc = _doc(SourceSystem.SLACK, {"ts": "banana"}, created_at=created)
    out = extract_source_ts(doc)
    assert out == created
