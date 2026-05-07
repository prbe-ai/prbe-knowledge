"""Tests for splice_regenerated — surgical doc-level replacement."""

from __future__ import annotations

from datetime import UTC, datetime

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.regen import splice_regenerated


def _doc(doc_id: str, source: Source, text: str) -> SynthDoc:
    return SynthDoc(
        id=doc_id,
        source=source,
        source_event_id=doc_id,
        text=text,
        occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        channel="#incidents" if source == Source.SLACK else None,
        page_id=doc_id if source == Source.NOTION else None,
        thread_parent_id=None,
        scenario_id="scn-1",
        archetype="INCIDENT",
        personas=("gh:alice",),
        services_mentioned=("payments",),
        priority=10,
    )


def test_splice_replaces_only_named_doc_text() -> None:
    docs = (
        _doc("d0", Source.SENTRY, "sentry-orig"),
        _doc("d1", Source.SLACK, "slack-orig"),
        _doc("d2", Source.NOTION, "notion-orig"),
    )
    spliced = splice_regenerated(docs, regenerated_text_by_doc_id={"d1": "slack-NEW"})
    assert spliced[0].text == "sentry-orig"
    assert spliced[1].text == "slack-NEW"
    assert spliced[2].text == "notion-orig"


def test_splice_preserves_doc_count_and_order() -> None:
    docs = tuple(_doc(f"d{i}", Source.SLACK, f"orig-{i}") for i in range(5))
    spliced = splice_regenerated(
        docs,
        regenerated_text_by_doc_id={"d2": "fixed-2", "d4": "fixed-4"},
    )
    assert len(spliced) == 5
    assert tuple(d.id for d in spliced) == ("d0", "d1", "d2", "d3", "d4")
    assert spliced[2].text == "fixed-2"
    assert spliced[4].text == "fixed-4"


def test_splice_preserves_thread_parent_and_other_fields() -> None:
    docs = (
        _doc("d0", Source.SLACK, "parent-orig"),
        SynthDoc(
            id="d1",
            source=Source.SLACK,
            source_event_id="d1",
            text="reply-orig",
            occurred_at=datetime(2026, 4, 12, 14, 5, 0, tzinfo=UTC),
            channel="#incidents",
            page_id=None,
            thread_parent_id="d0",
            scenario_id="scn-1",
            archetype="INCIDENT",
            personas=("gh:bob",),
            services_mentioned=("payments",),
            priority=11,
        ),
    )
    spliced = splice_regenerated(docs, regenerated_text_by_doc_id={"d1": "reply-NEW"})
    assert spliced[1].thread_parent_id == "d0"
    assert spliced[1].source_event_id == "d1"
    assert spliced[1].channel == "#incidents"
    assert spliced[1].priority == 11
    assert spliced[1].text == "reply-NEW"


def test_splice_unknown_doc_id_raises() -> None:
    docs = (_doc("d0", Source.SLACK, "orig"),)
    try:
        splice_regenerated(docs, regenerated_text_by_doc_id={"unknown": "x"})
    except ValueError as e:
        assert "unknown" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_splice_empty_replacement_returns_original_tuple() -> None:
    docs = (_doc("d0", Source.SLACK, "orig"),)
    spliced = splice_regenerated(docs, regenerated_text_by_doc_id={})
    assert spliced == docs
