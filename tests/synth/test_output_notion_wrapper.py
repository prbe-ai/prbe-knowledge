"""NotionWrapper round-trip tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.output.notion import wrap

_FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "notion" / "page_updated.json"


def _make_notion_doc(
    *,
    text: str = "## Incident: payments 500s\nOwner: @alice\nStatus: resolved",
    page_section: str = "Engineering > On-call rotation",
    occurred_at: datetime | None = None,
) -> SynthDoc:
    ts = occurred_at or datetime(2026, 5, 5, 9, 0, 0, tzinfo=UTC)
    return SynthDoc(
        id="scn-oncall-2026-05-05-notion-0",
        source=Source.NOTION,
        source_event_id="scn-oncall-2026-05-05-notion-0",
        text=text,
        occurred_at=ts,
        channel=None,
        page_id="page-scn-oncall-2026-05-05-notion-0",
        thread_parent_id=None,
        scenario_id="scn-oncall-2026-05-05",
        archetype="ON_CALL_HANDOFF",
        personas=("gh:alice", "gh:bob"),
        services_mentioned=("payments",),
        priority=100,
    )


def test_wrap_produces_valid_json() -> None:
    doc = _make_notion_doc()
    raw = wrap(doc)
    payload = json.loads(raw)
    assert payload["type"] == "page.updated"


def test_wrap_entity_shape() -> None:
    """Connector reads entity.type and entity.id."""
    doc = _make_notion_doc()
    payload = json.loads(wrap(doc))
    assert payload["entity"]["type"] == "page"
    assert payload["entity"]["id"] == doc.page_id


def test_wrap_data_has_last_edited_time() -> None:
    """Connector reads data.last_edited_time for source_event_id construction."""
    doc = _make_notion_doc(occurred_at=datetime(2026, 5, 5, 9, 0, 0, tzinfo=UTC))
    payload = json.loads(wrap(doc))
    assert "last_edited_time" in payload["data"]
    let = payload["data"]["last_edited_time"]
    assert "2026-05-05" in let


def test_fixture_shape_matches_wrapper_top_level_keys() -> None:
    """Wrapper top-level keys are a superset of the fixture's required keys."""
    fixture = json.loads(_FIXTURE.read_text())
    doc = _make_notion_doc()
    wrapper = json.loads(wrap(doc))
    required = {"type", "entity", "data", "workspace_id"}
    assert required.issubset(set(wrapper.keys()))
    assert {"type", "id"}.issubset(set(wrapper["entity"].keys()))
    # Sanity: the real fixture also satisfies these required keys.
    assert required.issubset(set(fixture.keys()))


def test_wrap_inlines_properties_on_entity_for_synth_bypass() -> None:
    """Synth-only bypass: entity carries properties.title so the prod handler
    can read the title without a live Notion API hydration call."""
    doc = _make_notion_doc(text="# My Heading\n\nFirst paragraph body.")
    payload = json.loads(wrap(doc))
    title = payload["entity"]["properties"]["title"]["title"][0]["plain_text"]
    assert title == "My Heading"


def test_wrap_inlines_body_markdown_on_entity_for_synth_bypass() -> None:
    """Synth-only bypass: entity carries pre-rendered body_markdown so the
    prod handler can populate document body without Notion API hydration."""
    doc = _make_notion_doc(text="## Section\n\nSome paragraph content.")
    payload = json.loads(wrap(doc))
    body_md = payload["entity"]["body_markdown"]
    assert "## Section" in body_md
    assert "Some paragraph content." in body_md
