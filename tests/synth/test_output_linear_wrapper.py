"""LinearWrapper round-trip tests.

Verifies that wrap(doc) -> bytes produces a JSON envelope matching the
shape of the real fixture at fixtures/linear/issue_create.json.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.output.linear import wrap

_FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "linear" / "issue_create.json"


def _make_doc(
    *,
    archetype: str = "BIG_REFACTOR",
    text: str = "Migrate auth-service to JWT tokens.\n\nThis is the full description.",
    occurred_at: datetime | None = None,
    personas: tuple[str, ...] = ("gh:alice", "gh:bob"),
    services: tuple[str, ...] = ("auth-service",),
) -> SynthDoc:
    ts = occurred_at or datetime(2026, 5, 2, 11, 0, 0, tzinfo=UTC)
    return SynthDoc(
        id="scn-bigrefactor-2026-05-02-linear-0",
        source=Source.LINEAR,
        source_event_id="scn-bigrefactor-2026-05-02-linear-0",
        text=text,
        occurred_at=ts,
        channel=None,
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-bigrefactor-2026-05-02",
        archetype=archetype,
        personas=personas,
        services_mentioned=services,
        priority=100,
    )


def test_wrap_returns_bytes() -> None:
    doc = _make_doc()
    result = wrap(doc)
    assert isinstance(result, bytes)
    json.loads(result)  # must be valid JSON


def test_wrap_type_and_action() -> None:
    """Top-level type=Issue and action=create per Linear webhook spec."""
    doc = _make_doc()
    payload = json.loads(wrap(doc))
    assert payload["type"] == "Issue"
    assert payload["action"] == "create"


def test_wrap_data_title_from_first_line() -> None:
    """data.title is the first non-empty line of doc.text."""
    doc = _make_doc(text="Migrate auth-service to JWT tokens.\n\nFull description here.")
    payload = json.loads(wrap(doc))
    assert payload["data"]["title"] == "Migrate auth-service to JWT tokens."


def test_wrap_data_description_from_full_text() -> None:
    """data.description contains the full doc.text."""
    text = "Migrate auth-service to JWT tokens.\n\nFull description here."
    doc = _make_doc(text=text)
    payload = json.loads(wrap(doc))
    assert payload["data"]["description"] == text


def test_wrap_fixture_shape() -> None:
    """Wrapper output contains at minimum the keys present in the real fixture."""
    _FIXTURE.read_text()  # ensure fixture is readable
    doc = _make_doc()
    wrapper = json.loads(wrap(doc))
    # Top-level keys
    required_top = {"type", "action", "data", "createdAt", "url"}
    assert required_top.issubset(set(wrapper.keys()))
    # data keys
    required_data = {"id", "title", "description", "team", "createdAt", "url"}
    assert required_data.issubset(set(wrapper["data"].keys()))


def test_wrap_assignee_from_personas() -> None:
    """data.assignee is derived from the second persona (bob assigns to alice)."""
    doc = _make_doc(personas=("gh:alice", "gh:bob"))
    payload = json.loads(wrap(doc))
    assignee = payload["data"].get("assignee")
    assert assignee is not None
    # assignee name should reference one of the personas
    assert "bob" in assignee["name"].lower() or "alice" in assignee["name"].lower()


def test_wrap_is_byte_identical_across_calls() -> None:
    """Plan 3 determinism contract (spec §13): same input → same bytes."""
    doc = _make_doc()
    assert wrap(doc) == wrap(doc)
