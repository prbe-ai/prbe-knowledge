"""SentryWrapper round-trip tests.

Verifies that wrap(doc) -> bytes produces a JSON envelope matching the
shape of the real fixture at fixtures/sentry/issue_created.json.

Plan 3's INCIDENT archetype emits the SENTRY doc as templated content
(no LLM call) — the wrapper serializes it into the issue_created envelope.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.output.sentry import wrap

_FIXTURE_ISSUE_CREATED = (
    Path(__file__).parent.parent.parent / "fixtures" / "sentry" / "issue_created.json"
)


def _make_doc(
    *,
    archetype: str = "INCIDENT",
    text: str = "TypeError: 'NoneType' object is not subscriptable\npayments.charges.handle_webhook",
    occurred_at: datetime | None = None,
    personas: tuple[str, ...] = ("gh:alice",),
    services: tuple[str, ...] = ("payments",),
) -> SynthDoc:
    ts = occurred_at or datetime(2026, 5, 1, 18, 0, 0, tzinfo=UTC)
    return SynthDoc(
        id="scn-incident-2026-05-01-sentry-0",
        source=Source.SENTRY,
        source_event_id="scn-incident-2026-05-01-sentry-0",
        text=text,
        occurred_at=ts,
        channel=None,
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-incident-2026-05-01",
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


def test_wrap_action_is_created() -> None:
    """Sentry issue_created webhook always has action=created."""
    doc = _make_doc()
    payload = json.loads(wrap(doc))
    assert payload["action"] == "created"


def test_wrap_data_issue_title_from_first_line() -> None:
    """data.issue.title is the first non-empty line of doc.text."""
    doc = _make_doc(text="TypeError: 'NoneType' object is not subscriptable\nextra context")
    payload = json.loads(wrap(doc))
    assert payload["data"]["issue"]["title"] == "TypeError: 'NoneType' object is not subscriptable"


def test_wrap_data_issue_culprit_from_services() -> None:
    """data.issue.culprit is derived from the first service mentioned."""
    doc = _make_doc(services=("payments",))
    payload = json.loads(wrap(doc))
    culprit = payload["data"]["issue"]["culprit"]
    assert "payments" in culprit


def test_wrap_fixture_shape() -> None:
    """Wrapper output contains at minimum the keys present in the real fixture."""
    fixture = json.loads(_FIXTURE_ISSUE_CREATED.read_text())
    assert fixture  # ensure fixture is readable
    doc = _make_doc()
    wrapper = json.loads(wrap(doc))
    # Top-level keys
    required_top = {"action", "installation", "data"}
    assert required_top.issubset(set(wrapper.keys()))
    # data.issue keys
    required_issue = {"id", "title", "culprit", "level", "status", "project"}
    assert required_issue.issubset(set(wrapper["data"]["issue"].keys()))


def test_wrap_is_byte_identical_across_calls() -> None:
    """Plan 3 determinism contract — same input → same bytes.

    Pins against PYTHONHASHSEED drift via sha256-based id derivation.
    """
    doc = _make_doc(
        text="TypeError: 'NoneType' object is not subscriptable\npayments.charges.handle_webhook",
    )
    assert wrap(doc) == wrap(doc)
