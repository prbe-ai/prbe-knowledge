"""SlackWrapper round-trip tests.

The wrapper must produce a byte payload that SlackConnector.parse_webhook_event
accepts. We test the parse_hint fields directly without importing the full
connector (which has heavy deps) — instead we parse the JSON ourselves and
assert the shape matches what the connector expects.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.output.slack import wrap

_FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "slack" / "message_simple.json"


def _make_slack_doc(
    *,
    channel: str = "#standup",
    text: str = "Yesterday: shipped payments. Today: auth - fix retry. Blockers: none.",
    thread_parent_id: str | None = None,
    occurred_at: datetime | None = None,
) -> SynthDoc:
    ts = occurred_at or datetime(2026, 5, 1, 9, 0, 0, tzinfo=UTC)
    return SynthDoc(
        id="scn-standup-gh-alice-2026-05-01-slack-0",
        source=Source.SLACK,
        source_event_id="scn-standup-gh-alice-2026-05-01-slack-0",
        text=text,
        occurred_at=ts,
        channel=channel,
        page_id=None,
        thread_parent_id=thread_parent_id,
        scenario_id="scn-standup-gh-alice-2026-05-01",
        archetype="STANDUP_UPDATE",
        personas=("gh:alice",),
        services_mentioned=("payments", "auth"),
        priority=100,
    )


def test_wrap_produces_valid_json() -> None:
    doc = _make_slack_doc()
    raw = wrap(doc)
    payload = json.loads(raw)
    assert payload["type"] == "event_callback"
    assert payload["event"]["type"] == "message"


def test_wrap_recovers_channel_text_ts() -> None:
    """Parsed envelope yields the same channel, text, and ts the doc had."""
    doc = _make_slack_doc(channel="#standup", text="hello world")
    payload = json.loads(wrap(doc))
    event = payload["event"]
    assert event["channel"] == "#standup"
    assert event["text"] == "hello world"
    expected_ts = f"{int(doc.occurred_at.timestamp())}.000000"
    assert event["ts"] == expected_ts


def test_wrap_thread_ts_present_when_reply() -> None:
    parent_id = "scn-oncall-2026-05-05-slack-0"
    doc = _make_slack_doc(thread_parent_id=parent_id)
    payload = json.loads(wrap(doc))
    assert "thread_ts" in payload["event"]


def test_wrap_no_thread_ts_for_root_message() -> None:
    doc = _make_slack_doc(thread_parent_id=None)
    payload = json.loads(wrap(doc))
    assert "thread_ts" not in payload["event"]


def test_fixture_shape_matches_wrapper_shape() -> None:
    """Wrapper output has the same top-level keys as the real fixture."""
    fixture = json.loads(_FIXTURE.read_text())
    doc = _make_slack_doc()
    wrapper_payload = json.loads(wrap(doc))
    fixture_keys = set(fixture.keys())
    wrapper_keys = set(wrapper_payload.keys())
    assert {"type", "event", "team_id"}.issubset(wrapper_keys)
    assert {"channel", "text", "ts", "type", "user"}.issubset(set(wrapper_payload["event"].keys()))
    # Sanity: the real fixture also has the expected top-level keys (verifies
    # our minimum-required-keys list is grounded in the actual connector input).
    assert {"type", "event", "team_id"}.issubset(fixture_keys)
    assert {"channel", "text", "ts", "type", "user"}.issubset(set(fixture["event"].keys()))
