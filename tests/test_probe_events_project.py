"""Projecting a stored session event onto probe-events/1.

The fixtures under tests/fixtures/probe_events/ are COPIES of research-os
`agent/tests/fixtures/probe_events/<harness>/expected.jsonl` (b5282848d, tap
0.9.11): what the four tap sanitizers upload today. Projection must leave them
exactly as they are, and must strip what taps before 0.9.10 sent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from engine.ingest.probe_events import project_event
from engine.shared.transcript_render import lines_from_events
from scripts.strip_session_payloads import strip_batch

FIXTURES = Path(__file__).parent / "fixtures" / "probe_events"
GOLDENS = sorted(FIXTURES.glob("*.expected.jsonl"))


def _events(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        value = json.loads(line)
        for event in value if isinstance(value, list) else [value] if value else []:
            out.append(event)
    return out


@pytest.mark.parametrize("path", GOLDENS, ids=lambda p: p.name)
def test_what_tap_0_9_11_uploads_is_left_exactly_as_it_is(path: Path) -> None:
    for event in _events(path):
        assert project_event(event) == event


# What a Claude Code tap before 0.9.10 stored: a deny-list let all of this through.
OLD_CC_EVENT = {
    "type": "user",
    "uuid": "u1",
    "sessionId": "s1",
    "cwd": "/w",
    "gitBranch": "main",
    "version": "2.0.1",
    "toolUseResult": {"stdout": "TOOL-OUTPUT-9b2c", "file": {"content": "FILE-CONTENT-7f3a"}},
    "message": {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "is_error": True, "result_bytes": 9,
             "content": "TOOL-OUTPUT-9b2c"},
            {"type": "text", "text": "and here"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": "iVBORw0KGgo" * 10}},
            {"type": "server_tool_use", "input": {"query": "SEARCH-RESULT-2e8d"}},
        ],
    },
}


def test_an_old_event_keeps_only_what_probe_events_1_names() -> None:
    out = project_event(OLD_CC_EVENT)
    dumped = json.dumps(out)
    for planted in ("TOOL-OUTPUT-9b2c", "FILE-CONTENT-7f3a", "SEARCH-RESULT-2e8d", "iVBORw0KGgo"):
        assert planted not in dumped
    assert set(out) == {"type", "uuid", "message"}
    tool_result, text, image, unknown = out["message"]["content"]
    assert tool_result == {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                           "result_bytes": 9}
    assert text == {"type": "text", "text": "and here"}
    assert image == {"type": "image", "mimeType": "image/png", "bytes": 110}
    assert unknown == {"type": "server_tool_use", "dropped": True}


def test_a_tool_call_keeps_its_summary_and_stats_never_its_input() -> None:
    event = {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "Edit", "summary": "a.py",
         "input": {"old_string": "x", "new_string": "SECRET-BODY"},
         "stats": {"added_lines": 3, "removed_lines": 1}}]}}
    [block] = project_event(event)["message"]["content"]
    assert block == {"type": "tool_use", "id": "t1", "name": "Edit", "summary": "a.py",
                     "stats": {"added_lines": 3, "removed_lines": 1}}


def test_an_attachment_keeps_its_type_and_only_a_file_reference_its_name() -> None:
    held = {"type": "attachment", "attachment": {"type": "file", "filename": "a.py",
                                                 "content": "FILE-CONTENT-7f3a"}}
    ref = {"type": "attachment", "attachment": {"type": "compact_file_reference",
                                                "filename": "a.py", "displayPath": "src/a.py",
                                                "content": "FILE-CONTENT-7f3a"}}
    assert project_event(held)["attachment"] == {"type": "file"}
    assert project_event(ref)["attachment"] == {"type": "compact_file_reference",
                                                "filename": "a.py", "displayPath": "src/a.py"}


def test_codex_loses_its_raw_shell_action_and_keeps_named_extras() -> None:
    event = {"type": "assistant", "_codex_extras": {
        "turn_id": "t", "action": {"command": ["bash", "-lc", "cat .env"], "env": {"K": "v"}}}}
    assert project_event(event)["_codex_extras"] == {"turn_id": "t"}


def test_projection_is_idempotent() -> None:
    once = project_event(OLD_CC_EVENT)
    assert project_event(once) == once


def _batch(events: list[dict[str, Any]]) -> bytes:
    return json.dumps({"payload": {"session_id": "s", "batch_seq": 0, "events": [
        {"line_no": i, "raw": raw} for i, raw in enumerate(events)]}}).encode()


def test_a_stripped_batch_renders_exactly_as_before() -> None:
    body = _batch([OLD_CC_EVENT, {"type": "assistant", "message": {"role": "assistant",
                  "content": [{"type": "text", "text": "done"}], "stop_reason": "max_tokens"}}])
    new, outcome, saved, dropped = strip_batch(body)
    assert outcome == "stripped" and saved > 0
    assert "toolUseResult" in dropped and "message.content[].input" in dropped
    before = [e for e in json.loads(body)["payload"]["events"]]
    after = json.loads(new)["payload"]["events"]
    assert lines_from_events(after) == lines_from_events(before)
    assert "TOOL-OUTPUT-9b2c" not in new.decode()
    assert strip_batch(new)[1] == "clean", "a stripped batch is a fixed point"


@pytest.mark.parametrize("path", GOLDENS, ids=lambda p: p.name)
def test_current_uploads_are_already_clean(path: Path) -> None:
    assert strip_batch(_batch(_events(path)))[1] == "clean"


def test_unreadable_and_empty_batches_are_left_alone() -> None:
    assert strip_batch(b"{not json")[:2] == (None, "unreadable")
    assert strip_batch(json.dumps({"payload": {"finalize": True}}).encode())[:2] == (None, "no_events")


def test_a_named_key_with_a_value_the_schema_does_not_allow_is_dropped() -> None:
    """Claude Code's own `origin` object (a peer message can carry another
    session's text) is not probe-events/1's `origin: "user_shell"`."""
    event = {"type": "user", "message": {"role": "user", "content": "hi"},
             "origin": {"kind": "peer", "body": "ANOTHER-SESSION-TEXT"}}
    out = project_event(event)
    assert "origin" not in out and "ANOTHER-SESSION-TEXT" not in json.dumps(out)
    assert project_event({**event, "origin": "user_shell"})["origin"] == "user_shell"
    pi = {"type": "assistant", "_pi_extras": {"model": {"leaked": 1}, "provider": "anthropic"}}
    assert project_event(pi)["_pi_extras"] == {"provider": "anthropic"}


def test_projected_old_events_validate_against_the_schema() -> None:
    import jsonschema

    schema = json.loads((Path(__file__).parents[1] / "engine/ingest/probe_events/probe-events-1.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    for event in (OLD_CC_EVENT, {"type": "user", "origin": {"kind": "human"},
                                 "message": {"role": "user", "content": "x"}}):
        errors = list(validator.iter_errors(project_event(event)))
        assert errors == [], [e.message for e in errors]
