"""Tests for the claude_code connector's _events_to_text renderer.

This is what the chunker + embedder consume — vector-search quality
depends on it being clean prose, not JSON dumps. These tests pin the
output shape per event/block type.
"""

from __future__ import annotations

from services.ingestion.handlers.claude_code import (
    _events_to_text,
    _render_event,
)


def _ev(raw: dict, line_no: int = 0) -> dict:
    """Wrap a raw transcript event in the {line_no, raw} envelope shape
    that fetch_supplementary produces."""
    return {"line_no": line_no, "raw": raw}


# ---------------------------------------------------------------------------
# user
# ---------------------------------------------------------------------------


def test_user_text_renders_as_USER_prefix() -> None:
    raw = {
        "type": "user",
        "uuid": "u",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "How does auth work?"}],
        },
    }
    assert _render_event(raw) == "USER: How does auth work?"


def test_user_string_content_legacy_shape() -> None:
    raw = {"type": "user", "message": {"role": "user", "content": "raw string"}}
    assert _render_event(raw) == "USER: raw string"


def test_user_tool_result_success_renders_as_one_line() -> None:
    """Sanitizer drops the result body; we keep tool_use_id + ok/error."""
    raw = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_x"},
            ],
        },
    }
    assert _render_event(raw) == "TOOL_RESULT (toolu_x): ok"


def test_user_tool_result_error() -> None:
    raw = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_x", "is_error": True},
            ],
        },
    }
    assert _render_event(raw) == "TOOL_RESULT (toolu_x): error"


def test_user_text_plus_tool_result_in_single_message() -> None:
    """A user message can mix text (e.g. interrupt) with a tool_result."""
    raw = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "is_error": True},
                {"type": "text", "text": "[Request interrupted by user]"},
            ],
        },
    }
    assert _render_event(raw) == (
        "TOOL_RESULT (t1): error\nUSER: [Request interrupted by user]"
    )


# ---------------------------------------------------------------------------
# assistant
# ---------------------------------------------------------------------------


def test_assistant_text() -> None:
    raw = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "We use JWT."}],
        },
    }
    assert _render_event(raw) == "ASSISTANT: We use JWT."


def test_assistant_thinking_then_text() -> None:
    raw = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me think carefully."},
                {"type": "text", "text": "Here's the answer."},
            ],
        },
    }
    assert _render_event(raw) == (
        "ASSISTANT (thinking): Let me think carefully.\nASSISTANT: Here's the answer."
    )


def test_assistant_empty_thinking_skipped() -> None:
    """Sanitizer should already drop empty thinking, but render is defensive."""
    raw = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "  \n  "},
                {"type": "text", "text": "answer"},
            ],
        },
    }
    assert _render_event(raw) == "ASSISTANT: answer"


def test_assistant_tool_use_with_summary() -> None:
    raw = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_x",
                    "name": "Bash",
                    "summary": "git status",
                },
            ],
        },
    }
    assert _render_event(raw) == "TOOL_USE: Bash — git status"


def test_assistant_tool_use_no_summary() -> None:
    raw = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "x", "name": "MysteryMcpTool"}],
        },
    }
    assert _render_event(raw) == "TOOL_USE: MysteryMcpTool"


def test_assistant_thinking_text_and_tool_use_in_one_turn() -> None:
    """Realistic mixed assistant turn — thinking, then text, then a tool call."""
    raw = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I need to check the schema."},
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Bash",
                    "summary": "psql -c '\\d users'",
                },
            ],
        },
    }
    assert _render_event(raw) == (
        "ASSISTANT (thinking): I need to check the schema.\n"
        "ASSISTANT: Let me check.\n"
        "TOOL_USE: Bash — psql -c '\\d users'"
    )


def test_assistant_max_tokens_stop_reason_annotated() -> None:
    raw = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "truncated"}],
            "stop_reason": "max_tokens",
        },
    }
    assert _render_event(raw) == "ASSISTANT: truncated\n[stop: max_tokens]"


def test_assistant_end_turn_stop_reason_NOT_annotated() -> None:
    """end_turn is the boring case; noting it would clutter every turn."""
    raw = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
        },
    }
    assert _render_event(raw) == "ASSISTANT: done"


# ---------------------------------------------------------------------------
# system + unknown + skip-on-empty
# ---------------------------------------------------------------------------


def test_system_event_with_string_content() -> None:
    raw = {"type": "system", "subtype": "user_warning", "content": "Heads up"}
    assert _render_event(raw) == "SYSTEM (user_warning): Heads up"


def test_system_event_no_content_no_subtype_is_skipped() -> None:
    raw = {"type": "system"}
    assert _render_event(raw) == ""


def test_unknown_event_with_string_content_renders() -> None:
    """Forward-compat: a future event type with top-level string content
    still gets surfaced (uppercased label) instead of dumped as JSON."""
    raw = {"type": "future-thing", "content": "hello"}
    assert _render_event(raw) == "FUTURE-THING: hello"


def test_unknown_event_without_string_content_is_dropped() -> None:
    """The whole point of this rewrite — no JSON dumps in the chunked text."""
    raw = {"type": "weird-cc-thing", "messageId": "abc", "snapshot": {"foo": [1, 2]}}
    assert _render_event(raw) == ""


# ---------------------------------------------------------------------------
# _events_to_text — sequence assembly + ordering
# ---------------------------------------------------------------------------


def test_events_to_text_joins_with_blank_lines() -> None:
    events = [
        _ev({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        }, 0),
        _ev({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
            },
        }, 1),
    ]
    assert _events_to_text(events) == "USER: hi\n\nASSISTANT: hello"


def test_events_to_text_skips_unrenderable_events_silently() -> None:
    """Empty / unknown events shouldn't leave gaps in the output. They get
    dropped, and the surrounding turns join cleanly."""
    events = [
        _ev({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "a"}]},
        }, 0),
        _ev({"type": "weird-thing", "foo": "bar"}, 1),  # unrenderable
        _ev({"type": "system"}, 2),  # empty system
        _ev({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "b"}],
            },
        }, 3),
    ]
    assert _events_to_text(events) == "USER: a\n\nASSISTANT: b"


def test_events_to_text_realistic_mixed_session() -> None:
    """End-to-end: a small but realistic session with all the block types."""
    events = [
        _ev({
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "check the auth setup"}],
            },
        }, 0),
        _ev({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I should grep for it."},
                    {"type": "tool_use", "id": "t1", "name": "Grep", "summary": "JWT|cookie"},
                ],
            },
        }, 1),
        _ev({
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1"}],
            },
        }, 2),
        _ev({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Found it in middleware.py."}],
            },
        }, 3),
    ]
    expected = (
        "USER: check the auth setup\n\n"
        "ASSISTANT (thinking): I should grep for it.\n"
        "TOOL_USE: Grep — JWT|cookie\n\n"
        "TOOL_RESULT (t1): ok\n\n"
        "ASSISTANT: Found it in middleware.py."
    )
    assert _events_to_text(events) == expected


def test_events_to_text_handles_non_dict_envelope() -> None:
    """fetch_supplementary deduplicates by line_no but tolerates malformed
    entries; the renderer should skip them rather than crash."""
    events = [
        "not a dict",  # type: ignore[list-item]
        _ev({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "x"}]},
        }, 0),
    ]
    assert _events_to_text(events) == "USER: x"
