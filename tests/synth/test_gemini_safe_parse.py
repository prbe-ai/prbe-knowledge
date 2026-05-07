"""Tests for _safe_parse_json — defensive parser for Gemini JSON output.

Despite response_mime_type='application/json', Gemini occasionally returns:
  1. Trailing commas before } or ]
  2. Trailing commentary or markdown after the JSON
  3. Truncated output that ends mid-value

For (1) and (2) we recover. For (3) the partial-recovery may produce a
dict that fails downstream Pydantic validation — which is the right
behavior (the operator should bump max_tokens).
"""

from __future__ import annotations

import pytest

from scripts.synth.llm.gemini_client import _safe_parse_json

# -----------------------------------------------------------------------------
# Happy path — well-formed JSON passes through
# -----------------------------------------------------------------------------


def test_well_formed_json_passes_through() -> None:
    raw = '{"passed": true, "violations": [{"doc_id": "d0", "issue": "x"}]}'
    assert _safe_parse_json(raw) == {
        "passed": True,
        "violations": [{"doc_id": "d0", "issue": "x"}],
    }


def test_well_formed_json_with_whitespace() -> None:
    assert _safe_parse_json('  \n  {"a": 1}  \n  ') == {"a": 1}


# -----------------------------------------------------------------------------
# Quirk 1 — trailing commas
# -----------------------------------------------------------------------------


def test_repairs_trailing_comma_in_object() -> None:
    raw = '{"a": 1, "b": 2,}'
    assert _safe_parse_json(raw) == {"a": 1, "b": 2}


def test_repairs_trailing_comma_in_array() -> None:
    raw = '{"items": [1, 2, 3,]}'
    assert _safe_parse_json(raw) == {"items": [1, 2, 3]}


def test_repairs_trailing_comma_with_whitespace() -> None:
    raw = '{"items": [1, 2, 3 , \n  ]}'
    assert _safe_parse_json(raw) == {"items": [1, 2, 3]}


# -----------------------------------------------------------------------------
# Quirk 2 — trailing commentary / markdown fences
# -----------------------------------------------------------------------------


def test_strips_trailing_commentary_after_json() -> None:
    raw = '{"a": 1}\n\nNote: this is what I generated based on the schema.'
    assert _safe_parse_json(raw) == {"a": 1}


def test_strips_markdown_code_fences() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert _safe_parse_json(raw) == {"a": 1}


def test_strips_markdown_fences_without_lang_tag() -> None:
    raw = '```\n{"a": 1}\n```'
    assert _safe_parse_json(raw) == {"a": 1}


def test_strips_trailing_commentary_with_brackets_in_strings() -> None:
    """Regression check: balance walker must respect string boundaries so that
    a `}` inside a string value doesn't cause early truncation."""
    raw = '{"text": "this contains } and ] braces"}\n\ntrailing junk'
    assert _safe_parse_json(raw) == {"text": "this contains } and ] braces"}


def test_strips_trailing_commentary_with_escaped_quotes_in_strings() -> None:
    raw = r'{"text": "she said \"hi\"!"}' + "\n\ngarbage"
    assert _safe_parse_json(raw) == {"text": 'she said "hi"!'}


# -----------------------------------------------------------------------------
# Quirk 3 — truncation
# -----------------------------------------------------------------------------


def test_truncated_json_raises_clear_error() -> None:
    """Truncated output where the JSON ends mid-value can't be recovered.
    The error message should point the operator at the right knob."""
    raw = '{"a": 1, "b": "this string was cut off in the mid'
    with pytest.raises(ValueError, match="malformed JSON"):
        _safe_parse_json(raw)


def test_error_message_includes_snippet_and_max_tokens_hint() -> None:
    raw = '{"a": 1, "broken'
    with pytest.raises(ValueError) as exc_info:
        _safe_parse_json(raw)
    msg = str(exc_info.value)
    assert "raw length" in msg
    assert "first 300 chars" in msg
    assert "max_output_tokens" in msg


# -----------------------------------------------------------------------------
# Combined quirks
# -----------------------------------------------------------------------------


def test_combined_markdown_fence_plus_trailing_comma() -> None:
    raw = '```json\n{"a": 1, "b": 2,}\n```'
    assert _safe_parse_json(raw) == {"a": 1, "b": 2}


def test_nested_object_with_trailing_commas_at_each_level() -> None:
    raw = '{"outer": {"inner": [1, 2,], "x": 3,}, "y": 4,}'
    assert _safe_parse_json(raw) == {
        "outer": {"inner": [1, 2], "x": 3},
        "y": 4,
    }
