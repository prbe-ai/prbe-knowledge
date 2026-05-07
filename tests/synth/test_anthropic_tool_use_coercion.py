"""Tests for _coerce_tool_use_output — defensive normalization of
Anthropic tool_use input dicts before schema validation.

Two quirks observed in real-LLM canonical recording runs:

1. Whole-response wrap: model returns ``{"parameter": {<actual>}}``
   instead of the schema fields at the top level (Haiku 4.5 quirk,
   first hit during PR #96 canonical recording).

2. Stringified nested fields: model returns ``{"passed": false,
   "violations": "[{...}]"}`` — the violations array as a JSON-string
   instead of a native list (Haiku 4.5, hit during medium_holistic
   recording on 2026-05-06).
"""

from __future__ import annotations

from scripts.synth.llm.anthropic_client import _coerce_tool_use_output

# -----------------------------------------------------------------------------
# Quirk 1 — envelope unwrap
# -----------------------------------------------------------------------------


def test_unwraps_parameter_envelope() -> None:
    raw = {"parameter": {"passed": True, "violations": []}}
    assert _coerce_tool_use_output(raw) == {"passed": True, "violations": []}


def test_unwraps_input_envelope() -> None:
    raw = {"input": {"passed": False, "violations": [{"doc_id": "d0", "issue": "x"}]}}
    assert _coerce_tool_use_output(raw) == {
        "passed": False,
        "violations": [{"doc_id": "d0", "issue": "x"}],
    }


def test_does_not_unwrap_when_envelope_value_is_not_a_dict() -> None:
    raw = {"parameter": "not a dict"}
    assert _coerce_tool_use_output(raw) == {"parameter": "not a dict"}


def test_does_not_unwrap_when_other_keys_present() -> None:
    raw = {"parameter": {"x": 1}, "extra": 2}
    assert _coerce_tool_use_output(raw) == {"parameter": {"x": 1}, "extra": 2}


# -----------------------------------------------------------------------------
# Quirk 2 — stringified JSON values
# -----------------------------------------------------------------------------


def test_parses_stringified_list_field() -> None:
    raw = {"passed": False, "violations": '[{"doc_id": "d0", "issue": "x"}]'}
    assert _coerce_tool_use_output(raw) == {
        "passed": False,
        "violations": [{"doc_id": "d0", "issue": "x"}],
    }


def test_parses_pretty_printed_stringified_list() -> None:
    """The exact shape from the 2026-05-06 medium-test recording failure:
    indented, multi-line JSON-string."""
    raw = {
        "passed": False,
        "violations": (
            '[\n  {\n    "doc_id": "scn-incident-...-slack-0",\n'
            '    "issue": "Document contradicts incident facts."\n  }\n]'
        ),
    }
    out = _coerce_tool_use_output(raw)
    assert out["passed"] is False
    assert out["violations"] == [
        {
            "doc_id": "scn-incident-...-slack-0",
            "issue": "Document contradicts incident facts.",
        }
    ]


def test_parses_stringified_dict_field() -> None:
    raw = {"meta": '{"k": "v"}', "passed": True}
    assert _coerce_tool_use_output(raw) == {"meta": {"k": "v"}, "passed": True}


def test_leaves_non_json_strings_alone() -> None:
    raw = {"summary": "regular text content", "count": 3}
    assert _coerce_tool_use_output(raw) == {"summary": "regular text content", "count": 3}


def test_leaves_invalid_json_strings_alone() -> None:
    """A string starting with [ or { but not parseable should be left as a
    string. The downstream Pydantic validator will then raise the right
    error."""
    raw = {"violations": "[broken"}
    assert _coerce_tool_use_output(raw) == {"violations": "[broken"}


# -----------------------------------------------------------------------------
# Combined / no-op
# -----------------------------------------------------------------------------


def test_combined_envelope_unwrap_then_stringified_field() -> None:
    """Worst case: envelope wrap AND a stringified inner field."""
    raw = {
        "parameter": {
            "passed": False,
            "violations": '[{"doc_id": "d0", "issue": "x"}]',
        }
    }
    assert _coerce_tool_use_output(raw) == {
        "passed": False,
        "violations": [{"doc_id": "d0", "issue": "x"}],
    }


def test_already_well_formed_dict_is_passthrough() -> None:
    raw = {"passed": True, "violations": [{"doc_id": "d0", "issue": "x"}]}
    assert _coerce_tool_use_output(raw) == raw


def test_empty_dict_is_passthrough() -> None:
    assert _coerce_tool_use_output({}) == {}
