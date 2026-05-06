"""Tests for _clean_schema_for_gemini.

Pydantic emits JSON schema fields that the Gemini API rejects at
request-build time (most commonly ``additionalProperties``). The cleaner
strips them before the schema is handed to GenerateContentConfig.
"""

from __future__ import annotations

from scripts.synth.llm.gemini_client import _clean_schema_for_gemini
from scripts.synth.llm.planner import PlannerOutputSchema
from scripts.synth.llm.validator_pass2 import Pass2OutputSchema


def _count_field(d: object, key: str) -> int:
    if isinstance(d, dict):
        return (1 if key in d else 0) + sum(_count_field(v, key) for v in d.values())
    if isinstance(d, list):
        return sum(_count_field(v, key) for v in d)
    return 0


# -----------------------------------------------------------------------------
# Synthetic shapes
# -----------------------------------------------------------------------------


def test_strips_additional_properties_false_at_top_level() -> None:
    schema = {"type": "object", "additionalProperties": False, "properties": {}}
    out = _clean_schema_for_gemini(schema)
    assert "additionalProperties" not in out
    assert out == {"type": "object", "properties": {}}


def test_strips_additional_properties_with_value_schema() -> None:
    """For dict[str, T] fields, Pydantic emits additionalProperties: {<schema>}.
    Gemini rejects this too. The semantic loss is acceptable — downstream
    Pydantic validation in the caller still catches type mismatches."""
    schema = {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }
    assert _clean_schema_for_gemini(schema) == {"type": "object"}


def test_strips_recursively_through_nested_objects() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "inner": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "deeper": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    }
                },
            }
        },
    }
    out = _clean_schema_for_gemini(schema)
    assert _count_field(out, "additionalProperties") == 0
    # Other structure preserved
    assert out["properties"]["inner"]["properties"]["deeper"]["type"] == "object"


def test_strips_through_array_items() -> None:
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"x": {"type": "integer"}},
        },
    }
    out = _clean_schema_for_gemini(schema)
    assert _count_field(out, "additionalProperties") == 0
    assert out["items"]["properties"] == {"x": {"type": "integer"}}


def test_strips_inside_defs() -> None:
    """Pydantic puts nested model definitions under $defs; the cleaner must
    descend into that block too."""
    schema = {
        "$defs": {
            "Inner": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"y": {"type": "string"}},
            }
        },
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/Inner"}},
    }
    out = _clean_schema_for_gemini(schema)
    assert _count_field(out, "additionalProperties") == 0
    assert "Inner" in out["$defs"]
    # $ref preserved (Gemini supports refs)
    assert out["properties"]["a"] == {"$ref": "#/$defs/Inner"}


def test_does_not_strip_other_keys() -> None:
    """title, default, type, properties, required, $ref, $defs all survive."""
    schema = {
        "title": "Foo",
        "type": "object",
        "default": {},
        "$defs": {},
        "$ref": "#/$defs/Foo",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    out = _clean_schema_for_gemini(schema)
    for k in ("title", "type", "default", "$defs", "$ref", "properties", "required"):
        assert k in out, f"unexpectedly stripped {k}"
    assert "additionalProperties" not in out


def test_passthrough_for_non_dict_non_list() -> None:
    assert _clean_schema_for_gemini("string") == "string"
    assert _clean_schema_for_gemini(42) == 42
    assert _clean_schema_for_gemini(None) is None
    assert _clean_schema_for_gemini(True) is True


# -----------------------------------------------------------------------------
# Real schemas — pin that the cleaner produces a Gemini-acceptable shape
# for the actual Pydantic models the codebase uses.
# -----------------------------------------------------------------------------


def test_planner_schema_has_no_additional_properties_after_cleaning() -> None:
    """The exact failure that surfaced 2026-05-06 BIG_REFACTOR planner call:
    PlannerOutputSchema's source_emissions field (dict[str, int]) emits
    additionalProperties: {type: integer} which Gemini rejects."""
    raw = PlannerOutputSchema.model_json_schema()
    assert _count_field(raw, "additionalProperties") >= 1, (
        "test premise broken: the planner schema should have at least one "
        "additionalProperties for the regression to mean anything"
    )
    cleaned = _clean_schema_for_gemini(raw)
    assert _count_field(cleaned, "additionalProperties") == 0


def test_pass2_schema_survives_cleaning_unchanged_in_substance() -> None:
    """Pass2OutputSchema doesn't trigger the bug (no dict[str, T] field) —
    cleaning should be a no-op apart from possibly absent additionalProperties
    keys. Pin this so we notice if a schema change introduces the issue."""
    raw = Pass2OutputSchema.model_json_schema()
    cleaned = _clean_schema_for_gemini(raw)
    assert _count_field(cleaned, "additionalProperties") == 0
    # Substantive structure preserved
    assert cleaned["properties"]["passed"]["type"] == "boolean"
    assert cleaned["properties"]["violations"]["type"] == "array"
