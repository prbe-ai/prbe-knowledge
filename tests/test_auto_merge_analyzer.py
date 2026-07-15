"""Unit tests for the AutoMergeAnalyzer's pure logic.

LLM judge + DB integration are exercised via a mocked acompletion + an
asyncpg test fixture in a separate integration suite (not in this file).
This file covers the deterministic pieces: path-canonical detection,
property-key conflict filtering, and prompt construction shape.
"""

from __future__ import annotations

from engine.ingest.auto_merge.analyzer import (
    Candidate,
    _build_prompt,
    _is_path_canonical,
    _properties_conflict,
)
from engine.ingest.auto_merge.models import AutoMergeVerdict

# --------------------------------------------------------------------------- #
# _is_path_canonical
# --------------------------------------------------------------------------- #


def test_path_canonical_rejects_pr_canonical_ids() -> None:
    assert _is_path_canonical("PR", "prbe-ai/prbe-knowledge#345")
    assert _is_path_canonical("Issue", "owner/repo#42")


def test_path_canonical_rejects_code_symbol_ids() -> None:
    assert _is_path_canonical(
        "Function",
        "prbe-ai/prbe-knowledge:engine.ingest.graph_writer.upsert_nodes",
    )
    assert _is_path_canonical(
        "Method", "prbe-ai/prbe-knowledge:app.services.foo.MyClass.__init__"
    )


def test_path_canonical_rejects_repo_owner_name() -> None:
    assert _is_path_canonical("Repo", "prbe-ai/prbe-knowledge")


def test_path_canonical_accepts_freeform_labels() -> None:
    assert not _is_path_canonical("Person", "richardwei6")
    assert not _is_path_canonical("Person", "Richard Wei")
    assert not _is_path_canonical("Topic", "litellm-proxy")
    assert not _is_path_canonical("WikiPerson", "ashwaryeyadav")
    assert not _is_path_canonical("Feature", "auto-merge")


def test_path_canonical_treats_blank_as_skip() -> None:
    # Blank canonical_id is degenerate — analyzer should skip to avoid noise.
    assert _is_path_canonical("Person", "")


# --------------------------------------------------------------------------- #
# _properties_conflict
# --------------------------------------------------------------------------- #


def test_properties_conflict_on_different_emails() -> None:
    assert _properties_conflict(
        {"email": "a@x.com", "name": "A"},
        {"email": "b@x.com", "name": "B"},
    )


def test_properties_match_on_same_emails() -> None:
    assert not _properties_conflict(
        {"email": "a@x.com", "name": "A"},
        {"email": "a@x.com", "name": "A2"},
    )


def test_properties_no_conflict_when_one_side_missing_email() -> None:
    # If only one side has the stable key, we can't decisively reject.
    # Let the LLM judge.
    assert not _properties_conflict(
        {"name": "A"},
        {"email": "a@x.com", "name": "A2"},
    )


def test_properties_conflict_on_repo_number_mismatch() -> None:
    assert _properties_conflict(
        {"repo": "owner/x", "number": 1},
        {"repo": "owner/y", "number": 1},
    )


def test_properties_conflict_on_owner_name_mismatch() -> None:
    assert _properties_conflict(
        {"owner": "a", "name": "x"},
        {"owner": "b", "name": "x"},
    )


# --------------------------------------------------------------------------- #
# _build_prompt
# --------------------------------------------------------------------------- #


def test_build_prompt_includes_new_entity_and_all_candidates() -> None:
    node = {
        "label": "Person",
        "canonical_id": "richardwei6",
        "properties": {"name": "Richard Wei", "source_system": "github"},
        "degree": 12,
    }
    candidates = [
        Candidate(
            canonical_id="Richard Wei",
            properties={"email": "richard@prbe.ai", "source_system": "slack"},
            degree=8,
            trigram_score=0.45,
            vector_distance=None,
        ),
        Candidate(
            canonical_id="00000000-0000-0000-0000-000000000001",
            properties={"name": "Richard W", "email": "richard@prbe.ai"},
            degree=4,
            trigram_score=None,
            vector_distance=0.12,
        ),
    ]
    prompt = _build_prompt(node, candidates)
    assert "richardwei6" in prompt
    assert "Richard Wei" in prompt
    assert "00000000-0000-0000-0000-000000000001" in prompt
    assert "trigram=0.45" in prompt
    assert "vector_distance=0.120" in prompt
    assert "richard@prbe.ai" in prompt


def test_build_prompt_with_empty_signals() -> None:
    node = {
        "label": "Topic",
        "canonical_id": "auto-merge",
        "properties": {},
        "degree": 0,
    }
    candidates = [
        Candidate(
            canonical_id="entity-dedup",
            properties={},
            degree=0,
            trigram_score=None,
            vector_distance=None,
        ),
    ]
    prompt = _build_prompt(node, candidates)
    assert "signals:      none" in prompt


# --------------------------------------------------------------------------- #
# AutoMergeVerdict schema
# --------------------------------------------------------------------------- #


def test_verdict_unique_with_no_primary() -> None:
    v = AutoMergeVerdict.model_validate_json(
        '{"verdict": "unique", "primary_canonical_id": null, '
        '"confidence": null, "rationale": "no overlap"}'
    )
    assert v.verdict == "unique"
    assert v.primary_canonical_id is None


def test_verdict_duplicate_high_confidence() -> None:
    v = AutoMergeVerdict.model_validate_json(
        '{"verdict": "duplicate", "primary_canonical_id": "Richard Wei", '
        '"confidence": "high", "rationale": "shared email richard@prbe.ai"}'
    )
    assert v.verdict == "duplicate"
    assert v.primary_canonical_id == "Richard Wei"
    assert v.confidence == "high"


def test_verdict_rejects_extra_fields() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AutoMergeVerdict.model_validate_json(
            '{"verdict": "unique", "rationale": "x", "extra_field": 1}'
        )
