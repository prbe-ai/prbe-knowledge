"""Schema tests for the debugging-KG class envelope.

These are pure-Pydantic tests — no DB, no async. They cover the happy path
plus every validation edge documented in services/kg/schema.py: blank-rule
must_match, short embedding_seed, out-of-range priority, slug-shape rules
on `Frontmatter.id`, and `extra='forbid'` rejection of typoed fields.

Refs: docs/superpowers/specs/2026-04-29-debugging-knowledge-graph-design.md
§4.1, §4.2, §5.2.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from services.kg.schema import (
    BugClass,
    ContextSource,
    Evidence,
    Frontmatter,
    Signature,
)


def _minimal_frontmatter(**overrides: Any) -> Frontmatter:
    base: dict[str, Any] = dict(
        id="auth-401-jwt-refresh",
        type="bug-class",
        description="401s on /api/* after JWT refresh fails",
        signature=Signature(
            must_match=["status_code == 401"],
            embedding_seed="jwt refresh",
        ),
    )
    base.update(overrides)
    return Frontmatter(**base)


def test_minimal_frontmatter_validates() -> None:
    fm = _minimal_frontmatter()
    assert fm.id == "auth-401-jwt-refresh"
    assert fm.related.analogous_to == []
    assert fm.evidence.match_count == 0


def test_signature_rejects_empty_must_match_after_strip() -> None:
    with pytest.raises(Exception):
        Signature(must_match=["   ", ""], embedding_seed="x" * 5)


def test_signature_rejects_short_embedding_seed() -> None:
    with pytest.raises(Exception):
        Signature(must_match=["status == 401"], embedding_seed="ab")


def test_context_source_rejects_invalid_priority() -> None:
    with pytest.raises(Exception):
        ContextSource(priority=4, name="x", tool="y", params={})  # type: ignore[arg-type]


def test_context_source_accepts_valid_priorities() -> None:
    valid: tuple[Literal[1, 2, 3], ...] = (1, 2, 3)
    for p in valid:
        cs = ContextSource(priority=p, name="x", tool="y", params={})
        assert cs.priority == p


def test_id_slug_pattern_accepts_valid() -> None:
    fm = _minimal_frontmatter(id="db-timeout-replica-lag")
    assert fm.id == "db-timeout-replica-lag"


def test_id_slug_rejects_uppercase() -> None:
    with pytest.raises(Exception):
        _minimal_frontmatter(id="Auth-401")


def test_id_slug_rejects_spaces() -> None:
    with pytest.raises(Exception):
        _minimal_frontmatter(id="auth 401")


def test_id_slug_rejects_starting_digit() -> None:
    with pytest.raises(Exception):
        _minimal_frontmatter(id="401-auth")


def test_id_slug_rejects_too_short() -> None:
    with pytest.raises(Exception):
        _minimal_frontmatter(id="ab")


def test_extra_field_in_frontmatter_rejected() -> None:
    with pytest.raises(Exception):
        Frontmatter(
            id="auth-401",
            type="bug-class",
            description="x",
            signature=Signature(must_match=["x rule"], embedding_seed="xxx"),
            unknown_field="should fail",  # type: ignore[call-arg]
        )


def test_round_trip_via_model_dump() -> None:
    fm = _minimal_frontmatter()
    cls = BugClass(frontmatter=fm, body="## When this fires\n401 ...")
    raw = cls.model_dump(mode="json")
    parsed = BugClass.model_validate(raw)
    assert parsed.model_dump(mode="json") == raw


def test_evidence_defaults() -> None:
    e = Evidence()
    assert e.match_count == 0
    assert e.last_updated is None
    assert e.recent_refinements == []
