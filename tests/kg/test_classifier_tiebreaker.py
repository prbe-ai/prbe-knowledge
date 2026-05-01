"""Tests for the classifier's Haiku tiebreaker step (spec §6 step 2).

The tiebreaker (``resolve_ambiguity``) is the third leg of the hybrid
match pipeline (rules → embedding → LLM). It calls Anthropic with the
incident plus a short list of candidate class IDs and lets the LLM
pick one — or ``"none"`` if none fit. The classifier orchestrator
(future task) catches a ``choice=None`` result and decides degraded-
mode policy; this module's only job is the LLM round-trip plus
fail-safe handling on any error path.

The unit-level concerns covered here:

1. Happy path — valid JSON with a candidate class_id is unwrapped.
2. ``"none"`` (string) collapses to ``None`` (Python).
3. Invalid JSON fails safe to ``choice=None`` and emits a structured
   ``kg.classifier.tiebreaker_failed`` warning.
4. Anthropic API exception fails safe to ``choice=None`` and emits the
   same warning.
5. A hallucinated ``choice`` (LLM picked an ID not in the candidates
   list) is rejected — fails safe to ``choice=None`` without firing the
   warning, since this is a guarded-against contract violation rather
   than a transport / parse failure.

The Anthropic client is mocked throughout; no real API calls.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from services.kg.classifier.tiebreaker import TiebreakerResult, resolve_ambiguity


def _fake_anthropic(text: str) -> MagicMock:
    """Build a duck-typed Anthropic client whose ``messages.create`` returns ``text``."""
    fake = MagicMock()
    fake.messages.create.return_value.content = [MagicMock(text=text)]
    return fake


def test_picks_named_class_from_response() -> None:
    """Valid JSON with an in-set ``choice`` is returned verbatim."""
    a = _fake_anthropic(
        '{"choice": "auth-401-jwt-refresh", "rationale": "matches symptoms exactly"}'
    )
    out = resolve_ambiguity(
        anthropic=a,
        incident={"x": 1},
        candidates=["auth-401-jwt-refresh", "auth-403-rbac"],
    )
    assert out == TiebreakerResult(
        choice="auth-401-jwt-refresh",
        rationale="matches symptoms exactly",
    )


def test_returns_none_when_judge_says_none() -> None:
    """``"none"`` (string) maps to Python ``None``; rationale is preserved."""
    a = _fake_anthropic('{"choice": "none", "rationale": "no good match"}')
    out = resolve_ambiguity(
        anthropic=a,
        incident={"x": 1},
        candidates=["auth-401-jwt-refresh"],
    )
    assert out == TiebreakerResult(choice=None, rationale="no good match")


def test_fail_safe_on_invalid_json(
    _route_structlog_to_stdlib: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Garbage from the model fails safe to no-match + warning event."""
    a = _fake_anthropic("not json")
    with caplog.at_level(logging.WARNING):
        out = resolve_ambiguity(
            anthropic=a,
            incident={"x": 1},
            candidates=["auth-401-jwt-refresh"],
        )
    assert out.choice is None
    assert out.rationale.startswith("tiebreaker error:")
    assert any(
        "kg.classifier.tiebreaker_failed" in r.getMessage()
        and r.levelno == logging.WARNING
        for r in caplog.records
    ), f"expected warning event in caplog; got {[r.getMessage() for r in caplog.records]}"


def test_fail_safe_on_api_error(
    _route_structlog_to_stdlib: None, caplog: pytest.LogCaptureFixture
) -> None:
    """A transport-layer exception fails safe to no-match + warning event."""
    a = MagicMock()
    a.messages.create.side_effect = RuntimeError("boom: 429 rate limited")
    with caplog.at_level(logging.WARNING):
        out = resolve_ambiguity(
            anthropic=a,
            incident={"x": 1},
            candidates=["auth-401-jwt-refresh"],
        )
    assert out.choice is None
    assert "boom: 429 rate limited" in out.rationale
    assert any(
        "kg.classifier.tiebreaker_failed" in r.getMessage()
        and r.levelno == logging.WARNING
        for r in caplog.records
    ), f"expected warning event in caplog; got {[r.getMessage() for r in caplog.records]}"


def test_unknown_class_id_rejected(
    _route_structlog_to_stdlib: None, caplog: pytest.LogCaptureFixture
) -> None:
    """A hallucinated class_id is rejected without firing the failure warning.

    This is a guarded-against contract violation, not a transport /
    parse error — so the rationale flags the unknown ID but no
    ``kg.classifier.tiebreaker_failed`` event is emitted.
    """
    a = _fake_anthropic(
        '{"choice": "made-up-class", "rationale": "fits perfectly"}'
    )
    with caplog.at_level(logging.WARNING):
        out = resolve_ambiguity(
            anthropic=a,
            incident={"x": 1},
            candidates=["auth-401-jwt-refresh", "auth-403-rbac"],
        )
    assert out.choice is None
    assert "made-up-class" in out.rationale
    assert not any(
        "kg.classifier.tiebreaker_failed" in r.getMessage()
        for r in caplog.records
    )
