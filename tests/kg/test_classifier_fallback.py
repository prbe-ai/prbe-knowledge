"""Tests for the classifier's low-confidence threshold (spec §6 step 2).

When the best embedding match's score is below ``LOW_CONF_THRESHOLD``
the classifier emits ``no_match`` and the debugging agent falls back to
freeform RAG over Probe MCP. The threshold and the empty-result
semantics are pure-Python — no DB needed.
"""

from __future__ import annotations

from services.kg.classifier.fallback import LOW_CONF_THRESHOLD, is_low_confidence


def test_low_confidence_below_threshold() -> None:
    assert is_low_confidence(LOW_CONF_THRESHOLD - 0.01) is True


def test_high_confidence_above_threshold() -> None:
    assert is_low_confidence(LOW_CONF_THRESHOLD + 0.01) is False


def test_low_confidence_when_no_matches() -> None:
    # Empty-result is low-confidence by definition: spec §6 step 2 says
    # ``no_match`` fires when the best score is below threshold. An empty
    # result has no best score, which is operationally below any threshold,
    # so callers pass ``best_score=0.0`` to mean "nothing came back".
    assert is_low_confidence(0.0) is True


def test_threshold_value_is_07() -> None:
    # Lock the threshold so a future tuning to 0.6 / 0.8 has to be a
    # deliberate test edit, not a silent constant change.
    assert LOW_CONF_THRESHOLD == 0.7
