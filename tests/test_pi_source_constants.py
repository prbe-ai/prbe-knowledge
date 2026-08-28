"""Tests for the pi SourceSystem constant and its source-registry tuning.

pi (earendil-works/pi) sessions arrive shimmed into Claude-Code shape by the
tap's sanitizer, exactly like Codex. Its per-source tuning (doc_type_prefix,
ingestion_priority, score_multiplier, half_life_days) is not a hand-picked
dict entry in this module — it is read through shared.source_registry,
populated at import time by PiConnector's inherited ClassVar attributes via
@register_connector (see kb/handlers/claude_code.py). A source missing its
registration doesn't KeyError at import: get_source_profile degrades to
generic defaults, so the miss would surface as silently-wrong ranking/queue
behavior in production rather than a test collection error — which is why
this file pins pi's registered profile to Codex's, value for value.
"""

import kb.handlers  # noqa: F401  (registers source profiles)
from engine.shared.constants import SOURCE_DISPLAY_NAMES, SourceSystem
from engine.shared.source_registry import get_source_profile


def test_pi_source_system_value() -> None:
    assert SourceSystem.PI.value == "pi"


def test_pi_in_display_names() -> None:
    assert SOURCE_DISPLAY_NAMES[SourceSystem.PI] == "pi"


def test_pi_tuning_matches_codex() -> None:
    pi_profile = get_source_profile(SourceSystem.PI.value)
    codex_profile = get_source_profile(SourceSystem.CODEX.value)

    assert pi_profile.doc_type_prefix == codex_profile.doc_type_prefix
    assert pi_profile.ingestion_priority == codex_profile.ingestion_priority
    assert pi_profile.score_multiplier == codex_profile.score_multiplier
    assert pi_profile.half_life_days == codex_profile.half_life_days


def test_pi_tuning_values() -> None:
    profile = get_source_profile(SourceSystem.PI.value)
    assert profile.doc_type_prefix == "claude_code."
    assert profile.ingestion_priority == 75
    assert profile.score_multiplier == 0.5
    assert profile.half_life_days == 7.0
