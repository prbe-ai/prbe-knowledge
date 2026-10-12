"""Tests for the Capy SourceSystem constant and its source-registry tuning.

Capy (capy.ai) sessions arrive shimmed into Claude-Code shape by the tap's
sanitizer, exactly like Codex, pi and Kimi Code. Its per-source tuning
(doc_type_prefix, ingestion_priority, score_multiplier, half_life_days) is
inherited from ClaudeCodeConnector and read through shared.source_registry,
populated at import time by @register_connector. A source missing its
registration does not KeyError: get_source_profile degrades to generic
defaults (live-integration priority, no demotion, no decay), so the miss would
surface as silently-wrong ranking and queue behaviour. This file pins Capy's
profile to Codex's, pi's and Kimi Code's, value for value.
"""

import kb.handlers  # noqa: F401  (registers source profiles)
from engine.shared.constants import (
    AGENT_SESSION_SOURCES,
    PRIORITY_AGENT_CAPTURE,
    SOURCE_DISPLAY_NAMES,
    SourceSystem,
)
from engine.shared.models import QueryRequest
from engine.shared.source_registry import get_source_profile


def test_capy_source_system_value() -> None:
    assert SourceSystem.CAPY.value == "capy"
    assert SourceSystem("capy") is SourceSystem.CAPY


def test_capy_in_display_names() -> None:
    assert SOURCE_DISPLAY_NAMES[SourceSystem.CAPY] == "Capy"


def test_capy_is_an_agent_session_source() -> None:
    """The one list the normalizer's session coalescing, the Jev tiers, session
    suppression and session deletion all read."""
    assert SourceSystem.CAPY in AGENT_SESSION_SOURCES


def test_capy_tuning_matches_the_other_shimmed_agents() -> None:
    capy = get_source_profile(SourceSystem.CAPY.value)
    for other in (SourceSystem.CODEX, SourceSystem.PI, SourceSystem.KIMI_CODE):
        profile = get_source_profile(other.value)
        assert capy.doc_type_prefix == profile.doc_type_prefix
        assert capy.ingestion_priority == profile.ingestion_priority
        assert capy.score_multiplier == profile.score_multiplier
        assert capy.half_life_days == profile.half_life_days


def test_capy_tuning_values() -> None:
    profile = get_source_profile(SourceSystem.CAPY.value)
    assert profile.doc_type_prefix == "claude_code."
    assert profile.ingestion_priority == PRIORITY_AGENT_CAPTURE
    assert profile.score_multiplier == 0.5
    assert profile.half_life_days == 7.0


def test_a_search_may_name_capy_as_a_source() -> None:
    """research-os sends every searchable agent id on a default search, and an
    unknown id is a 422 -- which is why the engine has to ship first."""
    request = QueryRequest(query="what did we decide", sources=["capy"])
    assert request.sources == [SourceSystem.CAPY]
