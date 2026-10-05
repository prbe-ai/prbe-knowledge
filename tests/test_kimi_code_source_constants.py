"""Tests for the Kimi Code SourceSystem constant and its source-registry tuning.

Kimi Code sessions arrive shimmed into Claude-Code shape by the tap's
sanitizer, exactly like Codex and pi. Its per-source tuning (doc_type_prefix,
ingestion_priority, score_multiplier, half_life_days) is inherited from
ClaudeCodeConnector and read through shared.source_registry, populated at
import time by @register_connector. A source missing its registration does not
KeyError: get_source_profile degrades to generic defaults (live-integration
priority, no demotion, no decay), so the miss would surface as silently-wrong
ranking and queue behaviour. This file pins Kimi Code's profile to Codex's and
pi's, value for value.
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


def test_kimi_code_source_system_value() -> None:
    assert SourceSystem.KIMI_CODE.value == "kimi_code"
    assert SourceSystem("kimi_code") is SourceSystem.KIMI_CODE


def test_kimi_code_in_display_names() -> None:
    assert SOURCE_DISPLAY_NAMES[SourceSystem.KIMI_CODE] == "Kimi Code"


def test_kimi_code_is_an_agent_session_source() -> None:
    """The one list the normalizer's session coalescing, the Jev tiers, session
    suppression and session deletion all read."""
    assert SourceSystem.KIMI_CODE in AGENT_SESSION_SOURCES


def test_kimi_code_tuning_matches_codex_and_pi() -> None:
    kimi = get_source_profile(SourceSystem.KIMI_CODE.value)
    for other in (SourceSystem.CODEX, SourceSystem.PI):
        profile = get_source_profile(other.value)
        assert kimi.doc_type_prefix == profile.doc_type_prefix
        assert kimi.ingestion_priority == profile.ingestion_priority
        assert kimi.score_multiplier == profile.score_multiplier
        assert kimi.half_life_days == profile.half_life_days


def test_kimi_code_tuning_values() -> None:
    profile = get_source_profile(SourceSystem.KIMI_CODE.value)
    assert profile.doc_type_prefix == "claude_code."
    assert profile.ingestion_priority == PRIORITY_AGENT_CAPTURE
    assert profile.score_multiplier == 0.5
    assert profile.half_life_days == 7.0


def test_a_search_may_name_kimi_code_as_a_source() -> None:
    """research-os sends every searchable agent id on a default search, and an
    unknown id is a 422 -- which is why the engine has to ship first."""
    request = QueryRequest(query="what did we decide", sources=["kimi_code"])
    assert request.sources == [SourceSystem.KIMI_CODE]
