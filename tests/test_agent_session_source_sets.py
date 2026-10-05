"""Every hand-written list of coding-agent sources names every coding agent.

`AGENT_SESSION_SOURCES` (engine/shared/constants.py) is the one list most
consumers read, but several sites still spell the agents out by hand: the
webhook's coalescing set, the idle-session finalizer, per-device stats, the
transcript-receipts door, the disconnect gate, grounding's entity types, the
extraction prompt's agent label, the synthesis source-preference rule and the
Jev doc-class table. pi was added to most of them in #516 and the rest turned
up as production stragglers two days later (#518). Each miss is silent: a
404/422 on one route, a generic prompt, a session that is never finalized.

This pins every one of them against the canonical tuple, so the next agent is
a red test here rather than a straggler in production.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import kb.handlers  # noqa: F401  (registers the connectors)
from engine.ingest.handlers import registry
from engine.shared.constants import AGENT_SESSION_SOURCES, SOURCE_DISPLAY_NAMES, SourceSystem
from kb.handlers.claude_code import ClaudeCodeConnector

_AGENTS = {s.value for s in AGENT_SESSION_SOURCES}


def test_the_canonical_tuple_names_all_four_agents() -> None:
    assert set(AGENT_SESSION_SOURCES) == {"claude_code", "codex", "pi", "kimi_code"}


@pytest.mark.parametrize("source", AGENT_SESSION_SOURCES, ids=str)
def test_each_agent_has_a_claude_code_shaped_connector(source: SourceSystem) -> None:
    cls = registry.get_connector_class(source)
    assert issubclass(cls, ClaudeCodeConnector)
    assert cls.source_system is source
    assert cls.display_name == SOURCE_DISPLAY_NAMES[source]
    # doc ids, AgentSession node ids (a cross-repo contract with research-os)
    # and the extraction prompt's label are all keyed on the source id.
    assert cls._doc_id_prefix == source.value
    assert cls._agent_label == source.value


def test_webhook_coalescing_set() -> None:
    from kb.ingestion_app import _COALESCING_AGENT_SOURCES

    assert set(_COALESCING_AGENT_SOURCES) == set(AGENT_SESSION_SOURCES)


def test_idle_session_finalizer_sweeps_every_agent() -> None:
    from kb.session_completer import AGENT_SOURCES

    assert set(AGENT_SOURCES) == set(AGENT_SESSION_SOURCES)


def test_per_device_stats_route_serves_every_agent() -> None:
    from kb.stats_routes import _DEVICE_PAIRED_SOURCES

    assert set(_DEVICE_PAIRED_SOURCES) == _AGENTS


def test_transcript_receipts_door_accepts_every_agent() -> None:
    from kb.session_receipts import _SOURCES

    assert set(_SOURCES) == _AGENTS


def test_session_deletion_covers_every_agent() -> None:
    from kb.session_deletion import AGENT_SOURCES

    assert set(AGENT_SOURCES) == _AGENTS


def test_disconnect_gate_leaves_every_agent_ungated() -> None:
    from engine.ingest.connectedness import _OAUTH_SOURCES, _UNGATED_SOURCES

    assert set(AGENT_SESSION_SOURCES) <= set(_UNGATED_SOURCES)
    assert not set(AGENT_SESSION_SOURCES) & set(_OAUTH_SOURCES)


def test_grounding_types_every_agent_as_a_session() -> None:
    from engine.retrieval.grounding import _SOURCE_SYSTEM_TO_ENTITY_TYPE

    for agent in _AGENTS:
        assert _SOURCE_SYSTEM_TO_ENTITY_TYPE.get(agent) == "session", agent


def test_extraction_prompt_names_every_agent() -> None:
    from engine.shared.claude_code_extraction import _AGENT_LABELS

    for source in AGENT_SESSION_SOURCES:
        assert _AGENT_LABELS.get(source.value) == SOURCE_DISPLAY_NAMES[source], source


def test_synthesis_rule_lists_every_agent_as_an_agent_session() -> None:
    from engine.retrieval.synthesis import _SOURCE_PREFERENCE_RULE

    bucket = _SOURCE_PREFERENCE_RULE.split("AGENT SESSION", 1)[1].split("Judge each chunk", 1)[0]
    for agent in _AGENTS:
        assert agent in bucket, agent


def test_jev_session_class_covers_every_agent() -> None:
    from engine.retrieval.agent.jev import _DOC_CLASS_TABLE

    doc_types, meaning = _DOC_CLASS_TABLE["agent_sessions"]
    for source in AGENT_SESSION_SOURCES:
        assert f"{source.value}.session" in doc_types, source
        assert SOURCE_DISPLAY_NAMES[source] in meaning, source


def test_metadata_backfill_script_accepts_every_agent() -> None:
    from scripts import backfill_cc_metadata_chunks as backfill

    src = inspect.getsource(backfill._amain)
    for source in AGENT_SESSION_SOURCES:
        assert f"SourceSystem.{source.name}.value" in src, source


def test_connectors_doc_lists_every_agent() -> None:
    doc = (Path(__file__).parents[1] / "docs" / "connectors.md").read_text()
    for agent in _AGENTS:
        assert f"| `{agent}`" in doc, agent
