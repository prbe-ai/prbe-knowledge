"""AgentSession is the join between a transcript and the runs it produced.

These assert the two things that make it reachable, both of which shipped broken
once: the reverse label mapping (grounding discards a candidate whose label does
not reverse) and the node's NAME (grounding matches on it).
"""

from __future__ import annotations

import pytest

from engine.shared.constants import (
    GROUNDING_ADDRESSABLE_ENTITY_TYPES,
    GROUNDING_ENTITY_LABELS,
    LLM_EXTRACTABLE_ENTITY_TYPES,
    ROUTER_ENTITY_TO_LABEL,
    NodeLabel,
    agent_session_canonical_id,
    agent_session_display_name,
    entity_type_for_node,
)

SESSION = "b2db51b4-3a03-4251-91ac-7dd2a0640781"


@pytest.mark.parametrize("label", list(GROUNDING_ENTITY_LABELS))
def test_every_groundable_label_reverses_to_an_entity_type(label: str) -> None:
    """Grounding scans these labels, then does `if not entity_type: continue`.

    A label it may scan but cannot reverse is silently discarded AFTER the SQL
    matched it: the node is addressable, extractable, matched, and unusable.
    AgentSession shipped in exactly that state.
    """
    assert entity_type_for_node(label, None) is not None


def test_agent_session_is_wired_through_every_derived_map() -> None:
    assert ROUTER_ENTITY_TO_LABEL["agent_session"] is NodeLabel.AGENT_SESSION
    assert entity_type_for_node("AgentSession", None) == "agent_session"
    assert "AgentSession" in GROUNDING_ENTITY_LABELS
    assert "agent_session" in GROUNDING_ADDRESSABLE_ENTITY_TYPES
    assert "agent_session" in LLM_EXTRACTABLE_ENTITY_TYPES


def test_ambiguous_labels_keep_their_hand_picked_winner() -> None:
    """Deriving the reverse map must not change what Document/CodeSymbol mean:
    several kind-less specs share those labels."""
    assert entity_type_for_node("Document", None) == "document"
    assert entity_type_for_node("CodeSymbol", None) == "symbol"


def test_canonical_id_is_the_cross_repo_join_key() -> None:
    """research-os composes this exact string independently. If it changes here
    without changing there, the run and its transcript stop being neighbours and
    nothing raises."""
    assert (
        agent_session_canonical_id("claude_code", SESSION)
        == f"agent_session:claude_code:{SESSION}"
    )


def test_display_name_carries_the_full_session_id() -> None:
    """The first version used the session document's title, which truncates the
    id to 8 characters, so a query naming the real session could not match it."""
    name = agent_session_display_name("claude_code", SESSION, "Richard Wei")
    assert SESSION in name, "an id query is the precise one; it must be matchable"
    assert "Richard Wei" in name, "human phrasing needs the person"
    assert "@" not in name, "an email's trigrams sink similarity below threshold"


def test_display_name_degrades_without_an_identity() -> None:
    """employee_name is genuinely optional. The session must still be reachable
    by id; only the person-flavoured query is lost."""
    name = agent_session_display_name("claude_code", SESSION)
    assert name == f"claude code session {SESSION}"


def test_agent_label_underscores_are_spaced_for_the_tsvector_channel() -> None:
    """`claude_code` is one token to plainto_tsquery; `claude code` is two."""
    assert agent_session_display_name("claude_code", SESSION).startswith("claude code ")
    # A label with no underscore is unaffected.
    assert agent_session_display_name("codex", SESSION).startswith("codex session ")
