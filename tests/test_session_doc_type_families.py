"""Every coding agent gets its own doc-type family.

Codex and pi sessions were typed `claude_code.*` for 10,191 documents. It was
deliberate -- same parsing, same coalescing, same staleness curve, only the
provenance label differs -- and it still meant `doc_type`, the field a reader
sees on a hit, named an agent that did not produce the document.
"""

from __future__ import annotations

import pytest

from engine.retrieval.doc_type_resolver import resolve_doc_type_token
from engine.shared.constants import DocType, SourceSystem
from kb.handlers.claude_code import (
    ClaudeCodeConnector,
    CodexConnector,
    PiConnector,
)

_KINDS = ("session", "qa", "code_change", "decision", "file_ref", "directive")
_CONNECTORS = (ClaudeCodeConnector, CodexConnector, PiConnector)


def test_each_connector_has_its_own_prefix() -> None:
    prefixes = {c.doc_type_prefix for c in _CONNECTORS}
    assert prefixes == {"claude_code.", "codex.", "pi."}
    assert len(prefixes) == len(_CONNECTORS), "two connectors share a family"


def test_a_connectors_prefix_names_its_own_source() -> None:
    """The invariant every OTHER source in the enum already holds: the dotted
    prefix is the source's own name. Inheriting one is how this broke."""
    for connector in _CONNECTORS:
        assert connector.doc_type_prefix == f"{connector.source_system.value}."


def test_every_family_is_complete() -> None:
    """A missing member makes `_doc_type` raise at INGEST, which is the loud
    failure we want -- but only if the member is actually there for all six
    kinds of all three agents."""
    for connector in _CONNECTORS:
        for kind in _KINDS:
            doc_type = connector._doc_type(kind)
            assert doc_type.value == f"{connector.doc_type_prefix}{kind}"


def test_a_connector_cannot_silently_emit_its_parents_family() -> None:
    """THE REGRESSION GUARD.

    `_doc_type` composes from `doc_type_prefix`, so a subclass that changes the
    prefix changes every emitted type with it. The old code named
    `DocType.CLAUDE_CODE_*` literally at each call site, so a subclass
    inherited the parent's family by simply not overriding anything -- which is
    exactly what happened, silently, for 10,191 documents.
    """
    assert CodexConnector._doc_type("session") is DocType.CODEX_SESSION
    assert PiConnector._doc_type("session") is DocType.PI_SESSION
    assert ClaudeCodeConnector._doc_type("session") is DocType.CLAUDE_CODE_SESSION

    # A subclass that sets only a prefix gets a whole correct family for free.
    class _FutureAgent(ClaudeCodeConnector):
        doc_type_prefix = "codex."  # stand-in for a real new family

    assert _FutureAgent._doc_type("qa") is DocType.CODEX_QA


def test_an_unknown_family_raises_rather_than_falling_back() -> None:
    """Silently falling back to the parent's family is the bug. A prefix with
    no enum members must fail at the first ingest."""

    class _Unregistered(ClaudeCodeConnector):
        doc_type_prefix = "nosuchagent."

    with pytest.raises(ValueError):
        _Unregistered._doc_type("session")


# ------------------------------------------------------------ the resolver

def test_a_bare_session_token_reaches_every_agent() -> None:
    """"session" is what a person says when they mean a coding-agent
    conversation, and they almost never mean one vendor's. Resolving it to
    Claude Code alone would now DROP the Codex and pi sessions beside it --
    a regression the split would otherwise have introduced."""
    resolved = resolve_doc_type_token("session")
    assert set(resolved) == {"claude_code.session", "codex.session", "pi.session"}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (SourceSystem.CLAUDE_CODE, "claude_code.session"),
        (SourceSystem.CODEX, "codex.session"),
        (SourceSystem.PI, "pi.session"),
    ],
)
def test_naming_a_source_narrows_to_that_agents_family(source, expected) -> None:
    assert resolve_doc_type_token("session", [source]) == [expected]


# ------------------------------------------------------------ the backfill

def test_the_backfill_is_driven_by_source_system_not_the_old_doc_type() -> None:
    """The existing doc_type is the thing that is WRONG, so it cannot be the
    input. A row is re-typed because `source_system` says which agent wrote
    it."""
    import inspect

    from scripts import backfill_session_doc_types as backfill

    src = inspect.getsource(backfill._backfill_tenant)
    assert "source_system = $2" in src


def test_the_backfill_round_trips() -> None:
    """`--revert` has to be lossless, because source_system still carries the
    truth: the rewrite can be undone and redone without reading the old value."""
    from scripts.backfill_session_doc_types import _mapping

    forward = _mapping(revert=False)
    back = _mapping(revert=True)
    for source in ("codex", "pi"):
        for old, new in forward[source].items():
            assert back[source][new] == old
