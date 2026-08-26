"""`_AGENT_SESSION_SOURCES` gates whether inferred-edge enqueue is deferred
to session completion (agent sources) or fires per-doc (everyone else).

Not one of the sites the pi-source task named, and not required to keep any
pre-existing test green -- there was no coverage of this predicate at all
before this file. Added because omitting pi here reintroduces the exact
cost problem the module docstring describes for Claude Code: "a single long
session ran dozens-to-hundreds of redundant LLM calls on a doc that wasn't
done growing." Without pi in this tuple, pi sessions would take the
per-append path (return doc_ids unconditionally, see engine/ingest/
normalizer.py::_inferred_edge_doc_ids) instead of deferring to
session_complete like claude_code/codex do.
"""

from __future__ import annotations

from types import SimpleNamespace

from engine.ingest.normalizer import _inferred_edge_doc_ids
from engine.shared.constants import SourceSystem


def _doc(session_complete: bool) -> SimpleNamespace:
    return SimpleNamespace(metadata={"session_complete": session_complete})


def test_pi_defers_to_session_complete_like_codex() -> None:
    """Mid-session (no doc yet carries session_complete=True): nothing
    enqueues for pi, exactly like codex and claude_code."""
    doc_ids = ["pi:cust-1:s-1"]
    documents = [_doc(session_complete=False)]

    assert _inferred_edge_doc_ids(SourceSystem.PI, doc_ids, documents) == []
    assert _inferred_edge_doc_ids(SourceSystem.CODEX, doc_ids, documents) == []


def test_pi_enqueues_everything_once_session_completes() -> None:
    doc_ids = ["pi:cust-1:s-1", "pi:cust-1:s-1:decision:0"]
    documents = [_doc(session_complete=True)]

    assert _inferred_edge_doc_ids(SourceSystem.PI, doc_ids, documents) == doc_ids


def test_non_agent_source_is_unaffected_by_pi_addition() -> None:
    """Sanity: widening _AGENT_SESSION_SOURCES to include pi must not touch
    the per-doc (non-agent) path other sources take."""
    doc_ids = ["slack:cust-1:msg-1"]
    documents = [_doc(session_complete=False)]

    assert _inferred_edge_doc_ids(SourceSystem.SLACK, doc_ids, documents) == doc_ids
