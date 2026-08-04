"""Request-level scope threading: `sources` + keyless tolerance.

Two bugs this file pins, both of which shipped green because nothing
asserted the wiring:

1. `QueryRequest.sources` was accepted, enum-validated, and then never
   threaded into the gatherer path — `execute_search` had no `sources`
   parameter at all. `/retrieve` with `sources=["claude_code"]` returned
   github docs, so every corpus-scoped caller post-filtered a diluted
   pool.

2. `source_keys_include_keyless` reached the retrieval CHANNELS but not
   the scope gates. The channels admitted keyless connector docs
   (github, claude_code), then `_doc_scope_sql` refused to let the agent
   read them and `_enforce_scope_on_chunks` dropped every one from the
   response — `dropped=10, kept=0` on a live trace.

Both are asserted at the predicate level, where the defect actually
lived, rather than through a mock that would pass either way.
"""

from __future__ import annotations

import inspect

from engine.retrieval.agent.adapter import to_query_response
from engine.retrieval.agent.loop import LoopState
from engine.retrieval.agent.tools import _doc_scope_sql, execute_search
from engine.retrieval.helpers import source_key_predicate

# ============================================================
# 1. `sources` reaches the search tool + every channel
# ============================================================

def test_execute_search_accepts_sources() -> None:
    """The tool the gatherer actually dispatches must be able to express
    the caller's source-system scope. Without this parameter the filter
    is unrepresentable and `sources` is a guaranteed no-op."""
    assert "sources" in inspect.signature(execute_search).parameters


def test_loop_state_carries_request_sources() -> None:
    """The loop injects request scope on every dispatch (the tool schema
    deliberately does not expose it). No state field => no injection."""
    assert "request_sources" in LoopState.__dataclass_fields__


def test_every_channel_retriever_accepts_sources() -> None:
    """All four channels must filter, or the unfiltered ones re-dilute
    the pool the filtered ones just narrowed."""
    from engine.retrieval.retrievers.bm25 import bm25_search
    from engine.retrieval.retrievers.graph import graph_search
    from engine.retrieval.retrievers.inferred_edges import inferred_edge_search
    from engine.retrieval.retrievers.vector import vector_search

    for fn in (vector_search, bm25_search, graph_search, inferred_edge_search):
        assert "sources" in inspect.signature(fn).parameters, fn.__name__


def test_adapter_gate_accepts_sources() -> None:
    """The response choke point re-verifies scope against live rows; it
    has to know about `sources` or graph/inferred hits bypass it."""
    assert "sources" in inspect.signature(to_query_response).parameters


# ============================================================
# 2. keyless tolerance is consistent across channel + gate
# ============================================================

def test_doc_scope_sql_hard_filters_by_default() -> None:
    """Default stays a hard filter — keyless docs excluded. This is the
    long-standing behaviour and must not drift."""
    params: list = []
    sql = _doc_scope_sql(
        params, alias="d", source_keys=["shared:probe"], doc_types=None
    )
    assert "source_key' = ANY($1::text[])" in sql
    assert "IS NULL" not in sql
    assert params == [["shared:probe"]]


def test_doc_scope_sql_admits_keyless_when_requested() -> None:
    """With keyless tolerance the gate must admit `source_key IS NULL`
    docs. When it did not, fetch_doc refused the github docs the same
    request's `search` had just returned."""
    params: list = []
    sql = _doc_scope_sql(
        params,
        alias="d",
        source_keys=["shared:probe"],
        doc_types=None,
        source_keys_include_keyless=True,
    )
    assert "IS NULL" in sql
    assert " OR " in sql


def test_gate_predicate_matches_channel_predicate() -> None:
    """The gate and the channels must agree on what keyless tolerance
    means. They are separate implementations, so pin them to the same
    shape — a divergence here is exactly the bug that emptied responses
    while total_candidates stayed non-zero."""
    gate_params: list = []
    gate = _doc_scope_sql(
        gate_params,
        alias="d",
        source_keys=["shared:probe"],
        doc_types=None,
        source_keys_include_keyless=True,
    )
    chan_params: list = []
    chan = source_key_predicate(
        chan_params, ["shared:probe"], alias="d", include_keyless=True
    )
    assert gate.strip() == chan.strip()
    assert gate_params == chan_params


def test_scope_gated_tools_accept_keyless_flag() -> None:
    """fetch_doc / fetch_chunk_window / subgraph each run the scope gate.
    A tool missing the flag silently reverts to the hard filter and
    refuses in-scope connector docs."""
    from engine.retrieval.agent.tools import (
        execute_fetch_chunk_window,
        execute_fetch_doc,
        execute_subgraph,
    )

    for fn in (execute_fetch_doc, execute_fetch_chunk_window, execute_subgraph):
        params = inspect.signature(fn).parameters
        assert "source_keys_include_keyless" in params, fn.__name__


def test_adapter_gate_accepts_keyless_flag() -> None:
    assert (
        "source_keys_include_keyless"
        in inspect.signature(to_query_response).parameters
    )
