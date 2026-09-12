"""Request-level PROJECT scope: pre-search, threaded everywhere (E7 phase 1).

`QueryRequest.scope.project_id` is a hard filter on
`documents.metadata->>'project_id'`. It must reach (1) every retrieval
channel's SQL, (2) the agent's in-loop content tools, (3) the identifier
lookup, and (4) the final live-row gate -- or the pool is scoped and the
answer is not, which is the post-filter this replaces. Pinned the way
`test_request_scope_threading.py` pins `sources`: at the signature and
predicate level, where a missing wire actually lives, plus one live-row
test for the gate itself.
"""

from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from engine.retrieval.agent import tools as tools_mod
from engine.retrieval.agent.adapter import _enforce_scope_on_chunks, to_query_response
from engine.retrieval.agent.loop import LoopState
from engine.retrieval.agent.models import GatheredChunk, GathererNotes, GathererOutput
from engine.retrieval.agent.tools import _doc_scope_sql, execute_search
from engine.retrieval.helpers import project_scope_predicate, source_key_predicate
from engine.retrieval.retrievers.id_lookup import lookup_identifiers
from engine.shared import db as db_module
from engine.shared.models import QueryRequest, RetrieveResponse, ScopeSpec

# ============================================================
# 1. The request carries a scope; the response echoes it
# ============================================================


def test_query_request_accepts_a_project_scope() -> None:
    req = QueryRequest(query="q", scope={"project_id": "proj-a"})
    assert req.scope is not None and req.scope.project_id == "proj-a"
    assert QueryRequest(query="q").scope is None


def test_scope_refuses_unknown_keys() -> None:
    """A misspelled scope key must fail loudly, not silently widen the
    search to the whole tenant."""
    with pytest.raises(ValidationError):
        QueryRequest(query="q", scope={"workspace": "w"})
    with pytest.raises(ValidationError):
        ScopeSpec(project_id="")


def test_response_can_echo_the_applied_scope() -> None:
    assert "applied_scope" in RetrieveResponse.model_fields


# ============================================================
# 2. The wire: every hop accepts `project_id`
# ============================================================


def test_execute_search_accepts_project_id() -> None:
    assert "project_id" in inspect.signature(execute_search).parameters


def test_loop_state_carries_request_project_id() -> None:
    assert "request_project_id" in LoopState.__dataclass_fields__


def test_every_channel_retriever_accepts_project_id() -> None:
    from engine.retrieval.retrievers.bm25 import bm25_search
    from engine.retrieval.retrievers.graph import graph_search
    from engine.retrieval.retrievers.inferred_edges import inferred_edge_search
    from engine.retrieval.retrievers.vector import vector_search

    for fn in (vector_search, bm25_search, graph_search, inferred_edge_search):
        assert "project_id" in inspect.signature(fn).parameters, fn.__name__


def test_identifier_lookup_accepts_project_id() -> None:
    """Pins are exact hits merged ahead of the ranked channels; an unscoped
    pin would put an out-of-project doc at rank 1 of a scoped answer."""
    assert "project_id" in inspect.signature(lookup_identifiers).parameters


def test_every_tool_that_takes_the_keyless_flag_takes_the_project_scope() -> None:
    """The loop injects request scope into a fixed tool set (`search`,
    `fetch_doc`, `fetch_chunk_window`, `subgraph`). The keyless flag marks
    exactly the tools that already take request scope, so any of them
    missing `project_id` would raise TypeError on injection -- or worse,
    silently fetch out-of-scope content if the injection were guarded."""
    checked = 0
    for name, fn in inspect.getmembers(tools_mod, inspect.iscoroutinefunction):
        params = inspect.signature(fn).parameters
        if "source_keys_include_keyless" in params:
            assert "project_id" in params, name
            checked += 1
    assert checked >= 3, "expected the scope-gated tools to be found"


def test_adapter_gate_accepts_project_id() -> None:
    assert "project_id" in inspect.signature(to_query_response).parameters
    assert "project_id" in inspect.signature(_enforce_scope_on_chunks).parameters


# ============================================================
# 3. One predicate, restated identically at every gate
# ============================================================


def test_project_predicate_is_a_hard_filter_on_metadata() -> None:
    params: list = ["cust"]
    sql = project_scope_predicate(params, "proj-a", alias="d")
    assert sql == "AND d.metadata->>'project_id' = $2"
    assert params == ["cust", "proj-a"]


def test_project_predicate_is_empty_without_a_scope() -> None:
    params: list = ["cust"]
    assert project_scope_predicate(params, None, alias="d") == ""
    assert params == ["cust"]


def test_in_loop_gate_restates_the_channel_predicate() -> None:
    """`_doc_scope_sql` (what fetch_doc/fetch_chunk_window/subgraph check
    before serving content) must build the SAME predicate the channels
    filter on. If they drift, the agent either reads what the channels
    excluded or is refused what they returned."""
    channel_params: list = []
    channel_sql = project_scope_predicate(channel_params, "proj-a", alias="d")
    gate_params: list = []
    gate_sql = _doc_scope_sql(
        gate_params, alias="d", source_keys=None, doc_types=None, project_id="proj-a"
    )
    assert gate_sql == channel_sql
    assert gate_params == channel_params == ["proj-a"]


def test_project_scope_composes_with_source_keys() -> None:
    """Both scopes AND together; parameter numbering stays contiguous."""
    params: list = []
    keyed = source_key_predicate(params, ["ws:1"], alias="d")
    proj = project_scope_predicate(params, "proj-a", alias="d")
    assert keyed == "AND d.metadata->>'source_key' = ANY($1::text[])"
    assert proj == "AND d.metadata->>'project_id' = $2"
    gate_params: list = []
    gate_sql = _doc_scope_sql(
        gate_params, alias="d", source_keys=["ws:1"], doc_types=None, project_id="proj-a"
    )
    assert "source_key" in gate_sql and "project_id" in gate_sql
    # source_keys binds as ONE array parameter; project_id as one scalar.
    assert gate_params == params == [["ws:1"], "proj-a"]


# ============================================================
# 4. The live-row gate
# ============================================================


def _chunk(doc_id: str) -> GatheredChunk:
    return GatheredChunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}:c0",
        content="...",
        matched_via=["vector"],
        why_relevant="test",
    )


async def _seed_project_docs(customer_id: str) -> None:
    now = datetime.now(UTC)
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'Project scope ' || $1, 'hash-' || $1)
            ON CONFLICT DO NOTHING
            """,
            customer_id,
        )
        await conn.executemany(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_preview, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, valid_to, ingested_at, acl,
                metadata
            ) VALUES (
                $1, 1, $2, 'custom_ingest', $1, '/x/' || $1,
                'raw_source', 'custom.experiment.run', 'text/plain',
                'h-' || $1, $1, 'body', 4, 1,
                $3, $3, $3, NULL, $3, '{}'::jsonb,
                $4::jsonb
            )
            """,
            [
                ("doc-a", customer_id, now, json.dumps({"project_id": "proj-a"})),
                ("doc-b", customer_id, now, json.dumps({"project_id": "proj-b"})),
                ("doc-none", customer_id, now, json.dumps({})),
            ],
        )


async def test_live_row_gate_keeps_only_the_scoped_project(live_db: None) -> None:
    """The gatherer is an LLM loop: navigation can hand it an out-of-scope
    doc_id. The gate re-verifies every emitted chunk against the live
    `documents` row. A document with NO project_id is out of a project
    scope -- a scope is a filter, not a boost."""
    cid = "test-cust-project-scope-gate"
    await _seed_project_docs(cid)
    gathered = GathererOutput(
        chunks=[_chunk("doc-a"), _chunk("doc-b"), _chunk("doc-none")],
        gatherer_notes=GathererNotes(),
    )
    await _enforce_scope_on_chunks(
        cid, gathered, source_keys=None, doc_types=None, trace_id="t", project_id="proj-a"
    )
    assert [c.doc_id for c in gathered.chunks] == ["doc-a"]


async def test_live_row_gate_without_a_project_scope_keeps_every_live_doc(
    live_db: None,
) -> None:
    cid = "test-cust-project-scope-gate-none"
    await _seed_project_docs(cid)
    gathered = GathererOutput(
        chunks=[_chunk("doc-a"), _chunk("doc-b"), _chunk("doc-none")],
        gatherer_notes=GathererNotes(),
    )
    await _enforce_scope_on_chunks(
        cid, gathered, source_keys=None, doc_types=None, trace_id="t"
    )
    assert [c.doc_id for c in gathered.chunks] == ["doc-a", "doc-b", "doc-none"]


async def test_response_echoes_the_project_scope() -> None:
    """A caller that pre-fills a project scope needs to tell "scoped and
    empty" from "the server ignored the scope"; `applied_scope` is that
    signal, mirroring `applied_sources`."""
    gathered = GathererOutput(chunks=[], gatherer_notes=GathererNotes())
    resp = await to_query_response(
        query="q",
        gathered=gathered,
        trace_id="t",
        timing_ms={},
        status=None,
        project_id="proj-a",
    )
    assert resp.applied_scope == {"project_id": "proj-a"}
    resp2 = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, status=None
    )
    assert resp2.applied_scope is None
