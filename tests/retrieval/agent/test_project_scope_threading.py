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
import re
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from engine.retrieval.agent import loop as loop_mod
from engine.retrieval.agent import tools as tools_mod
from engine.retrieval.agent.adapter import (
    _enforce_scope_on_chunks,
    _scope_graph_evidence,
    to_query_response,
)
from engine.retrieval.agent.loop import LoopState, _build_user_message, _execute_tool_call
from engine.retrieval.agent.models import GatheredChunk, GathererNotes, GathererOutput
from engine.retrieval.agent.tools import (
    _SCOPE_REFUSAL_NOTE,
    _doc_scope_sql,
    execute_fetch_chunk_window,
    execute_fetch_doc,
    execute_search,
)
from engine.retrieval.grounding import GroundingBundle
from engine.retrieval.helpers import project_scope_predicate, source_key_predicate
from engine.retrieval.retrievers.id_lookup import lookup_identifiers
from engine.shared import db as db_module
from engine.shared.models import GraphEvidence, QueryRequest, RetrieveResponse, ScopeSpec
from tests.retrieval.conftest import RecordingConn

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
    checked: set[str] = set()
    for name, fn in inspect.getmembers(tools_mod, inspect.iscoroutinefunction):
        params = inspect.signature(fn).parameters
        if "source_keys_include_keyless" in params:
            assert "project_id" in params, name
            checked.add(name)
    # The four executors the loop injects into must be among them (helpers
    # that take the flag are checked too, but these four are load-bearing).
    assert {
        "execute_search", "execute_fetch_doc", "execute_fetch_chunk_window", "execute_subgraph"
    } <= checked, checked


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
        await conn.executemany(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count, kind,
                embedding, first_seen_version, last_seen_version
            ) VALUES (
                $1 || ':c0', $1, $2, 0, 'body of ' || $1, 'chash-' || $1, 3, 'content',
                array_fill(0::real, ARRAY[3072])::halfvec, 1, 1
            )
            """,
            [("doc-a", customer_id), ("doc-b", customer_id), ("doc-none", customer_id)],
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


async def test_response_gate_fires_for_a_project_only_scope(live_db: None) -> None:
    """The gate's trigger is `(source_keys or doc_types or sources or
    project_id) and customer_id`. A project-only scope must fire it: the two
    direct gate tests would stay green if `or project_id` were reverted."""
    cid = "test-cust-project-scope-response"
    await _seed_project_docs(cid)
    gathered = GathererOutput(
        chunks=[_chunk("doc-a"), _chunk("doc-b"), _chunk("doc-none")],
        gatherer_notes=GathererNotes(),
    )
    resp = await to_query_response(
        query="q",
        gathered=gathered,
        trace_id="t",
        timing_ms={},
        status=None,
        customer_id=cid,
        project_id="proj-a",
    )
    assert {r.doc_id for r in resp.results} == {"doc-a"}
    assert resp.applied_scope == {"project_id": "proj-a"}


# ============================================================
# 5. Graph evidence is scoped like the chunks (review: neighbour leak)
# ============================================================


def _ev(via: str) -> GraphEvidence:
    return GraphEvidence(edge_type="motivates", confidence="INFERRED", via_entity=via)


async def test_graph_evidence_drops_out_of_scope_document_neighbours(live_db: None) -> None:
    """An in-scope result's inferred edge to an out-of-project DOCUMENT must
    not carry that document into a scoped response; an edge to an entity
    node (no documents row) has nothing to leak and is kept."""
    cid = "test-cust-project-scope-evidence"
    await _seed_project_docs(cid)
    doc_evidence = {
        "doc-a": [_ev("doc-b"), _ev("person:alice"), _ev("doc-none")],
        "doc-b": [_ev("doc-a")],
    }
    await _scope_graph_evidence(
        cid, doc_evidence, source_keys=None, doc_types=None, project_id="proj-a", trace_id="t"
    )
    assert [e.via_entity for e in doc_evidence["doc-a"]] == ["person:alice"]
    # doc-b's own entry survives (its neighbour doc-a IS in scope); the chunk
    # gate is what removes doc-b as a RESULT, not this filter.
    assert [e.via_entity for e in doc_evidence["doc-b"]] == ["doc-a"]


# ============================================================
# 6. The model is told a project scope is in force
# ============================================================


def test_user_message_names_the_project_scope() -> None:
    """Without this the model first learns of the scope from a refused
    fetch_doc and burns tool calls hunting for documents it can never see."""
    msg = _build_user_message("q", GroundingBundle(), None, project_id="proj-a")
    assert "<search_options>" in msg
    assert "project_id=proj-a" in msg
    assert "`project_id` scope is caller-enforced" in msg


def test_user_message_has_no_options_block_without_a_scope() -> None:
    """Prompt-cache stability: nothing new renders for a vanilla query."""
    msg = _build_user_message("q", GroundingBundle(), None)
    assert "<search_options>" not in msg


# ============================================================
# 7. Every channel's SQL carries the predicate with the value bound
# ============================================================


def _patch_conn(monkeypatch: pytest.MonkeyPatch, module: object) -> RecordingConn:
    from contextlib import asynccontextmanager

    conn = RecordingConn()

    @asynccontextmanager
    async def _fake_with_tenant(customer_id: str):  # type: ignore[no-untyped-def]
        yield conn

    monkeypatch.setattr(module, "with_tenant", _fake_with_tenant)
    return conn


def _assert_project_bound(conn: RecordingConn, channel: str) -> None:
    hits = [
        (sql, params)
        for sql, params in conn.fetched
        if re.search(r"metadata->>'project_id' = \$(\d+)", sql)
    ]
    assert hits, f"{channel}: no statement carried the project predicate:\n" + "\n---\n".join(
        s for s, _ in conn.fetched
    )
    for sql, params in hits:
        idx = int(re.search(r"metadata->>'project_id' = \$(\d+)", sql).group(1))
        assert params[idx - 1] == "proj-a", (channel, sql, params)


async def test_vector_channel_binds_the_project_predicate(recorded: RecordingConn) -> None:
    from engine.retrieval.retrievers.vector import vector_search

    await vector_search(customer_id="c1", query_text="q", top_k=5, project_id="proj-a")
    _assert_project_bound(recorded, "vector")


async def test_bm25_channel_binds_the_project_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.retrieval.retrievers import bm25 as bm25_mod

    conn = _patch_conn(monkeypatch, bm25_mod)
    await bm25_mod.bm25_search("c1", "payments api timeout", top_k=5, project_id="proj-a")
    _assert_project_bound(conn, "bm25")


async def test_inferred_edge_channel_binds_the_project_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engine.retrieval.retrievers import inferred_edges as ie_mod

    conn = _patch_conn(monkeypatch, ie_mod)
    await ie_mod.inferred_edge_search("c1", ["doc-x"], project_id="proj-a")
    _assert_project_bound(conn, "inferred_edge")


async def test_graph_channel_binds_the_project_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.retrieval.retrievers import graph as graph_mod

    conn = _patch_conn(monkeypatch, graph_mod)
    await graph_mod.graph_search("c1", [("issue", "linear:org:issue:prb-1")], project_id="proj-a")
    _assert_project_bound(conn, "graph")


async def test_id_lookup_binds_the_project_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.retrieval.retrievers import id_lookup as id_mod

    conn = _patch_conn(monkeypatch, id_mod)
    await id_mod.id_lookup_search("c1", ["PRB-1"], project_id="proj-a")
    _assert_project_bound(conn, "id_lookup")


# ============================================================
# 8. The in-loop content tools consult the scope when ONLY project_id is set
# ============================================================


async def test_fetch_doc_refuses_documents_outside_the_project(live_db: None) -> None:
    cid = "test-cust-project-scope-fetch"
    await _seed_project_docs(cid)
    served = await execute_fetch_doc(cid, doc_id="doc-a", project_id="proj-a")
    assert [c["chunk_id"] for c in served["chunks"]] == ["doc-a:c0"]
    for out_of_scope in ("doc-b", "doc-none"):
        refused = await execute_fetch_doc(cid, doc_id=out_of_scope, project_id="proj-a")
        assert refused["chunks"] == [], out_of_scope
        assert refused.get("note") == _SCOPE_REFUSAL_NOTE, out_of_scope


async def test_fetch_chunk_window_refuses_chunks_outside_the_project(live_db: None) -> None:
    cid = "test-cust-project-scope-window"
    await _seed_project_docs(cid)
    served = await execute_fetch_chunk_window(cid, chunk_id="doc-a:c0", project_id="proj-a")
    assert [c["chunk_id"] for c in served["chunks"]] == ["doc-a:c0"]
    refused = await execute_fetch_chunk_window(cid, chunk_id="doc-b:c0", project_id="proj-a")
    assert refused["chunks"] == []


# ============================================================
# 9. The loop injects the scope and strips model-supplied scope keys
# ============================================================


def _tool_call(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


@pytest.mark.parametrize("tool", ["search", "fetch_doc", "fetch_chunk_window", "subgraph"])
async def test_loop_injects_project_scope_and_strips_model_supplied_scope_keys(
    monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    """A non-strict provider can emit harness-owned keys. Every one is
    discarded and the request scope re-applied: the model can neither widen
    (a keyless flag on a keyed scope) nor redirect (its own project_id)."""
    dispatch = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(loop_mod, "dispatch_tool_call", dispatch)
    state = LoopState(
        customer_id="c1",
        trace_id="t",
        query="q",
        request_project_id="proj-a",
        request_source_keys=["ws:1"],
    )
    model_args = {
        "query" if tool == "search" else ("chunk_id" if tool == "fetch_chunk_window" else "doc_id"): "x",
        "project_id": "proj-b",
        "source_keys_include_keyless": True,
        "source_keys": ["ws:evil"],
    }
    await _execute_tool_call(state, _tool_call(tool, model_args))
    sent = dispatch.await_args.kwargs["arguments"]
    assert sent["project_id"] == "proj-a"
    assert sent["source_keys"] == ["ws:1"]
    assert "source_keys_include_keyless" not in sent


async def test_loop_sends_no_scope_keys_when_the_request_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(loop_mod, "dispatch_tool_call", dispatch)
    state = LoopState(customer_id="c1", trace_id="t", query="q")
    await _execute_tool_call(state, _tool_call("fetch_doc", {"doc_id": "d", "project_id": "proj-b"}))
    sent = dispatch.await_args.kwargs["arguments"]
    assert "project_id" not in sent and "source_keys" not in sent
