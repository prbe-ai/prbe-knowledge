"""Tool surface tests for the gatherer agent — fat-tool refactor.

Pure (no live DB) checks: schema sanity, registry coverage, dispatcher
edge cases, empty-input fast paths. Live execution paths are covered
indirectly by `test_loop.py` (mocked acompletion + dispatcher) and by
the per-retriever tests that already exist.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from services.retrieval.agent.tools import (
    NEED_DEEPER_TOOL_NAME,
    TERMINAL_TOOL_NAME,
    TOOL_REGISTRY,
    _clamp_top_k,
    _trim_properties,
    dispatch_tool_call,
    execute_fetch_doc,
    execute_search,
    execute_subgraph,
    tool_definitions,
)

# ============================================================
# Tool definitions: schema sanity
# ============================================================

def test_tool_definitions_contains_fat_tools_plus_terminal() -> None:
    """tool_definitions() exposes exactly: search, subgraph, fetch_doc,
    fetch_chunk_window, need_deeper, emit_gatherer_output. Drift breaks the
    prompt contract."""
    defs = tool_definitions()
    names = {d["function"]["name"] for d in defs}
    assert names == {
        "search",
        "subgraph",
        "fetch_doc",
        "fetch_chunk_window",
        NEED_DEEPER_TOOL_NAME,
        TERMINAL_TOOL_NAME,
    }


def test_registry_only_has_retrieval_tools() -> None:
    """TOOL_REGISTRY is the dispatcher table. need_deeper +
    emit_gatherer_output are handled by the loop itself and must NOT be
    in the registry (otherwise terminal-call detection breaks)."""
    assert set(TOOL_REGISTRY.keys()) == {
        "search",
        "subgraph",
        "fetch_doc",
        "fetch_chunk_window",
    }
    assert TERMINAL_TOOL_NAME not in TOOL_REGISTRY
    assert NEED_DEEPER_TOOL_NAME not in TOOL_REGISTRY


def test_fetch_doc_schema_has_offset_for_pagination() -> None:
    """fetch_doc is now a paginated reader — the `offset` param MUST be in
    the schema or the model can never page past the first chunk window."""
    defs = {d["function"]["name"]: d for d in tool_definitions()}
    props = defs["fetch_doc"]["function"]["parameters"]["properties"]
    assert "offset" in props
    assert props["offset"]["minimum"] == 0


def test_fetch_chunk_window_schema_shape() -> None:
    """fetch_chunk_window keys off the matched chunk_id (already in hand)
    and takes before/after neighbour counts."""
    defs = {d["function"]["name"]: d for d in tool_definitions()}
    fn = defs["fetch_chunk_window"]["function"]
    props = fn["parameters"]["properties"]
    assert set(fn["parameters"]["required"]) == {"chunk_id"}
    assert "before" in props and "after" in props


def test_tool_definitions_shape_is_openai_compatible() -> None:
    """Each tool def follows the OpenAI/Anthropic tool-use schema
    LiteLLM forwards verbatim."""
    for d in tool_definitions():
        assert d["type"] == "function"
        fn = d["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        if "required" in params:
            assert isinstance(params["required"], list)
            for r in params["required"]:
                assert isinstance(r, str)


def test_emit_gatherer_output_schema_is_gatherer_output_pydantic() -> None:
    """The terminal's parameters MUST be the GathererOutput JSON
    Schema. If we drift, the model can't terminate correctly."""
    from services.retrieval.agent.models import GathererOutput
    defs = tool_definitions()
    terminal = next(d for d in defs if d["function"]["name"] == TERMINAL_TOOL_NAME)
    schema = terminal["function"]["parameters"]
    expected = GathererOutput.model_json_schema()
    assert schema == expected


def test_top_k_clamping() -> None:
    """`_clamp_top_k` enforces [1, _HARD_TOP_K_CAP]; falls back to default."""
    assert _clamp_top_k(None, 15) == 15
    assert _clamp_top_k(0, 15) == 1
    assert _clamp_top_k(-5, 15) == 1
    assert _clamp_top_k(10, 15) == 10
    assert _clamp_top_k(99999, 15) == 100


# ============================================================
# Properties trimming
# ============================================================

def test_trim_properties_under_cap_returns_unchanged() -> None:
    small = {"name": "foo", "score": 0.95, "active": True}
    assert _trim_properties(small) == small


def test_trim_properties_over_cap_truncates_longest_string() -> None:
    big = {
        "name": "x",
        "long_field": "a" * 4000,
        "score": 0.99,
        "active": True,
    }
    out = _trim_properties(big)
    assert out["name"] == "x"
    assert out["score"] == 0.99
    assert out["active"] is True
    assert "TRUNCATED" in out["long_field"]
    assert len(out["long_field"]) < 4000


# ============================================================
# Dispatcher behavior
# ============================================================

@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_error_dict() -> None:
    result = await dispatch_tool_call("cust-1", "nonsense_tool", {})
    assert "error" in result
    assert "unknown" in result["error"].lower()


@pytest.mark.asyncio
async def test_dispatch_terminal_tool_returns_error_dict() -> None:
    """The loop intercepts emit_gatherer_output BEFORE dispatching.
    If it somehow reaches the dispatcher, the registry doesn't have it,
    so we get unknown-tool error — which is the right safety net."""
    result = await dispatch_tool_call("cust-1", TERMINAL_TOOL_NAME, {})
    assert "error" in result
    assert "unknown" in result["error"].lower()


@pytest.mark.asyncio
async def test_dispatch_handles_executor_exception() -> None:
    async def boom(*, customer_id: str, **kw: Any) -> dict[str, Any]:
        raise RuntimeError("postgres connection broken")

    with patch.dict(TOOL_REGISTRY, {"_test_boom": boom}, clear=False):
        result = await dispatch_tool_call("cust-1", "_test_boom", {})
        assert "error" in result
        assert "RuntimeError" in result["error"]


@pytest.mark.asyncio
async def test_dispatch_handles_argument_mismatch() -> None:
    async def strict(*, customer_id: str, expected_arg: str) -> dict[str, Any]:
        return {"got": expected_arg}

    with patch.dict(TOOL_REGISTRY, {"_test_strict": strict}, clear=False):
        result = await dispatch_tool_call(
            "cust-1", "_test_strict", {"wrong_arg_name": "x"}
        )
        assert "error" in result
        assert "argument" in result["error"].lower()


@pytest.mark.asyncio
async def test_dispatch_passes_customer_id_through() -> None:
    captured: dict[str, Any] = {}

    async def capture(**kw: Any) -> dict[str, Any]:
        captured.update(kw)
        return {}

    with patch.dict(TOOL_REGISTRY, {"_test_capture": capture}, clear=False):
        await dispatch_tool_call("cust-xyz", "_test_capture", {"foo": "bar"})
        assert captured.get("customer_id") == "cust-xyz"
        assert captured.get("foo") == "bar"


# ============================================================
# Empty / edge-case fast paths
# ============================================================

@pytest.mark.asyncio
async def test_search_empty_queries_short_circuits() -> None:
    """No queries → empty result; no DB or LLM call."""
    out = await execute_search("cust-1", queries=[])
    assert out == {"sub_queries": []}


@pytest.mark.asyncio
async def test_search_caps_at_five_subqueries() -> None:
    """Cap mirrors MAX_INTENTS+headroom. Extras dropped silently."""
    with patch(
        "services.retrieval.agent.tools._vector", new=AsyncMock(return_value=[])
    ), patch(
        "services.retrieval.agent.tools._bm25", new=AsyncMock(return_value=[])
    ), patch(
        "services.retrieval.agent.tools.build_bundle",
        new=AsyncMock(
            return_value=type("B", (), {
                "candidates": [], "bare_id_matches": [],
                "connected_sources": [], "timing_ms": 0.0,
            })()
        ),
    ):
        out = await execute_search(
            "cust-1", queries=[f"q{i}" for i in range(10)],
        )
        assert len(out["sub_queries"]) == 5


@pytest.mark.asyncio
async def test_search_threads_sort_and_author_ids_to_every_channel() -> None:
    """The `sort_by` + `author_ids` kwargs the harness derives from the
    extractor's `search_options` must reach EVERY channel — vector, bm25,
    graph, AND inferred_edge. A regression where only some channels
    honored the options would silently scatter results."""
    vec = AsyncMock(return_value=[])
    bm = AsyncMock(return_value=[])
    grp = AsyncMock(return_value=[])
    inf = AsyncMock(return_value=[])
    with patch(
        "services.retrieval.agent.tools._vector", new=vec
    ), patch(
        "services.retrieval.agent.tools._bm25", new=bm
    ), patch(
        "services.retrieval.agent.tools._graph", new=grp
    ), patch(
        "services.retrieval.agent.tools._inferred", new=inf
    ), patch(
        "services.retrieval.agent.tools._resolve_entities_to_anchor_docs",
        new=AsyncMock(return_value=["doc:1"]),  # non-empty so inferred fires
    ):
        await execute_search(
            "cust-1",
            queries=["what did mahit do last?"],
            entity_ids=[{"entity_type": "person", "canonical_id": "mahit@prbe.ai"}],
            author_ids=["mahit@prbe.ai"],
            sort_by="recency",
        )

    for chan, mock in (("vector", vec), ("bm25", bm), ("graph", grp), ("inferred", inf)):
        assert mock.await_count == 1, f"{chan} should fire once"
        kwargs = mock.await_args.kwargs
        assert kwargs.get("sort_by") == "recency", f"{chan} missing sort_by=recency"
        assert kwargs.get("author_ids") == ["mahit@prbe.ai"], f"{chan} missing author_ids"


@pytest.mark.asyncio
async def test_search_defaults_preserve_today_behavior() -> None:
    """Without sort_by / author_ids kwargs, channels see the same args
    as before the optimization landed — author_ids=None, sort_by="relevance".
    Regression guard for non-deterministic queries."""
    vec = AsyncMock(return_value=[])
    with patch(
        "services.retrieval.agent.tools._vector", new=vec
    ), patch(
        "services.retrieval.agent.tools._bm25", new=AsyncMock(return_value=[])
    ), patch(
        "services.retrieval.agent.tools._graph", new=AsyncMock(return_value=[])
    ), patch(
        "services.retrieval.agent.tools._inferred", new=AsyncMock(return_value=[])
    ), patch(
        "services.retrieval.agent.tools._resolve_entities_to_anchor_docs",
        new=AsyncMock(return_value=[]),
    ):
        await execute_search(
            "cust-1",
            queries=["how does auth work"],
            entity_ids=[{"entity_type": "feature", "canonical_id": "auth"}],
        )

    kwargs = vec.await_args.kwargs
    assert kwargs.get("sort_by") == "relevance"
    assert kwargs.get("author_ids") is None


def test_search_tool_schema_exposes_author_ids_and_sort_by() -> None:
    """The agent's `search` tool description must surface both new knobs
    so the model can use them on follow-up `search` calls (e.g. to
    refine after recognizing the harness picked a bad anchor)."""
    from services.retrieval.agent.tools import tool_definitions

    defs = {d["function"]["name"]: d for d in tool_definitions()}
    search_params = defs["search"]["function"]["parameters"]["properties"]

    assert "author_ids" in search_params
    assert search_params["author_ids"]["type"] == "array"
    assert "sort_by" in search_params
    assert sorted(search_params["sort_by"]["enum"]) == ["recency", "relevance"]


@pytest.mark.asyncio
async def test_subgraph_empty_anchor_returns_not_existed() -> None:
    """Empty anchor canonical_id can't be resolved → empty result, no crash."""
    with patch(
        "services.retrieval.main._resolve_anchor_alias",
        new=AsyncMock(return_value=""),
    ), patch(
        "services.retrieval.graph_explore.anchor_exists",
        new=AsyncMock(return_value=False),
    ):
        out = await execute_subgraph("cust-1", anchor_canonical_id="")
        assert out["anchor_existed"] is False
        assert out["nodes"] == []


@pytest.mark.asyncio
async def test_fetch_doc_minimal_call_returns_chunks_only() -> None:
    """Default args: no inferred edges, no evidence — just chunks."""
    class MockConn:
        async def fetch(self, sql: str, *args: Any) -> list:
            if "FROM chunks" in sql:
                return []
            return []

    class MockCtxMgr:
        async def __aenter__(self) -> MockConn:
            return MockConn()

        async def __aexit__(self, *args: Any) -> None:
            pass

    with patch(
        "services.retrieval.agent.tools.with_tenant",
        new=lambda customer_id: MockCtxMgr(),
    ):
        out = await execute_fetch_doc("cust-1", doc_id="doc:nonexistent")
        assert out["doc_id"] == "doc:nonexistent"
        assert out["chunks"] == []
        assert out["outbound_inferred_edges"] == []
        assert out["evidence_by_edge_id"] == {}
