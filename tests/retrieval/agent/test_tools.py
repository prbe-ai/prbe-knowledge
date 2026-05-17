"""Tool surface sanity tests for the gatherer agent.

Pure (no live DB) checks: every tool schema is valid against LiteLLM's
expected shape, the dispatcher gracefully handles unknown/bad input,
top_k clamping behaves, and the registry is complete relative to the
prompt's documented tool list. Live-DB execution paths are validated
indirectly through `test_loop.py` (mock acompletion + dispatch through
tool_call results) and via the per-retriever tests that already exist
(`test_bm25_*`, `test_inferred_edges_retriever`, etc.).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from services.retrieval.agent.tools import (
    TOOL_REGISTRY,
    _clamp_top_k,
    _trim_properties,
    dispatch_tool_call,
    tool_definitions,
)

# ============================================================
# Tool definitions: schema sanity
# ============================================================

def test_tool_definitions_count_matches_registry() -> None:
    """`tool_definitions()` must surface schemas for every retrieval tool
    in TOOL_REGISTRY plus `need_deeper` (which the loop handles directly,
    not via the registry)."""
    defs = tool_definitions()
    names = {d["function"]["name"] for d in defs}
    expected_retrieval_tools = set(TOOL_REGISTRY.keys())
    # need_deeper is a budget-extension signal, not a retrieval call —
    # the loop intercepts it before dispatching. Schema is still exposed.
    assert names == expected_retrieval_tools | {"need_deeper"}, (
        f"tool defs ({names}) drifted from registry ({expected_retrieval_tools})"
    )


def test_tool_definitions_shape_is_openai_compatible() -> None:
    """Each tool def must follow the OpenAI/Anthropic tool-use schema
    LiteLLM forwards verbatim. Anything missing trips a provider-side
    validation error at first call — fail in tests instead."""
    for d in tool_definitions():
        assert d["type"] == "function"
        fn = d["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        # `required` is optional in JSON Schema; if present must be list[str].
        if "required" in params:
            assert isinstance(params["required"], list)
            for r in params["required"]:
                assert isinstance(r, str)
                assert r in params["properties"], (
                    f"tool {fn['name']}: required key '{r}' missing from properties"
                )


def test_top_k_clamping() -> None:
    """`_clamp_top_k` must enforce [1, _HARD_TOP_K_CAP] and fall back to
    the per-tool default when None."""
    assert _clamp_top_k(None, 15) == 15
    assert _clamp_top_k(0, 15) == 1
    assert _clamp_top_k(-5, 15) == 1
    assert _clamp_top_k(10, 15) == 10
    assert _clamp_top_k(99999, 15) == 100  # _HARD_TOP_K_CAP


# ============================================================
# Properties trimming
# ============================================================

def test_trim_properties_under_cap_returns_unchanged() -> None:
    small = {"name": "foo", "score": 0.95, "active": True}
    assert _trim_properties(small) == small


def test_trim_properties_over_cap_truncates_longest_string() -> None:
    """When properties JSON exceeds ~2KB, the longest str-valued field
    gets truncated. Numeric / bool fields are preserved verbatim."""
    big = {
        "name": "x",
        "long_field": "a" * 4000,  # ~4KB string — should be truncated
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
    """Model can misfire and call a tool that doesn't exist. Dispatcher
    must NOT raise — return an error dict so the agent can recover."""
    result = await dispatch_tool_call("cust-1", "nonsense_tool", {})
    assert "error" in result
    assert "unknown" in result["error"].lower()


@pytest.mark.asyncio
async def test_dispatch_handles_executor_exception() -> None:
    """If an executor raises (DB blew up mid-call), dispatcher catches
    and packages — never propagates so the agent sees a graceful result."""
    async def boom(*, customer_id: str, **kw: Any) -> dict[str, Any]:
        raise RuntimeError("postgres connection broken")

    with patch.dict(TOOL_REGISTRY, {"_test_boom": boom}, clear=False):
        result = await dispatch_tool_call("cust-1", "_test_boom", {})
        assert "error" in result
        assert "RuntimeError" in result["error"]
        assert "postgres connection broken" in result["error"]


@pytest.mark.asyncio
async def test_dispatch_handles_argument_mismatch() -> None:
    """If the model passes kwargs the executor doesn't accept (schema
    bypass), dispatcher returns a clean argument-mismatch error rather
    than 500ing the loop."""
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
    """Every executor receives `customer_id` as kwarg. Sanity-check
    the dispatcher doesn't drop or rename it."""
    captured: dict[str, Any] = {}

    async def capture(**kw: Any) -> dict[str, Any]:
        captured.update(kw)
        return {}

    with patch.dict(TOOL_REGISTRY, {"_test_capture": capture}, clear=False):
        await dispatch_tool_call("cust-xyz", "_test_capture", {"foo": "bar"})
        assert captured.get("customer_id") == "cust-xyz"
        assert captured.get("foo") == "bar"


# ============================================================
# Specific tool behavior (empty-input fast paths)
# ============================================================

@pytest.mark.asyncio
async def test_graph_search_empty_entities_short_circuits() -> None:
    """Empty entity list must return {hits: []} WITHOUT hitting the DB.
    Turn-1 mandate fires graph_search even when grounding came back
    empty; the cheap empty-return is the contract."""
    from services.retrieval.agent.tools import execute_graph_search
    out = await execute_graph_search("cust-1", entities=[])
    assert out == {"hits": []}


@pytest.mark.asyncio
async def test_inferred_edge_search_no_anchors_short_circuits() -> None:
    """No entities and no doc_ids -> {hits: []} without DB or LLM call."""
    from services.retrieval.agent.tools import execute_inferred_edge_search
    out = await execute_inferred_edge_search("cust-1")
    assert out == {"hits": []}


@pytest.mark.asyncio
async def test_parallel_multi_query_empty_short_circuits() -> None:
    """Empty queries list -> {sub_queries: []}."""
    from services.retrieval.agent.tools import execute_parallel_multi_query
    out = await execute_parallel_multi_query("cust-1", queries=[])
    assert out == {"sub_queries": []}


@pytest.mark.asyncio
async def test_parallel_multi_query_caps_at_five() -> None:
    """Cap is 5 sub-queries (mirrors MAX_INTENTS + headroom). Extras dropped."""
    from services.retrieval.agent.tools import execute_parallel_multi_query
    with patch(
        "services.retrieval.agent.tools._vector",
        new=AsyncMock(return_value=[]),
    ), patch(
        "services.retrieval.agent.tools._bm25",
        new=AsyncMock(return_value=[]),
    ):
        out = await execute_parallel_multi_query(
            "cust-1", queries=[f"q{i}" for i in range(10)]
        )
        assert len(out["sub_queries"]) == 5


@pytest.mark.asyncio
async def test_expand_entity_cluster_empty_inputs_short_circuit() -> None:
    """Missing canonical_ids OR missing label -> {clusters: {}} without DB."""
    from services.retrieval.agent.tools import execute_expand_entity_cluster
    assert await execute_expand_entity_cluster(
        "cust-1", canonical_ids=[], label="Person"
    ) == {"clusters": {}}
    assert await execute_expand_entity_cluster(
        "cust-1", canonical_ids=["x"], label=""
    ) == {"clusters": {}}
