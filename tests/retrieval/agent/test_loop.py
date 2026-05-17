"""Gatherer agent loop tests — mocked acompletion, no live LLM.

Covers:
- Happy path: tool calls -> final emission -> QueryResponse adapter
- Turn-1 mandate logging when the agent skips a channel
- Tool budget exhaustion forces a final-emission turn
- need_deeper grants extensions up to the cap
- LLMError propagates as HTTPException(503) (no fallback by design)
- response_format unparseable -> harness emits empty GathererOutput with
  schema_violation status
- Cache hit rate is averaged across turns and written to request.state
- Session-affinity header is sent on every turn
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from services.retrieval.agent.loop import (
    _affinity_key,
    _empty_passthrough,
    _extract_cache_hit_rate,
    _parse_gatherer_output,
    run_gatherer,
)
from services.retrieval.agent.models import GathererOutput
from services.retrieval.grounding import GroundingBundle
from shared.models import QueryRequest

# ============================================================
# Test fixtures: fake LiteLLM response builder
# ============================================================

def _mk_resp(
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    content: str | None = None,
    prompt_tokens: int = 100,
    cached_tokens: int = 0,
) -> SimpleNamespace:
    """Build a SimpleNamespace mimicking a LiteLLM chat-completion response."""
    tcs = []
    for tc in tool_calls or []:
        tcs.append(SimpleNamespace(
            id=tc.get("id", "call_x"),
            function=SimpleNamespace(
                name=tc["name"],
                arguments=json.dumps(tc.get("arguments", {})),
            ),
        ))
    msg = SimpleNamespace(content=content, tool_calls=tcs)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=50,
            prompt_tokens_details={"cached_tokens": cached_tokens},
        ),
    )


def _final_emission_json(*, confidence: str = "high", chunks: int = 2) -> str:
    """Build a JSON string the agent could return as its final emission."""
    payload = {
        "entities": [],
        "chunks": [
            {
                "doc_id": f"doc-{i}",
                "chunk_id": f"chunk-{i}",
                "content": f"content body {i}",
                "matched_via": ["vector"],
                "why_relevant": f"surfaced via vector channel, rank {i+1}",
            }
            for i in range(chunks)
        ],
        "gatherer_notes": {
            "turns_used": 2,
            "tools_called": ["vector_search", "bm25_search", "graph_search", "inferred_edge_search"],
            "confidence": confidence,
            "dropped": [],
        },
    }
    return json.dumps(payload)


@pytest.fixture
def fake_request() -> SimpleNamespace:
    """A minimal FastAPI-Request-like object with a writable state."""
    return SimpleNamespace(state=SimpleNamespace())


@pytest.fixture
def fake_bundle() -> GroundingBundle:
    return GroundingBundle()


@pytest.fixture(autouse=True)
def _force_llm_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: every loop test runs as if a provider is configured.

    The new `_no_llm_configured()` short-circuit returns True in the test
    env (no API keys set), which would make every mocked-acompletion test
    short-circuit to an empty passthrough before reaching the mock.
    Pin it to False so tests exercise the real loop. Specific tests that
    want the short-circuit path can re-override.
    """
    monkeypatch.setattr(
        "services.retrieval.agent.loop._no_llm_configured", lambda: False
    )


# ============================================================
# Pure helpers
# ============================================================

def test_affinity_key_is_stable_per_query() -> None:
    a = _affinity_key("cust-1", "what is PRB-17")
    b = _affinity_key("cust-1", "what is PRB-17")
    assert a == b
    # Different customer or query -> different key (so cache scoping works).
    assert _affinity_key("cust-2", "what is PRB-17") != a
    assert _affinity_key("cust-1", "what is PRB-99") != a
    # 32 chars (sha256 truncated).
    assert len(a) == 32


def test_extract_cache_hit_rate_from_dict_details() -> None:
    resp = _mk_resp(prompt_tokens=100, cached_tokens=70)
    assert _extract_cache_hit_rate(resp) == pytest.approx(0.7)


def test_extract_cache_hit_rate_handles_missing() -> None:
    resp = SimpleNamespace(usage=None)
    assert _extract_cache_hit_rate(resp) is None
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=0))
    assert _extract_cache_hit_rate(resp) is None


def test_parse_gatherer_output_valid() -> None:
    out = _parse_gatherer_output(_final_emission_json(chunks=1))
    assert out is not None
    assert isinstance(out, GathererOutput)
    assert len(out.chunks) == 1
    assert out.gatherer_notes.confidence == "high"


def test_parse_gatherer_output_invalid_returns_none() -> None:
    assert _parse_gatherer_output(None) is None
    assert _parse_gatherer_output("") is None
    assert _parse_gatherer_output("{not valid json") is None
    # Valid JSON, wrong schema:
    assert _parse_gatherer_output('{"foo": "bar"}') is None


def test_empty_passthrough_constructs_low_confidence_dummy() -> None:
    out = _empty_passthrough("schema_violation")
    assert out.entities == []
    assert out.chunks == []
    assert out.gatherer_notes.confidence == "low"
    assert len(out.gatherer_notes.dropped) == 1
    assert "schema_violation" in out.gatherer_notes.dropped[0].reason


# ============================================================
# Loop integration (mocked acompletion + mocked grounding)
# ============================================================

@pytest.mark.asyncio
async def test_happy_path_curates_after_one_turn(
    fake_request: SimpleNamespace, fake_bundle: GroundingBundle
) -> None:
    """Agent emits no tool calls on turn 1 -> CURATE path -> QueryResponse."""
    req = QueryRequest(query="what is PRB-17", customer_id="cust-1", top_k=5)

    with patch(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        new=AsyncMock(return_value=fake_bundle),
    ), patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(
            content=_final_emission_json(chunks=2, confidence="high"),
            cached_tokens=80,
        )),
    ) as mock_acomp:
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert resp.total_candidates == 2
    assert resp.gatherer_notes is not None
    assert resp.gatherer_notes["confidence"] == "high"
    # Telemetry on request.state
    assert fake_request.state.gatherer_status == "ok"
    assert fake_request.state.confidence == "high"
    assert fake_request.state.tool_calls_count == 0  # zero tool calls; agent passed through
    assert fake_request.state.cache_hit_rate == pytest.approx(0.8)
    # Session-affinity header sent
    call_kwargs = mock_acomp.call_args.kwargs
    assert "extra_headers" in call_kwargs
    assert "x-session-affinity" in call_kwargs["extra_headers"]


@pytest.mark.asyncio
async def test_tool_call_then_final_emission(
    fake_request: SimpleNamespace, fake_bundle: GroundingBundle
) -> None:
    """Turn 1: fires 4 channels. Turn 2: emits final GathererOutput."""
    req = QueryRequest(query="why was PR #71 made", customer_id="cust-1", top_k=5)

    turn_1 = _mk_resp(tool_calls=[
        {"id": "c1", "name": "vector_search", "arguments": {"query": "why was PR #71 made"}},
        {"id": "c2", "name": "bm25_search", "arguments": {"query": "why was PR #71 made"}},
        {"id": "c3", "name": "graph_search", "arguments": {"entities": []}},
        {"id": "c4", "name": "inferred_edge_search", "arguments": {"entities": []}},
    ], cached_tokens=10)
    turn_2 = _mk_resp(content=_final_emission_json(chunks=3), cached_tokens=90)

    with patch(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        new=AsyncMock(return_value=fake_bundle),
    ), patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1, turn_2]),
    ), patch(
        "services.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"hits": []}),
    ) as mock_dispatch:
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert resp.total_candidates == 3
    # All 4 turn-1 channels dispatched in parallel
    assert mock_dispatch.call_count == 4
    fired_names = sorted(c.kwargs["tool_name"] for c in mock_dispatch.call_args_list)
    assert fired_names == ["bm25_search", "graph_search", "inferred_edge_search", "vector_search"]
    assert fake_request.state.tool_calls_count == 4
    # Cache hit rate averaged across both turns
    assert fake_request.state.cache_hit_rate == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_turn_1_skip_logs_warning(
    fake_request: SimpleNamespace, fake_bundle: GroundingBundle, capsys: pytest.CaptureFixture[str]
) -> None:
    """Agent skips the inferred_edge channel on turn 1 -> harness logs
    `agent.turn_1_mandate_skipped` for trace review (no enforcement,
    per the eng review's deliberate accept).

    The codebase uses structlog with a stdout renderer (not stdlib logging),
    so we capture stdout via `capsys` rather than pytest's `caplog` fixture.
    """
    req = QueryRequest(query="hi", customer_id="cust-1", top_k=5)
    turn_1 = _mk_resp(tool_calls=[
        {"id": "c1", "name": "vector_search", "arguments": {"query": "hi"}},
        {"id": "c2", "name": "bm25_search", "arguments": {"query": "hi"}},
        # Missing graph_search + inferred_edge_search
    ])
    turn_2 = _mk_resp(content=_final_emission_json(chunks=1))

    with patch(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        new=AsyncMock(return_value=fake_bundle),
    ), patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1, turn_2]),
    ), patch(
        "services.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"hits": []}),
    ):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    captured = capsys.readouterr()
    log_blob = captured.out + captured.err
    assert "turn_1_mandate_skipped" in log_blob, (
        "expected agent.turn_1_mandate_skipped warning when agent skipped channels on turn 1; "
        f"got log output: {log_blob[:500]}"
    )
    # The skipped channels should be enumerated in the structured log fields.
    assert "graph_search" in log_blob and "inferred_edge_search" in log_blob


@pytest.mark.asyncio
async def test_need_deeper_extends_budget(
    fake_request: SimpleNamespace, fake_bundle: GroundingBundle
) -> None:
    """need_deeper grants +10 to budget, up to 2 extensions. Counter tracked on request.state."""
    req = QueryRequest(query="big query", customer_id="cust-1", top_k=5)
    turn_1 = _mk_resp(tool_calls=[
        {"id": "n1", "name": "need_deeper", "arguments": {"reason": "found a long chain"}},
    ])
    turn_2 = _mk_resp(content=_final_emission_json(chunks=1))

    with patch(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        new=AsyncMock(return_value=fake_bundle),
    ), patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1, turn_2]),
    ):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert fake_request.state.need_deeper_extensions == 1
    assert fake_request.state.tool_calls_count == 1


@pytest.mark.asyncio
async def test_llm_error_raises_503(
    fake_request: SimpleNamespace, fake_bundle: GroundingBundle
) -> None:
    """Fatal provider error -> HTTPException(503), full_failure flag set
    (no fallback by design)."""
    from shared.llm import LLMError
    req = QueryRequest(query="boom", customer_id="cust-1", top_k=5)

    with patch(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        new=AsyncMock(return_value=fake_bundle),
    ), patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=LLMError("fireworks down")),
    ), pytest.raises(HTTPException) as exc_info:
        await run_gatherer(req, customer_id="cust-1", request=fake_request)
    assert exc_info.value.status_code == 503
    assert fake_request.state.full_failure is True


@pytest.mark.asyncio
async def test_unparseable_emission_emits_schema_violation(
    fake_request: SimpleNamespace, fake_bundle: GroundingBundle
) -> None:
    """response_format guarantees JSON, but if a provider quirk leaks bad
    output through, harness emits an empty GathererOutput with
    gatherer_status='schema_violation' rather than 500ing."""
    req = QueryRequest(query="x", customer_id="cust-1", top_k=5)
    bad_turn = _mk_resp(content="this is not valid json")

    with patch(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        new=AsyncMock(return_value=fake_bundle),
    ), patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=bad_turn),
    ):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert fake_request.state.gatherer_status == "schema_violation"
    assert resp.total_candidates == 0
    assert resp.gatherer_notes["confidence"] == "low"


@pytest.mark.asyncio
async def test_no_llm_configured_short_circuits_to_empty(
    fake_request: SimpleNamespace, fake_bundle: GroundingBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When neither the gateway URL nor any provider key is set, the
    gatherer returns an empty result with status 'no_llm_configured'
    instead of 503ing. Mirrors the pre-cutover router's graceful no-op
    in `_call_haiku` and lets test envs / bootstrap / self-host-without-
    keys keep `/retrieve` callable."""
    req = QueryRequest(query="anything", customer_id="cust-1", top_k=5)
    monkeypatch.setattr(
        "services.retrieval.agent.loop._no_llm_configured", lambda: True
    )

    boom_acompletion = AsyncMock(side_effect=AssertionError("acompletion should NOT be called"))
    with patch(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        new=AsyncMock(return_value=fake_bundle),
    ), patch(
        "services.retrieval.agent.loop.acompletion",
        new=boom_acompletion,
    ):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert resp.total_candidates == 0
    assert resp.gatherer_notes["confidence"] == "low"
    assert fake_request.state.gatherer_status == "no_llm_configured"
    assert fake_request.state.tool_calls_count == 0
    assert fake_request.state.failure_recovered is True
    boom_acompletion.assert_not_called()


@pytest.mark.asyncio
async def test_grounding_failure_does_not_break_loop(
    fake_request: SimpleNamespace
) -> None:
    """If grounding raises (DB hiccup), the loop continues with an empty
    bundle — recall comes from the agent's prompt-mandated turn-1
    fan-out, not from grounding."""
    req = QueryRequest(query="x", customer_id="cust-1", top_k=5)

    with patch(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        new=AsyncMock(side_effect=RuntimeError("DB down")),
    ), patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(content=_final_emission_json(chunks=1))),
    ):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert resp.total_candidates == 1
    assert fake_request.state.gatherer_status == "ok"


# ============================================================
# Trace blob stash (PR 1)
# ============================================================


@pytest.mark.asyncio
async def test_trace_stash_set_on_happy_path(
    fake_request: SimpleNamespace,
    fake_bundle: GroundingBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path stashes raw refs onto request.state for the BackgroundTask
    persister to consume."""
    monkeypatch.setattr("services.retrieval.agent.loop.random.random", lambda: 0.0)
    req = QueryRequest(query="what shipped", customer_id="cust-1", top_k=5)

    with patch(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        new=AsyncMock(return_value=fake_bundle),
    ), patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(content=_final_emission_json(chunks=1))),
    ):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert fake_request.state.search_agent_should_persist is True
    assert fake_request.state.search_agent_status == "ok"
    assert fake_request.state.search_agent_customer_id == "cust-1"
    assert fake_request.state.search_agent_query == "what shipped"
    # LoopState ref should be present and have the expected shape
    state = fake_request.state.search_agent_loop_state
    assert state is not None
    assert state.customer_id == "cust-1"
    assert isinstance(state.messages, list) and len(state.messages) > 0
    # Gathered should be set on the happy path
    assert fake_request.state.search_agent_gathered is not None


@pytest.mark.asyncio
async def test_trace_stash_skipped_when_sample_rate_zero(
    fake_request: SimpleNamespace,
    fake_bundle: GroundingBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """random.random > sample_rate => no stash. Pin sample_rate to 0.0
    and random to 0.5 (the inequality `random > 0.0` is True, so skip).
    """
    monkeypatch.setattr(
        "services.retrieval.agent.loop.SEARCH_AGENT_TRACE_SAMPLE_RATE", 0.0
    )
    monkeypatch.setattr("services.retrieval.agent.loop.random.random", lambda: 0.5)
    req = QueryRequest(query="x", customer_id="cust-1", top_k=5)

    with patch(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        new=AsyncMock(return_value=fake_bundle),
    ), patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(content=_final_emission_json(chunks=1))),
    ):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert not getattr(
        fake_request.state, "search_agent_should_persist", False
    )


@pytest.mark.asyncio
async def test_trace_stash_set_on_503_fatal_provider_error(
    fake_request: SimpleNamespace,
    fake_bundle: GroundingBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REGRESSION GUARD: 503 path must ALSO stash the trace. The
    fatal_provider_error transcripts are the most valuable ones to keep
    — without them we'd have a Sentry breadcrumb but no per-turn record
    of what the agent was doing when the provider died.
    """
    monkeypatch.setattr("services.retrieval.agent.loop.random.random", lambda: 0.0)
    from shared.llm import LLMError
    req = QueryRequest(query="boom", customer_id="cust-1", top_k=5)

    with patch(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        new=AsyncMock(return_value=fake_bundle),
    ), patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=LLMError("fireworks down")),
    ), pytest.raises(HTTPException):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    # Stash MUST have happened before the raise
    assert fake_request.state.search_agent_should_persist is True
    assert fake_request.state.search_agent_status == "fatal_provider_error"
    # gathered is None because we never got to the final emission
    assert fake_request.state.search_agent_gathered is None
    # State was constructed before the LLMError, so it should be present
    # with the partial fields the model managed to populate.
    state = fake_request.state.search_agent_loop_state
    assert state is not None
    assert state.customer_id == "cust-1"


@pytest.mark.asyncio
async def test_trace_stash_set_on_no_llm_configured(
    fake_request: SimpleNamespace,
    fake_bundle: GroundingBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """no_llm_configured short-circuit also stashes. These traces tell
    us whether self-host installations are bypassing the LLM accidentally."""
    monkeypatch.setattr("services.retrieval.agent.loop.random.random", lambda: 0.0)
    monkeypatch.setattr(
        "services.retrieval.agent.loop._no_llm_configured", lambda: True
    )
    req = QueryRequest(query="anything", customer_id="cust-1", top_k=5)

    boom = AsyncMock(side_effect=AssertionError("acompletion should NOT be called"))
    with patch(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        new=AsyncMock(return_value=fake_bundle),
    ), patch(
        "services.retrieval.agent.loop.acompletion",
        new=boom,
    ):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert fake_request.state.search_agent_should_persist is True
    assert fake_request.state.search_agent_status == "no_llm_configured"


@pytest.mark.asyncio
async def test_trace_id_unified_with_request_id(
    fake_request: SimpleNamespace,
    fake_bundle: GroundingBundle,
) -> None:
    """When request.state.request_id is set (middleware did its job),
    the gatherer's trace_id MUST match it — single id across logs,
    query_traces row, response, and R2 blob key.
    """
    fake_request.state.request_id = "11111111-2222-3333-4444-555555555555"
    req = QueryRequest(query="x", customer_id="cust-1", top_k=5)

    with patch(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        new=AsyncMock(return_value=fake_bundle),
    ), patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(content=_final_emission_json(chunks=1))),
    ):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert resp.trace_id == "11111111-2222-3333-4444-555555555555"
    # Stash should carry the same id (sampling is 1.0 by default).
    assert (
        fake_request.state.search_agent_trace_id
        == "11111111-2222-3333-4444-555555555555"
    )


@pytest.mark.asyncio
async def test_trace_id_falls_back_to_uuid_when_no_request_id(
    fake_request: SimpleNamespace,
    fake_bundle: GroundingBundle,
) -> None:
    """When request.state.request_id is missing AND caller didn't supply
    req.trace_id, generate a UUID. No more 'q-{ts}' placeholders.
    """
    # fake_request.state has no request_id attribute by default
    req = QueryRequest(query="x", customer_id="cust-1", top_k=5)

    with patch(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        new=AsyncMock(return_value=fake_bundle),
    ), patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(content=_final_emission_json(chunks=1))),
    ):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    # Should be a valid UUID, not "q-..."
    import uuid as _uuid
    parsed = _uuid.UUID(resp.trace_id)
    assert str(parsed) == resp.trace_id
    assert not resp.trace_id.startswith("q-")
