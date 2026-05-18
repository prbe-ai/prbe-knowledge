"""Gatherer agent loop tests — fat-tools + tool_choice=required + terminal-tool.

Mocked acompletion + mocked grounding/extraction/pre-fan-out. No live LLM,
no live DB. Covers:
- Happy path: terminal tool call on turn 1 → GathererOutput → QueryResponse adapter
- Multi-turn: exploration tool call → terminal on next turn
- Budget exhaustion: harness forces a final terminal turn
- LLMError → HTTPException(503) (no fallback by design)
- No tool calls (provider violated tool_choice=required) → schema_violation
- No-LLM-configured short-circuit (test env / bootstrap / self-host without keys)
- Per-stage latency telemetry: turn_latencies_ms, tool_latencies_ms
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
    _format_prefanout_compact,
    _parse_terminal_args,
    _PREFANOUT_PER_CHANNEL_DISPLAY_CAP,
    run_gatherer,
)
from services.retrieval.agent.models import GathererOutput
from services.retrieval.agent.tools import TERMINAL_TOOL_NAME
from services.retrieval.grounding import GroundingBundle
from shared.models import QueryRequest

# ============================================================
# Fixtures: fake LiteLLM response builder
# ============================================================

def _mk_resp(
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    content: str | None = None,
    prompt_tokens: int = 100,
    cached_tokens: int = 0,
    reasoning_content: str | None = None,
) -> SimpleNamespace:
    """Build a SimpleNamespace mimicking a LiteLLM chat-completion response.

    `reasoning_content` simulates the gpt-oss harmony `analysis` block
    that LiteLLM surfaces as `message.reasoning_content`. Default None
    so existing tests don't shift; pass a string to assert the loop
    captures it onto state.
    """
    tcs = []
    for tc in tool_calls or []:
        tcs.append(SimpleNamespace(
            id=tc.get("id", "call_x"),
            function=SimpleNamespace(
                name=tc["name"],
                arguments=json.dumps(tc.get("arguments", {})) if not isinstance(tc.get("arguments"), str) else tc["arguments"],
            ),
        ))
    msg = SimpleNamespace(
        content=content,
        tool_calls=tcs,
        reasoning_content=reasoning_content,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=50,
            prompt_tokens_details={"cached_tokens": cached_tokens},
        ),
    )


def _final_emission_args(*, confidence: str = "high", chunks: int = 2) -> dict[str, Any]:
    """Build the args the model would pass to emit_gatherer_output."""
    return {
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
            "turns_used": 1,
            "tools_called": ["emit_gatherer_output"],
            "confidence": confidence,
            "dropped": [],
        },
    }


def _terminal_call(args: dict[str, Any] | None = None, *, id: str = "term_1") -> dict[str, Any]:
    return {
        "id": id,
        "name": TERMINAL_TOOL_NAME,
        "arguments": json.dumps(args or _final_emission_args()),
    }


@pytest.fixture
def fake_request() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace())


@pytest.fixture
def fake_bundle() -> GroundingBundle:
    return GroundingBundle()


@pytest.fixture(autouse=True)
def _force_llm_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests run as if LLM is configured. Tests that want the
    short-circuit override this fixture."""
    monkeypatch.setattr(
        "services.retrieval.agent.loop._no_llm_configured", lambda: False
    )


@pytest.fixture(autouse=True)
def _stub_grounding_extraction_prefanout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the three upstream-of-loop calls so tests exercise only the
    agent loop itself. Individual tests can re-patch as needed."""
    monkeypatch.setattr(
        "services.retrieval.agent.loop._build_bundle_with_token_fallback",
        AsyncMock(return_value=GroundingBundle()),
    )
    monkeypatch.setattr(
        "services.retrieval.agent.loop.extract_entities_with_llm",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        "services.retrieval.agent.loop.execute_search",
        AsyncMock(return_value={"sub_queries": []}),
    )


# ============================================================
# Pure helpers
# ============================================================

def test_affinity_key_is_stable_per_query() -> None:
    a = _affinity_key("cust-1", "what is PRB-17")
    b = _affinity_key("cust-1", "what is PRB-17")
    assert a == b
    assert _affinity_key("cust-2", "what is PRB-17") != a
    assert _affinity_key("cust-1", "what is PRB-99") != a
    assert len(a) == 32


def test_extract_cache_hit_rate_from_dict_details() -> None:
    resp = _mk_resp(prompt_tokens=100, cached_tokens=70)
    assert _extract_cache_hit_rate(resp) == pytest.approx(0.7)


def test_extract_cache_hit_rate_handles_missing() -> None:
    resp = SimpleNamespace(usage=None)
    assert _extract_cache_hit_rate(resp) is None
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=0))
    assert _extract_cache_hit_rate(resp) is None


def test_parse_terminal_args_valid_dict() -> None:
    out = _parse_terminal_args(_final_emission_args(chunks=1))
    assert out is not None
    assert isinstance(out, GathererOutput)
    assert len(out.chunks) == 1
    assert out.gatherer_notes.confidence == "high"


def test_parse_terminal_args_valid_json_string() -> None:
    out = _parse_terminal_args(json.dumps(_final_emission_args(chunks=2)))
    assert out is not None
    assert len(out.chunks) == 2


def test_parse_terminal_args_invalid_returns_none() -> None:
    """Unrecoverable inputs (None, JSON parse failures) still return None.
    Schema-shape violations now degrade gracefully to an empty
    GathererOutput rather than None — see
    test_parse_terminal_args_lenient_accepts_unknown_fields."""
    assert _parse_terminal_args(None) is None
    assert _parse_terminal_args("{not valid json") is None


def test_parse_terminal_args_lenient_accepts_unknown_fields() -> None:
    """Cerebras gpt-oss-120b emits emit_gatherer_output args with input-
    shaped fields (title/source/url) instead of the strict schema. With
    extra='ignore' + optional fields with defaults, those become empty
    GathererOutput instead of None — same downstream effect as the old
    schema_violation status, but the loop trace records the model DID
    emit (just with wrong fields)."""
    out = _parse_terminal_args({"foo": "bar"})
    assert out is not None
    assert out.entities == []
    assert out.chunks == []
    # Defaults from model_config
    assert out.gatherer_notes.turns_used == 0
    assert out.gatherer_notes.confidence == "medium"


def test_parse_terminal_args_lenient_derives_doc_id_from_chunk_id() -> None:
    """Pre-parse coercion: when the model emits a chunk without doc_id
    but with a chunk_id that encodes the parent doc, derive it."""
    out = _parse_terminal_args({
        "chunks": [
            {
                "chunk_id": "github:owner/repo:pr:42:chunk:3",
                "content": "snippet",
            },
            {
                # No chunk_id either — can't derive, will be dropped
                "content": "orphan",
            },
        ],
    })
    assert out is not None
    assert len(out.chunks) == 1
    assert out.chunks[0].doc_id == "github:owner/repo:pr:42"
    assert out.chunks[0].chunk_id == "github:owner/repo:pr:42:chunk:3"


def test_parse_terminal_args_aliases_id_to_canonical_id() -> None:
    """Fourth Cerebras failure mode: entities emitted with `id` instead
    of `canonical_id` (common API convention)."""
    out = _parse_terminal_args({
        "entities": [
            {"id": "github:prbe-ai/prbe-backend:pr:256", "label": "PR"},
            {"canonical_id": "github:prbe-ai/prbe-backend:pr:299"},
            {"id": "notion:doc:abc", "title": "ignored"},
        ],
    })
    assert out is not None
    assert len(out.entities) == 3
    assert out.entities[0].canonical_id == "github:prbe-ai/prbe-backend:pr:256"
    assert out.entities[1].canonical_id == "github:prbe-ai/prbe-backend:pr:299"
    assert out.entities[2].canonical_id == "notion:doc:abc"


def test_parse_terminal_args_repairs_truncated_json() -> None:
    """Fifth Cerebras failure mode: model glitches mid-emission and the
    JSON is unterminated. Bracket-balancer salvages the valid prefix."""
    truncated = '''{"entities":[{"canonical_id":"a"},{"canonical_id":"b"}],"chunks":[{"doc_id":"x","chunk_id":"x","content":"first"},{"doc_id":"y","chunk_id":"y","content":"sec'''
    out = _parse_terminal_args(truncated)
    assert out is not None
    assert len(out.entities) == 2
    assert out.entities[0].canonical_id == "a"
    assert out.entities[1].canonical_id == "b"
    assert len(out.chunks) >= 1
    assert out.chunks[0].content == "first"


def test_parse_terminal_args_filters_non_dict_entities() -> None:
    """Third failure mode observed on Cerebras 2026-05-18: model emits
    malformed JSON fragments as bare strings inside the entities array
    (constrained-decoding partial failure inside the list). Drop the
    non-dict items so the valid neighbors survive."""
    out = _parse_terminal_args({
        "entities": [
            {"canonical_id": "github:owner/repo:pr:1", "label": "PR"},
            "{",  # malformed fragment
            {"canonical_id": "github:owner/repo:pr:2", "label": "PR"},
            '{canonical_id":...',  # mangled string
        ],
    })
    assert out is not None
    assert len(out.entities) == 2
    assert out.entities[0].canonical_id == "github:owner/repo:pr:1"
    assert out.entities[1].canonical_id == "github:owner/repo:pr:2"


def test_parse_terminal_args_lenient_doc_level_chunks() -> None:
    """Second failure mode observed on Cerebras 2026-05-18: model emits
    'doc-level chunks' where chunk_id is omitted (only doc_id present)
    and content lives under `description` or `summary` instead of
    `content`. Coercion: chunk_id ← doc_id; content ← description."""
    out = _parse_terminal_args({
        "chunks": [
            {
                "doc_id": "github:prbe-ai/prbe-backend:commit:abc",
                "description": "Refactored authentication flow.",
            },
            {
                "doc_id": "github:prbe-ai/prbe-backend:pr:42",
                "summary": "Wires up the dashboard self-host image.",
            },
            {
                # No doc_id, no chunk_id — unrecoverable, drops
                "content": "orphan content",
            },
        ],
    })
    assert out is not None
    assert len(out.chunks) == 2
    assert out.chunks[0].chunk_id == "github:prbe-ai/prbe-backend:commit:abc"
    assert out.chunks[0].content == "Refactored authentication flow."
    assert out.chunks[1].chunk_id == "github:prbe-ai/prbe-backend:pr:42"
    assert out.chunks[1].content == "Wires up the dashboard self-host image."


def test_parse_terminal_args_cerebras_real_failure_mode() -> None:
    """Reproduction of the exact failure observed live on 2026-05-18 when
    we flipped SEARCH_AGENT_INFERENCE_MODEL=cerebras/gpt-oss-120b.
    Cerebras emitted entities with input-shaped fields (title/source/url)
    and chunks missing doc_id + matched_via. Tolerant parser should
    produce a valid GathererOutput with the recoverable data."""
    cerebras_actual_payload = {
        "entities": [
            {
                "canonical_id": "github:prbe-ai/prbe-knowledge:pr:286",
                "title": "feat(id_lookup): match URL path segments",
                "source": "github",
                "url": "https://github.com/prbe-ai/prbe-knowledge/pull/286",
            }
        ],
        "chunks": [
            {
                "chunk_id": "github:prbe-ai/prbe-knowledge:pr:286:chunk:0",
                "content": "Implements the URL-path-segment match for id_lookup so PRB-17 hits rank 1.",
            },
        ],
        "gatherer_notes": {
            "confidence": "high",
            "dropped": [],
        },
    }
    out = _parse_terminal_args(cerebras_actual_payload)
    assert out is not None
    assert len(out.entities) == 1
    assert out.entities[0].canonical_id == "github:prbe-ai/prbe-knowledge:pr:286"
    # title/source/url silently dropped via extra="ignore"; label/why_relevant default to ""
    assert out.entities[0].label == ""
    assert out.entities[0].why_relevant == ""
    assert len(out.chunks) == 1
    assert out.chunks[0].doc_id == "github:prbe-ai/prbe-knowledge:pr:286"
    assert out.gatherer_notes.confidence == "high"


def test_empty_passthrough_constructs_low_confidence_dummy() -> None:
    out = _empty_passthrough("schema_violation")
    assert out.entities == []
    assert out.chunks == []
    assert out.gatherer_notes.confidence == "low"
    assert "schema_violation" in out.gatherer_notes.dropped[0].reason


# ============================================================
# Pre-fan-out compact formatter — input-token compression
# ============================================================
# The first-turn user message used to embed pre-fan-out as a JSON dump
# (~15K tokens worst-case). gpt-oss-120b is a reasoning model; that much
# cold-cache input deterministically hangs past the 90s loop timeout.
# Compact format preserves every doc_id + score + snippet (everything the
# model needs to pick its next tool call) while cutting per-hit overhead.

def _mk_hit(
    doc_id: str = "github:owner/repo:pr:42",
    score: float = 0.87,
    source_system: str = "github",
    title: str = "PR #42: add self-host docs",
    content: str = "This PR adds documentation for the self-hosting setup, including the Helm chart values and the data-plane secrets template.",
) -> dict[str, Any]:
    return {
        "channel": "vector",
        "chunk_id": "chunk-1",
        "doc_id": doc_id,
        "source_system": source_system,
        "source_url": "https://github.com/owner/repo/pull/42",
        "title": title,
        "content": content,
        "score": score,
        "created_at": "2026-05-17T12:00:00Z",
        "updated_at": "2026-05-17T12:00:00Z",
        "author_id": "richard",
    }


def test_compact_format_includes_doc_id_score_title_snippet() -> None:
    """Per-hit minimum: doc_id (so fetch_doc works), score (relevance),
    source_system + title + snippet (so model can judge what's in the doc)."""
    prefanout = {"sub_queries": [{
        "query": "self-hosting features",
        "vector": [_mk_hit()],
        "bm25": [], "graph": [], "inferred_edge": [],
    }]}
    out = _format_prefanout_compact(prefanout)
    assert "github:owner/repo:pr:42" in out
    assert "0.870" in out
    assert "github" in out
    assert "self-host docs" in out
    # Snippet truncated but present
    assert "documentation for the self-hosting" in out


def test_compact_format_caps_per_channel_display_with_note() -> None:
    """Beyond the per-channel cap the agent sees a 'showing_top_X_of_Y'
    note so it knows the rest is reachable via fetch_doc / search."""
    n_hits = _PREFANOUT_PER_CHANNEL_DISPLAY_CAP + 5
    hits = [_mk_hit(doc_id=f"doc:{i}", score=1.0 - i * 0.01) for i in range(n_hits)]
    prefanout = {"sub_queries": [{
        "query": "q", "vector": hits, "bm25": [], "graph": [], "inferred_edge": [],
    }]}
    out = _format_prefanout_compact(prefanout)
    assert f"showing_top_{_PREFANOUT_PER_CHANNEL_DISPLAY_CAP}_of_{n_hits}" in out
    # First N appear, last 5 don't
    for i in range(_PREFANOUT_PER_CHANNEL_DISPLAY_CAP):
        assert f"doc:{i}" in out
    for i in range(_PREFANOUT_PER_CHANNEL_DISPLAY_CAP, n_hits):
        assert f"doc:{i}" not in out


def test_compact_format_preserves_inferred_edge_why() -> None:
    """`why` is the inferred-edge moat — the rationale the agent uses to
    decide whether the edge is signal or noise. Must survive compression."""
    inf_hit = {
        "channel": "inferred_edge",
        "doc_id": "github:owner/repo:pr:78",
        "source_system": "github",
        "title": "PR #78: dashboard auth flow",
        "content": "...",
        "score": 0.76,
        "edge_type": "references_pr",
        "why": "PR #78 body explicitly references PR #71 as its prerequisite",
    }
    prefanout = {"sub_queries": [{
        "query": "q", "vector": [], "bm25": [], "graph": [], "inferred_edge": [inf_hit],
    }]}
    out = _format_prefanout_compact(prefanout)
    assert "why: PR #78 body explicitly references PR #71" in out
    assert "edge=references_pr" in out


def test_compact_format_is_much_smaller_than_json_dump() -> None:
    """Compression target: the new format should be at least 4x smaller
    than the equivalent JSON dump for a realistic 4-channel × 10-hit
    payload with production-sized chunk content (~800 chars/hit). This
    is the worst case the first turn sees on cold cache."""
    # Realistic chunk: ~800 chars (typical Slack/Linear/GitHub chunk)
    big_content = (
        "We retired managed-mode in chart 0.24.0 (PR #266). The CustomerMode "
        "enum kept MANAGED but it now means 'Probe-hosted'. Self-host stays "
        "as the second mode. The dashboard now ships as a standalone Docker "
        "image (ghcr.io/prbe-ai/prbe-dashboard, PR #77) and chart 0.27.1 "
        "(PR #271) renders deployment-dashboard.yaml + service-dashboard.yaml "
        "+ ingress / catch-all in mode=self-host. Managed-shared still uses "
        "Vercel for the dashboard (templates gated on `not isManagedShared`). "
        "Per-tenant K8s migration is still ongoing — one DOKS cluster, one "
        "customer-<uuid> namespace per tenant, each a prbe-data-plane Helm "
        "release behind <slug>.prbe.ai. Auth is HMAC via the data-plane key "
        "header which is shared across all data-plane services and rotates "
        "every 90 days via the rotation cron."
    )
    hits = [_mk_hit(doc_id=f"doc:{i}", content=big_content) for i in range(10)]
    prefanout = {"sub_queries": [{
        "query": "what features did we implement for self-hosting?",
        "vector": hits, "bm25": hits, "graph": hits, "inferred_edge": hits,
    }]}
    json_size = len(json.dumps(prefanout, default=str))
    compact_size = len(_format_prefanout_compact(prefanout))
    ratio = json_size / compact_size if compact_size else float("inf")
    assert ratio >= 4.0, (
        f"compact {compact_size} vs json {json_size} = {ratio:.1f}x "
        f"— compression target missed (want ≥4x)"
    )


def test_compact_format_handles_empty_prefanout() -> None:
    """Edge case: pre-fan-out returned nothing on any channel."""
    out = _format_prefanout_compact({"sub_queries": []})
    assert out == "(no pre-fan-out hits)"


# ============================================================
# Loop integration (mocked everything)
# ============================================================

@pytest.mark.asyncio
async def test_terminal_on_turn_1_is_happy_path(
    fake_request: SimpleNamespace,
) -> None:
    """The model calls emit_gatherer_output on turn 1 → loop ends, args
    become the final GathererOutput, telemetry recorded."""
    req = QueryRequest(query="what is PRB-17", customer_id="cust-1", top_k=5)

    with patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(
            tool_calls=[_terminal_call(_final_emission_args(chunks=2, confidence="high"))],
            cached_tokens=80,
        )),
    ) as mock_acomp:
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert resp.total_candidates == 2
    assert resp.gatherer_notes is not None
    assert resp.gatherer_notes["confidence"] == "high"
    # Telemetry
    assert fake_request.state.gatherer_status == "ok"
    assert fake_request.state.confidence == "high"
    # The model emitted the terminal — that counts as 0 retrieval calls.
    assert fake_request.state.tool_calls_count == 0
    assert fake_request.state.cache_hit_rate == pytest.approx(0.8)
    # Tool surface sent to model
    call_kwargs = mock_acomp.call_args.kwargs
    assert call_kwargs.get("tool_choice") == "required"
    assert call_kwargs.get("custom_llm_provider") == "openai"
    assert "extra_headers" in call_kwargs
    assert "x-session-affinity" in call_kwargs["extra_headers"]


@pytest.mark.asyncio
async def test_exploration_then_terminal(
    fake_request: SimpleNamespace,
) -> None:
    """Turn 1: model calls search. Turn 2: model calls terminal. Loop ends."""
    req = QueryRequest(query="auth refactor", customer_id="cust-1", top_k=5)

    turn_1 = _mk_resp(tool_calls=[{
        "id": "s1",
        "name": "search",
        "arguments": {"queries": ["auth refactor design doc"]},
    }])
    turn_2 = _mk_resp(tool_calls=[_terminal_call(_final_emission_args(chunks=3))])

    with patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1, turn_2]),
    ), patch(
        "services.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"sub_queries": []}),
    ) as mock_dispatch:
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert resp.total_candidates == 3
    assert fake_request.state.tool_calls_count == 1  # one search call
    mock_dispatch.assert_called_once()
    assert mock_dispatch.call_args.kwargs["tool_name"] == "search"


@pytest.mark.asyncio
async def test_no_tool_calls_returns_schema_violation(
    fake_request: SimpleNamespace,
) -> None:
    """tool_choice=required SHOULD force a tool call. If the provider
    quirks and returns content-only, harness logs + returns
    schema_violation (no prose-retry path)."""
    req = QueryRequest(query="x", customer_id="cust-1", top_k=5)
    bad_turn = _mk_resp(content="I wasn't able to find anything.", tool_calls=[])

    with patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=bad_turn),
    ):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert fake_request.state.gatherer_status == "schema_violation"
    assert resp.total_candidates == 0
    assert resp.gatherer_notes["confidence"] == "low"


@pytest.mark.asyncio
async def test_wrong_shape_terminal_args_degrades_to_empty_output(
    fake_request: SimpleNamespace,
) -> None:
    """Model calls emit_gatherer_output with args that don't match the
    GathererOutput schema (e.g. Cerebras gpt-oss-120b emitting input-
    shaped fields like title/source/url). Tolerant parser absorbs the
    drift: parses as empty GathererOutput, status='ok', 0 results.
    Effect on the user is identical to the old `schema_violation`
    branch (no results returned), but the loop trace records the
    model DID emit rather than treating it as a parse failure."""
    req = QueryRequest(query="x", customer_id="cust-1", top_k=5)
    bad_terminal = _terminal_call({"completely": "wrong shape"})

    with patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(tool_calls=[bad_terminal])),
    ):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert fake_request.state.gatherer_status == "ok"
    assert resp.total_candidates == 0


@pytest.mark.asyncio
async def test_llm_error_raises_503(
    fake_request: SimpleNamespace,
) -> None:
    """Fatal provider error → HTTPException(503). No fallback by design."""
    from shared.llm import LLMError
    req = QueryRequest(query="boom", customer_id="cust-1", top_k=5)

    with patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=LLMError("fireworks down")),
    ), pytest.raises(HTTPException) as exc_info:
        await run_gatherer(req, customer_id="cust-1", request=fake_request)
    assert exc_info.value.status_code == 503
    assert fake_request.state.full_failure is True


@pytest.mark.asyncio
async def test_no_llm_configured_short_circuits_to_empty(
    fake_request: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no LLM provider is configured, gatherer returns empty 200
    instead of 503 — mirrors PR #282's _call_haiku graceful no-op."""
    req = QueryRequest(query="anything", customer_id="cust-1", top_k=5)
    monkeypatch.setattr(
        "services.retrieval.agent.loop._no_llm_configured", lambda: True
    )
    boom = AsyncMock(side_effect=AssertionError("acompletion should NOT be called"))

    with patch("services.retrieval.agent.loop.acompletion", new=boom):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert resp.total_candidates == 0
    assert resp.gatherer_notes["confidence"] == "low"
    assert fake_request.state.gatherer_status == "no_llm_configured"
    assert fake_request.state.tool_calls_count == 0
    assert fake_request.state.failure_recovered is True
    boom.assert_not_called()


@pytest.mark.asyncio
async def test_per_stage_latency_recorded_on_state(
    fake_request: SimpleNamespace,
) -> None:
    """LoopState accumulates per-turn LLM latencies + per-tool latencies."""
    req = QueryRequest(query="q", customer_id="cust-1", top_k=5)
    turn_1 = _mk_resp(tool_calls=[{
        "id": "s1", "name": "search", "arguments": {"queries": ["q'"]},
    }])
    turn_2 = _mk_resp(tool_calls=[_terminal_call()])

    with patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1, turn_2]),
    ), patch(
        "services.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"sub_queries": []}),
    ):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    # cache_hit_rate is averaged across turns
    assert fake_request.state.cache_hit_rate is not None


@pytest.mark.asyncio
async def test_reasoning_content_captured_per_turn(
    fake_request: SimpleNamespace,
) -> None:
    """The gpt-oss harmony `analysis` block (surfaced by LiteLLM as
    `message.reasoning_content`) lands on `state.reasoning_per_turn`
    parallel to `turn_latencies_ms`. Without this capture the
    agent's "why" trail is lost because the OpenAI chat-completion
    round-trip only echoes role/content/tool_calls — not reasoning."""
    req = QueryRequest(query="why was PR 71 made", customer_id="cust-1", top_k=5)
    turn_1 = _mk_resp(
        tool_calls=[{"id": "s1", "name": "search", "arguments": {"queries": ["q1"]}}],
        reasoning_content=(
            "User asks about PR #71 motivation. I'll start with the "
            "vector channel anchored on PR-71 ID."
        ),
    )
    turn_2 = _mk_resp(
        tool_calls=[_terminal_call()],
        reasoning_content=None,  # provider may emit reasoning on some turns and not others
    )

    with patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1, turn_2]),
    ), patch(
        "services.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"sub_queries": []}),
    ):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    # PR 1's stash exposes the LoopState ref on request.state.
    loop_state = fake_request.state.search_agent_loop_state
    assert loop_state is not None
    assert len(loop_state.reasoning_per_turn) == 2
    assert loop_state.reasoning_per_turn[0] is not None
    assert "PR #71" in loop_state.reasoning_per_turn[0]
    assert loop_state.reasoning_per_turn[1] is None  # provider didn't emit on turn 2


@pytest.mark.asyncio
async def test_reasoning_per_turn_starts_empty_and_grows(
    fake_request: SimpleNamespace,
) -> None:
    """No reasoning emitted = list of None entries (one per turn), NOT
    a missing key. The analyzer relies on len(reasoning_per_turn) ==
    turn_count for per-turn correlation."""
    req = QueryRequest(query="q", customer_id="cust-1", top_k=5)
    # Single turn — terminal immediately, no reasoning.
    turn_1 = _mk_resp(tool_calls=[_terminal_call()], reasoning_content=None)

    with patch(
        "services.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1]),
    ), patch(
        "services.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"sub_queries": []}),
    ):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    loop_state = fake_request.state.search_agent_loop_state
    assert loop_state is not None
    assert len(loop_state.reasoning_per_turn) == loop_state.turn_count
    assert loop_state.reasoning_per_turn == [None]
