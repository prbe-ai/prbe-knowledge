"""Gatherer agent loop.

Entry point: `run_gatherer(req, customer_id, request)` -> QueryResponse.

Flow:
1. Deterministic grounding via `services/retrieval/grounding.py` (no LLM).
2. Agent loop on Fireworks gpt-oss-120B:
   - Turn 1 mandates parallel 4-channel fan-out (prompt only; harness logs anomalies).
   - Turn 2+ the agent reads results and either CURATEs or EXPLOREs.
   - Loop terminates when the agent emits no tool calls (final emission)
     OR hits the tool budget (forced final-call with `tools=None`).
3. `response_format=GathererOutput` constrains the final emission to a
   parseable Pydantic shape. Harness re-parses for defence in depth.
4. Telemetry: turn count, tool calls, cache hit rate written to
   `request.state.*` for the query_traces middleware to persist via
   migration 0078 columns.
5. Adapter converts `GathererOutput` -> existing `QueryResponse` shape.

Plan: docs/specs/agentic-search.md, section "Phased rollout: Phase 2".
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

if TYPE_CHECKING:
    from starlette.requests import Request

from services.retrieval.agent.adapter import to_query_response
from services.retrieval.agent.models import (
    DroppedCandidate,
    GathererNotes,
    GathererOutput,
    GathererStatus,
)
from services.retrieval.agent.prompt import build_system_prompt
from services.retrieval.agent.tools import dispatch_tool_call, tool_definitions
from services.retrieval.grounding import GroundingBundle
from services.retrieval.router import (
    _build_bundle_with_token_fallback,
    _escape_query_for_xml,
)
from shared.constants import (
    SEARCH_AGENT_EXTENSION_GRANT,
    SEARCH_AGENT_HARD_CAP,
    SEARCH_AGENT_INFERENCE_MODEL,
    SEARCH_AGENT_LOOP_TIMEOUT_SECONDS,
    SEARCH_AGENT_MAX_EXTENSIONS,
    SEARCH_AGENT_TOOL_BUDGET,
    SEARCH_AGENT_TURN_TIMEOUT_SECONDS,
)
from shared.llm import LLMError, acompletion
from shared.logging import get_logger
from shared.models import QueryRequest, QueryResponse

log = get_logger(__name__)


# Cached response_format payload — built once at import time. LiteLLM
# forwards this dict verbatim to Fireworks as the OpenAI
# `response_format: {type: json_schema, json_schema: {...}}` shape.
# Schema must be derived from `GathererOutput.model_json_schema()` (Pydantic
# v2) — passing the Pydantic class directly to LiteLLM's `response_format`
# kwarg gets silently dropped on the wire for Fireworks via the proxy.
_GATHERER_OUTPUT_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "GathererOutput",
        "schema": GathererOutput.model_json_schema(),
    },
}

# ============================================================
# Loop state
# ============================================================

@dataclass(slots=True)
class _LoopState:
    """Mutable per-request loop bookkeeping.

    Held entirely in memory for the duration of one query. Persisted
    only via the final summary that the harness writes to query_traces.
    """

    customer_id: str
    trace_id: str
    query: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools_fired: list[str] = field(default_factory=list)
    turn_count: int = 0
    tool_calls_count: int = 0
    extensions_used: int = 0
    budget: int = SEARCH_AGENT_TOOL_BUDGET
    cache_hit_rates: list[float] = field(default_factory=list)
    turn_1_tools_fired: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)


# ============================================================
# Helpers
# ============================================================

_REQUIRED_TURN_1_CHANNELS: frozenset[str] = frozenset({
    "vector_search",
    "bm25_search",
    "graph_search",
    "inferred_edge_search",
})


def _build_user_message(query: str, bundle: GroundingBundle) -> str:
    """Render the per-query user message.

    Mirrors PR #282's router user-message style (grounding first, query
    last for recency bias) so the cache prefix can remain stable.
    """
    grounding_lines = []
    for c in bundle.candidates:
        grounding_lines.append(
            f"  - {c.entity_type}:{c.canonical_id} ({c.display_name}) "
            f"[{c.match_source}]"
        )
    for m in bundle.bare_id_matches:
        grounding_lines.append(
            f"  - {m.entity_type}:{m.canonical_id} ({m.display_name}) [bare_id]"
        )
    grounding_block = "\n".join(grounding_lines) if grounding_lines else "  (no entities matched)"

    sources_block = (
        ", ".join(bundle.connected_sources) if bundle.connected_sources else "(none)"
    )
    safe_query = _escape_query_for_xml(query)
    return (
        f"<grounding>\n{grounding_block}\n</grounding>\n\n"
        f"<connected_sources>{sources_block}</connected_sources>\n\n"
        f"<query>\n{safe_query}\n</query>"
    )


def _affinity_key(customer_id: str, query: str) -> str:
    """Build a stable per-query affinity hash so Fireworks routes turns
    to the same replica (90% cache discount only applies within a replica).
    Per `feedback_litellm_gateway_gemini_405.md` we're forced to use the
    OpenAI wire shape; the header-pass-through is unchanged.
    """
    h = sha256()
    h.update(customer_id.encode("utf-8", errors="ignore"))
    h.update(b":")
    h.update(query.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:32]


def _extract_cache_hit_rate(resp: Any) -> float | None:
    """Compute cache_read_input_tokens / prompt_tokens for one response.

    Fireworks reports `prompt_tokens_details.cached_tokens` per OpenAI
    convention. LiteLLM normalizes this; some providers' adapters miss
    the field — return None when missing rather than 0 so the average
    isn't dragged down by missing-data rows.
    """
    try:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return None
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        if prompt_tokens <= 0:
            return None
        details = getattr(usage, "prompt_tokens_details", None) or {}
        cached = 0
        if isinstance(details, dict):
            cached = int(details.get("cached_tokens", 0) or 0)
        else:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        return cached / prompt_tokens
    except (AttributeError, TypeError, ValueError):
        return None


def _parse_gatherer_output(content: str | None) -> GathererOutput | None:
    """Re-parse the model's final emission for defence in depth.

    `response_format=GathererOutput` should guarantee a parseable JSON
    object, but provider quirks (incomplete decoding, structured-output
    bypass) can still leak through. Returns None on parse failure;
    caller surfaces `gatherer_status='schema_violation'` in that case.
    """
    if not content:
        return None
    try:
        return GathererOutput.model_validate_json(content)
    except Exception as exc:
        log.warning("agent.final_emission_parse_failed", error=str(exc))
        return None


def _no_llm_configured() -> bool:
    """True when no LLM provider is reachable.

    Mirrors `services/retrieval/router.py`'s pre-cutover guard in
    `_call_haiku`: short-circuits when neither a provider API key nor
    the LiteLLM gateway URL is available. Tests / bootstrap / self-host-
    without-keys hit this path and get an empty result (status
    `no_llm_configured`) instead of a 503. Provider outages with config
    present still bubble up as `LLMError` -> 503.
    """
    from shared.config import get_settings
    from shared.llm import gateway_url

    if gateway_url():
        return False
    try:
        settings = get_settings()
    except Exception:
        return True
    # Fireworks is the primary model — accept its key as sufficient.
    # Other provider keys (ANTHROPIC, OPENAI, GOOGLE) also count because
    # the LiteLLM SDK can route directly to them when gateway is absent.
    for attr in ("fireworks_api_key", "anthropic_api_key", "openai_api_key", "google_api_key"):
        key = getattr(settings, attr, None)
        if key is not None:
            value = key.get_secret_value() if hasattr(key, "get_secret_value") else key
            if value:
                return False
    return True


def _empty_passthrough(reason: GathererStatus) -> GathererOutput:
    """Synthesise an empty GathererOutput for fallback paths.

    Used when the agent fails fatally (tool budget exceeded with no final
    emission, schema violation, loop timeout) and we still need a
    structured response for the consumer.
    """
    return GathererOutput(
        entities=[],
        chunks=[],
        gatherer_notes=GathererNotes(
            turns_used=0,
            tools_called=[],
            confidence="low",
            dropped=[
                DroppedCandidate(
                    canonical_id="<harness>",
                    reason=f"harness_passthrough: {reason}",
                )
            ],
        ),
    )


# ============================================================
# Turn execution
# ============================================================

async def _execute_tool_call(
    state: _LoopState,
    tool_call: Any,
) -> dict[str, Any]:
    """Dispatch a single tool call, return the result dict + serialized JSON content."""
    fn = getattr(tool_call, "function", None)
    name = getattr(fn, "name", None) or "unknown"
    raw_args = getattr(fn, "arguments", "{}")
    try:
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
    except json.JSONDecodeError as exc:
        log.warning("agent.tool_arguments_invalid_json", tool_name=name, error=str(exc))
        return {"name": name, "result": {"error": f"invalid JSON arguments: {exc}"}}

    state.tools_fired.append(name)
    state.tool_calls_count += 1

    # need_deeper is the budget-extension signal — handled here so the
    # tool registry stays a clean retrieval-only surface.
    if name == "need_deeper":
        reason = str(arguments.get("reason", ""))[:200] or "no reason given"
        if state.extensions_used >= SEARCH_AGENT_MAX_EXTENSIONS:
            return {
                "name": name,
                "result": {
                    "granted": False,
                    "reason": f"max extensions ({SEARCH_AGENT_MAX_EXTENSIONS}) already used",
                },
            }
        state.extensions_used += 1
        state.budget += SEARCH_AGENT_EXTENSION_GRANT
        log.info(
            "agent.need_deeper_granted",
            customer_id=state.customer_id,
            trace_id=state.trace_id,
            extensions_used=state.extensions_used,
            new_budget=state.budget,
            reason=reason,
        )
        return {
            "name": name,
            "result": {
                "granted": True,
                "extensions_used": state.extensions_used,
                "new_budget": state.budget,
                "reason_logged": reason,
            },
        }

    result = await dispatch_tool_call(
        customer_id=state.customer_id,
        tool_name=name,
        arguments=arguments,
    )
    return {"name": name, "result": result}


async def _run_turn(
    state: _LoopState,
    *,
    force_final: bool,
) -> tuple[Any, str | None]:
    """Run one model turn. Returns (raw_response, content_str_or_none)."""
    tools = None if force_final else tool_definitions()
    tool_choice = None if force_final else "auto"

    call_kwargs: dict[str, Any] = {
        "model": SEARCH_AGENT_INFERENCE_MODEL,
        "messages": state.messages,
        # Use the explicit OpenAI-style json_schema form rather than passing
        # the Pydantic class directly. LiteLLM SDK's auto-translation of
        # Pydantic -> json_schema gets dropped on the wire to Fireworks via
        # the proxy (verified live 2026-05-17: passing the class returned
        # prose; passing the explicit json_schema returns valid JSON).
        # See _GATHERER_OUTPUT_RESPONSE_FORMAT below for the cached schema.
        "response_format": _GATHERER_OUTPUT_RESPONSE_FORMAT,
        "extra_headers": {"x-session-affinity": _affinity_key(state.customer_id, state.query)},
        "timeout": SEARCH_AGENT_TURN_TIMEOUT_SECONDS,
    }
    if tools is not None:
        call_kwargs["tools"] = tools
        call_kwargs["tool_choice"] = tool_choice

    try:
        resp = await acompletion(**call_kwargs)
    except LLMError as exc:
        log.warning(
            "agent.turn_llm_error",
            customer_id=state.customer_id,
            trace_id=state.trace_id,
            turn=state.turn_count,
            error=str(exc),
        )
        raise

    state.turn_count += 1
    rate = _extract_cache_hit_rate(resp)
    if rate is not None:
        state.cache_hit_rates.append(rate)

    choices = getattr(resp, "choices", None) or []
    if not choices:
        return resp, None
    msg = getattr(choices[0], "message", None)
    content = getattr(msg, "content", None) if msg is not None else None
    return resp, content


def _serialize_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """LiteLLM tool_calls -> OpenAI-shaped list for the next request's
    assistant message echo."""
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        out.append({
            "id": getattr(tc, "id", "?"),
            "type": "function",
            "function": {
                "name": getattr(fn, "name", "unknown") if fn is not None else "unknown",
                "arguments": getattr(fn, "arguments", "{}") if fn is not None else "{}",
            },
        })
    return out


# ============================================================
# Top-level entry point
# ============================================================

async def run_gatherer(
    req: QueryRequest,
    customer_id: str,
    request: Request | None = None,
) -> QueryResponse:
    """Run the gatherer agent against `req.query` and return a QueryResponse.

    Telemetry written to `request.state.*` when `request` is provided:
        gatherer_status, tool_calls_count, need_deeper_extensions,
        confidence, dropped_count, cache_hit_rate, intents_count (=1).

    Raises:
        HTTPException(503) on fatal LLM/provider failures (no fallback by
        design — consumers handle 503 cleanly).
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")

    trace_id = req.trace_id or f"q-{int(datetime.now().timestamp() * 1000)}"
    timing: dict[str, float] = {}

    # Step 1 — deterministic grounding (synchronous, ~25ms).
    t_grounding = time.perf_counter()
    try:
        bundle = await _build_bundle_with_token_fallback(customer_id, req.query)
    except Exception as exc:
        log.warning(
            "agent.grounding_failed",
            customer_id=customer_id,
            trace_id=trace_id,
            error=str(exc),
        )
        bundle = GroundingBundle()
    timing["grounding_ms"] = (time.perf_counter() - t_grounding) * 1000

    # Build the agent input message.
    user_msg = _build_user_message(req.query, bundle)
    system_prompt = build_system_prompt(datetime.now(UTC))

    state = _LoopState(
        customer_id=customer_id,
        trace_id=trace_id,
        query=req.query,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        # Fireworks ignores cache_control today, but
                        # LiteLLM forwards it for Anthropic too. Cheap to
                        # leave on; it'll auto-engage when we test other
                        # providers in the A/B set.
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": user_msg},
        ],
    )

    t_agent = time.perf_counter()
    status: GathererStatus = "ok"

    # Short-circuit when no LLM provider is configured (test env,
    # bootstrap, self-host without keys). Mirrors the pre-cutover router's
    # graceful no-op in `_call_haiku` — returns empty results with a clear
    # status rather than 503ing. Provider-side outages (ANTHROPIC_API_KEY
    # set + Fireworks down) still raise 503 via the LLMError catch below.
    if _no_llm_configured():
        log.info(
            "agent.no_llm_configured_short_circuit",
            customer_id=customer_id,
            trace_id=trace_id,
        )
        status = "no_llm_configured"
        gathered = _empty_passthrough("no_llm_configured")
        timing["agent_ms"] = (time.perf_counter() - t_agent) * 1000
        if request is not None:
            request.state.gatherer_status = status
            request.state.tool_calls_count = 0
            request.state.need_deeper_extensions = 0
            request.state.confidence = gathered.gatherer_notes.confidence
            request.state.dropped_count = len(gathered.gatherer_notes.dropped)
            request.state.cache_hit_rate = None
            request.state.intents_count = 1
            request.state.router_model = SEARCH_AGENT_INFERENCE_MODEL
            request.state.failure_recovered = True
        return to_query_response(
            query=req.query, gathered=gathered, trace_id=trace_id, timing_ms=timing
        )

    gathered: GathererOutput | None = None
    try:
        gathered = await asyncio.wait_for(
            _drive_loop(state),
            timeout=SEARCH_AGENT_LOOP_TIMEOUT_SECONDS,
        )
        if gathered is None:
            status = "schema_violation"
            gathered = _empty_passthrough("schema_violation")
        else:
            if state.tool_calls_count >= SEARCH_AGENT_HARD_CAP and not gathered.chunks and not gathered.entities:
                status = "tool_budget_exceeded"
    except TimeoutError:
        log.warning(
            "agent.loop_timeout",
            customer_id=customer_id,
            trace_id=trace_id,
            turn=state.turn_count,
            tool_calls=state.tool_calls_count,
        )
        status = "loop_timeout"
        gathered = _empty_passthrough("loop_timeout")
    except LLMError as exc:
        log.error(
            "agent.fatal_provider_error",
            customer_id=customer_id,
            trace_id=trace_id,
            error=str(exc),
        )
        if request is not None:
            request.state.full_failure = True
        raise HTTPException(status_code=503, detail="search agent unavailable") from exc

    timing["agent_ms"] = (time.perf_counter() - t_agent) * 1000

    # Defense-in-depth: log when turn-1 mandate slipped.
    missing = _REQUIRED_TURN_1_CHANNELS - set(state.turn_1_tools_fired)
    if missing:
        log.warning(
            "agent.turn_1_mandate_skipped",
            customer_id=customer_id,
            trace_id=trace_id,
            missing_channels=sorted(missing),
            fired=state.turn_1_tools_fired,
        )

    # Telemetry written to request.state for the query_traces middleware.
    if request is not None:
        request.state.gatherer_status = status
        request.state.tool_calls_count = state.tool_calls_count
        request.state.need_deeper_extensions = state.extensions_used
        request.state.confidence = gathered.gatherer_notes.confidence
        request.state.dropped_count = len(gathered.gatherer_notes.dropped)
        request.state.cache_hit_rate = (
            sum(state.cache_hit_rates) / len(state.cache_hit_rates)
            if state.cache_hit_rates
            else None
        )
        request.state.intents_count = 1  # gatherer is single-intent at the harness level
        request.state.router_model = SEARCH_AGENT_INFERENCE_MODEL
        request.state.failure_recovered = status != "ok"

    return to_query_response(
        query=req.query,
        gathered=gathered,
        trace_id=trace_id,
        timing_ms=timing,
    )


async def _drive_loop(state: _LoopState) -> GathererOutput | None:
    """Multi-turn loop: model -> tool calls -> tool results -> model -> ...
    Terminates on first turn with no tool_calls (final emission)."""
    while True:
        budget_exhausted = state.tool_calls_count >= state.budget
        # Force-final when budget is exhausted: strip tools so the model
        # MUST emit final GathererOutput rather than retry tool calls.
        force_final = budget_exhausted

        resp, content = await _run_turn(state, force_final=force_final)

        choices = getattr(resp, "choices", None) or []
        msg = getattr(choices[0], "message", None) if choices else None
        tool_calls = getattr(msg, "tool_calls", None) or []

        # Record turn-1 tools fired (for the mandate-tracking log).
        if state.turn_count == 1:
            state.turn_1_tools_fired = [
                getattr(getattr(tc, "function", None), "name", "?")
                for tc in tool_calls
            ]

        if not tool_calls:
            # Curate path — final emission.
            parsed = _parse_gatherer_output(content)
            return parsed

        if force_final:
            # We told the model "no more tools" — if it still emitted tool
            # calls, it ignored us. Try parsing whatever content came back.
            parsed = _parse_gatherer_output(content)
            if parsed is not None:
                return parsed
            return None

        # Echo the assistant turn back into history so the model sees its
        # own tool_calls on the next turn (OpenAI chat-completion contract).
        state.messages.append({
            "role": "assistant",
            "content": content or "",
            "tool_calls": _serialize_tool_calls(tool_calls),
        })

        # Execute all tool calls in parallel — the prompt enforces
        # parallel-by-default; we honor whatever the agent chose.
        results = await asyncio.gather(
            *(_execute_tool_call(state, tc) for tc in tool_calls),
            return_exceptions=False,
        )

        # Append each tool result as a `tool`-role message with the
        # call's id linking back to the assistant turn.
        for tc, res in zip(tool_calls, results, strict=True):
            payload = res["result"]
            # Trim massive payloads to keep per-turn context bounded.
            content_str = json.dumps(payload, default=str)
            if len(content_str) > 60_000:
                content_str = json.dumps({
                    "truncated": True,
                    "original_size_chars": len(content_str),
                    "head": content_str[:30_000],
                })
            state.messages.append({
                "role": "tool",
                "tool_call_id": getattr(tc, "id", "?"),
                "content": content_str,
            })

        if state.tool_calls_count >= SEARCH_AGENT_HARD_CAP:
            # Force a final emission next loop iteration.
            continue


__all__ = ["run_gatherer"]
