"""Gatherer agent loop.

Entry point: `run_gatherer(req, customer_id, request)` -> QueryResponse.

Flow:
1. SEQUENTIAL grounding → LLM entity extraction (extraction uses the
   grounding bundle as `<candidates>` context — same pattern as PR #282's
   Haiku router).
2. Pre-fan-out: harness calls `execute_search([query])` once before the
   LLM. The 4-channel fan-out result lands in the LLM's first user
   message as `<channel_results>`.
3. Agent loop on Fireworks gpt-oss-120B with `tool_choice="required"`:
   the model MUST call something — either a retrieval tool (search,
   subgraph, fetch_doc), the budget extension (need_deeper), or the
   terminal (emit_gatherer_output). No prose path. Loop ends when
   `emit_gatherer_output` is called; its arguments ARE the final
   GathererOutput.
4. Telemetry: per-stage latency log + R2 transcript blob (PR #301).
5. Adapter converts GathererOutput → existing QueryResponse shape.

Plan: docs/specs/agentic-search.md.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

if TYPE_CHECKING:
    from starlette.requests import Request

from services.retrieval.agent.adapter import to_query_response
from services.retrieval.agent.extractor import extract_entities_with_llm
from services.retrieval.agent.models import (
    DroppedCandidate,
    GathererNotes,
    GathererOutput,
    GathererStatus,
)
from services.retrieval.agent.prompt import build_system_prompt
from services.retrieval.agent.tools import (
    NEED_DEEPER_TOOL_NAME,
    TERMINAL_TOOL_NAME,
    dispatch_tool_call,
    execute_search,
    tool_definitions,
)
from services.retrieval.grounding import GroundingBundle
from services.retrieval.router import (
    Intent,
    RouterEntity,
    _build_bundle_with_token_fallback,
    _escape_query_for_xml,
    _reconcile_entities_with_bundle,
)
from shared.constants import (
    SEARCH_AGENT_EXTENSION_GRANT,
    SEARCH_AGENT_HARD_CAP,
    SEARCH_AGENT_INFERENCE_MODEL,
    SEARCH_AGENT_LOOP_TIMEOUT_SECONDS,
    SEARCH_AGENT_MAX_EXTENSIONS,
    SEARCH_AGENT_TOOL_BUDGET,
    SEARCH_AGENT_TRACE_SAMPLE_RATE,
    SEARCH_AGENT_TURN_TIMEOUT_SECONDS,
)
from shared.llm import LLMError, acompletion
from shared.logging import get_logger
from shared.models import QueryRequest, QueryResponse

log = get_logger(__name__)


# ============================================================
# Loop state
# ============================================================

@dataclass(slots=True)
class LoopState:
    """Mutable per-request loop bookkeeping.

    Held entirely in memory for the duration of one query. Captured by
    the trace_blob module (PR #301) for the per-query R2 transcript.
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
    # Per-turn LLM call latencies (ms).
    turn_latencies_ms: list[float] = field(default_factory=list)
    # Per-tool execution latencies (ms).
    tool_latencies_ms: list[float] = field(default_factory=list)
    # Retained for trace_blob schema compat. Always 0 under the
    # tool_choice="required" + emit_gatherer_output terminal — the model
    # can no longer emit prose, so there's nothing to retry.
    prose_retries: int = 0
    # Pre-fan-out search result, captured for the trace blob so the
    # nightly analyzer can correlate channel coverage with curated outcomes.
    prefanout: dict[str, Any] = field(default_factory=dict)
    prefanout_hit_counts: dict[str, int] = field(default_factory=dict)
    # Per-turn chain-of-thought from the model's `reasoning_content`
    # channel (gpt-oss harmony `analysis` block, surfaced by LiteLLM as
    # message.reasoning_content). One entry per acompletion call. None
    # when the provider didn't emit reasoning for that turn. Not echoed
    # back into the next turn's request (OpenAI chat-completion contract
    # only round-trips role/content/tool_calls), so without this list
    # the agent's "why did it pick this tool" trail is lost.
    reasoning_per_turn: list[str | None] = field(default_factory=list)


# ============================================================
# Message + helpers
# ============================================================

# Per-hit snippet cap (chars). Enough for the model to judge relevance;
# full content is one fetch_doc call away. 150 chars ≈ 1-2 sentences.
_PREFANOUT_SNIPPET_CHARS = 150

# Per-channel hit cap in the compact rendering. The pre-fan-out returns
# top-K hits per channel (config-controlled, currently up to ~20 each).
# Rendering all of them in the first-turn input bloats prompts to 15K+
# tokens of low-signal evidence. Cap at 10 — model sees the strongest
# hits per channel; weaker hits remain in the database for fetch_doc /
# explicit-search recovery (not lost, just not in the cold-cache prompt).
_PREFANOUT_PER_CHANNEL_DISPLAY_CAP = 10


def _truncate_snippet(text: str | None, n: int = _PREFANOUT_SNIPPET_CHARS) -> str:
    """One-line snippet for the compact channel rendering."""
    if not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= n:
        return flat
    return flat[: n - 1].rstrip() + "…"


def _format_prefanout_compact(prefanout: dict[str, Any]) -> str:
    """Render `execute_search` result as compact text instead of JSON dump.

    Why: the JSON form embeds every per-hit field (chunk_id, created_at,
    updated_at, author_id, source_url, full content, plus field-name
    overhead × N hits × 4 channels) and hits ~15K input tokens on cold
    cache. gpt-oss-120b is a reasoning model — that much input deterministically
    blows past the 90s loop timeout on cold cache. The compact form keeps
    every doc_id (so fetch_doc still works), every score, every title, and
    a 150-char snippet — everything the model needs to pick its next tool
    call. Verbose fields are one fetch_doc away.

    Format (per hit):
        [v1] doc_id score=0.87 src=slack title="..."
             "first 150 chars of content..."
    """
    out_lines: list[str] = []
    sub_queries = prefanout.get("sub_queries") or []
    for sq_idx, sq in enumerate(sub_queries, 1):
        if len(sub_queries) > 1:
            q_label = (sq.get("query") or "").strip()
            out_lines.append(f"\n=== sub_query {sq_idx}: {_truncate_snippet(q_label, 100)} ===")
        for channel_name, prefix in (
            ("vector", "v"),
            ("bm25", "b"),
            ("graph", "g"),
            ("inferred_edge", "i"),
        ):
            hits = sq.get(channel_name) or []
            if not hits:
                continue
            shown = hits[:_PREFANOUT_PER_CHANNEL_DISPLAY_CAP]
            omitted = len(hits) - len(shown)
            header = f"<{channel_name}>"
            if omitted > 0:
                header = (
                    f"<{channel_name} showing_top_{len(shown)}_of_{len(hits)} "
                    f"(remaining accessible via fetch_doc / search)>"
                )
            out_lines.append(header)
            for i, hit in enumerate(shown, 1):
                doc_id = hit.get("doc_id") or "?"
                score = hit.get("score")
                score_str = f"{score:.3f}" if isinstance(score, int | float) else "?"
                src = hit.get("source_system") or "?"
                title = _truncate_snippet(hit.get("title"), 80)
                snippet = _truncate_snippet(hit.get("content"))
                # inferred_edge carries the "why" rationale — the moat.
                why = hit.get("why")
                edge = hit.get("edge_type")
                tag_parts = [f"[{prefix}{i}]", doc_id, f"score={score_str}", f"src={src}"]
                if edge:
                    tag_parts.append(f"edge={edge}")
                if title:
                    tag_parts.append(f'title="{title}"')
                out_lines.append(" ".join(tag_parts))
                if snippet:
                    out_lines.append(f'    "{snippet}"')
                if why:
                    out_lines.append(f"    why: {_truncate_snippet(why, 200)}")
    return "\n".join(out_lines) if out_lines else "(no pre-fan-out hits)"


def _build_user_message(
    query: str,
    bundle: GroundingBundle,
    prefanout: dict[str, Any] | None = None,
) -> str:
    """Render the per-query user message.

    Layout (grounding-first for cache stability, then channel results,
    then the raw query last for recency bias):
        <grounding>           — entity bag from grounding + extraction
        <connected_sources>   — which source systems the tenant has wired
        <channel_results>     — output of pre-fan-out `execute_search`
        <query>               — raw user query, last for recency
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

    channel_results_block = ""
    if prefanout:
        channel_results_block = (
            f"\n\n<channel_results>\n"
            f"The harness already fired `search([raw_query])` before this turn — "
            f"results below (vector + bm25 + graph + inferred_edge, anchored on "
            f"the grounded entities). Use this as turn-1 evidence. For more "
            f"detail on any doc_id, call `fetch_doc(doc_id)`. For exploration, "
            f"call `search` with REFORMULATED queries or `subgraph(anchor)`. "
            f"When you've curated the answer, call `emit_gatherer_output`.\n"
            f"{_format_prefanout_compact(prefanout)}\n"
            f"</channel_results>"
        )

    safe_query = _escape_query_for_xml(query)
    return (
        f"<grounding>\n{grounding_block}\n</grounding>\n\n"
        f"<connected_sources>{sources_block}</connected_sources>"
        f"{channel_results_block}\n\n"
        f"<query>\n{safe_query}\n</query>"
    )


def _affinity_key(customer_id: str, query: str) -> str:
    """Per-query Fireworks session-affinity hash so consecutive turns
    cache-hit on the same replica (90% discount only applies in-replica)."""
    h = sha256()
    h.update(customer_id.encode("utf-8", errors="ignore"))
    h.update(b":")
    h.update(query.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:32]


def _extract_cache_hit_rate(resp: Any) -> float | None:
    """Compute cache_read_input_tokens / prompt_tokens for one response."""
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


def _empty_passthrough(reason: GathererStatus) -> GathererOutput:
    """Synthesise an empty GathererOutput for fallback paths
    (no_llm_configured, loop_timeout, terminal_args_invalid)."""
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


def _no_llm_configured() -> bool:
    """True when no LLM provider is reachable. Mirrors PR #282's
    `_call_haiku` graceful no-op for test env / bootstrap / self-host
    without keys."""
    from shared.config import get_settings
    from shared.llm import gateway_url

    if gateway_url():
        return False
    try:
        settings = get_settings()
    except Exception:
        return True
    for attr in ("fireworks_api_key", "anthropic_api_key", "openai_api_key", "google_api_key"):
        key = getattr(settings, attr, None)
        if key is not None:
            value = key.get_secret_value() if hasattr(key, "get_secret_value") else key
            if value:
                return False
    return True


# ============================================================
# Per-turn LLM call + dispatcher
# ============================================================

async def _run_turn(state: LoopState) -> Any:
    """Run one LLM turn with tool_choice='required'. Records latency +
    cache hit rate on state. Returns the raw response."""
    call_kwargs: dict[str, Any] = {
        "model": SEARCH_AGENT_INFERENCE_MODEL,
        "messages": state.messages,
        "tools": tool_definitions(),
        # Greedy decoding. Without this, Fireworks defaults to
        # temperature=1.0 and the same query produces different tool
        # trajectories run-to-run (3x test on "self-hosting features":
        # one run 13 results, one run dead-end, one run timeout). For a
        # retrieval gatherer same-question-same-evidence is the contract.
        "temperature": 0,
        # The whole point: model MUST call a tool. With this set, prose-
        # only output is not an option — the model picks a retrieval
        # tool (search/subgraph/fetch_doc/need_deeper) or the terminal
        # (emit_gatherer_output). No prose, no schema-violation path.
        "tool_choice": "required",
        # Force OpenAI wire shape so tool_choice + structured tool
        # schemas survive the LiteLLM proxy. See
        # `feedback_litellm_gateway_gemini_405` + the 4-layer Fireworks
        # gotcha memory for why this is per-call rather than global.
        "custom_llm_provider": "openai",
        "extra_headers": {"x-session-affinity": _affinity_key(state.customer_id, state.query)},
        "timeout": SEARCH_AGENT_TURN_TIMEOUT_SECONDS,
    }

    t_turn = time.perf_counter()
    try:
        resp = await acompletion(**call_kwargs)
    except LLMError as exc:
        elapsed_ms = (time.perf_counter() - t_turn) * 1000
        log.warning(
            "agent.turn_llm_error",
            customer_id=state.customer_id,
            trace_id=state.trace_id,
            turn=state.turn_count,
            elapsed_ms=round(elapsed_ms, 1),
            error=str(exc),
        )
        raise

    elapsed_ms = (time.perf_counter() - t_turn) * 1000
    state.turn_count += 1
    state.turn_latencies_ms.append(elapsed_ms)
    rate = _extract_cache_hit_rate(resp)
    if rate is not None:
        state.cache_hit_rates.append(rate)

    choices = getattr(resp, "choices", None) or []
    msg = getattr(choices[0], "message", None) if choices else None
    tool_calls = getattr(msg, "tool_calls", None) or [] if msg is not None else []
    content = getattr(msg, "content", None) if msg is not None else None
    # Capture the model's reasoning channel (gpt-oss harmony `analysis`
    # block, normalized by LiteLLM to `message.reasoning_content`). NOT
    # echoed back to the next turn; saved on state for the trace blob so
    # the nightly analyzer can see "why did the agent pick fetch_doc_chunks
    # here" instead of only the chosen arguments.
    reasoning = getattr(msg, "reasoning_content", None) if msg is not None else None
    state.reasoning_per_turn.append(reasoning if reasoning else None)
    log.info(
        "agent.turn_complete",
        customer_id=state.customer_id,
        trace_id=state.trace_id,
        turn=state.turn_count,
        elapsed_ms=round(elapsed_ms, 1),
        tool_calls_count=len(tool_calls),
        content_len=len(content) if content else 0,
        reasoning_len=len(reasoning) if reasoning else 0,
        cache_hit_rate=round(rate, 3) if rate is not None else None,
    )
    return resp


def _serialize_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """LiteLLM tool_calls → OpenAI-shape list for the next request's
    assistant-message echo."""
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


async def _execute_tool_call(
    state: LoopState,
    tool_call: Any,
) -> tuple[str, dict[str, Any]]:
    """Dispatch a single non-terminal tool call. Returns (tool_name, result_dict).

    Handles `need_deeper` inline (budget extension, no dispatch). Routes
    `search`, `subgraph`, `fetch_doc` to the registry. Terminal
    `emit_gatherer_output` is detected by the loop BEFORE this function
    runs and never reaches here.
    """
    fn = getattr(tool_call, "function", None)
    name = getattr(fn, "name", None) or "unknown"
    raw_args = getattr(fn, "arguments", "{}")
    try:
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
    except json.JSONDecodeError as exc:
        log.warning("agent.tool_arguments_invalid_json", tool_name=name, error=str(exc))
        return name, {"error": f"invalid JSON arguments: {exc}"}

    state.tools_fired.append(name)
    state.tool_calls_count += 1

    if name == NEED_DEEPER_TOOL_NAME:
        reason = str(arguments.get("reason", ""))[:200] or "no reason given"
        if state.extensions_used >= SEARCH_AGENT_MAX_EXTENSIONS:
            return name, {
                "granted": False,
                "reason": f"max extensions ({SEARCH_AGENT_MAX_EXTENSIONS}) already used",
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
        return name, {
            "granted": True,
            "extensions_used": state.extensions_used,
            "new_budget": state.budget,
            "reason_logged": reason,
        }

    t_tool = time.perf_counter()
    result = await dispatch_tool_call(
        customer_id=state.customer_id,
        tool_name=name,
        arguments=arguments,
    )
    elapsed_ms = (time.perf_counter() - t_tool) * 1000
    state.tool_latencies_ms.append(elapsed_ms)
    log.info(
        "agent.tool_complete",
        customer_id=state.customer_id,
        trace_id=state.trace_id,
        tool_name=name,
        elapsed_ms=round(elapsed_ms, 1),
        is_error="error" in result if isinstance(result, dict) else False,
    )
    return name, result


# ============================================================
# Terminal extraction
# ============================================================

def _derive_doc_id_from_chunk_id(chunk_id: str) -> str | None:
    """Some providers (Cerebras gpt-oss-120b in non-strict mode) emit
    chunks without `doc_id`. chunk_id typically encodes the parent doc
    as a prefix — split on ':chunk:' or strip trailing ':<index>'.

    Examples:
      `github:owner/repo:pr:42:chunk:3` → `github:owner/repo:pr:42`
      `notion:doc:abc123:chunk:0`       → `notion:doc:abc123`
      `linear:ticket:PRB-17`            → `linear:ticket:PRB-17` (chunk_id == doc_id)
    """
    if not chunk_id:
        return None
    if ":chunk:" in chunk_id:
        return chunk_id.split(":chunk:", 1)[0]
    # Fall through: chunk_id may already be doc_id (some retrievers index
    # whole-document chunks). Caller will dedupe.
    return chunk_id


# Alternate text-field names non-strict providers emit for chunk content.
# Cerebras gpt-oss-120b sometimes emits "doc-level chunks" with the body
# under `description`/`summary` instead of `content`. Order matters —
# the first non-empty match wins.
_CHUNK_CONTENT_ALIASES = ("description", "summary", "snippet", "text", "body", "title")


def _coerce_lenient(raw: dict[str, Any], state: "LoopState | None" = None) -> dict[str, Any]:
    """Pre-parse coercion for non-strict providers (Cerebras et al.).

    Cerebras's gpt-oss-120b emits emit_gatherer_output args with input-
    shaped fields (`title`/`source`/`url` from <channel_results>) instead
    of the strict GathererOutput schema, and sometimes "doc-level chunks"
    where `chunk_id` is omitted and `content` lives under
    `description`/`summary`. With `extra="ignore"` on the models the
    unknown extras drop silently; this helper additionally:

      - Derives `chunk.doc_id` from `chunk.chunk_id` prefix when missing.
      - Derives `chunk.chunk_id` from `chunk.doc_id` when missing
        (whole-doc citation fallback).
      - Sources `chunk.content` from alternate text fields (description,
        summary, snippet, text, body, title) when `content` is missing.
      - Harness-fills `gatherer_notes.turns_used` and `tools_called` from
        LoopState — that ledger is the canonical record per trace_blob.py.
      - Drops chunks for which no usable doc_id + content pair can be
        recovered (no resolvable citation).

    Safe to call even on Fireworks-strict output — only fills fields
    that are missing.
    """
    out = dict(raw) if isinstance(raw, dict) else {}
    # Entities: filter non-dict items. Cerebras gpt-oss-120b occasionally
    # emits malformed JSON fragments as bare strings inside arrays
    # (e.g. `entities=[{...valid...}, '{', '{canonical_id":...']`) — a
    # constrained-decoding partial-failure mode. Pydantic rejects the
    # whole emission on the first non-dict; drop the malformed items so
    # the valid neighbors survive.
    entities_in = out.get("entities") or []
    out["entities"] = [e for e in entities_in if isinstance(e, dict)]
    # Chunks: derive missing required fields where possible
    chunks_in = out.get("chunks") or []
    chunks_out: list[dict[str, Any]] = []
    for ch in chunks_in:
        if not isinstance(ch, dict):
            continue
        ch_out = dict(ch)
        # doc_id ↔ chunk_id derivation (whichever is present can supply
        # the other; without either we can't cite the chunk)
        if not ch_out.get("doc_id") and ch_out.get("chunk_id"):
            derived = _derive_doc_id_from_chunk_id(ch_out["chunk_id"])
            if derived:
                ch_out["doc_id"] = derived
        if not ch_out.get("chunk_id") and ch_out.get("doc_id"):
            # Whole-doc citation fallback — chunk_id == doc_id is a valid
            # convention (id_lookup retriever does this for short docs).
            ch_out["chunk_id"] = ch_out["doc_id"]
        if not ch_out.get("doc_id") or not ch_out.get("chunk_id"):
            continue  # Can't recover citation
        # Content — try aliases when missing
        if not ch_out.get("content"):
            for alias in _CHUNK_CONTENT_ALIASES:
                v = ch_out.get(alias)
                if isinstance(v, str) and v.strip():
                    ch_out["content"] = v
                    break
        if not ch_out.get("content"):
            continue  # No body to cite
        chunks_out.append(ch_out)
    if chunks_out or "chunks" in out:
        out["chunks"] = chunks_out
    # gatherer_notes: harness is authoritative for turns_used + tools_called
    notes = dict(out.get("gatherer_notes") or {})
    if state is not None:
        notes["turns_used"] = state.turn_count
        notes["tools_called"] = list(state.tools_fired) + [TERMINAL_TOOL_NAME]
    out["gatherer_notes"] = notes
    return out


def _parse_terminal_args(
    raw_args: str | dict[str, Any] | None,
    state: "LoopState | None" = None,
) -> GathererOutput | None:
    """Parse the emit_gatherer_output tool-call arguments as
    GathererOutput. Returns None on JSON-parse failure (unrecoverable).
    Pydantic-level schema drift from non-strict providers is absorbed
    via `_coerce_lenient` + the lenient `extra="ignore"` model config —
    no fallback parse needed.

    `state` is used to fill harness-authoritative fields (`turns_used`,
    `tools_called`) regardless of what the model emitted. Pre-#306
    (temperature=0) the model sometimes populated these; post-#306 it
    deterministically omits the optional `tools_called` field. Either
    way harness state is the canonical record (per trace_blob.py).
    """
    if raw_args is None:
        return None
    try:
        if isinstance(raw_args, str):
            raw_dict = json.loads(raw_args)
        else:
            raw_dict = dict(raw_args)
    except Exception as exc:
        log.warning("agent.terminal_args_json_parse_failed", error=str(exc))
        return None
    coerced = _coerce_lenient(raw_dict, state)
    try:
        return GathererOutput.model_validate(coerced)
    except Exception as exc:
        log.warning(
            "agent.terminal_args_parse_failed",
            error=str(exc),
            preview=str(coerced)[:1000],
        )
        return None


# ============================================================
# Trace-blob stash
# ============================================================

def _stash_for_trace_persist(
    request: Request | None,
    *,
    customer_id: str,
    trace_id: str,
    query: str,
    state: LoopState | None,
    gathered: GathererOutput | None,
    status: GathererStatus | None,
    timing: dict[str, float],
) -> None:
    """Stash raw refs onto request.state so middleware can persist the
    trace blob to R2 as a post-flush BackgroundTask. Sampling decided here."""
    if request is None:
        return
    try:
        if random.random() > SEARCH_AGENT_TRACE_SAMPLE_RATE:
            return
        request.state.search_agent_loop_state = state
        request.state.search_agent_gathered = gathered
        request.state.search_agent_status = status
        request.state.search_agent_timing = timing
        request.state.search_agent_query = query
        request.state.search_agent_model = SEARCH_AGENT_INFERENCE_MODEL
        request.state.search_agent_trace_id = trace_id
        request.state.search_agent_customer_id = customer_id
        request.state.search_agent_should_persist = True
    except Exception as exc:
        log.warning(
            "agent.trace_stash_failed",
            customer_id=customer_id,
            trace_id=trace_id,
            error=str(exc),
            error_class=type(exc).__name__,
        )


# ============================================================
# Top-level entry point
# ============================================================

async def run_gatherer(
    req: QueryRequest,
    customer_id: str,
    request: Request | None = None,
) -> QueryResponse:
    """Run the gatherer agent against `req.query` and return a QueryResponse.

    Raises HTTPException(503) on fatal LLM/provider failures (no fallback).
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")

    trace_id = req.trace_id or f"q-{int(datetime.now().timestamp() * 1000)}"
    timing: dict[str, float] = {}

    # Step 1 — SEQUENTIAL grounding → LLM extraction (bundle as context).
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

    t_extraction = time.perf_counter()
    extracted = await extract_entities_with_llm(customer_id, req.query, bundle)
    timing["extraction_ms"] = (time.perf_counter() - t_extraction) * 1000

    # Reconcile LLM-proposed entities against the bundle (safety net).
    if extracted:
        synthetic_intent = Intent(
            query_text=req.query,
            mode="search",
            confidence=0.0,
            entities=[
                RouterEntity(
                    entity_type=e.entity_type,
                    canonical_id=e.canonical_id,
                    display_name=e.display_name,
                    confidence=e.confidence,
                )
                for e in extracted
            ],
        )
        _reconcile_entities_with_bundle([synthetic_intent], bundle)
        reconciled = synthetic_intent.entities
    else:
        reconciled = []

    # Unified entity bag — grounded first (highest confidence anchor),
    # then deduped extracted picks. Used for the pre-fan-out search call.
    seen: set[tuple[str, str]] = set()
    entity_dicts: list[dict[str, str]] = []
    for c in (list(bundle.candidates) + list(bundle.bare_id_matches)):
        key = (c.entity_type, c.canonical_id)
        if key in seen:
            continue
        seen.add(key)
        entity_dicts.append({"entity_type": c.entity_type, "canonical_id": c.canonical_id})
    for e in reconciled:
        key = (e.entity_type, e.canonical_id)
        if key in seen:
            continue
        seen.add(key)
        entity_dicts.append({"entity_type": e.entity_type, "canonical_id": e.canonical_id})

    log.info(
        "agent.entity_bag_assembled",
        customer_id=customer_id,
        trace_id=trace_id,
        grounded=len(bundle.candidates) + len(bundle.bare_id_matches),
        extracted=len(extracted),
        final=len(entity_dicts),
        grounding_ms=round(timing["grounding_ms"], 1),
        extraction_ms=round(timing["extraction_ms"], 1),
    )

    # Step 2 — Pre-fan-out: single `execute_search` call covers all 4
    # channels (vector + bm25 + graph + inferred_edge) anchored on the
    # unified entity bag. Result is the LLM's turn-1 evidence.
    t_prefanout = time.perf_counter()
    prefanout_result = await execute_search(
        customer_id=customer_id,
        queries=[req.query],
        entity_ids=entity_dicts or None,
    )
    timing["prefanout_ms"] = (time.perf_counter() - t_prefanout) * 1000

    # Capture per-channel hit counts for the trace + summary log.
    sub = (prefanout_result.get("sub_queries") or [{}])[0]
    prefanout_hit_counts = {
        "vector": len(sub.get("vector") or []),
        "bm25": len(sub.get("bm25") or []),
        "graph": len(sub.get("graph") or []),
        "inferred_edge": len(sub.get("inferred_edge") or []),
    }
    log.info(
        "agent.prefanout_complete",
        customer_id=customer_id,
        trace_id=trace_id,
        elapsed_ms=round(timing["prefanout_ms"], 1),
        hits=prefanout_hit_counts,
    )

    # Step 3 — Build the user message and short-circuit if no LLM.
    user_msg = _build_user_message(req.query, bundle, prefanout_result)
    system_prompt = build_system_prompt(datetime.now(UTC))

    state = LoopState(
        customer_id=customer_id,
        trace_id=trace_id,
        query=req.query,
        prefanout=prefanout_result,
        prefanout_hit_counts=prefanout_hit_counts,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": user_msg},
        ],
    )

    t_agent = time.perf_counter()
    status: GathererStatus = "ok"

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
        _stash_for_trace_persist(
            request,
            customer_id=customer_id,
            trace_id=trace_id,
            query=req.query,
            state=state,
            gathered=gathered,
            status=status,
            timing=timing,
        )
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
        _stash_for_trace_persist(
            request,
            customer_id=customer_id,
            trace_id=trace_id,
            query=req.query,
            state=state,
            gathered=None,
            status="fatal_provider_error",
            timing=timing,
        )
        raise HTTPException(status_code=503, detail="search agent unavailable") from exc

    timing["agent_ms"] = (time.perf_counter() - t_agent) * 1000
    timing["agent_loop_ms"] = sum(state.turn_latencies_ms)
    timing["agent_tools_ms"] = sum(state.tool_latencies_ms)

    log.info(
        "agent.query_summary",
        customer_id=customer_id,
        trace_id=trace_id,
        status=status,
        confidence=gathered.gatherer_notes.confidence,
        turns=state.turn_count,
        tool_calls=state.tool_calls_count,
        extensions=state.extensions_used,
        results=len(gathered.chunks) + len(gathered.entities),
        grounding_ms=round(timing.get("grounding_ms", 0), 1),
        extraction_ms=round(timing.get("extraction_ms", 0), 1),
        prefanout_ms=round(timing.get("prefanout_ms", 0), 1),
        agent_total_ms=round(timing.get("agent_ms", 0), 1),
        agent_llm_ms=round(timing["agent_loop_ms"], 1),
        agent_tool_ms=round(timing["agent_tools_ms"], 1),
        per_turn_ms=[round(t, 1) for t in state.turn_latencies_ms],
        per_tool_ms=[round(t, 1) for t in state.tool_latencies_ms],
        cache_hit_rate=(
            round(sum(state.cache_hit_rates) / len(state.cache_hit_rates), 3)
            if state.cache_hit_rates
            else None
        ),
    )

    # Telemetry to request.state for the query_traces middleware.
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
        request.state.intents_count = 1
        request.state.router_model = SEARCH_AGENT_INFERENCE_MODEL
        request.state.failure_recovered = status != "ok"

    _stash_for_trace_persist(
        request,
        customer_id=customer_id,
        trace_id=trace_id,
        query=req.query,
        state=state,
        gathered=gathered,
        status=status,
        timing=timing,
    )

    return to_query_response(
        query=req.query,
        gathered=gathered,
        trace_id=trace_id,
        timing_ms=timing,
    )


async def _drive_loop(state: LoopState) -> GathererOutput | None:
    """Multi-turn loop: model calls tool → execute → loop back, until
    the model calls `emit_gatherer_output` (terminal).

    `tool_choice="required"` guarantees the model picks SOME tool on
    every turn — no prose path.
    """
    while True:
        budget_exhausted = state.tool_calls_count >= state.budget
        resp = await _run_turn(state)

        choices = getattr(resp, "choices", None) or []
        msg = getattr(choices[0], "message", None) if choices else None
        tool_calls = getattr(msg, "tool_calls", None) or []
        content = getattr(msg, "content", None)

        if state.turn_count == 1:
            state.turn_1_tools_fired = [
                getattr(getattr(tc, "function", None), "name", "?")
                for tc in tool_calls
            ]

        if not tool_calls:
            # With tool_choice="required" the model should ALWAYS emit at
            # least one tool call. If it didn't (provider quirk), and we
            # have parseable content, give it one last chance to be the
            # terminal payload — otherwise return None to mark
            # schema_violation.
            log.warning(
                "agent.no_tool_calls_despite_required",
                customer_id=state.customer_id,
                trace_id=state.trace_id,
                content_len=len(content) if content else 0,
            )
            return None

        # Check for the terminal in this turn's tool calls. If present,
        # take its args as the final GathererOutput and stop. (The model
        # could theoretically emit emit_gatherer_output ALONGSIDE other
        # tool calls — we treat the terminal as authoritative and ignore
        # the rest.)
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None)
            if name == TERMINAL_TOOL_NAME:
                raw_args = getattr(fn, "arguments", None)
                parsed = _parse_terminal_args(raw_args, state=state)
                log.info(
                    "agent.terminal_emit",
                    customer_id=state.customer_id,
                    trace_id=state.trace_id,
                    turn=state.turn_count,
                    parsed_ok=parsed is not None,
                )
                return parsed

        if budget_exhausted:
            # Budget gone but the model didn't terminate. Inject a forcing
            # nudge and run one more turn. tool_choice="required" still
            # applies — model must pick a tool — but with the explicit
            # "emit_gatherer_output now" instruction it should pick the
            # terminal. If it doesn't, we return None.
            log.info(
                "agent.budget_exhausted_force_terminate",
                customer_id=state.customer_id,
                trace_id=state.trace_id,
                tool_calls=state.tool_calls_count,
            )
            state.messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": _serialize_tool_calls(tool_calls),
            })
            state.messages.append({
                "role": "user",
                "content": (
                    "Tool-call budget exhausted. Call `emit_gatherer_output` "
                    "NOW with the final GathererOutput based on the evidence "
                    "you already have. Do not call any other tool."
                ),
            })
            resp2 = await _run_turn(state)
            choices2 = getattr(resp2, "choices", None) or []
            msg2 = getattr(choices2[0], "message", None) if choices2 else None
            tcs2 = getattr(msg2, "tool_calls", None) or []
            for tc in tcs2:
                fn = getattr(tc, "function", None)
                if getattr(fn, "name", None) == TERMINAL_TOOL_NAME:
                    return _parse_terminal_args(getattr(fn, "arguments", None), state=state)
            return None

        # Echo the assistant turn so the model sees its own tool_calls
        # on the next iteration (OpenAI chat-completion contract).
        state.messages.append({
            "role": "assistant",
            "content": content or "",
            "tool_calls": _serialize_tool_calls(tool_calls),
        })

        # Execute non-terminal tool calls in parallel.
        results = await asyncio.gather(
            *(_execute_tool_call(state, tc) for tc in tool_calls),
            return_exceptions=False,
        )

        # Append each tool result as a `tool`-role message linked by id.
        for tc, (_name, payload) in zip(tool_calls, results, strict=True):
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


__all__ = ["LoopState", "run_gatherer"]
