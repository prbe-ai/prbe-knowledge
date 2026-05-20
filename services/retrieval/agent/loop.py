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
    SEARCH_AGENT_SOFT_TURN_CAP,
    SEARCH_AGENT_TOOL_BUDGET,
    SEARCH_AGENT_TRACE_SAMPLE_RATE,
    SEARCH_AGENT_TURN_TIMEOUT_SECONDS,
    SourceSystem,
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
    # One entry per turn (None when the provider response omitted
    # `usage.prompt_tokens_details.cached_tokens`). Aligned by index
    # with turn_latencies_ms / reasoning_per_turn / system_fingerprints
    # so analyzers can join per-turn telemetry.
    cache_hit_rates: list[float | None] = field(default_factory=list)
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
    # Deterministic 32-bit `seed` sent to the provider on every turn of
    # this query. Same value for all turns so the tool trajectory stays
    # stable; derived from sha256(customer_id, query). Recorded so the
    # nightly analyzer can correlate seed vs reproducibility outcomes.
    seed: int = 0
    # Per-turn `system_fingerprint` from the provider response. Cerebras
    # returns a string identifying the backend config the request ran
    # against; pair with `seed` to detect when a fingerprint flip broke
    # reproducibility (live-traced 2026-05-19: identical input, same
    # seed, two distinct outputs because Q2/Q3 landed on a different
    # backend than Q1 — cache hit dropped 99.96% → 8.67% as the
    # smoking gun, but no fingerprint was captured). None when the
    # provider omits it for that turn.
    system_fingerprints_per_turn: list[str | None] = field(default_factory=list)


# ============================================================
# Message + helpers
# ============================================================

def _format_prefanout_compact(prefanout: dict[str, Any]) -> str:
    """Render `execute_search` result as a full JSON dump for the LLM.

    History: PR #307 introduced a compact rendering (top-10 per channel,
    150-char snippet truncation, fields stripped) because Fireworks
    gpt-oss-120b deterministically blew past the 90s loop timeout on
    cold-cache 15K-token inputs. Cerebras's higher input throughput
    removed that constraint — and the cap was masking non-GitHub hits
    from the LLM whenever bm25/vector's top 10 happened to be all
    GitHub. Same query, same corpus → "GitHub-only chunks" symptom.

    Full uncompress: dump every channel hit with every field. The agent
    now sees the complete pre-fan-out, can curate freely across all
    sources, and only the LLM-emitted final selection feeds downstream
    consumers. Function name kept (callsite stability) but the
    "compact" semantic is gone.
    """
    if not prefanout or not prefanout.get("sub_queries"):
        return "(no pre-fan-out hits)"
    return json.dumps(prefanout, default=str, indent=2)


# Total chain-line cap. Each line ~100-300 chars; capping ~30 keeps the
# section under ~10KB on the worst-case 5-sub_query x 10-hit fan-out.
# The full JSON dump in `_format_prefanout_compact` is the source of truth
# for raw hits; this section is a complementary structural view (grouped
# by anchor) — its job is making chain shape visible, not exhaustively
# enumerating hits.
_PREFANOUT_INFERRED_CHAINS_CAP = 30


def _truncate_snippet(text: str | None, n: int) -> str:
    """Whitespace-collapse + ellipsize for chain-section rendering.

    PR #328 removed the original compact-rendering `_truncate_snippet`
    along with the per-channel display cap (full JSON dump replaced
    compact rendering). The chains section deliberately keeps its
    truncation: chain lines optimize for at-a-glance chain shape, not
    full hit fidelity (the JSON dump has the full fidelity).
    """
    if not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= n:
        return flat
    return flat[: n - 1].rstrip() + "…"


def _format_inferred_chains(prefanout: dict[str, Any]) -> str:
    """Re-group `inferred_edge` channel hits by `anchor_doc_id`.

    The inferred-edge channel surfaces LLM-asserted cross-source links —
    PR → Linear issue, PR → Slack discussion, PR → Notion design doc.
    Each hit carries `anchor_doc_id` (the originating doc) + `why` (the
    rationale). The full pre-fan-out JSON dump already shows every hit
    (PR #328); this helper presents the SAME inferred-edge hits
    additionally regrouped by anchor so the chain shape (one source doc
    motivates / cites / references multiple downstream docs) is visible
    at-a-glance — the structural view the agent needs for "why was X
    created" / "what led to Y" queries that the flat JSON layout doesn't
    expose.

    Dedup: an `(anchor, doc_id, edge_type)` triple can appear under
    multiple sub_queries when fan-out anchors overlap; we keep only the
    first occurrence (highest-ranked sub-query position).

    Bound: total emitted hit lines are capped at
    `_PREFANOUT_INFERRED_CHAINS_CAP`. Excess hits drop with a single
    "showing top N of M" note so the agent knows more is reachable via
    `subgraph(anchor)` or `fetch_doc(... with_inferred_edges=true)`.

    Returns empty string when no inferred-edge hits exist anywhere in
    the pre-fan-out.
    """
    # Insertion-order dict of anchor → list of hits; dedup by (doc_id,
    # edge_type) within each anchor block.
    chains: dict[str, list[dict[str, Any]]] = {}
    seen_per_anchor: dict[str, set[tuple[str, str]]] = {}
    total_hits = 0
    for sq in prefanout.get("sub_queries") or []:
        for hit in sq.get("inferred_edge") or []:
            anchor = hit.get("anchor_doc_id")
            if not anchor:
                continue
            doc_id = hit.get("doc_id") or ""
            edge_type = hit.get("edge_type") or ""
            seen = seen_per_anchor.setdefault(anchor, set())
            key = (doc_id, edge_type)
            if key in seen:
                continue
            seen.add(key)
            chains.setdefault(anchor, []).append(hit)
            total_hits += 1
    if not chains:
        return ""
    out_lines: list[str] = []
    rendered = 0
    truncated = False
    for anchor, hits in chains.items():
        out_lines.append(f"anchor: {anchor}")
        for hit in hits:
            if rendered >= _PREFANOUT_INFERRED_CHAINS_CAP:
                truncated = True
                break
            linked = hit.get("doc_id") or "?"
            src = hit.get("source_system") or "?"
            edge = hit.get("edge_type") or "?"
            title = _truncate_snippet(hit.get("title"), 60)
            why = _truncate_snippet(hit.get("why"), 200)
            line = f"  -> {linked} [{src}] edge={edge}"
            if title:
                line += f' title="{title}"'
            out_lines.append(line)
            if why:
                out_lines.append(f"     why: {why}")
            rendered += 1
        if truncated:
            break
    if truncated:
        out_lines.append(
            f"  (showing top {rendered} of {total_hits} chain hits; "
            f"call subgraph(anchor) or fetch_doc(..., with_inferred_edges=true) "
            f"for the remainder)"
        )
    return "\n".join(out_lines)


def _build_user_message(
    query: str,
    bundle: GroundingBundle,
    prefanout: dict[str, Any] | None = None,
) -> str:
    """Render the per-query user message.

    Layout (grounding-first for cache stability, then channel results,
    then the chain-shaped re-grouping, then the raw query last for
    recency bias):
        <grounding>          — entity bag from grounding + extraction
        <connected_sources>  — which source systems the tenant has wired
        <channel_results>    — output of pre-fan-out `execute_search`
        <inferred_chains>    — same inferred-edge hits regrouped by
                               anchor doc, so the why-chain structure
                               (A → B → C with `why` per hop) is visible
                               at-a-glance. Only present when the pre-
                               fan-out surfaced any inferred-edge hits.
        <query>              — raw user query, last for recency
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
    chains_section = ""
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
        chains_block = _format_inferred_chains(prefanout)
        if chains_block:
            chains_section = (
                f"\n\n<inferred_chains>\n"
                f"The same inferred-edge hits from `<channel_results>` above, "
                f"additionally regrouped by their `anchor_doc_id` (the "
                f"originating doc of each LLM-asserted link). Each anchor "
                f"motivates / cites / references the listed downstream docs "
                f"with a `why` rationale. For 'why was X created' / 'what led "
                f"to Y' / 'what is the context behind Z' queries THIS is the "
                f"answer chain — emit each linked doc as a `GatheredChunk` "
                f"with the `why` quoted verbatim in `why_relevant`. If an "
                f"`anchor:` value matches a doc_id whose graph entity also "
                f"appears in `<grounding>`, ALSO emit that entity as a "
                f"`GatheredEntity` (use its `canonical_id` from `<grounding>`, "
                f"not the anchor doc_id).\n"
                f"{chains_block}\n"
                f"</inferred_chains>"
            )

    safe_query = _escape_query_for_xml(query)
    return (
        f"<grounding>\n{grounding_block}\n</grounding>\n\n"
        f"<connected_sources>{sources_block}</connected_sources>"
        f"{channel_results_block}"
        f"{chains_section}\n\n"
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


def _seed_for_query(customer_id: str, query: str) -> int:
    """Deterministic 31-bit non-negative seed derived from (customer_id, query).

    Sent on every turn of one query so the tool trajectory stays
    stable; same input → same seed across reruns. Cerebras's API
    documents `seed` as "best-effort deterministic sampling" — it
    only holds within a fixed `system_fingerprint`, so the trace
    blob captures both fields and the analyzer flags reruns where
    fingerprint flipped under us.

    Range: masked to signed-int31 (0 to 2^31-1) so the seed validates
    against any provider that treats `seed` as signed int32 (some
    gateway proxies do); also keeps it inside Python int → JSON int
    representable range without losing precision.
    """
    h = sha256()
    h.update(customer_id.encode("utf-8", errors="ignore"))
    h.update(b":")
    h.update(query.encode("utf-8", errors="ignore"))
    return int.from_bytes(h.digest()[:4], "big") & 0x7FFFFFFF


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
        # Deterministic seed. temperature=0 alone is NOT enough on
        # Cerebras — live-traced 2026-05-19: identical input, three
        # reruns of the same query produced 2 distinct outputs because
        # requests landed on different backend replicas (cache hit rate
        # dropped 99.96% → 8.67% on the divergent run). `seed` is
        # provider-honored best-effort; it stabilises sampling within a
        # given `system_fingerprint` and the trace blob captures the
        # fingerprint per turn so the analyzer can flag fingerprint
        # drift as a reproducibility breaker.
        "seed": state.seed,
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
    # Cardinality contract: every per-turn list has exactly turn_count
    # entries. Use None for "no signal this turn" so the analyzer can
    # join cache / fingerprint / reasoning / latency by turn index.
    rate = _extract_cache_hit_rate(resp)
    state.cache_hit_rates.append(rate)
    # Provider's backend fingerprint — Cerebras returns this on every
    # response; pairs with `seed` to make replica drift visible.
    fingerprint = getattr(resp, "system_fingerprint", None)
    state.system_fingerprints_per_turn.append(fingerprint if fingerprint else None)

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
        system_fingerprint=fingerprint,
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


def _derive_source_system_from_doc_id(doc_id: str | None) -> str:
    """Doc IDs are namespaced `<source>:<...>` (e.g. `slack:thread:T123`,
    `github:owner/repo:pr:42`, `linear:ticket:PRB-17`). Pull the prefix
    and map it to the SourceSystem enum. Returns empty string when the
    prefix doesn't match a known source — the adapter then keeps the
    field blank rather than mislabelling.
    """
    if not doc_id or ":" not in doc_id:
        return ""
    prefix = doc_id.split(":", 1)[0].strip().lower()
    try:
        return SourceSystem(prefix).value
    except ValueError:
        return ""


def _build_prefanout_doc_meta(prefanout: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Flatten `execute_search` result into `doc_id → {source_system, title,
    source_url, created_at, updated_at, author_id}`. Used by `_coerce_lenient`
    to fill doc-level fields on emitted chunks when the model omits them.

    First-write wins: vector channel typically lands first; later channels
    won't overwrite. (The doc-level fields are identical across channels
    for the same doc_id; ordering only matters when one channel happens
    to omit a field — keep the most complete record.)
    """
    out: dict[str, dict[str, Any]] = {}
    for sq in (prefanout or {}).get("sub_queries") or []:
        if not isinstance(sq, dict):
            continue
        for channel in ("vector", "bm25", "graph", "inferred_edge"):
            for hit in sq.get(channel) or []:
                if not isinstance(hit, dict):
                    continue
                doc_id = hit.get("doc_id")
                if not doc_id:
                    continue
                meta = out.setdefault(doc_id, {})
                for meta_field in ("source_system", "title", "source_url",
                                   "created_at", "updated_at", "author_id"):
                    if not meta.get(meta_field) and hit.get(meta_field):
                        meta[meta_field] = hit[meta_field]
    return out


def _build_prefanout_chunk_content(
    prefanout: dict[str, Any] | None,
) -> dict[tuple[str, str], str]:
    """Flatten `execute_search` result into `(chunk_id, doc_id) → content`.

    Used by `_coerce_lenient` to OVERWRITE the model-emitted content
    with the verbatim chunk body from the search hit. The agent's job
    is to pick which chunks are relevant; the chunk bodies themselves
    are not the agent's to rewrite. Pre-fix, the agent paraphrased
    chunks down to ~150-char summary blurbs, starving downstream
    synthesis of the factual content needed to answer.

    The key is the (chunk_id, doc_id) PAIR rather than chunk_id alone.
    Today `chunks.chunk_id` is content-addressed and unique per
    customer, so collisions don't occur — but keying on the pair is
    defensive against any future ingest path that mints
    non-content-addressed chunk_ids (e.g., id-lookup aliases). Without
    the pair check, a chunk_id collision would let the harness coerce
    the model into citing the right body under the wrong doc.

    First-write wins per channel ordering (vector / bm25 / graph).
    Inferred-edge channel hits do NOT carry chunk-level content
    (they're doc-level chain references) and are skipped — the
    agent's emission for inferred-edge chunks is whole-doc and may
    pick up content via fetch_doc instead.
    """
    out: dict[tuple[str, str], str] = {}
    for sq in (prefanout or {}).get("sub_queries") or []:
        if not isinstance(sq, dict):
            continue
        for channel in ("vector", "bm25", "graph"):
            for hit in sq.get(channel) or []:
                if not isinstance(hit, dict):
                    continue
                chunk_id = hit.get("chunk_id")
                doc_id = hit.get("doc_id")
                content = hit.get("content")
                if (
                    not chunk_id
                    or not doc_id
                    or not isinstance(content, str)
                    or not content.strip()
                ):
                    continue
                key = (chunk_id, doc_id)
                if key not in out:
                    out[key] = content
    return out


def _coerce_lenient(raw: dict[str, Any], state: LoopState | None = None) -> dict[str, Any]:
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
      - Fills doc-level pass-through fields (source_system, title,
        source_url, created_at, updated_at, author_id) from the
        pre-fan-out doc meta when the model omitted them. Falls back to
        deriving `source_system` from the doc_id prefix. Pre-extension,
        the adapter hard-coded `source_system="github"` for every
        result because there was no source field to read.

    Safe to call even on Fireworks-strict output — only fills fields
    that are missing.
    """
    out = dict(raw) if isinstance(raw, dict) else {}
    # Doc-meta lookup from the harness's pre-fan-out so we can fill
    # source_system/title/url on chunks the model emitted by doc_id only.
    prefanout_meta: dict[str, dict[str, Any]] = (
        _build_prefanout_doc_meta(state.prefanout) if state is not None else {}
    )
    # Chunk-content lookup — the harness is authoritative for chunk
    # bodies. When the agent emits a (chunk_id, doc_id) pair that's
    # in the prefanout, we OVERWRITE its `content` with the verbatim
    # chunk body from the search hit. This eliminates the failure mode
    # where the model paraphrased chunks down to summary blurbs that
    # starved downstream synthesis of the factual content needed to
    # answer. Chunks discovered via fetch_doc / subgraph aren't in
    # this map; their content stays as the model emitted it (the
    # prompt instructs the model to copy those verbatim from the tool
    # response, so trust the emission).
    prefanout_chunks: dict[tuple[str, str], str] = (
        _build_prefanout_chunk_content(state.prefanout) if state is not None else {}
    )
    # Entities: filter non-dict items + alias `id` → `canonical_id`.
    # Cerebras gpt-oss-120b occasionally emits malformed JSON fragments as
    # bare strings inside arrays (e.g. `entities=[{...valid...}, '{',
    # '{canonical_id":...']`) — a constrained-decoding partial-failure
    # mode. Pydantic rejects the whole emission on the first non-dict;
    # drop the malformed items so the valid neighbors survive. Also
    # accept `id` as an alias for `canonical_id` (common API convention
    # — Cerebras emits this for github/notion-shaped docs).
    entities_in = out.get("entities") or []
    entities_out: list[dict[str, Any]] = []
    for e in entities_in:
        if not isinstance(e, dict):
            continue
        e_out = dict(e)
        if not e_out.get("canonical_id") and e_out.get("id"):
            e_out["canonical_id"] = e_out["id"]
        entities_out.append(e_out)
    out["entities"] = entities_out
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
        # Content — HARNESS-AUTHORITATIVE. If the (chunk_id, doc_id)
        # pair is in prefanout, overwrite whatever the model wrote
        # with the verbatim chunk body. Eliminates the agent-
        # paraphrasing failure mode (live-traced 2026-05-20: agent
        # emitted ~150-char summary blurbs instead of full ~1800-char
        # chunk bodies, starving downstream synthesis). Falls back to
        # the model's content (or alias fields) for chunks the agent
        # discovered via fetch_doc/subgraph that aren't in prefanout —
        # the prompt instructs the model to copy those verbatim from
        # the tool response, so trust the emission.
        verbatim = prefanout_chunks.get((ch_out["chunk_id"], ch_out["doc_id"]))
        if verbatim:
            ch_out["content"] = verbatim
        else:
            if not ch_out.get("content"):
                for alias in _CHUNK_CONTENT_ALIASES:
                    v = ch_out.get(alias)
                    if isinstance(v, str) and v.strip():
                        ch_out["content"] = v
                        break
        if not ch_out.get("content"):
            continue  # No body to cite
        # Fill doc-level pass-through fields. The prefanout meta is the
        # canonical record from the DB (titles, URLs, timestamps the
        # ingestion pipeline persisted), so it WINS over the model's
        # emission whenever it's present — Cerebras's gpt-oss-120b is
        # known to fabricate / schema-drift on these fields. When the
        # doc isn't in prefanout (e.g., the agent followed an inferred
        # edge via fetch_doc to a new doc), the model's emission is the
        # only source — keep it. Final fallback for source_system is the
        # doc_id prefix.
        meta = prefanout_meta.get(ch_out["doc_id"], {})
        if meta.get("source_system"):
            ch_out["source_system"] = meta["source_system"]
        elif not ch_out.get("source_system"):
            ch_out["source_system"] = (
                _derive_source_system_from_doc_id(ch_out["doc_id"]) or ""
            )
        for meta_field in ("title", "source_url"):
            if meta.get(meta_field):
                ch_out[meta_field] = meta[meta_field]
        for meta_field in ("created_at", "updated_at", "author_id"):
            if meta.get(meta_field) is not None:
                ch_out[meta_field] = meta[meta_field]
        chunks_out.append(ch_out)
    if chunks_out or "chunks" in out:
        out["chunks"] = chunks_out
    # gatherer_notes: harness is authoritative for turns_used + tools_called
    notes = dict(out.get("gatherer_notes") or {})
    if state is not None:
        notes["turns_used"] = state.turn_count
        notes["tools_called"] = [*state.tools_fired, TERMINAL_TOOL_NAME]
    out["gatherer_notes"] = notes
    return out


def _repair_truncated_json(s: str) -> str | None:
    """Best-effort repair of unterminated JSON. Cerebras gpt-oss-120b
    occasionally cuts a long emit mid-string (observed live: single
    per-turn-ms hit 12.5s vs typical 1-3s, then emitted ~5400 chars of
    JSON ending mid-string). The valid prefix usually has many complete
    entities/chunks worth recovering.

    Walks the prefix tracking quote state + bracket stack, records the
    position after every closing `}` or `]` keyed by the depth AFTER
    closure. When the walk ends in a broken state (open brackets and/or
    unterminated string), finds the deepest clean boundary at or above
    the broken depth, truncates there, then appends just enough closers
    to balance. Returns repaired string if it parses, else None.
    """
    if not s:
        return None

    def walk(text: str) -> tuple[list[str], bool, dict[int, int]]:
        stack: list[str] = []
        in_string = False
        escape = False
        last_clean_at_depth: dict[int, int] = {}
        for i, c in enumerate(text):
            if escape:
                escape = False
                continue
            if in_string:
                if c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
                continue
            if c in "{[":
                stack.append(c)
            elif c in "}]" and stack and (
                (c == "}" and stack[-1] == "{") or (c == "]" and stack[-1] == "[")
            ):
                stack.pop()
                last_clean_at_depth[len(stack)] = i + 1
        return stack, in_string, last_clean_at_depth

    stack, in_string, last_clean = walk(s)
    if not stack and not in_string:
        return s  # already valid

    broken_depth = len(stack)
    cut_at: int | None = None
    for d in range(broken_depth, -1, -1):
        if d in last_clean:
            cut_at = last_clean[d]
            break
    if cut_at is None or cut_at == 0:
        return None

    prefix = s[:cut_at]
    stack, in_string, _ = walk(prefix)
    if in_string:
        return None

    closer = ""
    while stack:
        c = stack.pop()
        closer += "}" if c == "{" else "]"
    candidate = prefix + closer
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        return None


def _parse_terminal_args(
    raw_args: str | dict[str, Any] | None,
    state: LoopState | None = None,
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
            try:
                raw_dict = json.loads(raw_args)
            except json.JSONDecodeError:
                repaired = _repair_truncated_json(raw_args)
                if repaired is None:
                    raise
                raw_dict = json.loads(repaired)
                log.info(
                    "agent.terminal_args_json_repaired",
                    original_len=len(raw_args),
                    repaired_len=len(repaired),
                )
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
    # Seed the extractor too — without it, entity extraction variance
    # changes the prefanout anchors and the variance attribution in the
    # trace gets misassigned to the (now-seeded) gatherer loop.
    extraction_seed = _seed_for_query(customer_id, req.query)
    extracted = await extract_entities_with_llm(
        customer_id, req.query, bundle, seed=extraction_seed
    )
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
        seed=_seed_for_query(customer_id, req.query),
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

    # Pre-flight zero-recall fast-path. If every pre-fan-out channel
    # (vector + BM25 + graph + inferred edges, run in parallel) returned
    # zero hits, the LLM has nothing to curate from. Skip the loop,
    # return an empty GathererOutput. Saves ~67-90s per truly-hopeless
    # query (e.g. "blue yeti microphones" against a tech KB) — the loop
    # otherwise oscillates hitting `search` / `fetch_doc` with rephrases
    # of the same query until the wall-clock kills it.
    #
    # We do NOT also require entity_dicts == [] here. `entity_dicts`
    # combines grounded entities (bundle.candidates + bare_id_matches,
    # KB-anchored via alias resolution) and reconciled LLM-extracted
    # entities, and the whole bag was already passed to execute_search
    # above as `entity_ids` — so the graph + inferred_edge channels
    # have already exercised anchor-driven exploration. A 0-hit result
    # across all 4 channels means even the entity-anchored paths found
    # nothing; no in-loop tool call can recover.
    #
    # Trade-off: a degraded-prefanout window where bundle.bare_id_matches
    # has a real doc canonical_id but all 4 channels coincidentally
    # returned [] (e.g. transient embedder + AGE outage) used to recover
    # via in-loop fetch_doc. Acceptable: that's a multi-channel failure,
    # not a query-shape issue, and the previous behavior was a 90s
    # death loop for the much more common "query has no answer" case.
    prefanout_total = sum(prefanout_hit_counts.values())
    if prefanout_total == 0:
        log.info(
            "agent.zero_recall_short_circuit",
            customer_id=customer_id,
            trace_id=trace_id,
            entity_count=len(entity_dicts),
            prefanout_hit_counts=prefanout_hit_counts,
        )
        status = "zero_recall_short_circuit"
        gathered = _empty_passthrough("zero_recall_short_circuit")
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
        return await to_query_response(
            query=req.query,
            gathered=gathered,
            trace_id=trace_id,
            timing_ms=timing,
            prefanout=state.prefanout,
            customer_id=customer_id,
        )

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
        return await to_query_response(
            query=req.query,
            gathered=gathered,
            trace_id=trace_id,
            timing_ms=timing,
            prefanout=state.prefanout,
            customer_id=customer_id,
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
            round(sum(_r) / len(_r), 3)
            if (_r := [r for r in state.cache_hit_rates if r is not None])
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
        _rates = [r for r in state.cache_hit_rates if r is not None]
        request.state.cache_hit_rate = (
            sum(_rates) / len(_rates) if _rates else None
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

    return await to_query_response(
        query=req.query,
        gathered=gathered,
        trace_id=trace_id,
        timing_ms=timing,
        prefanout=state.prefanout,
        customer_id=customer_id,
    )


async def _drive_loop(state: LoopState) -> GathererOutput | None:
    """Multi-turn loop: model calls tool → execute → loop back, until
    the model calls `emit_gatherer_output` (terminal).

    `tool_choice="required"` guarantees the model picks SOME tool on
    every turn — no prose path.
    """
    while True:
        budget_exhausted = state.tool_calls_count >= state.budget
        # Soft turn cap: once the model has burned through SOFT_TURN_CAP
        # exploration turns without emitting, force the next turn to be the
        # terminal. Broad open-ended queries (e.g. "what features did we
        # implement for self-hosting?") oscillate fetch_doc / search and
        # never call emit_gatherer_output on their own; each retained
        # tool-result balloons the prefill, the final turn hits 30-40s,
        # and the loop wall-clock (90s) kills it with status=loop_timeout
        # + chunk_count=0. Tripping the existing forcing-nudge path on
        # turn count instead of waiting for the wall clock keeps the
        # output non-empty and bounds p95 turns. The first turn always
        # gets a free shot at the terminal (turn_count incremented inside
        # _run_turn); the cap only matters once the model has chosen
        # exploration at least SOFT_TURN_CAP times.
        turn_cap_reached = state.turn_count >= SEARCH_AGENT_SOFT_TURN_CAP
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

        if budget_exhausted or turn_cap_reached:
            # Tool-call budget gone OR soft turn cap tripped, and the model
            # still didn't terminate. Inject a forcing nudge and run one
            # more turn. tool_choice="required" still applies — the model
            # must pick a tool — but with the explicit "emit_gatherer_output
            # now" instruction it should pick the terminal. If it doesn't,
            # we return None.
            reason = "budget_exhausted" if budget_exhausted else "soft_turn_cap"
            log.info(
                "agent.force_terminate",
                customer_id=state.customer_id,
                trace_id=state.trace_id,
                tool_calls=state.tool_calls_count,
                turns=state.turn_count,
                reason=reason,
            )
            state.messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": _serialize_tool_calls(tool_calls),
            })
            state.messages.append({
                "role": "user",
                "content": (
                    "Stop exploring. Call `emit_gatherer_output` NOW with "
                    "the final GathererOutput based on the evidence you "
                    "already have in `<channel_results>` and prior tool "
                    "results. Do not call any other tool."
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
