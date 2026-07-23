"""Gatherer agent loop.

Entry point: `run_gatherer(req, customer_id, request)` -> RetrieveResponse.

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
5. Adapter converts GathererOutput → existing RetrieveResponse shape.

Plan: docs/specs/agentic-search.md.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any, get_args

from fastapi import HTTPException

if TYPE_CHECKING:
    from starlette.requests import Request

from engine.retrieval.agent.adapter import to_query_response
from engine.retrieval.agent.extractor import extract_entities_with_llm
from engine.retrieval.agent.models import (
    ConfidenceLabel,
    DroppedCandidate,
    GatheredChunk,
    GathererNotes,
    GathererOutput,
    GathererStatus,
    MatchedViaChannel,
    SearchOptions,
)
from engine.retrieval.agent.prompt import build_system_prompt
from engine.retrieval.agent.tools import (
    NEED_DEEPER_TOOL_NAME,
    TERMINAL_TOOL_NAME,
    dispatch_tool_call,
    execute_search,
    tool_definitions,
)
from engine.retrieval.grounding import GroundingBundle
from engine.retrieval.helpers import expand_to_author_id_set
from engine.retrieval.router import (
    Intent,
    RouterEntity,
    _build_bundle_with_token_fallback,
    _escape_query_for_xml,
    _reconcile_entities_with_bundle,
)
from engine.shared.constants import (
    DEFAULT_RECENCY_HALF_LIFE_DAYS,
    SEARCH_AGENT_EXTENSION_GRANT,
    SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS,
    SEARCH_AGENT_HARD_CAP,
    SEARCH_AGENT_INFERENCE_MODEL,
    SEARCH_AGENT_LOOP_TIMEOUT_SECONDS,
    SEARCH_AGENT_MAX_CONTEXT_TOKENS,
    SEARCH_AGENT_MAX_EXTENSIONS,
    SEARCH_AGENT_PREFANOUT_TOKEN_BUDGET,
    SEARCH_AGENT_SOFT_TURN_CAP,
    SEARCH_AGENT_TOOL_BUDGET,
    SEARCH_AGENT_TRACE_SAMPLE_RATE,
    SourceSystem,
)
from engine.shared.db import with_tenant
from engine.shared.llm import LLMError, acompletion, gateway_url
from engine.shared.llm_tools import is_context_overflow, is_transient_provider_error
from engine.shared.logging import get_logger
from engine.shared.models import QueryRequest, RetrieveResponse
from engine.shared.source_registry import half_life_days_for, score_multiplier_for

log = get_logger(__name__)

# Allowed enum members for the two Literal-typed fields the model
# routinely fabricates values for (`chunks[].matched_via`,
# `gatherer_notes.confidence`). Sourced from the schema via get_args so
# adding a new value in models.py automatically flows through.
_MATCHED_VIA_VALID = frozenset(get_args(MatchedViaChannel))
_CONFIDENCE_VALID = frozenset(get_args(ConfidenceLabel))


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
    # Failed LLM-call latencies (ms), kept separate from the successful-turn
    # arrays above. A failed provider attempt has no cache/fingerprint/
    # reasoning response with which to preserve the per-turn cardinality
    # contract, but its wall time still belongs in query/trace telemetry.
    failed_turn_latencies_ms: list[float] = field(default_factory=list)
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
    # Resolved search_options from the LLM extractor — sort directive +
    # downstream filter args the harness applied to the pre-fan-out. Kept
    # on state so the trace blob captures intent without re-running the
    # extractor at inspection time.
    search_options: SearchOptions = field(default_factory=SearchOptions)
    pre_fanout_author_ids: list[str] = field(default_factory=list)
    # Caller-provided hard scope from QueryRequest. When set, the loop
    # injects these into every in-loop `search` dispatch (the tool schema
    # deliberately does not expose them) so the agent can reformulate
    # queries but never widen the caller's scope. None = unscoped, which
    # keeps the dispatch arguments byte-identical to pre-filter behavior.
    request_source_keys: list[str] | None = None
    request_doc_types: list[str] | None = None

    # QueryRequest.discovery. Widens the graph channel's budget on every
    # in-loop `search` dispatch (not exposed on the tool schema — the agent
    # reformulates queries, it does not choose the retrieval posture).
    request_discovery: bool = False
    request_source_keys_include_keyless: bool = False
    request_per_source_top_k: int | None = None


# ============================================================
# Message + helpers
# ============================================================

# Lazy cl100k tokenizer, shared across calls. False = tiktoken unavailable
# (fall back to a chars/4 heuristic). cl100k is an ESTIMATE for gpt-oss
# (different tokenizer); every budget that uses it carries headroom.
_TOKEN_ENCODING: Any = None


def _count_tokens(text: str) -> int:
    """Approximate token count via cl100k_base (lazy-loaded).

    Budgeting only (prefanout render + running context gate), never exact
    accounting — gpt-oss's tokenizer differs, so callers leave headroom.
    Falls back to chars/4 if tiktoken can't load.
    """
    global _TOKEN_ENCODING
    if not text:
        return 0
    if _TOKEN_ENCODING is None:
        try:
            import tiktoken

            _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover - tiktoken is a hard dep in prod
            _TOKEN_ENCODING = False
    if _TOKEN_ENCODING is False:
        return (len(text) + 3) // 4
    return len(_TOKEN_ENCODING.encode(text))


def _render_prefanout_budgeted(prefanout: dict[str, Any]) -> str:
    """Render the pre-fan-out for the LLM, capped at a token budget.

    Previously an UNCAPPED `json.dumps` of every hit with full content. That
    dump rides in the message history on every turn and was the primary
    driver of the Cerebras 131K context overflow.

    History: PR #307 added a top-10-PER-CHANNEL cap for Fireworks timeouts;
    PR #328 reverted it because per-channel capping masked non-GitHub sources
    whenever bm25/vector's top slots were all GitHub ("GitHub-only chunks").
    This cap avoids that trap: it keeps the full nested {sub_queries:
    [{vector,bm25,graph,inferred_edge}]} shape, but drops the lowest-value
    CONTENT hits (vector + bm25 only) until they fit
    `SEARCH_AGENT_PREFANOUT_TOKEN_BUDGET`. "Value" is the doc's fused RRF
    rank across ALL (sub_query, channel) lists — a global Top-N, so no single
    source is masked by another filling a channel. graph + inferred_edge hits
    are left intact (small, and they carry the why-chain context). Output is
    deterministic (fused-score order) so the cached prompt prefix stays
    stable turn to turn.
    """
    if not prefanout or not prefanout.get("sub_queries"):
        return "(no pre-fan-out hits)"

    sub_queries = prefanout.get("sub_queries") or []

    # Charge each doc its content tokens times how many times it will actually
    # be RENDERED in the two channels we trim — a doc in both vector and bm25
    # renders twice — so the budget matches the real dumped size instead of
    # under-counting. Metadata-only hits (no content body) are ~free and always
    # kept. graph + inferred_edge hits are always kept whole (small, and they
    # carry the why-chains), so their content is a fixed baseline: this is why
    # selection is scoped to vector/bm25 only — ranking across all four
    # channels would let high-fused graph docs consume the budget and starve
    # the vector/bm25 content the filter then wipes.
    trim_occurrences: dict[str, int] = {}
    baseline_tokens = 0
    for sq in sub_queries:
        if not isinstance(sq, dict):
            continue
        for channel in ("vector", "bm25"):
            for hit in sq.get(channel) or []:
                if not isinstance(hit, dict):
                    continue
                content = hit.get("content")
                doc_id = hit.get("doc_id")
                if doc_id and isinstance(content, str) and content.strip():
                    trim_occurrences[doc_id] = trim_occurrences.get(doc_id, 0) + 1
        for channel in ("graph", "inferred_edge"):
            for hit in sq.get(channel) or []:
                if isinstance(hit, dict):
                    baseline_tokens += _count_tokens(hit.get("content") or "")

    total_trim_docs = len(trim_occurrences)
    # Rank content docs by fused RRF (deterministic) and keep them, cheapest-
    # rank-first, until the budget fills. graph/inferred content is pre-charged
    # as the baseline so the total rendered size — not just vector/bm25 —
    # respects the cap.
    kept_docs: set[str] = set()
    budget_used = baseline_tokens
    for entry in _fuse_prefanout_docs(prefanout):
        doc_id = entry["doc_id"]
        occ = trim_occurrences.get(doc_id, 0)
        if occ == 0:
            continue  # graph/inferred-only doc — already in the baseline
        cost = _count_tokens(entry["hit"].get("content") or "") * occ
        if kept_docs and budget_used + cost > SEARCH_AGENT_PREFANOUT_TOKEN_BUDGET:
            break
        kept_docs.add(doc_id)
        budget_used += cost

    if len(kept_docs) >= total_trim_docs:
        # Every content doc fits — original behaviour, no filtering overhead.
        return json.dumps(prefanout, default=str, indent=2)

    # Trim vector + bm25 to kept docs. Keep every metadata-only hit (no content
    # body -> ~free, and dropping it would narrow PR#328's "show every hit"
    # guarantee) and every graph/inferred hit (why-chains).
    def _keep(h: Any) -> bool:
        if not isinstance(h, dict):
            return True
        content = h.get("content")
        if not (isinstance(content, str) and content.strip()):
            return True  # metadata-only, ~free
        return h.get("doc_id") in kept_docs

    filtered_sqs: list[dict[str, Any]] = []
    for sq in sub_queries:
        if not isinstance(sq, dict):
            continue
        new_sq = dict(sq)
        for channel in ("vector", "bm25"):
            hits = sq.get(channel)
            if isinstance(hits, list):
                new_sq[channel] = [h for h in hits if _keep(h)]
        filtered_sqs.append(new_sq)
    filtered = dict(prefanout)
    filtered["sub_queries"] = filtered_sqs
    dropped = total_trim_docs - len(kept_docs)
    note = (
        f"\n(pre-fan-out trimmed to fit context: showing the top "
        f"{len(kept_docs)} of {total_trim_docs} content docs by fused relevance "
        f"across all sources; {dropped} lower-ranked docs omitted. Call `search` "
        f"with a reformulated query if you need something not shown.)"
    )
    return json.dumps(filtered, default=str, indent=2) + note


# Total chain-line cap. Each line ~100-300 chars; capping ~30 keeps the
# section under ~10KB on the worst-case 5-sub_query x 10-hit fan-out.
# The (budgeted) JSON dump in `_render_prefanout_budgeted` is the source of
# truth for raw hits; this section is a complementary structural view
# (grouped by anchor) — its job is making chain shape visible, not
# exhaustively enumerating hits.
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


async def _resolve_person_author_ids(
    customer_id: str,
    entity_dicts: list[dict[str, str]],
) -> list[str]:
    """Pick `person` canonical_ids out of the unified entity bag and expand
    each via `expand_to_author_id_set`, which unions:

      1. The full entity_aliases cluster (primary + every alias of every
         cluster the input ids belong to)
      2. The Lane E enrichment property values (`employee_id`, `login`,
         `email`) on every Person row in that cluster — these are the raw
         identifiers that landed in `documents.author_id` from
         non-canonical sources (claude_code better-auth uuid, github login,
         granola email, etc.)

    Returns `[]` (not None) when nothing applies — `execute_search`'s
    `author_ids` arg treats both as "no filter", so the empty case is the
    natural pass-through.

    Without (2), claude_code documents authored by Mahit
    (`author_id='08578d48-…'`) would never match a query that grounded on
    his Slack-rooted Person canonical_id, because the uuid lives only as
    a property on that Person row, not as a graph_node of its own.
    """
    person_ids = [
        e["canonical_id"]
        for e in entity_dicts
        if (e.get("entity_type") or "").lower() == "person" and e.get("canonical_id")
    ]
    if not person_ids:
        return []
    try:
        async with with_tenant(customer_id) as conn:
            return await expand_to_author_id_set(
                conn,
                customer_id,
                person_canonical_ids=person_ids,
            )
    except Exception as exc:
        log.warning(
            "agent.author_id_cluster_expand_failed",
            customer_id=customer_id,
            person_ids=person_ids,
            error=str(exc),
        )
        return person_ids  # Fall back to unexpanded — never NULL the intent.


def _build_user_message(
    query: str,
    bundle: GroundingBundle,
    prefanout: dict[str, Any] | None = None,
    *,
    options: SearchOptions | None = None,
    author_ids: list[str] | None = None,
    source_keys: list[str] | None = None,
    doc_types: list[str] | None = None,
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

    # Render `<search_options>` only when something deviates from defaults
    # (sort=relevance, no author filter). Suppressing the tag in the default
    # case keeps the cache prefix bit-identical to pre-PR so prompt cache
    # hit rates don't drop for vanilla queries.
    options_block = ""
    sort_nondefault = options is not None and options.sort != "relevance"
    has_author_filter = bool(author_ids)
    has_scope_filter = bool(source_keys) or bool(doc_types)
    if sort_nondefault or has_author_filter or has_scope_filter:
        # Render each option INDEPENDENTLY — only emit `sort=...` when the
        # sort directive deviates from the default; only emit `author_ids=...`
        # when the harness actually applied one. Without this guard, a
        # person-mentioning relevance-sorted query would render
        # `sort=relevance` into the prompt — a string that never appeared
        # pre-PR — and break prompt-cache prefix stability for every such
        # query (measurable cost regression at scale).
        parts: list[str] = []
        if sort_nondefault:
            parts.append(f"sort={options.sort}")  # type: ignore[union-attr]
        if has_author_filter:
            parts.append(f"author_ids={list(author_ids)}")
        # Caller-enforced scope (QueryRequest.source_keys / .doc_types).
        # Rendered so the model understands WHY channels look narrow and
        # doesn't burn tool calls hunting for out-of-scope sources. The
        # harness re-applies this scope to every in-loop `search`, so the
        # model cannot widen it -- say so explicitly.
        scope_note = ""
        if has_scope_filter:
            if source_keys:
                parts.append(f"source_keys={list(source_keys)}")
            if doc_types:
                parts.append(f"doc_types={list(doc_types)}")
            scope_note = (
                " The `source_keys` / `doc_types` scope is caller-enforced: "
                "every `search` you issue is automatically constrained to "
                "it, so do not retry searches hoping to reach other "
                "sources — curate from what the scope returns."
            )
        options_block = (
            f"\n\n<search_options>\n"
            f"The harness applied these options to the pre-fan-out below: "
            f"{' '.join(parts)}. When sort=recency, each channel's hits "
            f"are ordered by `updated_at DESC` after entity / token / "
            f"embedding narrowing. When `author_ids` is set, all channels "
            f"hard-filter `documents.author_id = ANY(...)`. Trust the "
            f"channel ordering — don't re-rank by your own intuition."
            f"{scope_note}\n"
            f"</search_options>"
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
            f"{_render_prefanout_budgeted(prefanout)}\n"
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
        f"{options_block}"
        f"{channel_results_block}"
        f"{chains_section}\n\n"
        f"<query>\n{safe_query}\n</query>"
    )


def _affinity_key(customer_id: str, query: str) -> str:
    """Per-customer Cerebras session-affinity hash so the static system
    prompt + tool-defs prefix cache-hits ACROSS queries (not just across
    turns of one query).

    Original design (Fireworks): hash `(customer_id, query)` so the two
    turns of a single query share a replica. That worked because the
    only cache reuse Fireworks gave you was within a single multi-turn
    query — and per-query affinity guaranteed it.

    Why this changed (Cerebras, 2026-05-20): yesterday's digest of 126
    single-turn queries showed turn-0 mean cache_hit_rate of 0.13. The
    static system prompt + tool-defs (~2.4K tokens) gets a cold KV-cache
    on turn 0 of every query because every new query hashed to a fresh
    replica. The 4 traces that warmed (chr >= 0.5) ran 2.3x faster
    (886ms vs 2029ms). 104/126 traces are single-turn, so per-query
    affinity buys nothing in those cases — it just guarantees we cold-
    start the prefix on every query.

    With customer-only affinity all of one customer's queries route to
    the same replica, so the system prompt + tool-defs prefix stays
    warm. Multi-turn cache hit is preserved because Cerebras's prefix
    cache is content-addressed (vLLM-style): turn 1 of a multi-turn
    query still hits the warm token-prefix that turn 0 wrote to the
    same replica's cache.

    Signature keeps `query: str` so existing call sites compile
    unchanged; the argument is intentionally unused
    (see PRB-12; PR digest 2026-05-19 cited request_ids
    fe737834-647c-43fa-9c8f-790dc271cee4,
    4c55749c-d984-413b-8b8a-ff0458b21346,
    a6edb1bc-37f6-404c-8107-856a7ec1f01a,
    6d62d15a-285c-47d0-832d-b7c78d1fc0dc,
    587edbbf-93f1-4679-ba0e-e41dd6fb0b4b).
    """
    h = sha256()
    h.update(customer_id.encode("utf-8", errors="ignore"))
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


def _empty_passthrough(
    reason: GathererStatus,
    state: LoopState | None = None,
) -> GathererOutput:
    """Synthesize an empty, low-confidence output for degraded paths.

    Recall-floor backfill may subsequently populate its chunks from citable
    pre-fan-out evidence. When the loop already ran, preserve its
    harness-authoritative turn/tool ledger in the response metadata.
    """
    return GathererOutput(
        entities=[],
        chunks=[],
        gatherer_notes=GathererNotes(
            turns_used=state.turn_count if state is not None else 0,
            tools_called=list(state.tools_fired) if state is not None else [],
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
    from engine.shared.config import get_settings
    from engine.shared.llm import gateway_url

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

def _message_tokens(m: dict[str, Any]) -> int:
    """Approximate token cost of one chat message (content + tool_calls)."""
    total = 0
    content = m.get("content")
    if isinstance(content, str):
        total += _count_tokens(content)
    elif isinstance(content, list):
        # system prompt shape: [{"type": "text", "text": ...}]
        total += sum(
            _count_tokens(p.get("text", "")) for p in content if isinstance(p, dict)
        )
    tool_calls = m.get("tool_calls")
    if tool_calls:
        total += _count_tokens(json.dumps(tool_calls, default=str))
    return total


# Stub that replaces an evicted tool result. Keeps the `tool` message (and
# its tool_call_id linkage) so the assistant<->tool pairing the OpenAI
# contract requires stays intact — only the bulky content is dropped.
_EVICTED_TOOL_CONTENT = json.dumps(
    {"truncated": True, "reason": "older tool result trimmed to fit context window"}
)


def _enforce_context_budget(state: LoopState) -> None:
    """Trim the oldest tool results in place until the history fits the window.

    Backstop for the context overflow: the prefanout render is budgeted and
    fetch_doc is paginated, but an agent can still page several long docs past
    `SEARCH_AGENT_MAX_CONTEXT_TOKENS`. Rather than DELETE messages (which would
    orphan an assistant tool_call from its tool response and 400), this shrinks
    the CONTENT of the oldest `tool`-role messages to a stub. System prompt,
    turn-1 evidence, and assistant tool_calls are never touched. If even
    stubbing every tool message can't get under budget, the next LLM call may
    still 400 — caught by the context_overflow handler and degraded to 200.
    """
    # One tokenizer pass over the history; reuse the per-message counts in the
    # eviction loop so tool bodies aren't encoded twice.
    per_msg = [_message_tokens(m) for m in state.messages]
    total = sum(per_msg)
    if total <= SEARCH_AGENT_MAX_CONTEXT_TOKENS:
        return
    stub_cost = _count_tokens(_EVICTED_TOOL_CONTENT)
    evicted = 0
    for idx, m in enumerate(state.messages):
        if total <= SEARCH_AGENT_MAX_CONTEXT_TOKENS:
            break
        if m.get("role") != "tool" or not isinstance(m.get("content"), str):
            continue
        cur = per_msg[idx]  # already counted above — a tool message is content-only
        if cur <= stub_cost:
            continue
        m["content"] = _EVICTED_TOOL_CONTENT
        total -= cur - stub_cost
        evicted += 1
    if evicted:
        log.info(
            "agent.context_budget_evicted",
            customer_id=state.customer_id,
            trace_id=state.trace_id,
            turn=state.turn_count,
            evicted_tool_results=evicted,
            approx_tokens_after=total,
        )


async def _run_turn(state: LoopState) -> Any:
    """Run one LLM turn with tool_choice='required'. Records latency +
    cache hit rate on state. Returns the raw response."""
    # Backstop: keep the accumulated history under the model's context window
    # before every call (prefanout budget + fetch pagination handle the common
    # case; this catches deep multi-doc paging).
    _enforce_context_budget(state)
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
        # LiteLLM's SDK treats this as a client transport deadline (the OpenAI
        # client sends x-stainless-read-timeout); it is not the proxy request
        # body's provider timeout. Per-deployment limits remain gateway-owned
        # and can therefore trigger Cerebras -> Fireworks.
        "timeout": SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS,
    }
    if gateway_url():
        # The managed proxy owns provider retries and failover. If it exhausts
        # both routes, the client must not replay this high-token turn through
        # the full chain. Direct self-hosted calls retain normal retries.
        call_kwargs["max_retries"] = 0

    t_turn = time.perf_counter()
    try:
        resp = await acompletion(**call_kwargs)
    except LLMError as exc:
        elapsed_ms = (time.perf_counter() - t_turn) * 1000
        state.failed_turn_latencies_ms.append(elapsed_ms)
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

    # Enforce the caller's request-level scope on every in-loop content
    # tool. `source_keys` / `doc_types` are not on the agent-facing tool
    # schema, so this never overwrites a model-provided value -- it
    # re-applies the QueryRequest scope the prefanout already ran under.
    # Covers the navigation bypass: without this, `subgraph` can surface
    # an out-of-scope Document node and `fetch_doc` / `fetch_chunk_window`
    # would happily haul its content into the agent's context. (The
    # adapter's scope gate re-verifies the final output regardless --
    # this keeps out-of-scope content from ever entering the loop.)
    if name in ("search", "fetch_doc", "fetch_chunk_window", "subgraph"):
        if state.request_source_keys:
            arguments["source_keys"] = state.request_source_keys
        if state.request_doc_types:
            arguments["doc_types"] = state.request_doc_types
    if name == "search" and state.request_discovery:
        arguments["discovery"] = True
    if name == "search":
        if state.request_source_keys_include_keyless:
            arguments["source_keys_include_keyless"] = True
        if state.request_per_source_top_k is not None:
            arguments["per_source_top_k"] = state.request_per_source_top_k

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


# Recall floor: append top pre-fan-out docs the gatherer dropped until the
# response carries at least this many DISTINCT docs. The gatherer runs
# single-turn (SOFT_TURN_CAP=1, tool_choice="required") and curates the
# wide pre-fan-out pool by hand; on breadth questions (multi-session /
# temporal / commonsense) it under-emits, dropping gold docs that ARE in
# the top-K pool. The adapter surfaces ONLY emitted chunks, so a dropped
# gold doc is unrecoverable. Graded latency is set by the NUMBER of
# sequential LLM turns (replay fixed-delay per call), so appending pool
# docs the model already had in front of it is free; recall is the graded
# metric and precision is not, so appending AFTER the gatherer's own picks
# is strictly safe.
_RECALL_FLOOR_DOCS = 10

# Reciprocal-rank-fusion constant. Standard value; rank is 0-based so the
# top hit of any list contributes 1/(_RRF_K + 1).
_RRF_K = 60

# ln(2), for the exponential recency half-life below.
_LN2 = math.log(2)


def _source_weight(hit: dict[str, Any], ref_now: datetime) -> float:
    """Per-source score multiplier + recency decay for one hit.

    Ported from the deleted fusion._apply_source_decay. These two signals
    were the ONLY consumers of source_registry's `score_multiplier` and
    `half_life_days`, so when the agentic cutover removed fusion.py from the
    pipeline they stopped being applied at all -- silently. That regression
    is why high-volume agent transcripts (claude_code / codex, both weighted
    0.5) stopped being demoted below authored artifacts like PR descriptions
    and Linear tickets.

    Multiplier BEFORE decay, deliberately: otherwise a brand-new transcript
    sits at age 0, contributes no decay, and bypasses its demotion entirely.

    A hit missing source_system or updated_at contributes weight 1.0 rather
    than being dropped -- an unattributed hit should rank neutrally, not
    vanish.
    """
    source_system = hit.get("source_system")
    if not isinstance(source_system, str) or not source_system:
        return 1.0

    weight = score_multiplier_for(source_system)

    updated_at = hit.get("updated_at")
    if isinstance(updated_at, str) and updated_at:
        try:
            parsed = datetime.fromisoformat(updated_at)
        except ValueError:
            return weight
    elif isinstance(updated_at, datetime):
        parsed = updated_at
    else:
        return weight

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    half_life = half_life_days_for(source_system, DEFAULT_RECENCY_HALF_LIFE_DAYS)
    age_days = (ref_now - parsed).total_seconds() / 86400.0
    if age_days >= 0 and half_life > 0:
        weight *= math.exp(-_LN2 * age_days / half_life)
    return weight


def _fuse_prefanout_docs(prefanout: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Reciprocal-rank-fuse every (sub_query, channel) ranked list in the
    pre-fan-out into a single doc-deduped ranking.

    A doc that surfaces high across multiple sub-queries / channels scores
    higher than one that appears once deep in a single list — exactly the
    breadth signal the single-turn gatherer can't weigh by hand. Returns
    `[{doc_id, hit, channel}]` ordered by fused score DESC; `hit` is the
    first (highest-ranked) channel hit seen for that doc, carrying the
    content + doc-level meta the backfill synthesizes a chunk from. Docs
    with no usable content are skipped (e.g. inferred-edge-only hits).

    Each doc's fused score is scaled by `_source_weight` (per-source
    multiplier + recency decay), restoring the two ranking signals that the
    agentic cutover dropped along with fusion.py.
    """
    scores: dict[str, float] = {}
    best_hit: dict[str, dict[str, Any]] = {}
    best_channel: dict[str, str] = {}
    ref_now = datetime.now(UTC)
    for sq in (prefanout or {}).get("sub_queries") or []:
        if not isinstance(sq, dict):
            continue
        for channel in ("vector", "bm25", "graph", "inferred_edge"):
            for rank, hit in enumerate(sq.get(channel) or []):
                if not isinstance(hit, dict):
                    continue
                doc_id = hit.get("doc_id")
                content = hit.get("content")
                if not doc_id or not (isinstance(content, str) and content.strip()):
                    continue
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
                if doc_id not in best_hit:
                    best_hit[doc_id] = hit
                    best_channel[doc_id] = channel
    for doc_id in scores:
        scores[doc_id] *= _source_weight(best_hit[doc_id], ref_now)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {"doc_id": d, "hit": best_hit[d], "channel": best_channel[d]}
        for d, _ in ranked
    ]


def _has_citable_prefanout_evidence(prefanout: dict[str, Any] | None) -> bool:
    """Whether pre-fan-out contains a document safe to return without the LLM.

    ``_fuse_prefanout_docs`` already enforces the citation boundary used by
    recall-floor backfill: a non-empty ``doc_id`` plus nonblank ``content``.
    Reusing it here keeps the provider-error gate and response construction in
    lockstep, including inferred-edge hits that carry no citable body.
    """
    return bool(_fuse_prefanout_docs(prefanout))


def _backfill_recall_floor(
    gathered: GathererOutput, prefanout: dict[str, Any] | None
) -> int:
    """Append top fused pre-fan-out docs the gatherer didn't emit until the
    response carries at least `_RECALL_FLOOR_DOCS` distinct docs.

    No-op when the pool is empty or the gatherer already cleared the floor.
    Mutates `gathered.chunks` in place; returns the number of docs appended.
    """
    emitted_docs = {c.doc_id for c in gathered.chunks if c.doc_id}
    needed = _RECALL_FLOOR_DOCS - len(emitted_docs)
    if needed <= 0:
        return 0
    appended = 0
    for entry in _fuse_prefanout_docs(prefanout):
        if appended >= needed:
            break
        doc_id = entry["doc_id"]
        if doc_id in emitted_docs:
            continue
        hit = entry["hit"]
        channel = entry["channel"]
        gathered.chunks.append(
            GatheredChunk(
                doc_id=doc_id,
                chunk_id=hit.get("chunk_id") or doc_id,
                content=hit.get("content") or "",
                matched_via=[channel] if channel in _MATCHED_VIA_VALID else [],
                why_relevant="",
                source_system=(
                    hit.get("source_system")
                    or _derive_source_system_from_doc_id(doc_id)
                    or ""
                ),
                title=hit.get("title") or "",
                source_url=hit.get("source_url") or "",
                created_at=hit.get("created_at"),
                updated_at=hit.get("updated_at"),
                author_id=hit.get("author_id"),
            )
        )
        emitted_docs.add(doc_id)
        appended += 1
    return appended


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
      - Filters `chunks[].matched_via` to schema-allowed channels — the
        model sometimes fabricates labels here, and a single bad value
        would otherwise tank the whole emission via Pydantic Literal
        validation. Unknowns logged via `agent.literal_clamped`.
      - Clamps `gatherer_notes.confidence` to schema-allowed values
        (high/medium/low), defaulting to "medium" when the model
        emits a fabricated label or explicit null. Same telemetry.
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
        # Content — try aliases when missing
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
        # Filter `matched_via` to the schema's allowed channel set. The
        # model (Cerebras gpt-oss-120b in particular) sometimes invents
        # labels here ("telepathy" etc.) and occasionally emits non-
        # string members in the list (dicts, nested lists — same
        # constrained-decoding partial-failure mode that produces the
        # bare-string entities[] items handled above). Restrict the
        # `in` check to strings so an unhashable member can't raise
        # TypeError. Unknown values are dropped; if nothing survives,
        # the field falls back to its empty-list default.
        mv = ch_out.get("matched_via")
        if isinstance(mv, list):
            kept = [v for v in mv if isinstance(v, str) and v in _MATCHED_VIA_VALID]
            dropped = [v for v in mv if not (isinstance(v, str) and v in _MATCHED_VIA_VALID)]
            if dropped:
                log.info(
                    "agent.literal_clamped",
                    customer_id=state.customer_id if state is not None else None,
                    trace_id=state.trace_id if state is not None else None,
                    field="chunks.matched_via",
                    dropped=[str(d)[:50] for d in dropped[:5]],
                )
            ch_out["matched_via"] = kept
        elif mv is not None:
            # Non-list value (rare): drop it, let the default kick in.
            ch_out.pop("matched_via", None)
        chunks_out.append(ch_out)
    if chunks_out or "chunks" in out:
        out["chunks"] = chunks_out
    # gatherer_notes: harness is authoritative for turns_used + tools_called
    notes = dict(out.get("gatherer_notes") or {})
    if state is not None:
        notes["turns_used"] = state.turn_count
        notes["tools_called"] = [*state.tools_fired, TERMINAL_TOOL_NAME]
    # Clamp `confidence` to the schema's allowed set. The model
    # occasionally emits fabricated labels here ("definitely_unknown_label",
    # "uncertain", etc.) or explicit JSON null; without clamping, the
    # Literal validation fails and we lose the entire emission. Bad
    # values fall back to "medium" (the schema default — neither
    # pessimistic nor optimistic). Absent key is left alone so Pydantic
    # uses the field default.
    if "confidence" in notes and notes["confidence"] not in _CONFIDENCE_VALID:
        log.info(
            "agent.literal_clamped",
            customer_id=state.customer_id if state is not None else None,
            trace_id=state.trace_id if state is not None else None,
            field="gatherer_notes.confidence",
            original=str(notes["confidence"])[:50],
        )
        notes["confidence"] = "medium"
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


def _finalize_agent_timing(
    timing: dict[str, float],
    state: LoopState,
    *,
    started_at: float,
) -> None:
    """Populate aggregate agent timing for both success and failure exits."""
    timing["agent_ms"] = (time.perf_counter() - started_at) * 1000
    timing["agent_failed_llm_ms"] = sum(state.failed_turn_latencies_ms)
    timing["agent_loop_ms"] = (
        sum(state.turn_latencies_ms) + timing["agent_failed_llm_ms"]
    )
    timing["agent_tools_ms"] = sum(state.tool_latencies_ms)


# ============================================================
# Top-level entry point
# ============================================================

async def run_gatherer(
    req: QueryRequest,
    customer_id: str,
    request: Request | None = None,
) -> RetrieveResponse:
    """Run the gatherer agent against `req.query` and return a RetrieveResponse.

    Raises HTTPException(503) on fatal LLM/provider failures. Transient
    provider-chain exhaustion degrades to citable pre-fan-out evidence when
    available; provider routing and failover remain owned by LiteLLM.
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
    extracted_entities = extracted.entities
    search_options = extracted.search_options

    # Reconcile LLM-proposed entities against the bundle (safety net).
    if extracted_entities:
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
                for e in extracted_entities
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

    # Resolve `documents.author_id` filter from `person` entities in the bag.
    # Expanded via the person's alias cluster so post-merge canonical_ids
    # still match pre-merge raw author_id rows (raw historical text, never
    # rewritten on merge). When no person entity is present, returns [] —
    # the retrievers treat that as "no filter."
    author_ids = await _resolve_person_author_ids(customer_id, entity_dicts)

    # Caller-provided hard scope from QueryRequest. `req.doc_types`
    # OVERRIDES the extractor's inferred doc_types (the documented
    # QueryRequest contract); `req.source_keys` has no extractor
    # counterpart. Both thread into the prefanout below and into every
    # in-loop `search` dispatch via LoopState. When neither is set the
    # values collapse to exactly what the extractor produced, keeping
    # unfiltered requests byte-identical to pre-scope behavior.
    request_source_keys = req.source_keys or None
    request_doc_types = req.doc_types or None
    request_discovery = bool(req.discovery)
    request_source_keys_include_keyless = bool(req.source_keys_include_keyless)
    request_per_source_top_k = req.per_source_top_k
    effective_doc_types = request_doc_types or search_options.doc_types or None

    log.info(
        "agent.entity_bag_assembled",
        customer_id=customer_id,
        trace_id=trace_id,
        grounded=len(bundle.candidates) + len(bundle.bare_id_matches),
        extracted=len(extracted_entities),
        final=len(entity_dicts),
        sort=search_options.sort,
        doc_types=effective_doc_types,
        source_keys=request_source_keys,
        author_id_count=len(author_ids),
        grounding_ms=round(timing["grounding_ms"], 1),
        extraction_ms=round(timing["extraction_ms"], 1),
    )

    # Step 2 — Pre-fan-out: single `execute_search` call covers all 4
    # channels (vector + bm25 + graph + inferred_edge) anchored on the
    # unified entity bag. Result is the LLM's turn-1 evidence. The
    # extractor's search_options thread into every channel so all four
    # honor the same sort + author-filter + doc-type discipline.
    # Fold the extractor's reformulations into the pre-fan-out as
    # ADDITIONAL sub-queries (raw query first, then deduped reformulations).
    # Each runs vector + bm25 in parallel on its own text while sharing the
    # same entity anchors, widening candidate recall for breadth-limited
    # axes (single-session vocabulary gap, multi-hop decomposition,
    # temporal/event focus). Latency-neutral on the LLM-turn budget — the
    # prefanout is one parallel non-LLM fan-out, not a model turn.
    prefanout_queries = [req.query]
    seen_q = {req.query.strip().lower()}
    for sq in extracted.sub_queries:
        key = sq.strip().lower()
        if key and key not in seen_q:
            seen_q.add(key)
            prefanout_queries.append(sq)

    t_prefanout = time.perf_counter()
    prefanout_result = await execute_search(
        customer_id=customer_id,
        queries=prefanout_queries,
        entity_ids=entity_dicts or None,
        author_ids=author_ids or None,
        sort_by=search_options.sort,
        doc_types=effective_doc_types,
        source_keys=request_source_keys,
        discovery=request_discovery,
        source_keys_include_keyless=request_source_keys_include_keyless,
        per_source_top_k=request_per_source_top_k,
    )
    timing["prefanout_ms"] = (time.perf_counter() - t_prefanout) * 1000

    # Capture per-channel hit counts for the trace + summary log. Sum
    # across ALL sub-queries (not just sub_queries[0]) so the downstream
    # zero-recall short-circuit sees hits surfaced by a reformulation even
    # when the raw query alone returned nothing.
    prefanout_hit_counts = {"vector": 0, "bm25": 0, "graph": 0, "inferred_edge": 0}
    for sub in prefanout_result.get("sub_queries") or []:
        for _ch in prefanout_hit_counts:
            prefanout_hit_counts[_ch] += len(sub.get(_ch) or [])
    log.info(
        "agent.prefanout_complete",
        customer_id=customer_id,
        trace_id=trace_id,
        elapsed_ms=round(timing["prefanout_ms"], 1),
        hits=prefanout_hit_counts,
    )

    # Step 3 — Build the user message and short-circuit if no LLM.
    user_msg = _build_user_message(
        req.query,
        bundle,
        prefanout_result,
        options=search_options,
        author_ids=author_ids,
        source_keys=request_source_keys,
        doc_types=request_doc_types,
    )
    system_prompt = build_system_prompt(datetime.now(UTC))

    state = LoopState(
        customer_id=customer_id,
        trace_id=trace_id,
        query=req.query,
        seed=_seed_for_query(customer_id, req.query),
        prefanout=prefanout_result,
        prefanout_hit_counts=prefanout_hit_counts,
        search_options=search_options,
        pre_fanout_author_ids=list(author_ids),
        request_source_keys=request_source_keys,
        request_doc_types=request_doc_types,
        request_discovery=request_discovery,
        request_source_keys_include_keyless=request_source_keys_include_keyless,
        request_per_source_top_k=request_per_source_top_k,
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
        gathered = _empty_passthrough("zero_recall_short_circuit", state)
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
            top_k_related=req.top_k_related,
            source_keys=request_source_keys,
            doc_types=request_doc_types,
        )

    if _no_llm_configured():
        log.info(
            "agent.no_llm_configured_short_circuit",
            customer_id=customer_id,
            trace_id=trace_id,
        )
        status = "no_llm_configured"
        gathered = _empty_passthrough("no_llm_configured", state)
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
            top_k_related=req.top_k_related,
            source_keys=request_source_keys,
            doc_types=request_doc_types,
        )

    gathered: GathererOutput | None = None
    try:
        gathered = await asyncio.wait_for(
            _drive_loop(state),
            timeout=SEARCH_AGENT_LOOP_TIMEOUT_SECONDS,
        )
        if gathered is None:
            status = "schema_violation"
            gathered = _empty_passthrough("schema_violation", state)
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
        gathered = _empty_passthrough("loop_timeout", state)
    except LLMError as exc:
        if is_context_overflow(exc):
            # Deterministic input-too-large (provider 400), NOT an outage —
            # retrying the identical request always fails. Degrade to a 200
            # passthrough: the recall-floor backfill below fills `gathered`
            # from state.prefanout, so the caller still gets relevant results
            # (minus the agent's curation) instead of a 503 that falsely reads
            # as "retryable". The prefanout budget + context gate make this
            # rare; this is the last-resort net.
            log.warning(
                "agent.context_overflow",
                customer_id=customer_id,
                trace_id=trace_id,
                turn=state.turn_count,
                tool_calls=state.tool_calls_count,
                error=str(exc),
            )
            status = "context_overflow"
            gathered = _empty_passthrough("context_overflow", state)
        elif is_transient_provider_error(exc) and _has_citable_prefanout_evidence(
            state.prefanout
        ):
            # The configured provider call/chain has already ended. Do not
            # replay this high-token turn in-process. In managed mode the
            # gateway owns Cerebras -> Fireworks; direct mode retains SDK
            # behavior. The recall-floor backfill below converts the citable
            # pre-fan-out pool into a low-confidence response without masking
            # fatal auth, configuration, validation, or unknown errors.
            log.warning(
                "agent.provider_error_prefanout_fallback",
                customer_id=customer_id,
                trace_id=trace_id,
                turn=state.turn_count,
                tool_calls=state.tool_calls_count,
                status_code=exc.status_code,
                provider=exc.provider,
                gateway_enabled=gateway_url() is not None,
                error=str(exc),
            )
            status = "provider_error_prefanout_fallback"
            gathered = _empty_passthrough(
                "provider_error_prefanout_fallback",
                state,
            )
        else:
            log.error(
                "agent.fatal_provider_error",
                customer_id=customer_id,
                trace_id=trace_id,
                error=str(exc),
            )
            if request is not None:
                request.state.full_failure = True
                request.state.gatherer_status = "fatal_provider_error"
                request.state.tool_calls_count = state.tool_calls_count
                request.state.need_deeper_extensions = state.extensions_used
                # No GathererOutput exists on a fatal path, so final-output
                # fields stay NULL instead of fabricating confidence/drop data.
                request.state.confidence = None
                request.state.dropped_count = None
                _rates = [r for r in state.cache_hit_rates if r is not None]
                request.state.cache_hit_rate = (
                    sum(_rates) / len(_rates) if _rates else None
                )
                request.state.intents_count = 1
                request.state.router_model = SEARCH_AGENT_INFERENCE_MODEL
                request.state.failure_recovered = False
            _finalize_agent_timing(timing, state, started_at=t_agent)
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

    # Recall-floor backfill. The single-turn gatherer curates the wide
    # pre-fan-out pool by hand and under-emits on breadth questions; append
    # the top fused pool docs it dropped (after its own picks) so the
    # graded recall isn't capped by hand-curation. Latency-neutral — no
    # added LLM turn. Also recovers recall on degraded paths (loop_timeout
    # / schema_violation) where `gathered` is empty but the pool has hits.
    backfilled = _backfill_recall_floor(gathered, state.prefanout)
    if backfilled:
        log.info(
            "agent.recall_floor_backfill",
            customer_id=customer_id,
            trace_id=trace_id,
            status=status,
            appended=backfilled,
            total_chunks=len(gathered.chunks),
        )

    _finalize_agent_timing(timing, state, started_at=t_agent)

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
        failed_llm_ms=round(timing["agent_failed_llm_ms"], 1),
        agent_tool_ms=round(timing["agent_tools_ms"], 1),
        per_turn_ms=[round(t, 1) for t in state.turn_latencies_ms],
        failed_turn_ms=[round(t, 1) for t in state.failed_turn_latencies_ms],
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
        top_k_related=req.top_k_related,
        source_keys=request_source_keys,
        doc_types=request_doc_types,
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
        terminal_tc = next(
            (
                tc for tc in tool_calls
                if getattr(getattr(tc, "function", None), "name", None)
                == TERMINAL_TOOL_NAME
            ),
            None,
        )
        if terminal_tc is not None:
            fn = getattr(terminal_tc, "function", None)
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
