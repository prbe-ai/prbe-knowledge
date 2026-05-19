"""LLM-based inferred-edge extractor.

Sends one structured call to the configured LLM and validates the output.
Every validation drop reason has a counter in ExtractionResult.dropped.

Model dispatch is by prefix on `INFERRED_EDGES_MODEL`:
  - "claude-*"  -> Anthropic, assistant-prefill JSON-array trick
  - "gemini-*"  -> Google, response_schema-constrained JSON output

Both providers route through `shared.llm.acompletion` (Phase-0b chunk C
LiteLLM migration). Managed-isolated tenants without provider keys
get the gateway URL via `LLM_GATEWAY_URL`; everyone else uses the
provider env vars (`ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`).

Validation pipeline per edge (model-agnostic):
  1. Both endpoints resolve to existing graph_nodes for bundle.customer_id.
  2. edge_type is in the extended EdgeType enum.
  3. confidence in {INFERRED, AMBIGUOUS}; EXTRACTED -> forced to AMBIGUOUS.
  4. why present and <= 200 chars.
  5. from != to (no self-edges).

Kill-switch: if dropped["unknown_endpoint"] / total > 0.5, fail the entire
bundle (probable bad LLM run -- do not pollute the graph).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

import asyncpg

from services.ingestion.inferred_edges.bundle import Bundle
from services.ingestion.inferred_edges.prompts.v1 import PROMPT_VERSION, SYSTEM_PROMPT
from shared.constants import (
    INFERRED_EDGES_MODEL,
    INFERRED_EDGES_MODEL_PRICES,
    EdgeType,
)
from shared.llm import LLMError, acompletion, gateway_url
from shared.llm_tools import usage_tokens
from shared.logging import get_logger

# Pre-migration this module imported `anthropic` and `google.genai` at
# load time. After Phase-0b chunk C the production path goes through
# `shared.llm.acompletion`, so neither SDK is required for the
# extractor to function. Rate-limit detection uses class-name matching
# (see `_is_rate_limit_error`), which doesn't need the SDK class
# itself.

log = get_logger(__name__)

# Maximum output tokens from the LLM for the edge-extraction call.
_MAX_OUTPUT_TOKENS = 4096

# Rate-limit backoff. The first backfill on probe-founders dropped 1383/3257
# bundles (42%) when 64 concurrent extractors blew through Haiku's per-minute
# rate limit. Without backoff, a single attempt fails -> bundle marked failed
# -> queue worker retries on next claim, but the rate-limit window persists
# longer than the inter-claim interval, so all 3 attempts eat the same wall.
# With exponential backoff inside a single attempt, the call rides out the
# transient rate-limit window before giving up.
_RATE_LIMIT_MAX_RETRIES = 4
_RATE_LIMIT_BACKOFF_BASE_SECONDS = 5.0
_RATE_LIMIT_BACKOFF_CAP_SECONDS = 60.0

# Valid confidence values the LLM may emit.
_VALID_CONFIDENCES = {"INFERRED", "AMBIGUOUS"}

# Kill-switch threshold: if more than 50% of all proposed edges have
# unknown endpoints, the whole bundle is failed.
#
# Threshold history: 0.5 was the original spec; in practice it killed
# the entire batch whenever the LLM's body had a mix of real cross-refs
# (PR/issue/file mentions that exist in the graph) and ad-hoc free-text
# references ("user-reported symptoms", "loop_timeout traces") that
# don't have a canonical_id. Live trace from 2026-05-18 on PRs #328 /
# #329 / #332 (auto-rationale bodies citing 3-6 entities each) showed
# ratios of 0.75-0.80 — every single batch was killed, valid edges lost.
# Raised to 0.9 so the kill-switch only fires on near-total
# hallucination (genuine "bad LLM run" signal); mixed runs now emit the
# valid edges and let `dropped["unknown_endpoint"]` carry the warning.
_UNKNOWN_ENDPOINT_FAIL_RATIO = 0.9


@dataclass(slots=True)
class InferredEdge:
    """One validated, upsert-ready edge from the LLM."""

    from_label: str
    from_canonical_id: str
    to_label: str
    to_canonical_id: str
    edge_type: str  # EdgeType.value
    confidence: str  # INFERRED | AMBIGUOUS
    why: str
    extractor_id: str
    extracted_at: datetime
    # Which LLM produced this edge. Stored on graph_edges.properties.model
    # for audit (which model wrote which edges) and for A/B comparison
    # without bumping extractor_id (the prompt+pipeline is unchanged; only
    # the model changed in the v1 -> Flash Lite cutover).
    model: str = ""


@dataclass
class ExtractionResult:
    """Validated output of one LLM extraction call."""

    edges: list[InferredEdge] = field(default_factory=list)
    # reason -> count for telemetry
    dropped: dict[str, int] = field(default_factory=dict)
    # USD cost estimate for the metric
    cost_usd: float = 0.0
    # Whether the kill-switch fired (too many unknown_endpoint drops)
    bundle_failed: bool = False
    bundle_fail_reason: str = ""


def _inc(dropped: dict[str, int], reason: str) -> None:
    dropped[reason] = dropped.get(reason, 0) + 1


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Per-1M USD pricing lookup. Unknown models cost $0 (telemetry-only;
    pipeline correctness doesn't depend on this)."""
    in_per_1m, out_per_1m = INFERRED_EDGES_MODEL_PRICES.get(model, (0.0, 0.0))
    return (
        input_tokens / 1_000_000 * in_per_1m
        + output_tokens / 1_000_000 * out_per_1m
    )


def _provider_for(model: str) -> str:
    """Map a model id to its SDK provider key by prefix.

    Centralised here so extract_edges and tests both use the same rule.
    Unknown prefixes raise -- there's no sensible default.
    """
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "google"
    raise ValueError(f"unsupported INFERRED_EDGES_MODEL prefix: {model!r}")


# ---- valid edge types set (extended with Lane B types) ---------------------

_VALID_EDGE_TYPES: set[str] = {e.value for e in EdgeType}


# ---- bundle serialisation --------------------------------------------------


def _bundle_to_user_message(bundle: Bundle) -> str:
    """Render bundle contents as a structured user message."""
    lines: list[str] = [
        f"# Bundle for anchor document: {bundle.anchor_doc_id}",
        f"# Customer: {bundle.customer_id}",
        f"# Total documents in bundle: {len(bundle.docs)}",
        "",
    ]
    for i, doc in enumerate(bundle.docs, 1):
        lines.append(f"## Document {i}: {doc.doc_id}")
        lines.append(f"   source_system: {doc.source_system}")
        if doc.title:
            lines.append(f"   title: {doc.title}")
        lines.append("")
        lines.append(doc.content)
        lines.append("")

    # Append the node manifest so the LLM knows which canonical_ids exist.
    lines.append("## Node manifest (use ONLY these canonical_ids):")
    # Dedupe by (doc_id -> label=Document, canonical_id=doc_id)
    for doc in bundle.docs:
        lines.append(f"  - label=Document  canonical_id={doc.doc_id}")

    return "\n".join(lines)


# ---- LLM call wrapper (rate-limit backoff) --------------------------------


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Recognise an Anthropic rate-limit error.

    We avoid an `isinstance` check against `anthropic.RateLimitError` so the
    caller doesn't have to import-and-handle if the anthropic package is
    missing (already guarded in the outer try). Class-name match is enough
    here -- the anthropic SDK reliably uses `RateLimitError` for 429s.
    """
    return type(exc).__name__ == "RateLimitError"


async def _backoff_sleep(attempt: int) -> None:
    """Exponential backoff with jitter: 5, 10, 20, 40 seconds (cap 60)."""
    backoff = min(
        _RATE_LIMIT_BACKOFF_BASE_SECONDS * (2**attempt),
        _RATE_LIMIT_BACKOFF_CAP_SECONDS,
    )
    backoff += random.uniform(0, 1.0)
    await asyncio.sleep(backoff)


@dataclass(slots=True)
class _LLMResponse:
    """Provider-agnostic response shape for the extractor wrapper."""

    raw_text: str
    input_tokens: int
    output_tokens: int


async def _call_anthropic_with_backoff(
    *,
    model: str,
    customer_id: str,
    anchor_doc_id: str,
    user_message: str,
) -> _LLMResponse:
    """Anthropic call (LiteLLM-routed) with exponential backoff on rate limit.

    Uses the assistant-prefill `[` trick to force JSON-array output —
    Claude reliably continues from `[` instead of emitting a preamble
    or markdown fence. The `[` is NOT included in the returned
    raw_text; the caller re-prepends it before json.loads.

    Phase-0b: routes through `shared.llm.acompletion`. The Anthropic
    assistant-prefill trick survives the migration because LiteLLM
    accepts a trailing `{"role": "assistant", "content": "..."}` and
    forwards it to Anthropic's `messages` API verbatim (Anthropic
    treats it as the start of the model's reply, which is exactly
    what the prefill trick relies on).
    """
    last_exc: Exception | None = None
    for attempt in range(_RATE_LIMIT_MAX_RETRIES):
        try:
            response = await acompletion(
                model=_anthropic_litellm_model(model),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": "["},
                ],
                max_tokens=_MAX_OUTPUT_TOKENS,
            )
            choices = getattr(response, "choices", None) or []
            text = ""
            if choices:
                message = getattr(choices[0], "message", None)
                text = getattr(message, "content", None) or ""
            tokens = usage_tokens(response)
            return _LLMResponse(
                raw_text=text,
                input_tokens=tokens["prompt_tokens"],
                output_tokens=tokens["completion_tokens"],
            )
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise
            last_exc = exc
            log.warning(
                "inferred_edges.extractor.rate_limited",
                customer=customer_id,
                anchor=anchor_doc_id,
                provider="anthropic",
                attempt=attempt + 1,
                max_attempts=_RATE_LIMIT_MAX_RETRIES,
            )
            await _backoff_sleep(attempt)

    assert last_exc is not None
    raise last_exc


# Edge-extraction JSON Schema for the structured-output call. Mirrors the
# prompt's edge object shape. Gemini constrains generation to fit this
# schema, so `edge_type` and `confidence` enums get enforced at
# generation time — the validator's per-edge enum check still runs as
# defense-in-depth (e.g. EXTRACTED -> AMBIGUOUS demotion still happens;
# we just won't see the LLM emit something outside the closed sets).
#
# Plain dict (not `google.genai.types.Schema`): LiteLLM accepts both,
# but the dict form is portable across LiteLLM versions and doesn't
# require the google-genai SDK at module load time.
_GEMINI_EDGE_SCHEMA: dict[str, object] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "from": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "canonical_id": {"type": "STRING"},
                },
                "required": ["label", "canonical_id"],
            },
            "to": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "canonical_id": {"type": "STRING"},
                },
                "required": ["label", "canonical_id"],
            },
            "edge_type": {
                "type": "STRING",
                "enum": [
                    "DISCUSSES", "DOCUMENTS", "RESOLVES",
                    "MENTIONS_ENTITY", "RELATES_TO", "REFERENCES",
                ],
            },
            "confidence": {
                "type": "STRING",
                "enum": ["INFERRED", "AMBIGUOUS"],
            },
            "why": {"type": "STRING"},
        },
        "required": ["from", "to", "edge_type", "confidence", "why"],
    },
}


def _is_gemini_rate_limit_error(exc: BaseException) -> bool:
    """Recognise a Gemini quota / 429 error.

    Post-Phase-0b the underlying call goes through LiteLLM, which
    raises `LLMError(status_code=429)` on quota / rate-limit errors.
    We also keep the legacy class-name match so test suites that
    still raise a fake google-genai `ClientError` (or any class
    named like one of these) continue to trigger the backoff path.
    """
    if isinstance(exc, LLMError) and exc.status_code == 429:
        return True
    name = type(exc).__name__
    msg = str(exc)
    return (
        name in ("ResourceExhausted", "TooManyRequests", "ClientError")
        and ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower())
    )


async def _call_gemini_with_backoff(
    *,
    model: str,
    customer_id: str,
    anchor_doc_id: str,
    user_message: str,
) -> _LLMResponse:
    """Gemini call (LiteLLM-routed) with structured-output + backoff on quota.

    Uses `response_schema` (forwarded as a Gemini-native kwarg via
    LiteLLM's provider passthrough) to constrain the model to emit a
    JSON array of edges conforming to the closed enums. The
    structured-output mode typically returns valid JSON; we still run
    the parser+validator downstream as defense-in-depth (handles the
    rare empty-array case and any edge-type drift).
    """
    last_exc: Exception | None = None
    for attempt in range(_RATE_LIMIT_MAX_RETRIES):
        try:
            resp = await acompletion(
                model=_gemini_litellm_model(model),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=_MAX_OUTPUT_TOKENS,
                # Provider-native passthrough — LiteLLM forwards
                # these to Gemini's GenerateContentConfig unchanged.
                response_schema=_GEMINI_EDGE_SCHEMA,
                response_mime_type="application/json",
            )
            choices = getattr(resp, "choices", None) or []
            text = ""
            if choices:
                message = getattr(choices[0], "message", None)
                text = getattr(message, "content", None) or ""
            tokens = usage_tokens(resp)
            return _LLMResponse(
                raw_text=text,
                input_tokens=tokens["prompt_tokens"],
                output_tokens=tokens["completion_tokens"],
            )
        except Exception as exc:
            if not _is_gemini_rate_limit_error(exc):
                raise
            last_exc = exc
            log.warning(
                "inferred_edges.extractor.rate_limited",
                customer=customer_id,
                anchor=anchor_doc_id,
                provider="google",
                attempt=attempt + 1,
                max_attempts=_RATE_LIMIT_MAX_RETRIES,
            )
            await _backoff_sleep(attempt)

    assert last_exc is not None
    raise last_exc


def _anthropic_litellm_model(model: str) -> str:
    """Return a LiteLLM-prefixed Anthropic model id. Idempotent."""
    if "/" in model:
        return model
    return f"anthropic/{model}"


def _gemini_litellm_model(model: str) -> str:
    """Return a LiteLLM-prefixed Gemini model id. Idempotent."""
    if "/" in model:
        return model
    return f"gemini/{model}"


async def _call_llm(
    *,
    model: str,
    customer_id: str,
    anchor_doc_id: str,
    user_message: str,
) -> _LLMResponse:
    """Provider-dispatched LLM call. Picks Anthropic or Gemini by prefix."""
    provider = _provider_for(model)
    if provider == "anthropic":
        return await _call_anthropic_with_backoff(
            model=model, customer_id=customer_id,
            anchor_doc_id=anchor_doc_id, user_message=user_message,
        )
    if provider == "google":
        return await _call_gemini_with_backoff(
            model=model, customer_id=customer_id,
            anchor_doc_id=anchor_doc_id, user_message=user_message,
        )
    raise AssertionError("unreachable")  # pragma: no cover


# ---- main extraction function ---------------------------------------------


async def extract_edges(
    bundle: Bundle,
    conn: asyncpg.Connection,
    *,
    model: str | None = None,
) -> ExtractionResult:
    """Call the LLM and return validated inferred edges.

    `conn` must be a tenant-scoped connection (with_tenant already called)
    for the endpoint existence checks in validation.

    `model` defaults to `INFERRED_EDGES_MODEL` from shared.constants. Tests
    can override per-call (e.g. force Haiku for a regression case). The
    provider SDK is picked by prefix -- "claude-*" -> anthropic,
    "gemini-*" -> google-genai.

    Returns an empty result if the relevant API key isn't set so the worker
    doesn't crash in credential-less environments.
    """
    result = ExtractionResult()
    model_id = model or INFERRED_EDGES_MODEL

    if not bundle.docs:
        log.debug("inferred_edges.extractor.empty_bundle", customer=bundle.customer_id)
        return result

    # Per-provider key check up front. Anthropic and Google have
    # different secret names; bail early so we don't pay the bundle
    # work cost only to crash on the LiteLLM call. Managed-isolated
    # tenants ride `LLM_GATEWAY_URL` instead of provider env vars —
    # in that case the gateway holds the credential and we proceed.
    provider = _provider_for(model_id)
    expected_env = (
        "ANTHROPIC_API_KEY" if provider == "anthropic" else "GOOGLE_API_KEY"
    )
    if not os.environ.get(expected_env) and not gateway_url():
        log.warning(
            "inferred_edges.extractor.no_api_key",
            customer=bundle.customer_id,
            anchor=bundle.anchor_doc_id,
            model=model_id,
            missing_env=expected_env,
        )
        return result

    # ---- LLM call ----------------------------------------------------------
    user_message = _bundle_to_user_message(bundle)
    try:
        response = await _call_llm(
            model=model_id,
            customer_id=bundle.customer_id,
            anchor_doc_id=bundle.anchor_doc_id,
            user_message=user_message,
        )
    except Exception as exc:
        log.error(
            "inferred_edges.extractor.llm_call_failed",
            customer=bundle.customer_id,
            anchor=bundle.anchor_doc_id,
            model=model_id,
            error=str(exc),
        )
        result.bundle_failed = True
        result.bundle_fail_reason = f"llm_call_failed: {type(exc).__name__}"
        return result

    result.cost_usd = _estimate_cost(
        model_id, response.input_tokens, response.output_tokens,
    )
    raw_text = response.raw_text

    # ---- Reconstruct + parse the JSON array --------------------------------
    # Two response shapes converge here:
    #   - Anthropic: assistant prefilled with `[`, raw_text is the body
    #     CONTINUATION (starts with edges or `]`). We re-prepend `[`.
    #   - Gemini: structured output, raw_text is a complete JSON array
    #     starting with `[` on its own.
    # We normalize: strip, optionally prepend `[`, optionally append `]`
    # for truncated-mid-element recovery. Then json.loads.
    stripped = raw_text.strip()
    # Strip markdown code fences if a model added them (Gemini occasionally
    # wraps with ```json despite response_mime_type).
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
    # Empty or close-bracket-first means "no edges" -- valid.
    if not stripped or stripped.startswith("]"):
        return result

    candidate = stripped if stripped.startswith("[") else "[" + stripped
    if not candidate.endswith("]"):
        candidate = candidate.rstrip(",") + "]"

    try:
        raw_edges = json.loads(candidate)
        if not isinstance(raw_edges, list):
            log.warning(
                "inferred_edges.extractor.non_list_response",
                customer=bundle.customer_id,
                anchor=bundle.anchor_doc_id,
            )
            result.bundle_failed = True
            result.bundle_fail_reason = "non_list_response"
            return result
    except json.JSONDecodeError as exc:
        log.warning(
            "inferred_edges.extractor.json_parse_failed",
            customer=bundle.customer_id,
            anchor=bundle.anchor_doc_id,
            error=str(exc),
            raw_text_preview=raw_text[:200],
        )
        result.bundle_failed = True
        result.bundle_fail_reason = f"json_parse_failed: {exc}"
        return result

    if not raw_edges:
        return result  # Empty array is valid

    # ---- Validation pipeline -----------------------------------------------
    total = len(raw_edges)
    now = datetime.now(UTC)

    # Pre-load existing graph nodes for this customer for endpoint validation.
    # Each entry maps (label, canonical_id) -> display_title (joined from
    # documents.title for Document nodes, falling back to
    # properties->>'name'). The title is consumed by `_topic_overlaps`
    # to drop edges where the LLM's `why` rationale shares zero topical
    # content words with either endpoint's title — catches hallucinations
    # that pick plausible canonical_ids but assert a topical connection
    # the endpoint doc doesn't actually carry.
    existing_nodes: dict[tuple[str, str], str] = await _load_existing_nodes(
        conn, bundle.customer_id
    )

    for raw in raw_edges:
        if not isinstance(raw, dict):
            _inc(result.dropped, "bad_format")
            continue

        from_node = raw.get("from") or {}
        to_node = raw.get("to") or {}
        from_label = str(from_node.get("label") or "")
        from_cid = str(from_node.get("canonical_id") or "")
        to_label = str(to_node.get("label") or "")
        to_cid = str(to_node.get("canonical_id") or "")
        edge_type = str(raw.get("edge_type") or "")
        confidence = str(raw.get("confidence") or "")
        why = str(raw.get("why") or "")

        # Rule 5: self-edge
        if from_cid and from_cid == to_cid:
            _inc(result.dropped, "self_edge")
            continue

        # Rule 2: edge type
        if edge_type not in _VALID_EDGE_TYPES:
            _inc(result.dropped, "unknown_type")
            continue

        # Rule 3: confidence
        if confidence == "EXTRACTED":
            # Force-demote to AMBIGUOUS (never trust an LLM claiming EXTRACTED)
            confidence = "AMBIGUOUS"
            _inc(result.dropped, "forced_confidence_demoted")
            # Note: we continue processing this edge after demotion
        elif confidence not in _VALID_CONFIDENCES:
            _inc(result.dropped, "unknown_confidence")
            continue

        # Rule 4: why
        if not why or len(why) > 200:
            _inc(result.dropped, "bad_justification")
            continue

        # Rule 1: endpoint existence
        from_title = existing_nodes.get((from_label, from_cid))
        to_title = existing_nodes.get((to_label, to_cid))
        from_exists = from_title is not None
        to_exists = to_title is not None
        if not from_exists or not to_exists:
            _inc(result.dropped, "unknown_endpoint")
            log.debug(
                "inferred_edges.extractor.unknown_endpoint",
                customer=bundle.customer_id,
                from_label=from_label,
                from_cid=from_cid,
                to_label=to_label,
                to_cid=to_cid,
                from_exists=from_exists,
                to_exists=to_exists,
            )
            continue

        # Rule 6: topic-relevance sanity check. The kill-switch
        # threshold raise (0.5 → 0.9) let mixed-quality runs through,
        # which surfaced a hallucination class where the LLM picks a
        # valid-looking canonical_id but the `why` describes topics
        # the endpoint doc doesn't actually cover. Catch the egregious
        # cases per-edge: require at least one informative endpoint
        # title to share a content word with the rationale. Edges
        # where both endpoints have empty titles (stub-upserted Entity
        # nodes) skip this check and fall back to the kill-switch.
        if not _topic_overlaps(why, [from_title or "", to_title or ""]):
            _inc(result.dropped, "unrelated_topic")
            log.debug(
                "inferred_edges.extractor.unrelated_topic",
                customer=bundle.customer_id,
                from_cid=from_cid,
                to_cid=to_cid,
                from_title=from_title,
                to_title=to_title,
                why_preview=why[:120],
            )
            continue

        result.edges.append(
            InferredEdge(
                from_label=from_label,
                from_canonical_id=from_cid,
                to_label=to_label,
                to_canonical_id=to_cid,
                edge_type=edge_type,
                confidence=confidence,
                why=why,
                extractor_id=PROMPT_VERSION,
                extracted_at=now,
                model=model_id,
            )
        )

    # ---- Kill-switch: >50% unknown_endpoint -> fail bundle -----------------
    # `total` is the count of ALL proposed edges (including non-unknown_endpoint
    # drops like self_edge / bad_justification). This dilutes the ratio
    # intentionally — the spec is "fraction of total proposals that hallucinate
    # endpoints," not "fraction of validation failures." Don't change without
    # updating the spec.
    unknown_count = result.dropped.get("unknown_endpoint", 0)
    if total > 0 and unknown_count / total > _UNKNOWN_ENDPOINT_FAIL_RATIO:
        log.warning(
            "inferred_edges.extractor.bundle_killed_unknown_endpoints",
            customer=bundle.customer_id,
            anchor=bundle.anchor_doc_id,
            unknown_count=unknown_count,
            total=total,
        )
        result.edges = []
        result.bundle_failed = True
        result.bundle_fail_reason = (
            f"unknown_endpoint_ratio={unknown_count}/{total}"
        )

    return result


async def _load_existing_nodes(
    conn: asyncpg.Connection,
    customer_id: str,
) -> dict[tuple[str, str], str]:
    """Load `(label, canonical_id) -> display_title` for every graph node.

    `display_title` is, in priority order:
      1. `documents.title` for Document-class nodes (joined on
         `canonical_id = documents.doc_id`)
      2. `graph_nodes.properties->>'name'` (the human-readable display
         name set by ingestion when the node first lands)
      3. empty string when neither is set (e.g. stub-upserted Entity
         nodes that were never enriched)

    The conn must already be scoped via with_tenant(customer_id) so RLS
    on graph_nodes / documents filters by tenant; the explicit WHERE on
    customer_id is defence-in-depth.

    Used downstream by the topic-relevance sanity check in
    `_topic_overlaps` — without the title we have no signal that the
    LLM's `why` actually describes the endpoint doc, and the threshold
    can't catch hallucinations where the LLM picked a plausible-looking
    canonical_id whose actual content has nothing to do with the `why`.
    """
    rows = await conn.fetch(
        """
        SELECT gn.label,
               gn.canonical_id,
               COALESCE(d.title, gn.properties->>'name', '') AS display_title
        FROM graph_nodes gn
        LEFT JOIN documents d
               ON d.customer_id = gn.customer_id
              AND d.doc_id = gn.canonical_id
        WHERE gn.customer_id = $1
        """,
        customer_id,
    )
    return {
        (r["label"], r["canonical_id"]): r["display_title"] or ""
        for r in rows
    }


# Content-word tokens shared by `why` and one of the endpoint titles
# need to clear this floor for the edge to be kept. 1 token is permissive
# — most hallucinations we've observed lose ALL topical overlap (e.g.
# LLM proposes "PR #X implements graph traversals" but X's actual title
# is "fix(DSN routing)" — zero topic overlap → reject).
#
# Live-trace context (2026-05-18): the kill-switch threshold was raised
# 0.5 → 0.9 so mixed-quality runs land. Without this per-edge check, a
# hallucinated-but-plausible edge (right canonical_id form, wrong topic)
# slips into graph_edges and shows up in the dashboard's chain-of-
# reasoning panel pointing at an unrelated doc.
_MIN_TITLE_OVERLAP_TOKENS = 1

# Tokens to strip from `why` before comparing — these are the identifier
# parts of the LLM's own endpoint citation (e.g. "PR #327" or "pr:327")
# and they create trivial false-positive overlap when the canonical_id
# digits appear in both sides. The overlap check should be on the
# topical content, not the cite itself.
_WHY_TOPIC_STRIPPED_PREFIXES: Final[frozenset[str]] = frozenset({
    "pr", "issue", "ticket", "doc", "id", "edge", "the", "session",
    "this", "that", "those", "these", "implements", "implement",
    "implemented", "implementing", "implementation", "implementations",
    "discusses", "discussed", "discussing", "discussion", "discussions",
    "references", "referenced", "referencing", "reference",
    "describes", "described", "describing", "description",
    "with", "from", "about", "regarding",
})


def _content_words(text: str) -> set[str]:
    """Lower-case, len>=3, alphabetic-only content words.

    Mirrors `grounding._extract_tokens` semantics for stop-word filtering
    but is stricter — drops short tokens and pure-numeric tokens (which
    is the canonical_id leakage path we explicitly want to suppress).
    Kept local to this module so it stays calibrated for the topic-match
    check; tuning it for retrieval-side query parsing is a separate
    concern.
    """
    if not text:
        return set()
    out: set[str] = set()
    for raw in re.split(r"[^A-Za-z0-9]+", text.lower()):
        if len(raw) < 3:
            continue
        if not any(c.isalpha() for c in raw):
            continue  # pure-numeric: skip
        if raw in _WHY_TOPIC_STRIPPED_PREFIXES:
            continue
        out.add(raw)
    return out


def _topic_overlaps(why: str, endpoint_titles: list[str]) -> bool:
    """True when `why` shares >= _MIN_TITLE_OVERLAP_TOKENS content
    words with at least one endpoint's title.

    Topic-only check: identifier slugs ("pr", "session", "327", etc.)
    are stripped from both sides before comparing. Edges with at least
    one informative endpoint title still need that title to share
    content words with the rationale — otherwise the LLM is asserting
    a connection between docs whose topics don't overlap.

    Returns True (PASS) when every endpoint has an EMPTY title — we
    can't validate topic match without any signal, so we defer to the
    other validators (existence, edge_type, confidence, kill-switch).
    """
    why_tokens = _content_words(why)
    if not why_tokens:
        return True  # rationale itself was just identifiers — nothing to check
    informative = [t for t in endpoint_titles if t]
    if not informative:
        return True  # both endpoints are stub-upserted — defer
    for title in informative:
        title_tokens = _content_words(title)
        if len(why_tokens & title_tokens) >= _MIN_TITLE_OVERLAP_TOKENS:
            return True
    return False
