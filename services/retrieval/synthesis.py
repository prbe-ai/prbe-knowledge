"""LLM synthesis over retrieval chunks for the /query endpoint.

Three providers (Anthropic, OpenAI, Google) behind one async function:

    result = await synthesize(query, chunks, model, max_tokens)

Each provider uses its native structured-output mechanism so the wrapper
shape is enforced at the API level, not by asking the model nicely:

    Anthropic — forced tool call with a `render_answer` function
    OpenAI    — response_format = json_schema strict mode
    Google    — config.response_schema

Phase-0b: the non-streaming paths route through `shared.llm.acompletion`
so managed-isolated tenants without provider keys can use them via the
central LiteLLM gateway. The streaming path (`synthesize_stream`) still
uses the provider SDKs directly — chunk D handles that migration.

This avoids the "did the model wrap in JSON or return prose?" guessing
game that the old prompt-only approach hit on every provider.

Citation format inside the answer string is still free-form. Models
sometimes drop the [bracket] formatting; normalize_citations_in_answer()
canonicalizes bare `chunk:N` to `[chunk:N]` so downstream rendering
works regardless.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from shared import llm as shared_llm
from shared.config import get_settings
from shared.constants import SYNTHESIS_MODELS
from shared.exceptions import PrbeError
from shared.llm import LLMError
from shared.llm_tools import ToolCallParseError, forced_tool_call
from shared.logging import get_logger
from shared.models import QueryDocumentResult, QueryResult

log = get_logger(__name__)

SYNTHESIS_TIMEOUT_SECONDS = 30.0


class SynthesisError(PrbeError):
    """Raised when an LLM call fails or returns unparseable output."""


@dataclass(slots=True)
class SynthesisChunk:
    """Minimal chunk shape the synthesizer needs. Independent of QueryChunk
    so the synthesizer is reusable outside the /query handler.
    """

    chunk_id: str
    title: str | None
    content: str
    source_system: str
    source_url: str
    updated_at: str  # ISO8601


@dataclass(slots=True)
class SynthesisResult:
    answer: str  # prose with [chunk:N] citations inline
    citations: list[dict[str, object]]  # [{"index": 1, "chunk_id": "..."}]
    insufficient_context: bool
    model: str
    raw_provider_response: str  # debugging / observability only


# ---------------------------------------------------------------------------
# Schema (one definition, three providers)
# ---------------------------------------------------------------------------


# additionalProperties: False is required by OpenAI's strict json_schema mode
# and harmless on Anthropic + Google. All three properties live in `required`
# for the same reason.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {
            "type": "string",
            "description": (
                "Cited prose answer. 1-3 short paragraphs. Every factual claim "
                "ends with one or more inline citations. Each citation is its "
                "own bracketed tag: write [chunk:1][chunk:5] or [chunk:1] and "
                "[chunk:5], NOT [chunk:1, 5]. N is the 1-indexed chunk number. "
                "Use ONLY information present in the chunks. Do not invent "
                "dates, names, PR numbers, decisions, or relationships. Do not "
                "restate the question. No preamble. Use markdown for emphasis "
                "when helpful."
            ),
        },
        "citations_used": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "description": (
                "Every chunk index referenced in the answer text. Used as a "
                "self-check by the renderer."
            ),
        },
        "insufficient_context": {
            "type": "boolean",
            "description": (
                "True iff the chunks don't contain enough information to "
                "confidently answer. When true, `answer` should be a one-line "
                "explanation of what's missing rather than a fabricated answer."
            ),
        },
    },
    "required": ["answer", "citations_used", "insufficient_context"],
}


# The retrieval router resolves relative dates in the user's query into a
# temporal filter, but the synthesizer still needs to know `today` so it can
# write phrases like "in the past 7 days" or "this month" coherently in the
# answer prose.
_DATE_CONTEXT = """The user's current date (UTC) is: {today_iso}
Use this when the question or chunks involve relative time ("last week",
"in the past 7 days", "this month"). Resolve such phrases against today
rather than refusing for lack of a date."""


def _build_system_prompt(now: datetime) -> str:
    today_iso = now.strftime("%Y-%m-%d")
    return f"""You are a careful retrieval-augmented assistant. You answer the user's
question using ONLY the chunks you've been given. The runtime enforces a
structured output schema; just produce the values it asks for.

{_DATE_CONTEXT.format(today_iso=today_iso)}

Hard rules:
- Use ONLY information present in the chunks. Do not invent facts.
- Every sentence that makes a claim must end with at least one [chunk:N].
- If the chunks don't support a confident answer, set insufficient_context
  to true and write a one-line explanation in `answer` instead of guessing.
- Be concise. 1-3 short paragraphs. No preamble.
- Markdown formatting (bold, italic, code) is fine when it helps clarity.
"""


# Streaming variant: no forced tool call (the Anthropic streaming API for
# tool-input-delta is fragile to parse incrementally). Plain text out, with
# `<<INSUFFICIENT>>` as a sentinel the caller strips after the stream ends.
def _build_streaming_system_prompt(now: datetime) -> str:
    today_iso = now.strftime("%Y-%m-%d")
    return f"""You are a careful retrieval-augmented assistant. Answer the user's
question using ONLY the chunks you've been given.

{_DATE_CONTEXT.format(today_iso=today_iso)}

Hard rules:
- Use ONLY information present in the chunks. Do not invent facts.
- Every sentence that makes a claim must end with at least one [chunk:N].
- If the chunks don't support a confident answer, START your reply with the
  literal token <<INSUFFICIENT>> on its own line, then a one-line
  explanation of what's missing. Do not fabricate.
- Be concise. 1-3 short paragraphs. No preamble.
- Markdown formatting (bold, italic, code) is fine when it helps clarity.
"""


_INSUFFICIENT_SENTINEL = "<<INSUFFICIENT>>"


# ---------------------------------------------------------------------------
# Polymorphic-results adapter
# ---------------------------------------------------------------------------


def flatten_documents_for_synthesis(
    results: list[QueryResult],
) -> list[SynthesisChunk]:
    """Flatten the polymorphic `QueryResponse.results` into a flat chunk
    list the synthesizer can cite.

    Skips Entity results -- entities have no body content to cite. Each
    Document's chunks expand into one SynthesisChunk per chunk in
    `chunk_index`-style order (already sorted by score desc within doc by
    the search pipeline). Citation indices into this flattened list use
    `[chunk:N]` referring to the 1-indexed position; that's what the
    synthesis prompt asks for and what `_extract_citations` interprets.
    """
    out: list[SynthesisChunk] = []
    for r in results:
        if not isinstance(r, QueryDocumentResult):
            continue  # Entity results carry no content -- skip
        for c in r.chunks:
            out.append(
                SynthesisChunk(
                    chunk_id=c.chunk_id,
                    title=r.title,
                    content=c.content,
                    source_system=r.source_system.value,
                    source_url=r.source_url,
                    updated_at=r.updated_at.isoformat(),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def synthesize(
    query: str,
    chunks: list[SynthesisChunk],
    model: str,
    max_tokens: int = 600,
) -> SynthesisResult:
    """Run an LLM over `chunks` to synthesize an answer to `query`.

    Empty `chunks` short-circuits to insufficient_context — no model call.
    Provider failures raise SynthesisError; the route handler converts to 502.
    Model output that doesn't match the schema falls back to plain-text
    handling so the user always gets *something* renderable.
    """
    if not chunks:
        return SynthesisResult(
            answer="No chunks were retrieved for this query, so there's nothing to summarize.",
            citations=[],
            insufficient_context=True,
            model=model,
            raw_provider_response="",
        )

    if model not in SYNTHESIS_MODELS:
        raise SynthesisError(
            f"unsupported synthesis model: {model}. Allowed: {sorted(SYNTHESIS_MODELS)}"
        )
    provider_name = SYNTHESIS_MODELS[model]
    model_id = model.split("/", 1)[1]

    user_prompt = _format_user_prompt(query, chunks)
    parsed = await _dispatch(
        provider_name,
        system=_build_system_prompt(datetime.now(UTC)),
        user=user_prompt,
        model=model_id,
        max_tokens=max_tokens,
    )

    answer = normalize_citations_in_answer(str(parsed.get("answer", "")))
    declared = parsed.get("citations_used") or parsed.get("citations") or None
    declared_list = declared if isinstance(declared, list) else None
    citations = _extract_citations(answer, chunks, declared=declared_list)
    return SynthesisResult(
        answer=answer,
        citations=citations,
        insufficient_context=bool(parsed.get("insufficient_context", False)),
        model=model,
        raw_provider_response=json.dumps(parsed, default=str)[:4000],
    )


# ---------------------------------------------------------------------------
# Streaming synthesis (Anthropic only) — used by /query/stream
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StreamDelta:
    """A piece of generated text from the model."""

    text: str


@dataclass(slots=True)
class StreamFinal:
    """End-of-stream payload: parsed final answer + extracted citations.

    Emitted exactly once after all StreamDeltas. The streaming endpoint
    converts this into the SSE `done` event so the UI can swap its
    in-flight buffer for the canonical answer + citation list.
    """

    answer: str
    citations: list[dict[str, object]]
    insufficient_context: bool
    model: str


async def synthesize_stream(
    query: str,
    chunks: list[SynthesisChunk],
    model: str,
    max_tokens: int = 600,
) -> AsyncIterator[StreamDelta | StreamFinal]:
    """Streaming variant of `synthesize`.

    Yields StreamDelta(...) for each token chunk, then exactly one
    StreamFinal(...) with the parsed answer + citations.

    Routes through `shared.llm.acompletion(stream=True)` so managed-
    isolated and self-host tenants (no local provider keys; only an
    `LLM_GATEWAY_URL` pointing at the central LiteLLM proxy) can use it
    transparently. LiteLLM normalizes Anthropic + Google streaming into
    a single OpenAI-shaped chunk iterator.

    Anthropic + Google supported today; OpenAI streaming could be added
    by extending SYNTHESIS_MODELS without further code change. The
    non-streaming `synthesize` path covers all three providers.
    """
    if not chunks:
        yield StreamFinal(
            answer="No chunks were retrieved for this query, so there's nothing to summarize.",
            citations=[],
            insufficient_context=True,
            model=model,
        )
        return

    if model not in SYNTHESIS_MODELS:
        raise SynthesisError(
            f"unsupported synthesis model: {model}. Allowed: {sorted(SYNTHESIS_MODELS)}"
        )
    provider_name = SYNTHESIS_MODELS[model]
    if provider_name not in ("anthropic", "google"):
        raise SynthesisError(
            f"streaming synthesis only supports Anthropic and Google models today (got {provider_name})"
        )
    model_id = model.split("/", 1)[1]

    user_prompt = _format_user_prompt(query, chunks)
    system_prompt = _build_streaming_system_prompt(datetime.now(UTC))

    # Route through `shared.llm.acompletion(..., stream=True)` so the
    # call automatically forwards to the customer's LiteLLM proxy when
    # `LLM_GATEWAY_URL` is set (managed-isolated / self-host tenants
    # have no provider keys locally). LiteLLM normalizes Anthropic's
    # `messages.stream` and Google's `generate_content_stream` into
    # one OpenAI-shaped chunk iterator: each chunk exposes
    # `chunk.choices[0].delta.content`; final usage rides on the last
    # chunk's `chunk.usage`. See `docs/llm-migration-inventory.md`
    # rows for `synthesize_stream`.
    #
    # SYNTHESIS_MODELS keys use `google/` as the internal provider tag
    # but LiteLLM routes Gemini (AI Studio, API-key auth) via the
    # `gemini/` prefix. Bare ids and `google/` route to Vertex AI which
    # needs full GCP service-account creds we don't ship. Translate
    # here; Anthropic's prefix is already canonical.
    litellm_provider = "gemini" if provider_name == "google" else provider_name
    litellm_model = f"{litellm_provider}/{model_id}"
    completion_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "timeout": SYNTHESIS_TIMEOUT_SECONDS,
    }
    if provider_name == "google":
        # Gemini 3 Flash thinks by default. Thinking tokens are billed
        # against `max_output_tokens` and produce no visible output;
        # for retrieval-grounded synthesis the answer must come from the
        # chunks, so extended reasoning is pure latency + wasted budget.
        # The legacy call used `thinking_config: {thinking_budget: 0}`.
        # LiteLLM normalizes this to `reasoning_effort="none"`; for
        # Gemini 2.x that maps to budget=0, but for **Gemini 3+** it
        # maps to `thinking_level="minimal"` because Google removed the
        # ability to fully disable thinking on the 3.x line. "Minimal"
        # is the closest available approximation — managed-tenant
        # streaming may incur a small thinking-token overhead the old
        # direct-SDK path did not, but correctness (cited prose grounded
        # in the chunks) is unchanged. The optimization was about
        # latency, not correctness.
        # TODO(phase-0b-thinking-config): revisit when LiteLLM exposes
        # a passthrough for Gemini 3's `thinking_level` budget knob, or
        # when Google re-enables true budget=0 on 3.x.
        completion_kwargs["reasoning_effort"] = "none"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    accumulated = ""
    try:
        resp = await shared_llm.acompletion(
            model=litellm_model,
            messages=messages,
            stream=True,
            **completion_kwargs,
        )
        async for chunk in resp:
            # OpenAI-shape: each chunk has `.choices[0].delta.content`
            # (string or None). Final chunk may carry `.usage` for
            # token accounting (Phase D15 metering); we accept it
            # silently today — wire-format hasn't grown a usage slot
            # on `StreamFinal`, so we don't surface it. Recorded here
            # for the eventual metering pass.
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            text = getattr(delta, "content", None) if delta is not None else None
            if not text:
                continue
            accumulated += text
            yield StreamDelta(text=text)
    except shared_llm.LLMError as exc:
        raise SynthesisError(
            f"{provider_name} streaming api error: {exc}"
        ) from exc

    # Detect the insufficient-context sentinel and strip it before parsing
    # citations. Models occasionally emit it lowercased or with surrounding
    # whitespace — be tolerant.
    raw = accumulated.strip()
    insufficient = False
    cleaned = raw
    if raw.upper().startswith(_INSUFFICIENT_SENTINEL):
        insufficient = True
        cleaned = raw[len(_INSUFFICIENT_SENTINEL) :].lstrip(" \t\r\n:-")
    else:
        # Heuristic fallback for models that ignore the sentinel rule.
        lowered = raw.lower()
        if any(marker in lowered for marker in _INSUFFICIENT_MARKERS) and len(raw) < 200:
            insufficient = True

    answer = normalize_citations_in_answer(cleaned)
    citations = _extract_citations(answer, chunks, declared=None)
    yield StreamFinal(
        answer=answer,
        citations=citations,
        insufficient_context=insufficient,
        model=model,
    )


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _format_user_prompt(query: str, chunks: list[SynthesisChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        title = f" — {c.title}" if c.title else ""
        body = (c.content or "").strip()
        if len(body) > 1500:
            body = body[:1500] + "…"
        blocks.append(
            f"[chunk:{i}] ({c.source_system}{title} | {c.source_url} | "
            f"updated {c.updated_at})\n{body}"
        )
    chunk_block = "\n\n".join(blocks)
    return f"""Query: {query}

Chunks:
{chunk_block}

Answer using only these chunks. Cite every claim with [chunk:N]."""


# ---------------------------------------------------------------------------
# Provider dispatch — each adapter returns the parsed structured dict
# ---------------------------------------------------------------------------


async def _dispatch(
    provider_name: str,
    *,
    system: str,
    user: str,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    if provider_name == "anthropic":
        return await _call_anthropic(system, user, model, max_tokens)
    if provider_name == "openai":
        return await _call_openai(system, user, model, max_tokens)
    if provider_name == "google":
        return await _call_google(system, user, model, max_tokens)
    raise SynthesisError(f"unknown provider: {provider_name}")


async def _call_anthropic(
    system: str, user: str, model: str, max_tokens: int
) -> dict[str, Any]:
    """Forced `render_answer` tool call via LiteLLM (Phase-0b chunk C).

    Mirrors the pre-migration Anthropic-shape forced tool-use: same
    JSON Schema (`ANSWER_SCHEMA`), same tool name (`render_answer`),
    same forced `tool_choice`. LiteLLM normalises Anthropic's
    `tool_use` block into the OpenAI-shaped `tool_calls[0].function`
    that `forced_tool_call` reads.
    """
    _check_provider_credentials(provider="anthropic", env_name="ANTHROPIC_API_KEY")
    try:
        args, _resp = await forced_tool_call(
            model=_litellm_model(provider="anthropic", model_id=model),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tool_name="render_answer",
            tool_description=(
                "Output the answer in the required structured format. "
                "Always invoke this tool — never reply in plain text."
            ),
            tool_schema=ANSWER_SCHEMA,
            max_tokens=max_tokens,
            timeout=SYNTHESIS_TIMEOUT_SECONDS,
        )
    except ToolCallParseError:
        # Model declined to call the tool — best-effort fall back to
        # text content. Same fallback the SDK-shaped path used.
        return _fallback_parse_text("")
    except LLMError as exc:
        raise SynthesisError(f"anthropic api error: {exc}") from exc
    return args


async def _call_openai(
    system: str, user: str, model: str, max_tokens: int
) -> dict[str, Any]:
    """OpenAI strict JSON-schema response via LiteLLM (Phase-0b chunk C).

    Uses `response_format={"type": "json_schema", ..., "strict": True}`
    — LiteLLM forwards this verbatim to OpenAI; OpenAI's strict mode
    constrains generation to the schema. The strict-mode contract
    (`additionalProperties: false`, every property in `required`)
    is enforced upstream in `ANSWER_SCHEMA`.
    """
    from shared.llm import acompletion

    _check_provider_credentials(provider="openai", env_name="OPENAI_API_KEY")
    try:
        resp = await acompletion(
            model=_litellm_model(provider="openai", model_id=model),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": ANSWER_SCHEMA,
                    "strict": True,
                },
            },
            timeout=SYNTHESIS_TIMEOUT_SECONDS,
        )
    except LLMError as exc:
        raise SynthesisError(f"openai api error: {exc}") from exc

    choices = getattr(resp, "choices", None) or []
    content = ""
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) or ""
    if not content:
        return _fallback_parse_text("")
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return _fallback_parse_text(content)


def _strip_keys_recursive(
    schema: Any, keys_to_strip: tuple[str, ...]
) -> Any:
    """Return a deep copy of `schema` with the named keys removed at every
    object level. Used to sanitize a JSON-Schema dict for providers whose
    schema dialect rejects keys others require.
    """
    if isinstance(schema, dict):
        return {
            k: _strip_keys_recursive(v, keys_to_strip)
            for k, v in schema.items()
            if k not in keys_to_strip
        }
    if isinstance(schema, list):
        return [_strip_keys_recursive(v, keys_to_strip) for v in schema]
    return schema


async def _call_google(
    system: str, user: str, model: str, max_tokens: int
) -> dict[str, Any]:
    """Google response_schema-constrained JSON via LiteLLM (Phase-0b chunk C).

    Uses LiteLLM's provider-passthrough kwargs to forward
    `response_schema` (Gemini's native structured-output spec),
    `response_mime_type`, and `thinking_config` straight to Gemini.
    The schema is sanitised first — Google rejects
    `additionalProperties` outright while OpenAI strict mode requires
    it, so the schema is identical to `ANSWER_SCHEMA` minus that key.
    """
    from shared.llm import acompletion

    _check_provider_credentials(provider="google", env_name="GOOGLE_API_KEY")
    # Google has no separate system slot; LiteLLM merges system into the
    # contents when routing to Gemini. Send the OpenAI-shaped messages
    # list and let LiteLLM do the join.
    google_schema = _strip_keys_recursive(ANSWER_SCHEMA, ("additionalProperties",))
    try:
        resp = await acompletion(
            model=_litellm_model(provider="google", model_id=model),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            # Provider-passthrough — LiteLLM forwards these unchanged
            # to google-genai's GenerateContentConfig.
            response_schema=google_schema,
            response_mime_type="application/json",
            # Disable Gemini 3 thinking so the full token budget is
            # available for structured JSON output (see streaming
            # branch for full rationale).
            thinking_config={"thinking_budget": 0},
            timeout=SYNTHESIS_TIMEOUT_SECONDS,
        )
    except LLMError as exc:
        raise SynthesisError(f"google api error: {exc}") from exc

    choices = getattr(resp, "choices", None) or []
    text = ""
    if choices:
        message = getattr(choices[0], "message", None)
        text = getattr(message, "content", None) or ""
    if not text:
        return _fallback_parse_text("")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return _fallback_parse_text(text)


def _check_provider_credentials(*, provider: str, env_name: str) -> None:
    """Raise SynthesisError when neither the provider key nor a LiteLLM
    gateway is configured. After Phase-0b chunk A, managed-isolated
    tenants run without provider keys — the gateway URL carries the
    credential — so we must accept either condition.
    """
    from shared.llm import gateway_url

    secret = getattr(get_settings(), f"{provider}_api_key", None)
    key = secret.get_secret_value() if secret is not None else ""
    if not key and not gateway_url():
        raise SynthesisError(f"{env_name} not configured")


def _litellm_model(*, provider: str, model_id: str) -> str:
    """Map our `provider` key + bare model id to a LiteLLM-prefixed model.

    LiteLLM uses these provider prefixes (see shared/llm.py docstring):
      * Anthropic -> ``anthropic/...``
      * OpenAI    -> ``openai/...``
      * Google    -> ``gemini/...`` (note: NOT ``google/...``)

    If `model_id` already carries a slash we assume it's pre-prefixed
    and pass it through.
    """
    if "/" in model_id:
        return model_id
    prefix_for = {"anthropic": "anthropic", "openai": "openai", "google": "gemini"}
    prefix = prefix_for.get(provider)
    if prefix is None:
        raise SynthesisError(f"unknown provider for LiteLLM routing: {provider}")
    return f"{prefix}/{model_id}"


# ---------------------------------------------------------------------------
# Fallback — when a provider returns free-form text instead of structured
# output (model declined the tool, schema enforcement glitched, etc.).
# ---------------------------------------------------------------------------


_INSUFFICIENT_MARKERS = (
    "insufficient context",
    "cannot answer",
    "can't answer",
    "not enough information",
    "no relevant information",
    "no information",
    "do not have",
    "don't have",
    "unable to answer",
)


def _fallback_parse_text(text: str) -> dict[str, Any]:
    """Wrap a raw text body into the structured shape with best-effort
    insufficient_context inference. Used only when the provider's structured
    output mode fails or returns nothing.
    """
    body = text.strip()
    # Tolerate markdown-fenced JSON.
    if body.startswith("```"):
        stripped = body.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        body = stripped
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    if not body:
        return {
            "answer": "Provider returned no content.",
            "insufficient_context": True,
            "citations_used": [],
        }
    lower = body.lower()
    insufficient = any(marker in lower for marker in _INSUFFICIENT_MARKERS)
    log.info(
        "synthesis.fallback_parse",
        raw_len=len(text),
        insufficient=insufficient,
    )
    return {
        "answer": body,
        "insufficient_context": insufficient,
        "citations_used": [],
    }


# ---------------------------------------------------------------------------
# Citation handling
# ---------------------------------------------------------------------------


# Match [chunk:N] (preferred) and bare chunk:N. Models in particular tend to
# drop the brackets despite the prompt; we accept both and normalize on render.
_CITATION_RE = re.compile(r"\[?chunk:(\d+)\]?")

# Some models (notably Gemini) read "one or more inline citations of the form
# [chunk:N]" as permission to write a comma list inside one bracket pair, e.g.
# `[chunk:1, 5, 7]`. Split those into separate `[chunk:N]` tags before the
# single-citation normalizer runs, otherwise `_CITATION_RE` greedily consumes
# `[chunk:1` and leaves `, 5, 7]` dangling in the rendered output.
_MULTI_CITATION_RE = re.compile(r"\[chunk:\s*(\d+(?:\s*,\s*\d+)+)\s*\]")


def normalize_citations_in_answer(answer: str) -> str:
    """Wrap bare `chunk:N` into `[chunk:N]` and split multi-chunk citations
    like `[chunk:1, 5, 7]` so downstream renderers see a single canonical
    format.
    """

    def _split_multi(m: re.Match[str]) -> str:
        nums = [n.strip() for n in m.group(1).split(",")]
        return "".join(f"[chunk:{n}]" for n in nums if n)

    answer = _MULTI_CITATION_RE.sub(_split_multi, answer)

    def _repl(m: re.Match[str]) -> str:
        return f"[chunk:{m.group(1)}]"

    return _CITATION_RE.sub(_repl, answer)


def _extract_citations(
    answer: str,
    chunks: list[SynthesisChunk],
    declared: list[int] | None = None,
) -> list[dict[str, object]]:
    """Pull citations from the answer text plus any `citations_used` list
    the model declared. Dedupe, drop out-of-range, map index → chunk_id.
    """
    seen: set[int] = set()
    out: list[dict[str, object]] = []

    def _add(idx: int) -> None:
        if idx < 1 or idx > len(chunks) or idx in seen:
            return
        seen.add(idx)
        out.append({"index": idx, "chunk_id": chunks[idx - 1].chunk_id})

    for match in _CITATION_RE.finditer(answer):
        _add(int(match.group(1)))
    if declared:
        for idx in declared:
            try:
                _add(int(idx))
            except (TypeError, ValueError):
                continue
    return out
