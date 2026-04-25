"""LLM synthesis over retrieval chunks for the /query endpoint.

Three providers (Anthropic, OpenAI, Google) behind one async function:

    result = await synthesize(query, chunks, model, max_tokens)

Each provider uses its native structured-output mechanism so the wrapper
shape is enforced at the API level, not by asking the model nicely:

    Anthropic — tool use with a forced `render_answer` tool call
    OpenAI    — response_format = json_schema strict mode
    Google    — config.response_schema

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
from dataclasses import dataclass
from typing import Any

from shared.config import get_settings
from shared.constants import SYNTHESIS_MODELS
from shared.exceptions import PrbeError
from shared.logging import get_logger

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
                "ends with one or more inline citations of the form [chunk:N], "
                "where N is the 1-indexed chunk number. Use ONLY information "
                "present in the chunks. Do not invent dates, names, PR numbers, "
                "decisions, or relationships. Do not restate the question. No "
                "preamble. Use markdown for emphasis when helpful."
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


_SYSTEM_PROMPT = """You are a careful retrieval-augmented assistant. You answer the user's
question using ONLY the chunks you've been given. The runtime enforces a
structured output schema; just produce the values it asks for.

Hard rules:
- Use ONLY information present in the chunks. Do not invent facts.
- Every sentence that makes a claim must end with at least one [chunk:N].
- If the chunks don't support a confident answer, set insufficient_context
  to true and write a one-line explanation in `answer` instead of guessing.
- Be concise. 1-3 short paragraphs. No preamble.
- Markdown formatting (bold, italic, code) is fine when it helps clarity.
"""


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
        system=_SYSTEM_PROMPT,
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
    from anthropic import APIError, AsyncAnthropic

    api_key = get_settings().anthropic_api_key.get_secret_value()
    if not api_key:
        raise SynthesisError("ANTHROPIC_API_KEY not configured")
    client = AsyncAnthropic(api_key=api_key, timeout=SYNTHESIS_TIMEOUT_SECONDS)
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": "render_answer",
                    "description": (
                        "Output the answer in the required structured format. "
                        "Always invoke this tool — never reply in plain text."
                    ),
                    "input_schema": ANSWER_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": "render_answer"},
        )
    except APIError as exc:
        raise SynthesisError(f"anthropic api error: {exc}") from exc

    for block in resp.content:
        if (
            getattr(block, "type", "") == "tool_use"
            and getattr(block, "name", "") == "render_answer"
        ):
            payload = getattr(block, "input", None)
            if isinstance(payload, dict):
                return payload
    # Model declined to call the tool — best-effort fall back to text content.
    text = "".join(
        getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
    )
    return _fallback_parse_text(text)


async def _call_openai(
    system: str, user: str, model: str, max_tokens: int
) -> dict[str, Any]:
    from openai import APIError, AsyncOpenAI

    api_key = get_settings().openai_api_key.get_secret_value()
    if not api_key:
        raise SynthesisError("OPENAI_API_KEY not configured")
    client = AsyncOpenAI(api_key=api_key, timeout=SYNTHESIS_TIMEOUT_SECONDS)
    try:
        resp = await client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": ANSWER_SCHEMA,
                    "strict": True,
                },
            },
        )
    except APIError as exc:
        raise SynthesisError(f"openai api error: {exc}") from exc

    content = resp.choices[0].message.content or ""
    if not content:
        return _fallback_parse_text("")
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return _fallback_parse_text(content)


async def _call_google(
    system: str, user: str, model: str, max_tokens: int
) -> dict[str, Any]:
    try:
        from google import genai
    except ImportError as exc:
        raise SynthesisError(
            "google-genai not installed; run pip install -e '.[dev]'"
        ) from exc

    api_key = get_settings().google_api_key.get_secret_value()
    if not api_key:
        raise SynthesisError("GOOGLE_API_KEY not configured")
    client = genai.Client(api_key=api_key)
    # Google has no separate system slot; prepend the system message.
    contents = f"{system}\n\n---\n\n{user}"
    try:
        resp = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config={
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json",
                "response_schema": ANSWER_SCHEMA,
            },
        )
    except Exception as exc:
        raise SynthesisError(f"google api error: {exc}") from exc

    text = getattr(resp, "text", None) or ""
    if not text:
        return _fallback_parse_text("")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return _fallback_parse_text(text)


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


def normalize_citations_in_answer(answer: str) -> str:
    """Wrap bare `chunk:N` into `[chunk:N]` so downstream renderers see a
    single canonical format.
    """

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
