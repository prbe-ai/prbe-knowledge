"""LLM synthesis over retrieval chunks for the /query endpoint.

Three providers (Anthropic, OpenAI, Google) behind one async function:

    result = await synthesize(query, chunks, model, max_tokens)

The model selector is a "<provider>/<model>" string. Provider keys are
defined in shared.constants.SYNTHESIS_MODELS — adding a new model is
one line there if it shares a provider, or a new branch in `_dispatch`
for a fresh provider.

Output contract is identical across providers: every claim in `answer`
ends with one or more `[chunk:N]` citations (1-indexed against the
input chunk list). When chunks don't support a confident answer, the
LLM returns `insufficient_context: true` instead of hallucinating.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

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
    raw_provider_response: str  # untouched body from the provider for debugging


class _Provider(Protocol):
    async def call(self, system: str, user: str, model: str, max_tokens: int) -> str: ...


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """You are a careful retrieval-augmented assistant. You answer the user's question
using ONLY the chunks you've been given. Every factual claim in your answer must
end with one or more inline citations of the form [chunk:N], where N is the
1-indexed chunk number.

Rules:
- Use ONLY information present in the chunks. Do not invent dates, names,
  PR numbers, decisions, or relationships.
- Every sentence that makes a claim must end with at least one [chunk:N].
- If the chunks do not contain enough information to confidently answer,
  set "insufficient_context": true and write `answer` as a one-sentence
  explanation of what's missing.
- Be concise. 1-3 short paragraphs. Don't restate the question. No preamble.
- No bullet lists unless the user explicitly asks for a list.

Return ONLY this JSON, nothing else:
{
  "answer": "string with inline [chunk:N] citations",
  "citations_used": [1, 2, 5],
  "insufficient_context": false
}
"""


async def synthesize(
    query: str,
    chunks: list[SynthesisChunk],
    model: str,
    max_tokens: int = 600,
) -> SynthesisResult:
    """Run an LLM over `chunks` to synthesize an answer to `query`.

    Raises SynthesisError on provider failure or unparseable output.
    Empty `chunks` short-circuits to an explicit "no context" result so
    we don't waste a model call on nothing.
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
    raw = await _dispatch(
        provider_name, system=_SYSTEM_PROMPT, user=user_prompt,
        model=model_id, max_tokens=max_tokens,
    )

    parsed = _parse_response(raw)
    citations = _extract_citations(parsed["answer"], chunks)
    return SynthesisResult(
        answer=parsed["answer"],
        citations=citations,
        insufficient_context=bool(parsed.get("insufficient_context", False)),
        model=model,
        raw_provider_response=raw,
    )


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _format_user_prompt(query: str, chunks: list[SynthesisChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        title = f" — {c.title}" if c.title else ""
        body = (c.content or "").strip()
        # Truncate very long chunks so we don't blow the context window.
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

Now answer using only these chunks. Cite every claim with [chunk:N]."""


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


async def _dispatch(
    provider_name: str, *, system: str, user: str, model: str, max_tokens: int
) -> str:
    if provider_name == "anthropic":
        return await _call_anthropic(system, user, model, max_tokens)
    if provider_name == "openai":
        return await _call_openai(system, user, model, max_tokens)
    if provider_name == "google":
        return await _call_google(system, user, model, max_tokens)
    raise SynthesisError(f"unknown provider: {provider_name}")


async def _call_anthropic(system: str, user: str, model: str, max_tokens: int) -> str:
    from anthropic import APIError, AsyncAnthropic

    api_key = get_settings().anthropic_api_key.get_secret_value()
    if not api_key:
        raise SynthesisError("ANTHROPIC_API_KEY not configured")
    client = AsyncAnthropic(api_key=api_key, timeout=SYNTHESIS_TIMEOUT_SECONDS)
    try:
        # Prefill the assistant turn with `{` to bias toward JSON. Claude
        # respects this prefix; the response continues from there. Re-prepend
        # the `{` to the returned text since the SDK strips it.
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[
                {"role": "user", "content": user},
                {"role": "assistant", "content": "{"},
            ],
        )
    except APIError as exc:
        raise SynthesisError(f"anthropic api error: {exc}") from exc
    body = "".join(
        block.text for block in resp.content if getattr(block, "type", "") == "text"
    )
    return "{" + body


async def _call_openai(system: str, user: str, model: str, max_tokens: int) -> str:
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
            response_format={"type": "json_object"},
        )
    except APIError as exc:
        raise SynthesisError(f"openai api error: {exc}") from exc
    choice = resp.choices[0]
    return choice.message.content or ""


async def _call_google(system: str, user: str, model: str, max_tokens: int) -> str:
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
    # Google doesn't have a separate system slot; prepend system to the user msg.
    contents = f"{system}\n\n---\n\n{user}"
    try:
        resp = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config={
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json",
            },
        )
    except Exception as exc:
        raise SynthesisError(f"google api error: {exc}") from exc
    return getattr(resp, "text", None) or ""


# ---------------------------------------------------------------------------
# Output parsing + citation extraction
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


def _parse_response(raw: str) -> dict[str, Any]:
    """Best-effort: prefer JSON when the model obeys the schema, fall back
    to treating the raw output as plain prose. Models occasionally answer
    in prose despite the prompt — that's still a useful answer, no reason
    to 502 the caller over it.
    """
    text = raw.strip()
    # Tolerate markdown-fenced JSON.
    if text.startswith("```"):
        stripped = text.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        text = stripped
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: treat the raw text as the answer body. Citation extraction
    # downstream still works on [chunk:N] tags inside prose. Infer
    # insufficient_context heuristically from common refusal phrasing.
    lower = raw.lower()
    insufficient = any(marker in lower for marker in _INSUFFICIENT_MARKERS)
    log.info(
        "synthesis.fallback_parse",
        raw_len=len(raw),
        insufficient=insufficient,
    )
    return {
        "answer": raw.strip(),
        "insufficient_context": insufficient,
    }


_CITATION_RE = re.compile(r"\[chunk:(\d+)\]")


def _extract_citations(
    answer: str, chunks: list[SynthesisChunk]
) -> list[dict[str, object]]:
    """Pull [chunk:N] tags from the answer, dedupe, map to chunk_id.

    Out-of-range or duplicate indices are dropped silently — the dashboard
    can render the [chunk:N] tags inline without worrying about validity.
    """
    seen: set[int] = set()
    out: list[dict[str, object]] = []
    for match in _CITATION_RE.finditer(answer):
        idx = int(match.group(1))
        if idx < 1 or idx > len(chunks) or idx in seen:
            continue
        seen.add(idx)
        out.append({"index": idx, "chunk_id": chunks[idx - 1].chunk_id})
    return out
