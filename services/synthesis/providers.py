"""Provider Protocols for cheap-model synthesis stages.

Two abstractions live here:

  TRIAGE              -> shared.constants.WIKI_TRIAGE_MODEL
  DIRECTED PHRASES    -> shared.constants.DIRECTED_PHRASES_MODEL

Both stages support Anthropic Haiku and Gemini variants. The wiki agent
itself goes through `services.synthesis.gemini_agent_client` and does
not use this module — its surface (CachedContent + cached generate
calls) doesn't translate to Anthropic's prompt-cache model.

Selection: the model name is read from the constants above. To flip a
stage from Haiku -> a Gemini variant (or back), edit the constant and
redeploy. There is no env-var override path; per-stage tuning lives in
shared/constants.py alongside RRF_K, source half-lives, and the rest of
the LLM-id registry.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Protocol

from anthropic import AsyncAnthropic

from services.synthesis.models import (
    TriageInput,
    TriageOutput,
)
from services.synthesis.prompts import (
    build_directed_phrases_prompt,
    build_triage_prompt,
    directed_tool_name,
    triage_tool_name,
)
from shared.config import get_settings
from shared.constants import (
    DIRECTED_PHRASES_MODEL,
    HAIKU_MODEL,
    MAX_DIRECTED_PHRASE_CHARS,
    MAX_DIRECTED_VECTORS_PER_DOC,
    WIKI_TRIAGE_MODEL,
)
from shared.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Errors — shared across providers so call sites match one type
# ---------------------------------------------------------------------------


class TriageParseError(RuntimeError):
    """Provider returned output we couldn't parse into TriageOutput."""


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class TriageProvider(Protocol):
    async def triage(self, events: list[TriageInput], *, now: datetime) -> TriageOutput: ...


# ---------------------------------------------------------------------------
# Provider name resolution
# ---------------------------------------------------------------------------


_ANTHROPIC_TRIAGE_NAMES = {"haiku", "claude-haiku", HAIKU_MODEL}
_GEMINI_FLASH_LITE_NAMES = {
    "gemini-flash-lite",
    "gemini-flash-lite-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
}


# ---------------------------------------------------------------------------
# Anthropic implementation (tool_use blocks)
# ---------------------------------------------------------------------------


def _extract_tool_use_input(
    blocks: list[Any], *, expected_name: str, error_cls: type[RuntimeError]
) -> dict[str, Any]:
    for block in blocks:
        if (
            getattr(block, "type", "") == "tool_use"
            and getattr(block, "name", "") == expected_name
        ):
            payload = getattr(block, "input", None)
            if isinstance(payload, dict):
                return payload
            raise error_cls(
                f"tool_use input was not a dict: {type(payload).__name__}"
            )
    raise error_cls(f"response had no {expected_name} tool_use block")


class _AnthropicTriage:
    """Anthropic Haiku via tool_use forced output."""

    def __init__(self, client: AsyncAnthropic, *, model: str = HAIKU_MODEL) -> None:
        self._client = client
        self._model = model

    async def triage(self, events: list[TriageInput], *, now: datetime) -> TriageOutput:
        if not events:
            return TriageOutput(verdicts={})
        kwargs = build_triage_prompt(events, now=now)
        resp = await self._client.messages.create(model=self._model, **kwargs)
        payload = _extract_tool_use_input(
            resp.content,
            expected_name=triage_tool_name(),
            error_cls=TriageParseError,
        )
        try:
            return TriageOutput(**payload)
        except Exception as exc:
            raise TriageParseError(
                f"triage tool input failed validation: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Gemini implementation (response_schema + application/json)
# ---------------------------------------------------------------------------


def _strip_keys_recursive(schema: Any, keys: tuple[str, ...]) -> Any:
    """Drop dict keys that Google's response_schema rejects (e.g.
    `additionalProperties`, `pattern`). Mirrors
    `services/retrieval/synthesis.py:_strip_keys_recursive`.
    """
    if isinstance(schema, dict):
        return {
            k: _strip_keys_recursive(v, keys) for k, v in schema.items() if k not in keys
        }
    if isinstance(schema, list):
        return [_strip_keys_recursive(v, keys) for v in schema]
    return schema


_GEMINI_REJECTED_SCHEMA_KEYS = (
    "additionalProperties",
    "pattern",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "cache_control",
)


# Gemini 3.x defaults to thinking-on. Reasoning tokens are deducted from
# `max_output_tokens` and silently truncate the structured-output JSON when
# the answer + reasoning exceed the budget. The eval at
# scripts/eval_directed_phrases.py surfaced this for Pro and Flash on
# 2026-05-09 (reproducible: same prompt, identical model except family).
#   - Pro:        rejects budget=0; needs explicit non-zero.
#   - Flash:      tolerates budget=0 and produces the same answer faster.
#   - Flash Lite: tolerates budget=0; default already minimal.
def _thinking_budget_for(model: str) -> int:
    name = model.lower()
    if "pro" in name:
        # Pro can't disable thinking. Give it slack so the JSON answer
        # always fits even after reasoning consumes part of the budget.
        return 4096
    return 0


def _gemini_client() -> Any:
    """Build a Gemini client. Raises a tagged error if google-genai isn't
    importable or the API key isn't configured.
    """
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai not installed; cannot use Gemini provider"
        ) from exc
    api_key = get_settings().google_api_key.get_secret_value()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not configured for Gemini provider")
    return genai.Client(api_key=api_key)


async def _gemini_call_json(
    *,
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    """Issue one Gemini structured-output call. Returns the parsed dict.

    Call shape rationale (post-2026-05-09 model-shootout eval):
      * `system_instruction=system` instead of concatenating system+user
        into `contents`. Gemini's first-class system slot is processed
        differently from user prompts (instruction-following is stronger)
        and matches the eval harness's call shape, so the eval's quality
        numbers actually predict production quality.
      * `temperature=0.0` for determinism. Default Gemini temperature
        (~1.0) adds run-to-run variance that hurts the deterministic
        regen contract for directed-vector phrases.
      * `response_schema=sanitized` enforces structured output via the
        SDK's response-schema slot.
    """
    client = _gemini_client()
    sanitized = _strip_keys_recursive(schema, _GEMINI_REJECTED_SCHEMA_KEYS)
    # Build the config via the typed objects so thinking_config lands
    # correctly across SDK versions (the dict-shaped config did not always
    # propagate thinking_config through google-genai's coercion path).
    # Local import: same lazy pattern as `_gemini_client()` so a missing
    # google-genai install only bites Gemini callers.
    from google.genai import types as genai_types

    config = genai_types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        response_schema=sanitized,
        temperature=0.0,
        thinking_config=genai_types.ThinkingConfig(
            thinking_budget=_thinking_budget_for(model)
        ),
    )
    resp = await client.aio.models.generate_content(
        model=model,
        contents=user,
        config=config,
    )
    text = getattr(resp, "text", None) or ""
    if not text:
        raise RuntimeError("gemini response was empty")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gemini response was not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"gemini response was not a JSON object: {type(parsed).__name__}"
        )
    return parsed


def _flatten_anthropic_kwargs(
    kwargs: dict[str, Any],
) -> tuple[str, str, dict[str, Any], int]:
    """Pull the system text, user text, tool input_schema, and max_tokens
    out of an Anthropic-shaped `messages.create` kwargs dict so the same
    prompt builder feeds Gemini.
    """
    system_blocks = kwargs.get("system") or []
    if isinstance(system_blocks, list) and system_blocks:
        system = system_blocks[0].get("text", "")
    else:
        system = system_blocks if isinstance(system_blocks, str) else ""
    messages = kwargs.get("messages") or []
    user = messages[0].get("content", "") if messages else ""
    tools = kwargs.get("tools") or []
    schema: dict[str, Any] = {}
    if tools:
        schema = tools[0].get("input_schema", {}) or {}
    max_tokens = int(kwargs.get("max_tokens") or 2048)
    return system, user, schema, max_tokens


class _GeminiTriage:
    """Gemini Flash Lite via response_schema."""

    def __init__(self, *, model: str = "gemini-flash-lite-preview") -> None:
        self._model = model

    async def triage(self, events: list[TriageInput], *, now: datetime) -> TriageOutput:
        if not events:
            return TriageOutput(verdicts={})
        kwargs = build_triage_prompt(events, now=now)
        system, user, schema, max_tokens = _flatten_anthropic_kwargs(kwargs)
        try:
            payload = await _gemini_call_json(
                model=self._model,
                system=system,
                user=user,
                schema=schema,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise TriageParseError(f"gemini triage call failed: {exc}") from exc
        try:
            return TriageOutput(**payload)
        except Exception as exc:
            raise TriageParseError(
                f"gemini triage output failed validation: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Factory function used by the triage worker
# ---------------------------------------------------------------------------


def get_triage_provider(
    anthropic_client: AsyncAnthropic | None = None,
    *,
    model_override: str | None = None,
) -> TriageProvider:
    """Return the configured triage provider.

    `anthropic_client` is required only if the configured model is
    Anthropic; the caller already owns one. `model_override` lets tests
    pin the choice without env vars.
    """
    name = (model_override or WIKI_TRIAGE_MODEL).lower()
    if name in _GEMINI_FLASH_LITE_NAMES:
        return _GeminiTriage(
            model=name if name.startswith("gemini") else "gemini-flash-lite-preview"
        )
    if name in _ANTHROPIC_TRIAGE_NAMES:
        if anthropic_client is None:
            raise ValueError("Anthropic triage requires an AsyncAnthropic client")
        return _AnthropicTriage(anthropic_client, model=HAIKU_MODEL)
    raise ValueError(f"unknown WIKI_TRIAGE_MODEL: {name}")


# ===========================================================================
# Directed-phrase generation provider
# ===========================================================================
#
# Mirrors the triage abstraction. One Gemini call per wiki page during
# synthesis emits 5-10 trigger phrases that boost retrieval ranking when
# an engineer's symptom-style query semantically matches them. The eval
# at scripts/eval_directed_phrases.py (2026-05-09) showed Gemini 3 Flash
# beats Haiku 4.5 on every quality metric (specificity 8.6 vs 7.8,
# retrieval-fit 8.2 vs 7.8) at a quarter of the cost.
#
# Same routing pattern as triage: model name -> impl. Add a new alias to
# the relevant set if a future Gemini variant should be selectable.
# ---------------------------------------------------------------------------


class DirectedPhrasesParseError(RuntimeError):
    """Provider returned output we couldn't coerce into list[str]."""


class DirectedPhrasesProvider(Protocol):
    async def generate(self, *, page_title: str, page_body: str) -> list[str]: ...


_ANTHROPIC_DIRECTED_NAMES = {"haiku", "claude-haiku", HAIKU_MODEL}

# Alias -> canonical model id sent to the Google API. Aliases let
# operators set DIRECTED_PHRASES_MODEL to a friendly name; the canonical
# value is what the SDK actually uses.
#
# IMPORTANT: a canonical id MUST also map to itself, so flipping
# DIRECTED_PHRASES_MODEL to either the alias or the canonical value
# both resolve correctly.
_GEMINI_FLASH_CANONICAL = {
    "gemini-flash":            "gemini-3-flash-preview",
    "gemini-3-flash":          "gemini-3-flash-preview",
    "gemini-3-flash-preview":  "gemini-3-flash-preview",
}

# Directed-phrases-specific Flash-Lite registry. Intentionally distinct
# from the triage-side `_GEMINI_FLASH_LITE_NAMES` so a future PR adding a
# triage-only Flash-Lite alias doesn't silently route directed-phrase
# traffic through an unevaluated model.
_GEMINI_FLASH_LITE_CANONICAL = {
    "gemini-flash-lite":              "gemini-flash-lite-preview",
    "gemini-flash-lite-preview":      "gemini-flash-lite-preview",
    "gemini-3.1-flash-lite-preview":  "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite":          "gemini-2.5-flash-lite",
}


def _resolve_alias(name: str, registry: dict[str, str]) -> str | None:
    """Return the canonical model id for `name`, or None if `name` is not
    a registered alias. Uses lookup, not substring matching, so a typoed
    constant fails loud rather than silently routing somewhere unintended.
    """
    return registry.get(name)


def _coerce_phrases(raw: Any) -> list[str]:
    """Normalize a 'phrases' payload into a clean list[str].

    Both providers route through this so the post-call rules (length cap,
    whitespace strip, MAX_DIRECTED_VECTORS_PER_DOC truncation) live in
    one place.
    """
    if not isinstance(raw, list):
        raise DirectedPhrasesParseError(
            f"phrases payload was not a list: {type(raw).__name__}"
        )
    cleaned: list[str] = []
    for p in raw:
        if not isinstance(p, str):
            continue
        s = p.strip()
        if not s:
            continue
        if len(s) > MAX_DIRECTED_PHRASE_CHARS:
            log.warning(
                "directed.llm_phrase_too_long",
                length=len(s),
                limit=MAX_DIRECTED_PHRASE_CHARS,
            )
            continue
        cleaned.append(s)
    return cleaned[:MAX_DIRECTED_VECTORS_PER_DOC]


class _AnthropicDirectedPhrases:
    """Anthropic Haiku via tool_use (legacy default; kept for fallback /
    A-B comparison if Gemini regresses).
    """

    def __init__(self, client: AsyncAnthropic, *, model: str = HAIKU_MODEL) -> None:
        self._client = client
        self._model = model

    async def generate(self, *, page_title: str, page_body: str) -> list[str]:
        kwargs = build_directed_phrases_prompt(
            page_title=page_title, page_body=page_body
        )
        resp = await self._client.messages.create(model=self._model, **kwargs)
        expected = directed_tool_name()
        for block in resp.content:
            if (
                getattr(block, "type", "") == "tool_use"
                and getattr(block, "name", "") == expected
            ):
                payload = getattr(block, "input", None)
                if not isinstance(payload, dict):
                    raise DirectedPhrasesParseError(
                        f"directed tool input was not a dict: {type(payload).__name__}"
                    )
                # Same rule as the Gemini path: missing-key is a parse
                # failure, NOT a successful zero-phrase result, so the
                # orchestrator preserves prior LLM rows.
                if "phrases" not in payload:
                    raise DirectedPhrasesParseError(
                        "anthropic tool input missing 'phrases' key (got: "
                        f"{sorted(payload.keys()) or 'empty object'})"
                    )
                return _coerce_phrases(payload["phrases"])
        raise DirectedPhrasesParseError(
            f"directed response had no {expected} tool_use block"
        )


class _GeminiDirectedPhrases:
    """Gemini structured output via response_schema. Default impl for new
    deploys per the 2026-05-09 model-shootout eval.
    """

    def __init__(self, *, model: str = "gemini-3-flash-preview") -> None:
        self._model = model

    async def generate(self, *, page_title: str, page_body: str) -> list[str]:
        kwargs = build_directed_phrases_prompt(
            page_title=page_title, page_body=page_body
        )
        system, user, schema, max_tokens = _flatten_anthropic_kwargs(kwargs)
        try:
            payload = await _gemini_call_json(
                model=self._model,
                system=system,
                user=user,
                schema=schema,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise DirectedPhrasesParseError(
                f"gemini directed-phrase call failed: {exc}"
            ) from exc
        # Treat "phrases key missing entirely" as a parse failure, NOT as
        # "successfully returned zero phrases". The orchestrator's
        # `result.llm_failed` branch preserves prior LLM rows on parse
        # failure but DELETES them on a successful empty result -- so a
        # prompt drift to e.g. {"trigger_phrases": [...]} would otherwise
        # silently wipe every doc's directed_vectors on the next regen.
        if "phrases" not in payload:
            raise DirectedPhrasesParseError(
                "gemini response missing 'phrases' key (got: "
                f"{sorted(payload.keys()) or 'empty object'})"
            )
        return _coerce_phrases(payload["phrases"])


def get_directed_phrases_provider(
    anthropic_client: AsyncAnthropic | None = None,
    *,
    model_override: str | None = None,
) -> DirectedPhrasesProvider:
    """Return the configured directed-phrases provider.

    `anthropic_client` is required only when the configured model resolves
    to an Anthropic alias (today: HAIKU_MODEL). For Gemini variants,
    the helper builds its own client internally.

    `model_override` lets tests pin the choice without touching constants.
    """
    name = (model_override or DIRECTED_PHRASES_MODEL).lower()
    # Alias -> canonical Google model id. Reviewing reviewers caught that
    # the previous `name if name.startswith("gemini") else <fallback>`
    # ternary had a dead else-branch (every alias starts with "gemini"),
    # which would have shipped the alias string verbatim as the API model
    # id and 4xx'd. Keep the resolution explicit + auditable.
    flash_canonical = _resolve_alias(name, _GEMINI_FLASH_CANONICAL)
    if flash_canonical is not None:
        return _GeminiDirectedPhrases(model=flash_canonical)
    flash_lite_canonical = _resolve_alias(name, _GEMINI_FLASH_LITE_CANONICAL)
    if flash_lite_canonical is not None:
        return _GeminiDirectedPhrases(model=flash_lite_canonical)
    if name in _ANTHROPIC_DIRECTED_NAMES:
        # Lazy-construct an AsyncAnthropic from settings if the caller
        # didn't supply one. Mirrors `_gemini_client()`'s lazy pattern so
        # flipping `DIRECTED_PHRASES_MODEL = "haiku"` for rollback
        # actually works end-to-end without threading a client through
        # `wiki_agent` -> `persist_directed_vectors`. The triage path's
        # caller (services/synthesis/triage.py:224) owns its own client
        # for the worker's lifetime; the directed path has no such
        # owner, hence the lazy fallback here.
        if anthropic_client is None:
            api_key = get_settings().anthropic_api_key.get_secret_value()
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not configured; cannot use Anthropic "
                    "directed-phrases provider"
                )
            anthropic_client = AsyncAnthropic(api_key=api_key)
        return _AnthropicDirectedPhrases(anthropic_client, model=HAIKU_MODEL)
    raise ValueError(f"unknown DIRECTED_PHRASES_MODEL: {name}")
