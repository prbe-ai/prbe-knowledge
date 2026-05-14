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

Phase-0b: every provider call goes through `shared.llm.acompletion` so
tenants without provider API keys route through the central LiteLLM
gateway. Prompt caching (cache_control: ephemeral)
survives — LiteLLM forwards it on Anthropic provider calls.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Protocol

from services.synthesis.models import (
    TriageInput,
    TriageOutput,
)
from services.synthesis.prompts import (
    build_directed_phrases_prompt,
    build_triage_prompt,
    directed_tool_name,
)
from shared.constants import (
    DIRECTED_PHRASES_MODEL,
    HAIKU_MODEL,
    MAX_DIRECTED_PHRASE_CHARS,
    MAX_DIRECTED_VECTORS_PER_DOC,
    WIKI_TRIAGE_MODEL,
)
from shared.llm import LLMError
from shared.llm_tools import ToolCallParseError, forced_tool_call
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
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
}


# ---------------------------------------------------------------------------
# Anthropic-shape -> LiteLLM/OpenAI-shape kwargs adapter
# ---------------------------------------------------------------------------


def _anthropic_kwargs_to_messages_and_schema(
    kwargs: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],  # messages
    str,                    # tool_name
    str,                    # tool_description
    dict[str, Any],         # tool_schema (OpenAI-shaped parameters; same JSON Schema)
    int,                    # max_tokens
]:
    """Translate the Anthropic-`messages.create` kwargs dict (produced by
    `services.synthesis.prompts.build_*_prompt`) into LiteLLM-compatible
    arguments for `shared.llm_tools.forced_tool_call`.

    The translation:

      * Anthropic `system=[{"type":"text","text":..., "cache_control":{...}}]`
        -> OpenAI-shaped system message whose `content` is a list of
        content blocks. LiteLLM preserves the `cache_control` field
        on Anthropic provider calls (see
        litellm/llms/anthropic/chat/transformation.py::is_cache_control_set
        and the surrounding system-message handler).

      * Anthropic `tools=[{"name":..., "description":..., "input_schema":...}]`
        -> OpenAI-shaped `{"type":"function","function":{...}}` is built
        by `forced_tool_call`. Here we just pull the name, description,
        and JSON Schema (which is identical between Anthropic
        `input_schema` and OpenAI `parameters`).

      * Anthropic `tool_choice={"type":"tool","name":...}` -> we always
        force the named call inside `forced_tool_call`, so this is
        implicit (we assert the names match).
    """
    # System block + cache_control survives — LiteLLM forwards it to
    # Anthropic verbatim and ignores it for OpenAI/Gemini.
    system_blocks = kwargs.get("system") or []
    system_message: dict[str, Any] | None
    if isinstance(system_blocks, list) and system_blocks:
        # Preserve content-block shape AND cache_control. Build typed
        # text blocks so LiteLLM's Anthropic transformer recognizes the
        # cache_control hint.
        content_blocks = []
        for block in system_blocks:
            if not isinstance(block, dict):
                continue
            content_blocks.append(
                {
                    "type": "text",
                    "text": block.get("text", ""),
                    **(
                        {"cache_control": block["cache_control"]}
                        if "cache_control" in block
                        else {}
                    ),
                }
            )
        system_message = {"role": "system", "content": content_blocks}
    elif isinstance(system_blocks, str) and system_blocks:
        system_message = {"role": "system", "content": system_blocks}
    else:
        system_message = None

    # User/assistant messages pass through (the prompt builders only
    # produce a single-user message, no assistant turns).
    user_messages = kwargs.get("messages") or []

    messages: list[dict[str, Any]] = []
    if system_message is not None:
        messages.append(system_message)
    messages.extend(user_messages)

    tools = kwargs.get("tools") or []
    if not tools:
        raise RuntimeError("anthropic prompt kwargs missing tools list")
    tool = tools[0]
    tool_name = tool.get("name") or ""
    tool_description = tool.get("description") or ""
    tool_schema = tool.get("input_schema") or {}

    # Sanity-check the prompt builder agreed with the caller on the tool
    # name we'll force.
    tool_choice = kwargs.get("tool_choice") or {}
    declared_name = tool_choice.get("name") if isinstance(tool_choice, dict) else None
    if declared_name and declared_name != tool_name:
        raise RuntimeError(
            f"tool_choice name {declared_name!r} disagrees with tools[0].name "
            f"{tool_name!r}; refusing to migrate ambiguous prompt"
        )

    max_tokens = int(kwargs.get("max_tokens") or 2048)
    return messages, tool_name, tool_description, tool_schema, max_tokens


def _anthropic_litellm_model(model: str) -> str:
    """Return a LiteLLM-prefixed Anthropic model id. Idempotent — a
    model id that's already prefixed (e.g. ``anthropic/<id>``) passes
    through unchanged.
    """
    if "/" in model:
        return model
    return f"anthropic/{model}"


def _gemini_litellm_model(model: str) -> str:
    """Return a LiteLLM-prefixed Gemini model id. Per the Google
    convention LiteLLM uses ``gemini/<id>`` (NOT ``google/<id>``); see
    shared/llm.py docstring for the routing rules.
    """
    if "/" in model:
        return model
    return f"gemini/{model}"


# ---------------------------------------------------------------------------
# Anthropic implementation (forced tool-call via LiteLLM)
# ---------------------------------------------------------------------------


class _AnthropicTriage:
    """Anthropic Haiku via forced tool-call (LiteLLM-routed)."""

    def __init__(self, client: Any | None = None, *, model: str = HAIKU_MODEL) -> None:
        # `client` is kept for backward-compat with the constructor
        # signature `get_triage_provider` passes in (the triage worker
        # used to own an AsyncAnthropic); after the LiteLLM migration
        # we don't own a client — every call goes through
        # `shared.llm.acompletion`. Tests pass a sentinel value here.
        self._client = client  # unused; kept so test mocks construct cleanly
        self._model = model

    async def triage(self, events: list[TriageInput], *, now: datetime) -> TriageOutput:
        if not events:
            return TriageOutput(verdicts={})
        kwargs = build_triage_prompt(events, now=now)
        (
            messages,
            tool_name,
            tool_description,
            tool_schema,
            max_tokens,
        ) = _anthropic_kwargs_to_messages_and_schema(kwargs)
        try:
            args, _resp = await forced_tool_call(
                model=_anthropic_litellm_model(self._model),
                messages=messages,
                tool_name=tool_name,
                tool_description=tool_description,
                tool_schema=tool_schema,
                max_tokens=max_tokens,
            )
        except ToolCallParseError as exc:
            # Same failure mode as the old "no record_triage tool_use
            # block" path. Preserve the message phrasing so the
            # split-retry parse-overflow regex
            # `services.synthesis.triage._PARSE_OVERFLOW_REGEXES`
            # ("no <name> tool_use block") still matches.
            raise TriageParseError(
                f"response had no {tool_name} tool_use block: {exc}"
            ) from exc
        # Pydantic validation of the verdicts payload — same shape and
        # same error type as before.
        try:
            return TriageOutput(**args)
        except Exception as exc:
            raise TriageParseError(
                f"triage tool input failed validation: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Gemini implementation — response_schema via google-genai pre-Phase-0b;
# now goes through `shared.llm.acompletion` with native passthrough.
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


async def _gemini_call_json(
    *,
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    """Issue one Gemini structured-output call via LiteLLM. Returns the
    parsed dict.

    Call shape rationale (post-2026-05-09 model-shootout eval +
    Phase-0b LiteLLM migration):

      * `system` is sent as a separate OpenAI-shaped system message;
        LiteLLM forwards it to Gemini's `system_instruction` slot.
        Gemini's first-class system slot is processed differently from
        user prompts (instruction-following is stronger) and matches
        the eval harness's call shape, so the eval's quality numbers
        actually predict production quality.

      * `temperature=0.0` for determinism. Default Gemini temperature
        (~1.0) adds run-to-run variance that hurts the deterministic
        regen contract for directed-vector phrases.

      * `response_schema=<sanitized JSON Schema>` is forwarded to
        Gemini as the structured-output spec via LiteLLM's
        provider-passthrough kwarg. We prefer the provider-native
        kwarg here over OpenAI's `response_format=json_schema` because
        (a) the existing call site already uses Gemini's schema
        sanitization rules (strips `additionalProperties` etc) and
        (b) the eval was calibrated against the native schema slot,
        so deviating would invalidate the model-shootout numbers.

      * `thinking_config={"thinking_budget": N}` — Phase-0b passes
        through to Gemini via LiteLLM's `thinking` kwarg
        (provider-specific extra). See `_thinking_budget_for` for the
        per-model rule.
    """
    sanitized = _strip_keys_recursive(schema, _GEMINI_REJECTED_SCHEMA_KEYS)
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    # Provider-native passthrough. LiteLLM forwards unknown kwargs to
    # the provider so `response_schema` and `thinking_config` land in
    # Gemini's GenerateContentConfig unchanged. `response_mime_type`
    # is the JSON-output hint Gemini wants when a schema is supplied.
    extra_kwargs: dict[str, Any] = {
        "response_schema": sanitized,
        "response_mime_type": "application/json",
        "thinking_config": {"thinking_budget": _thinking_budget_for(model)},
    }

    from shared.llm import acompletion

    try:
        resp = await acompletion(
            model=_gemini_litellm_model(model),
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            **extra_kwargs,
        )
    except LLMError as exc:
        # Preserve the pre-migration exception shape callers expect
        # (the call sites wrap this in a domain-specific *ParseError).
        raise RuntimeError(f"gemini call failed: {exc}") from exc

    # Gemini structured-output content surfaces as JSON text on
    # `choices[0].message.content`. LiteLLM doesn't pre-parse it.
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise RuntimeError("gemini response was empty (no choices)")
    message = getattr(choices[0], "message", None)
    text = getattr(message, "content", None) or ""
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

    def __init__(self, *, model: str = "gemini-3.1-flash-lite") -> None:
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
    anthropic_client: Any | None = None,
    *,
    model_override: str | None = None,
) -> TriageProvider:
    """Return the configured triage provider.

    `anthropic_client` is accepted (and ignored after the Phase-0b
    migration) for call-site compatibility — the triage_worker still
    passes a sentinel value through. Both Anthropic and Gemini paths
    route through `shared.llm.acompletion` now.

    `model_override` lets tests pin the choice without env vars.
    """
    name = (model_override or WIKI_TRIAGE_MODEL).lower()
    if name in _GEMINI_FLASH_LITE_NAMES:
        # "gemini-flash-lite" is the friendly alias for the GA 3.1 model;
        # versioned ids ("gemini-3.1-flash-lite", "gemini-2.5-flash-lite")
        # are sent to the Google API unchanged.
        return _GeminiTriage(
            model="gemini-3.1-flash-lite" if name == "gemini-flash-lite" else name
        )
    if name in _ANTHROPIC_TRIAGE_NAMES:
        # Pre-migration this raised when no client was supplied; after
        # the migration we don't need a client (LiteLLM owns the
        # transport). Test fixtures that rely on the constructor
        # receiving a client still work — the parameter passes through
        # to `_AnthropicTriage.__init__` and is stored but unused.
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
    "gemini-flash-lite":          "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite":      "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite":      "gemini-2.5-flash-lite",
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
    """Anthropic Haiku via forced tool-call (legacy default; kept for
    fallback / A-B comparison if Gemini regresses).
    """

    def __init__(self, client: Any | None = None, *, model: str = HAIKU_MODEL) -> None:
        # `client` is accepted for call-site compatibility; the LiteLLM
        # migration removed the dependency on a caller-supplied
        # AsyncAnthropic — every call goes through
        # `shared.llm.acompletion`.
        self._client = client  # unused; kept so test mocks pass through
        self._model = model

    async def generate(self, *, page_title: str, page_body: str) -> list[str]:
        kwargs = build_directed_phrases_prompt(
            page_title=page_title, page_body=page_body
        )
        (
            messages,
            tool_name,
            tool_description,
            tool_schema,
            max_tokens,
        ) = _anthropic_kwargs_to_messages_and_schema(kwargs)
        expected = directed_tool_name()
        if tool_name != expected:
            # Defensive: would only fire if `build_directed_phrases_prompt`
            # drifted from `directed_tool_name()`. Surface the mismatch
            # loud rather than letting tests pass with a wrong-tool
            # forced call.
            raise DirectedPhrasesParseError(
                f"prompt tool name {tool_name!r} disagrees with "
                f"directed_tool_name() {expected!r}"
            )
        try:
            args, _resp = await forced_tool_call(
                model=_anthropic_litellm_model(self._model),
                messages=messages,
                tool_name=tool_name,
                tool_description=tool_description,
                tool_schema=tool_schema,
                max_tokens=max_tokens,
            )
        except ToolCallParseError as exc:
            # Preserve "no <name> tool_use block" phrasing so consumers
            # that match on that string (none today, but the symmetry
            # with the triage error message is a regression guard)
            # keep matching.
            raise DirectedPhrasesParseError(
                f"directed response had no {tool_name} tool_use block: {exc}"
            ) from exc

        # Same rule as the Gemini path: missing-key is a parse failure,
        # NOT a successful zero-phrase result, so the orchestrator
        # preserves prior LLM rows.
        if "phrases" not in args:
            raise DirectedPhrasesParseError(
                "anthropic tool input missing 'phrases' key (got: "
                f"{sorted(args.keys()) or 'empty object'})"
            )
        return _coerce_phrases(args["phrases"])


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
    anthropic_client: Any | None = None,
    *,
    model_override: str | None = None,
) -> DirectedPhrasesProvider:
    """Return the configured directed-phrases provider.

    `anthropic_client` is accepted for call-site compatibility. After
    Phase-0b, neither the Anthropic nor the Gemini path needs a
    caller-supplied SDK client — both route through
    `shared.llm.acompletion`. Tests may still pass a client to assert
    against; the parameter passes through to
    `_AnthropicDirectedPhrases.__init__` and is stored but unused.

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
        # Pre-Phase-0b this lazy-constructed an AsyncAnthropic from
        # settings.anthropic_api_key when no client was supplied, so the
        # wiki_agent caller (which passes none) could roll back to
        # Anthropic via a one-line constant flip. After the migration
        # there's nothing to construct — every call goes through
        # `shared.llm.acompletion` which picks up the API key from the
        # env (`ANTHROPIC_API_KEY`) or routes via `LLM_GATEWAY_URL`.
        return _AnthropicDirectedPhrases(anthropic_client, model=HAIKU_MODEL)
    raise ValueError(f"unknown DIRECTED_PHRASES_MODEL: {name}")
