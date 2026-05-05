"""Provider Protocol for the triage stage only.

v4 collapses the synthesis pipeline to:

  TRIAGE (cheap model) -> WIKI AGENT (Gemini 3.1 Pro)

The triage provider abstraction stays — Anthropic Haiku and Gemini
Flash Lite are still both viable for the binary-ish triage call. The
agent uses Gemini directly via `services.synthesis.gemini_agent_client`;
no provider abstraction there because the agent harness's surface
(CachedContent + cached generate calls) doesn't translate to
Anthropic's prompt-cache model.

Selection: the model name is read from `shared.constants.WIKI_TRIAGE_MODEL`.
To flip a stage from Haiku -> Flash Lite (or back), edit the constant
and redeploy. There is no env-var override path; the prior
`getattr(settings, ...)` plumbing referenced fields that didn't exist
on Settings, so the env var was silently inert.

Default: Anthropic Haiku.
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
    build_triage_prompt,
    triage_tool_name,
)
from shared.config import get_settings
from shared.constants import (
    HAIKU_MODEL,
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
    """Issue one Gemini structured-output call. Returns the parsed dict."""
    client = _gemini_client()
    contents = f"{system}\n\n---\n\n{user}"
    sanitized = _strip_keys_recursive(schema, _GEMINI_REJECTED_SCHEMA_KEYS)
    resp = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config={
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
            "response_schema": sanitized,
        },
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
