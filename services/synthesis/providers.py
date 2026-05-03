"""Provider Protocol for triage + synthesis + verifier stages.

Each stage of the wiki pipeline can independently target a different
provider (Anthropic vs Google) via env var. The Protocol lets the
worker code depend on a uniform interface; provider-specific shape
conversion (Anthropic tool_use blocks vs Gemini response_schema +
application/json) lives inside the implementations.

Selection: read the model name from `shared.constants` (defaults) but
allow `shared.config.Settings` env overrides at runtime — that's how we
flip a stage from Haiku → Flash Lite without code deploy.

Default behavior matches the pre-redesign pipeline (Anthropic for all
three stages). Set `WIKI_TRIAGE_MODEL`, `WIKI_SYNTHESIS_MODEL`, or
`WIKI_VERIFIER_MODEL` env vars to flip.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Protocol

from anthropic import AsyncAnthropic

from services.synthesis.models import (
    SynthesisInput,
    SynthesisOutput,
    TriageInput,
    TriageOutput,
    VerifierInput,
    VerifierOutput,
)
from services.synthesis.prompts import (
    build_synthesis_prompt,
    build_triage_prompt,
    build_verifier_prompt,
    synthesis_tool_name,
    triage_tool_name,
    verifier_tool_name,
)
from shared.config import get_settings
from shared.constants import (
    HAIKU_MODEL,
    SONNET_MODEL,
    WIKI_SYNTHESIS_MODEL,
    WIKI_TRIAGE_MODEL,
    WIKI_VERIFIER_MODEL,
)
from shared.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Errors — shared across providers so call sites match one type
# ---------------------------------------------------------------------------


class TriageParseError(RuntimeError):
    """Provider returned output we couldn't parse into TriageOutput."""


class SynthesisParseError(RuntimeError):
    """Provider returned output we couldn't parse into SynthesisOutput."""


class VerifierParseError(RuntimeError):
    """Provider returned output we couldn't parse into VerifierOutput."""


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class TriageProvider(Protocol):
    async def triage(self, events: list[TriageInput], *, now: datetime) -> TriageOutput: ...


class SynthesisProvider(Protocol):
    async def synthesize(self, cluster: SynthesisInput, *, now: datetime) -> SynthesisOutput: ...


class VerifierProvider(Protocol):
    async def verify(self, cluster: VerifierInput, *, now: datetime) -> VerifierOutput: ...


# ---------------------------------------------------------------------------
# Provider name resolution
# ---------------------------------------------------------------------------


_ANTHROPIC_TRIAGE_NAMES = {"haiku", "claude-haiku", HAIKU_MODEL}
_ANTHROPIC_SYNTH_NAMES = {"sonnet", "claude-sonnet", SONNET_MODEL}
_ANTHROPIC_VERIFIER_NAMES = _ANTHROPIC_SYNTH_NAMES
_GEMINI_FLASH_LITE_NAMES = {
    "gemini-flash-lite",
    "gemini-flash-lite-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
}
_GEMINI_PRO_NAMES = {
    "gemini-pro",
    "gemini-3.1-pro",
    "gemini-3.1-pro-preview",
}


def _resolve_triage_model() -> str:
    settings = get_settings()
    return getattr(settings, "wiki_triage_model", None) or WIKI_TRIAGE_MODEL


def _resolve_synthesis_model() -> str:
    settings = get_settings()
    return getattr(settings, "wiki_synthesis_model", None) or WIKI_SYNTHESIS_MODEL


def _resolve_verifier_model() -> str:
    settings = get_settings()
    return getattr(settings, "wiki_verifier_model", None) or WIKI_VERIFIER_MODEL


# ---------------------------------------------------------------------------
# Anthropic implementations (tool_use blocks)
# ---------------------------------------------------------------------------


def _extract_tool_use_input(
    blocks: list[Any], *, expected_name: str, error_cls: type[RuntimeError]
) -> dict[str, Any]:
    for block in blocks:
        if getattr(block, "type", "") == "tool_use" and getattr(block, "name", "") == expected_name:
            payload = getattr(block, "input", None)
            if isinstance(payload, dict):
                return payload
            raise error_cls(f"tool_use input was not a dict: {type(payload).__name__}")
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
            raise TriageParseError(f"triage tool input failed validation: {exc}") from exc


class _AnthropicSynthesis:
    """Anthropic Sonnet via tool_use forced output."""

    def __init__(self, client: AsyncAnthropic, *, model: str = SONNET_MODEL) -> None:
        self._client = client
        self._model = model

    async def synthesize(self, cluster: SynthesisInput, *, now: datetime) -> SynthesisOutput:
        kwargs = build_synthesis_prompt(cluster, now=now)
        resp = await self._client.messages.create(model=self._model, **kwargs)
        payload = _extract_tool_use_input(
            resp.content,
            expected_name=synthesis_tool_name(),
            error_cls=SynthesisParseError,
        )
        try:
            return SynthesisOutput(**payload)
        except Exception as exc:
            raise SynthesisParseError(f"synthesis tool input failed validation: {exc}") from exc


class _AnthropicVerifier:
    """Anthropic Sonnet running the verifier prompt."""

    def __init__(self, client: AsyncAnthropic, *, model: str = SONNET_MODEL) -> None:
        self._client = client
        self._model = model

    async def verify(self, cluster: VerifierInput, *, now: datetime) -> VerifierOutput:
        kwargs = build_verifier_prompt(cluster, now=now)
        resp = await self._client.messages.create(model=self._model, **kwargs)
        payload = _extract_tool_use_input(
            resp.content,
            expected_name=verifier_tool_name(),
            error_cls=VerifierParseError,
        )
        try:
            return VerifierOutput(**payload)
        except Exception as exc:
            raise VerifierParseError(f"verifier tool input failed validation: {exc}") from exc


# ---------------------------------------------------------------------------
# Gemini implementations (response_schema + application/json)
# ---------------------------------------------------------------------------


def _strip_keys_recursive(schema: Any, keys: tuple[str, ...]) -> Any:
    """Drop dict keys that Google's response_schema rejects (e.g.
    `additionalProperties`, `pattern`). Mirrors
    `services/retrieval/synthesis.py:_strip_keys_recursive`.
    """
    if isinstance(schema, dict):
        return {k: _strip_keys_recursive(v, keys) for k, v in schema.items() if k not in keys}
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
        raise RuntimeError("google-genai not installed; cannot use Gemini provider") from exc
    api_key = get_settings().google_api_key.get_secret_value()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not configured for Gemini provider")
    return genai.Client(api_key=api_key)


def _gemini_call_json(
    *,
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    max_tokens: int,
) -> Any:
    """Issue one Gemini structured-output call. Returns the parsed dict.

    Marked as a thin helper so the three Gemini provider classes share
    error handling and schema sanitization. Caller is responsible for
    Pydantic validation.
    """

    async def _go() -> dict[str, Any]:
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
            raise RuntimeError(f"gemini response was not a JSON object: {type(parsed).__name__}")
        return parsed

    return _go()


def _flatten_anthropic_kwargs(kwargs: dict[str, Any]) -> tuple[str, str, dict[str, Any], int]:
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
            raise TriageParseError(f"gemini triage output failed validation: {exc}") from exc


class _GeminiSynthesis:
    """Gemini 3.1 Pro Preview synthesis via response_schema."""

    def __init__(self, *, model: str = "gemini-3.1-pro-preview") -> None:
        self._model = model

    async def synthesize(self, cluster: SynthesisInput, *, now: datetime) -> SynthesisOutput:
        kwargs = build_synthesis_prompt(cluster, now=now)
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
            raise SynthesisParseError(f"gemini synthesis call failed: {exc}") from exc
        try:
            return SynthesisOutput(**payload)
        except Exception as exc:
            raise SynthesisParseError(f"gemini synthesis output failed validation: {exc}") from exc


class _GeminiVerifier:
    """Gemini 3.1 Pro Preview running the verifier prompt."""

    def __init__(self, *, model: str = "gemini-3.1-pro-preview") -> None:
        self._model = model

    async def verify(self, cluster: VerifierInput, *, now: datetime) -> VerifierOutput:
        kwargs = build_verifier_prompt(cluster, now=now)
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
            raise VerifierParseError(f"gemini verifier call failed: {exc}") from exc
        try:
            return VerifierOutput(**payload)
        except Exception as exc:
            raise VerifierParseError(f"gemini verifier output failed validation: {exc}") from exc


# ---------------------------------------------------------------------------
# Factory functions used by the workers
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
    name = (model_override or _resolve_triage_model()).lower()
    if name in _GEMINI_FLASH_LITE_NAMES:
        return _GeminiTriage(
            model=name if name.startswith("gemini") else "gemini-flash-lite-preview"
        )
    if name in _ANTHROPIC_TRIAGE_NAMES:
        if anthropic_client is None:
            raise ValueError("Anthropic triage requires an AsyncAnthropic client")
        return _AnthropicTriage(anthropic_client, model=HAIKU_MODEL)
    raise ValueError(f"unknown WIKI_TRIAGE_MODEL: {name}")


def get_synthesis_provider(
    anthropic_client: AsyncAnthropic | None = None,
    *,
    model_override: str | None = None,
) -> SynthesisProvider:
    name = (model_override or _resolve_synthesis_model()).lower()
    if name in _GEMINI_PRO_NAMES:
        return _GeminiSynthesis(
            model=name if name.startswith("gemini") else "gemini-3.1-pro-preview"
        )
    if name in _ANTHROPIC_SYNTH_NAMES:
        if anthropic_client is None:
            raise ValueError("Anthropic synthesis requires an AsyncAnthropic client")
        return _AnthropicSynthesis(anthropic_client, model=SONNET_MODEL)
    raise ValueError(f"unknown WIKI_SYNTHESIS_MODEL: {name}")


def get_verifier_provider(
    anthropic_client: AsyncAnthropic | None = None,
    *,
    model_override: str | None = None,
) -> VerifierProvider:
    name = (model_override or _resolve_verifier_model()).lower()
    if name in _GEMINI_PRO_NAMES:
        return _GeminiVerifier(
            model=name if name.startswith("gemini") else "gemini-3.1-pro-preview"
        )
    if name in _ANTHROPIC_VERIFIER_NAMES:
        if anthropic_client is None:
            raise ValueError("Anthropic verifier requires an AsyncAnthropic client")
        return _AnthropicVerifier(anthropic_client, model=SONNET_MODEL)
    raise ValueError(f"unknown WIKI_VERIFIER_MODEL: {name}")
