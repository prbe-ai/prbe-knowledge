"""AnthropicClient — replaces Plan 1's LlmClient with provider-aware shape.

Adds generate_structured() via tool_use API and optional cache_control
injection on the system prompt for Anthropic prompt caching.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from scripts.synth.llm.base import LlmRequest, LlmResponse


def _coerce_tool_use_output(raw: dict[str, Any]) -> dict[str, Any]:
    """Defensively normalize Anthropic tool_use input dicts before schema validation.

    The Anthropic tool_use API has two recurring serialization quirks
    (most often surfaced by haiku-4-5 but occasionally by other models)
    that produce dicts that don't pass downstream Pydantic validation
    even when the model's intent was correct:

    1. Whole-response wrap: the model returns ``{"parameter": {<actual>}}``
       (or ``{"input": {<actual>}}``) instead of returning the schema
       fields at the top level. Defensively unwrap when the dict has
       exactly one key matching that pattern and the value is itself a
       dict.

    2. Stringified nested fields: the model returns scalar fields as
       expected but serializes a complex field (list / dict) as a JSON
       string instead of a native value, e.g.
       ``{"passed": false, "violations": "[{...}, {...}]"}``. Try to
       ``json.loads`` any top-level string value that LOOKS like JSON
       (starts with ``[`` or ``{``); leave non-JSON strings alone.

    Both quirks are reproducible against the live API and have shown up
    in real-LLM canonical recording runs.
    """
    # (1) unwrap envelope
    if isinstance(raw, dict) and len(raw) == 1:
        key = next(iter(raw))
        if key in {"parameter", "input"} and isinstance(raw[key], dict):
            raw = raw[key]

    # (2) coerce stringified JSON values for top-level fields
    if isinstance(raw, dict):
        coerced: dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, str):
                stripped = v.lstrip()
                if stripped.startswith(("[", "{")):
                    try:
                        coerced[k] = json.loads(v)
                        continue
                    except json.JSONDecodeError:
                        pass
            coerced[k] = v
        raw = coerced

    return raw


class AnthropicClient:
    """Real Anthropic-backed client implementing LlmClientProtocol."""

    def __init__(self, api_key: str) -> None:
        self._client = AsyncAnthropic(api_key=api_key)

    def _system_arg(self, system: str) -> str | list[dict[str, Any]]:
        """Return system prompt with cache_control if non-empty, else plain string."""
        if not system:
            return system
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    def _create_kwargs(self, req: LlmRequest) -> dict[str, Any]:
        """Build messages.create kwargs.

        Anthropic's newer reasoning models (claude-opus-4-7+) deprecated the
        temperature parameter entirely and reject any request that includes it
        — they pick an internal temperature that callers can't override. We
        therefore drop ``req.temperature`` here unconditionally; the
        determinism contract for synthetic data on these models comes from
        seed-driven prompt content + prompt cache, not from a temperature knob
        callers can set. Older models (e.g. sonnet-4-6, haiku-4-5) accept
        temperature but happily run without it too, so always-omit keeps the
        wire shape consistent across the roster.
        """
        return {
            "model": req.model,
            "system": self._system_arg(req.system),
            "max_tokens": req.max_tokens,
            "messages": [{"role": "user", "content": req.prompt}],
        }

    async def generate(self, req: LlmRequest) -> LlmResponse:
        msg = await self._client.messages.create(**self._create_kwargs(req))
        text_parts: list[str] = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        return LlmResponse(text="".join(text_parts))

    async def generate_structured(self, req: LlmRequest, schema: type[BaseModel]) -> dict:
        """Call Anthropic tool_use API and return the tool_use block's input dict."""
        json_schema = schema.model_json_schema()
        tool_def = {
            "name": "structured_output",
            "description": "Return structured output conforming to the provided JSON schema.",
            "input_schema": json_schema,
        }
        msg = await self._client.messages.create(
            **self._create_kwargs(req),
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "structured_output"},
        )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                return _coerce_tool_use_output(dict(getattr(block, "input", {})))
        raise ValueError("Anthropic response contained no tool_use block")

    async def close(self) -> None:
        with contextlib.suppress(AttributeError):
            await self._client.aclose()  # type: ignore[attr-defined]
