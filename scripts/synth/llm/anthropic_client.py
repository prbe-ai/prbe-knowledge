"""AnthropicClient — replaces Plan 1's LlmClient with provider-aware shape.

Adds generate_structured() via tool_use API and optional cache_control
injection on the system prompt for Anthropic prompt caching.
"""

from __future__ import annotations

import contextlib
from typing import Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from scripts.synth.llm.base import LlmRequest, LlmResponse


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
                return dict(getattr(block, "input", {}))
        raise ValueError("Anthropic response contained no tool_use block")

    async def close(self) -> None:
        with contextlib.suppress(AttributeError):
            await self._client.aclose()  # type: ignore[attr-defined]
