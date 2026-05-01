"""LLM client. Plan 1 only needs basic single-shot generate() and a
StaticLlmClient used by tests / CompanyContext auto-inferrer.

Plan 3 will extend this with prompt-cache control blocks (Anthropic SDK
cache_control), retries via tenacity, and a fixture-keyed mock client
for the `--mock-llm` CLI flag.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Protocol

from anthropic import AsyncAnthropic


@dataclass(frozen=True)
class LlmRequest:
    model: str
    system: str
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.4


@dataclass(frozen=True)
class LlmResponse:
    text: str


class LlmClientProtocol(Protocol):
    async def generate(self, req: LlmRequest) -> LlmResponse: ...
    async def close(self) -> None: ...


class LlmClient:
    """Real Anthropic-backed client."""

    def __init__(self, api_key: str) -> None:
        self._client = AsyncAnthropic(api_key=api_key)

    async def generate(self, req: LlmRequest) -> LlmResponse:
        msg = await self._client.messages.create(
            model=req.model,
            system=req.system,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            messages=[{"role": "user", "content": req.prompt}],
        )
        text_parts: list[str] = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        return LlmResponse(text="".join(text_parts))

    async def close(self) -> None:
        with contextlib.suppress(AttributeError):
            await self._client.aclose()  # type: ignore[attr-defined]


class StaticLlmClient:
    """Test client: prompts → canned responses by exact match."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    async def generate(self, req: LlmRequest) -> LlmResponse:
        if req.prompt not in self._mapping:
            raise KeyError(f"no canned response for prompt: {req.prompt!r}")
        return LlmResponse(text=self._mapping[req.prompt])

    async def close(self) -> None:
        return None
