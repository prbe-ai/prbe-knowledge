"""LLM provider abstraction — base types, protocol, and routing.

LlmRequest.temperature defaults to 0.0 (Plan 3 determinism contract).
Plan 1's LlmClient used 0.4; this is intentionally tightened.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


def provider_from_model(model_id: str) -> Provider:
    """Derive provider from model name prefix.

    Examples:
        claude-opus-4-7 -> Provider.ANTHROPIC
        gemini-2.5-pro  -> Provider.GEMINI

    Raises ValueError for unknown prefixes (no silent default).
    """
    if model_id.startswith("claude"):
        return Provider.ANTHROPIC
    if model_id.startswith("gemini"):
        return Provider.GEMINI
    raise ValueError(f"unknown model prefix — cannot derive provider from model_id={model_id!r}")


@dataclass(frozen=True)
class LlmRequest:
    model: str
    system: str
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.0


@dataclass(frozen=True)
class LlmResponse:
    text: str


@runtime_checkable
class LlmClientProtocol(Protocol):
    async def generate(self, req: LlmRequest) -> LlmResponse: ...
    async def generate_structured(self, req: LlmRequest, schema: type[BaseModel]) -> dict: ...
    async def close(self) -> None: ...
