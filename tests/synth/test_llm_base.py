"""Tests for LlmClientProtocol, Provider routing, LlmRequest, LlmResponse."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel

from scripts.synth.llm.base import (
    LlmClientProtocol,
    LlmRequest,
    LlmResponse,
    Provider,
    provider_from_model,
)


def test_provider_from_model_anthropic() -> None:
    assert provider_from_model("claude-opus-4-7") == Provider.ANTHROPIC
    assert provider_from_model("claude-3-5-sonnet-20241022") == Provider.ANTHROPIC


def test_provider_from_model_gemini() -> None:
    assert provider_from_model("gemini-2.5-pro") == Provider.GEMINI
    assert provider_from_model("gemini-1.5-flash") == Provider.GEMINI


def test_provider_from_model_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown model prefix"):
        provider_from_model("gpt-4o")


def test_llm_request_defaults() -> None:
    req = LlmRequest(model="claude-opus-4-7", system="sys", prompt="hi")
    assert req.max_tokens == 2048
    assert req.temperature == 0.0


def test_llm_request_frozen() -> None:
    req = LlmRequest(model="claude-opus-4-7", system="sys", prompt="hi")
    with pytest.raises(FrozenInstanceError):
        req.prompt = "mutated"  # type: ignore[misc]


def test_llm_response_frozen() -> None:
    resp = LlmResponse(text="hello")
    with pytest.raises(FrozenInstanceError):
        resp.text = "mutated"  # type: ignore[misc]


def test_protocol_satisfied_by_stub() -> None:
    """A class with the right async methods satisfies LlmClientProtocol structurally."""

    class _FakeClient:
        async def generate(self, req: LlmRequest) -> LlmResponse:
            return LlmResponse(text="fake")

        async def generate_structured(self, req: LlmRequest, schema: type[BaseModel]) -> dict:
            return {}

        async def close(self) -> None:
            pass

    client = _FakeClient()
    assert isinstance(client, LlmClientProtocol)
