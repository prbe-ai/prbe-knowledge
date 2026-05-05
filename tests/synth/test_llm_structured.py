"""Tests for the provider-agnostic structured output adapter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from scripts.synth.llm.base import LlmRequest, LlmResponse
from scripts.synth.llm.structured import StructuredOutputValidationError, generate_typed


class _Answer(BaseModel):
    answer: str
    score: int


class _FakeClient:
    def __init__(self, response: dict) -> None:
        self._response = response

    async def generate(self, req: LlmRequest) -> LlmResponse:
        return LlmResponse(text="")

    async def generate_structured(self, req: LlmRequest, schema: type[BaseModel]) -> dict:
        return self._response

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_generate_typed_returns_model_instance() -> None:
    client = _FakeClient({"answer": "42", "score": 9})
    req = LlmRequest(model="claude-opus-4-7", system="", prompt="classify")
    result = await generate_typed(client, req, _Answer)
    assert isinstance(result, _Answer)
    assert result.answer == "42"
    assert result.score == 9


@pytest.mark.asyncio
async def test_generate_typed_raises_on_invalid_dict() -> None:
    client = _FakeClient({"answer": "42"})  # missing required "score"
    req = LlmRequest(model="claude-opus-4-7", system="", prompt="classify")
    with pytest.raises(StructuredOutputValidationError):
        await generate_typed(client, req, _Answer)


@pytest.mark.asyncio
async def test_generate_typed_unknown_fields_lax() -> None:
    """Extra fields in the response dict are ignored by Pydantic's lax mode."""
    client = _FakeClient({"answer": "ok", "score": 1, "extra_field": "ignored"})
    req = LlmRequest(model="claude-opus-4-7", system="", prompt="classify")
    result = await generate_typed(client, req, _Answer)
    assert result.answer == "ok"
