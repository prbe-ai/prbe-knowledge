"""Tests for GeminiClient using a mocked google.genai Client."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from scripts.synth.llm.base import LlmRequest, LlmResponse
from scripts.synth.llm.gemini_client import GeminiClient


class _MySchema(BaseModel):
    label: str
    confidence: float


def _make_genai_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    return resp


@pytest.fixture()
def mock_genai_client():
    with patch("scripts.synth.llm.gemini_client.genai") as mock_genai:
        mock_client_instance = MagicMock()
        mock_client_instance.aio = MagicMock()
        mock_client_instance.aio.models = MagicMock()
        mock_client_instance.aio.models.generate_content = AsyncMock()
        mock_genai.Client.return_value = mock_client_instance
        yield mock_genai, mock_client_instance


@pytest.mark.asyncio
async def test_generate_returns_llm_response(mock_genai_client) -> None:
    _, mock_client = mock_genai_client
    mock_client.aio.models.generate_content.return_value = _make_genai_response("result text")
    client = GeminiClient(api_key="test-key")
    req = LlmRequest(model="gemini-2.5-pro", system="You are helpful.", prompt="Say hi")
    resp = await client.generate(req)
    assert isinstance(resp, LlmResponse)
    assert resp.text == "result text"


@pytest.mark.asyncio
async def test_generate_passes_correct_kwargs(mock_genai_client) -> None:
    _, mock_client = mock_genai_client
    mock_client.aio.models.generate_content.return_value = _make_genai_response("ok")
    client = GeminiClient(api_key="test-key")
    req = LlmRequest(model="gemini-2.5-pro", system="sys", prompt="prompt", temperature=0.0)
    await client.generate(req)
    call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-pro"
    assert call_kwargs["contents"] == "prompt"


@pytest.mark.asyncio
async def test_generate_structured_passes_response_schema(mock_genai_client) -> None:
    _, mock_client = mock_genai_client
    mock_client.aio.models.generate_content.return_value = _make_genai_response(
        json.dumps({"label": "positive", "confidence": 0.95})
    )
    client = GeminiClient(api_key="test-key")
    req = LlmRequest(model="gemini-2.5-pro", system="", prompt="classify")
    result = await client.generate_structured(req, _MySchema)
    assert result == {"label": "positive", "confidence": 0.95}
    call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
    config = call_kwargs["config"]
    assert config.response_mime_type == "application/json"
    # The client passes a CLEANED dict (not the Pydantic class) so that
    # Gemini-incompatible JSON-schema fields (additionalProperties) are
    # stripped before request build. See _clean_schema_for_gemini.
    assert isinstance(config.response_schema, dict)
    assert config.response_schema["type"] == "object"
    assert set(config.response_schema["properties"].keys()) == {"label", "confidence"}
    assert config.response_schema["properties"]["label"]["type"] == "string"
    assert config.response_schema["properties"]["confidence"]["type"] == "number"
    assert "additionalProperties" not in config.response_schema


@pytest.mark.asyncio
async def test_generate_temperature_passed_through(mock_genai_client) -> None:
    _, mock_client = mock_genai_client
    mock_client.aio.models.generate_content.return_value = _make_genai_response("ok")
    client = GeminiClient(api_key="test-key")
    req = LlmRequest(model="gemini-2.5-pro", system="", prompt="test", temperature=0.0)
    await client.generate(req)
    call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].temperature == 0.0


@pytest.mark.asyncio
async def test_close_is_safe_noop(mock_genai_client) -> None:
    client = GeminiClient(api_key="test-key")
    await client.close()  # must not raise
