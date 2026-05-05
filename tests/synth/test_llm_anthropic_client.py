"""Tests for AnthropicClient using a mocked AsyncAnthropic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from scripts.synth.llm.anthropic_client import AnthropicClient
from scripts.synth.llm.base import LlmRequest, LlmResponse


class _MySchema(BaseModel):
    answer: str
    score: int


def _make_text_message(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    return msg


def _make_tool_use_message(tool_input: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.input = tool_input
    msg = MagicMock()
    msg.content = [block]
    return msg


@pytest.fixture()
def mock_anthropic_class():
    with patch("scripts.synth.llm.anthropic_client.AsyncAnthropic") as cls:
        instance = MagicMock()
        instance.messages = MagicMock()
        instance.messages.create = AsyncMock()
        instance.aclose = AsyncMock()
        cls.return_value = instance
        yield cls, instance


@pytest.mark.asyncio
async def test_generate_returns_text(mock_anthropic_class) -> None:
    _, mock_client = mock_anthropic_class
    mock_client.messages.create.return_value = _make_text_message("hello world")
    client = AnthropicClient(api_key="test-key")
    req = LlmRequest(model="claude-opus-4-7", system="sys", prompt="hi")
    resp = await client.generate(req)
    assert isinstance(resp, LlmResponse)
    assert resp.text == "hello world"


@pytest.mark.asyncio
async def test_generate_structured_returns_dict(mock_anthropic_class) -> None:
    _, mock_client = mock_anthropic_class
    mock_client.messages.create.return_value = _make_tool_use_message(
        {"answer": "42", "score": 9}
    )
    client = AnthropicClient(api_key="test-key")
    req = LlmRequest(model="claude-opus-4-7", system="sys", prompt="classify this")
    result = await client.generate_structured(req, _MySchema)
    assert result == {"answer": "42", "score": 9}


@pytest.mark.asyncio
async def test_close_calls_aclose(mock_anthropic_class) -> None:
    _, mock_client = mock_anthropic_class
    client = AnthropicClient(api_key="test-key")
    await client.close()
    mock_client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_cache_control_injected_when_system_set(mock_anthropic_class) -> None:
    _, mock_client = mock_anthropic_class
    mock_client.messages.create.return_value = _make_text_message("ok")
    client = AnthropicClient(api_key="test-key")
    req = LlmRequest(model="claude-opus-4-7", system="You are helpful.", prompt="hi")
    await client.generate(req)
    call_kwargs = mock_client.messages.create.call_args.kwargs
    system_arg = call_kwargs["system"]
    assert isinstance(system_arg, list)
    assert system_arg[0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_generate_structured_tool_name_in_call(mock_anthropic_class) -> None:
    _, mock_client = mock_anthropic_class
    mock_client.messages.create.return_value = _make_tool_use_message({"answer": "x", "score": 1})
    client = AnthropicClient(api_key="test-key")
    req = LlmRequest(model="claude-opus-4-7", system="", prompt="classify")
    await client.generate_structured(req, _MySchema)
    call_kwargs = mock_client.messages.create.call_args.kwargs
    tools = call_kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "structured_output"
