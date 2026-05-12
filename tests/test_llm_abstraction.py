"""Unit tests for `shared/llm.py` — the LiteLLM abstraction layer.

These tests do NOT make real network calls. We patch
``litellm.acompletion`` and ``litellm.aembedding`` to assert the wrapper
forwards model + messages + input verbatim, honors ``LLM_GATEWAY_URL``,
and translates LiteLLM errors into the stable ``LLMError`` shape.

Phase 0a contract being verified:
  1. acompletion forwards (model, messages, **kwargs) to litellm
  2. aembedding forwards (model, input, **kwargs) to litellm
  3. LLM_GATEWAY_URL env var sets api_base when caller doesn't override
  4. Caller-supplied api_base wins over the env var
  5. LiteLLM errors are wrapped in LLMError with __cause__ preserved
  6. status_code / provider attributes flow through onto LLMError
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from litellm.exceptions import APIError, RateLimitError

from shared import llm
from shared.llm import LLMError, acompletion, aembedding, gateway_url

# ---------------------------------------------------------------------------
# acompletion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acompletion_forwards_model_and_messages() -> None:
    sentinel = object()
    fake = AsyncMock(return_value=sentinel)
    messages = [{"role": "user", "content": "hello"}]
    with patch.object(llm.litellm, "acompletion", fake):
        result = await acompletion(
            "anthropic/claude-sonnet-4-6",
            messages,
            max_tokens=64,
        )
    assert result is sentinel
    fake.assert_awaited_once()
    kwargs = fake.await_args.kwargs
    assert kwargs["model"] == "anthropic/claude-sonnet-4-6"
    assert kwargs["messages"] == messages
    assert kwargs["max_tokens"] == 64


@pytest.mark.asyncio
async def test_acompletion_passes_through_tools_and_tool_choice() -> None:
    """Tool-use kwargs survive the wrapper untouched.

    Production call sites (router, triage, claude_code_extraction) pass
    `tools=[...]` + `tool_choice={...}`. Regressing this would break
    structured-output routing silently.
    """
    fake = AsyncMock(return_value="resp")
    tools = [{"name": "route_query", "input_schema": {"type": "object"}}]
    tool_choice = {"type": "tool", "name": "route_query"}
    with patch.object(llm.litellm, "acompletion", fake):
        await acompletion(
            "openai/gpt-4o-mini",
            [{"role": "user", "content": "x"}],
            tools=tools,
            tool_choice=tool_choice,
        )
    kwargs = fake.await_args.kwargs
    assert kwargs["tools"] is tools
    assert kwargs["tool_choice"] is tool_choice


# ---------------------------------------------------------------------------
# aembedding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aembedding_forwards_model_and_input() -> None:
    sentinel = object()
    fake = AsyncMock(return_value=sentinel)
    inputs = ["chunk one", "chunk two"]
    with patch.object(llm.litellm, "aembedding", fake):
        result = await aembedding(
            "openai/text-embedding-3-large",
            inputs,
            dimensions=3072,
        )
    assert result is sentinel
    fake.assert_awaited_once()
    kwargs = fake.await_args.kwargs
    assert kwargs["model"] == "openai/text-embedding-3-large"
    assert kwargs["input"] == inputs
    assert kwargs["dimensions"] == 3072


# ---------------------------------------------------------------------------
# LLM_GATEWAY_URL routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_url_sets_api_base_for_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://customer-proxy.example.com")
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "acompletion", fake):
        await acompletion(
            "anthropic/claude-sonnet-4-6",
            [{"role": "user", "content": "x"}],
        )
    assert fake.await_args.kwargs["api_base"] == "https://customer-proxy.example.com"


@pytest.mark.asyncio
async def test_gateway_url_sets_api_base_for_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://customer-proxy.example.com")
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "aembedding", fake):
        await aembedding("openai/text-embedding-3-large", ["x"])
    assert fake.await_args.kwargs["api_base"] == "https://customer-proxy.example.com"


@pytest.mark.asyncio
async def test_no_gateway_url_means_no_api_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "acompletion", fake):
        await acompletion("anthropic/claude-sonnet-4-6", [])
    assert "api_base" not in fake.await_args.kwargs


@pytest.mark.asyncio
async def test_empty_gateway_url_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A self-host installer that explicitly clears the var (sets it to '')
    should fall back to direct provider calls, not pass api_base=''."""
    monkeypatch.setenv("LLM_GATEWAY_URL", "")
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "acompletion", fake):
        await acompletion("anthropic/claude-sonnet-4-6", [])
    assert "api_base" not in fake.await_args.kwargs
    assert gateway_url() is None


@pytest.mark.asyncio
async def test_caller_api_base_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit per-call ``api_base`` override beats the global env var.
    Eval harness pattern: point at a staging gateway without unsetting
    the production env var."""
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://prod-proxy.example.com")
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "acompletion", fake):
        await acompletion(
            "anthropic/claude-sonnet-4-6",
            [],
            api_base="https://staging-proxy.example.com",
        )
    assert (
        fake.await_args.kwargs["api_base"] == "https://staging-proxy.example.com"
    )


# ---------------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acompletion_wraps_litellm_error() -> None:
    original = APIError(
        status_code=502,
        message="bad gateway from upstream",
        llm_provider="anthropic",
        model="claude-sonnet-4-6",
    )
    fake = AsyncMock(side_effect=original)
    with patch.object(llm.litellm, "acompletion", fake), pytest.raises(LLMError) as exc_info:
        await acompletion("anthropic/claude-sonnet-4-6", [])
    err = exc_info.value
    assert "bad gateway from upstream" in str(err)
    assert err.status_code == 502
    assert err.provider == "anthropic"
    # Original LiteLLM exception preserved on __cause__ for callers that
    # need provider-specific handling.
    assert err.__cause__ is original


@pytest.mark.asyncio
async def test_aembedding_wraps_rate_limit_error() -> None:
    original = RateLimitError(
        message="rate limited",
        llm_provider="openai",
        model="text-embedding-3-large",
    )
    fake = AsyncMock(side_effect=original)
    with patch.object(llm.litellm, "aembedding", fake), pytest.raises(LLMError) as exc_info:
        await aembedding("openai/text-embedding-3-large", ["x"])
    err = exc_info.value
    assert err.provider == "openai"
    assert err.__cause__ is original


@pytest.mark.asyncio
async def test_acompletion_wraps_unexpected_error() -> None:
    """Belt-and-suspenders: a non-LiteLLM exception (e.g. transport
    error from httpx, JSON decode error from a malformed response)
    still surfaces as ``LLMError`` so call sites have one type to
    catch."""

    class WeirdLeakedException(Exception):
        pass

    fake = AsyncMock(side_effect=WeirdLeakedException("transport blew up"))
    with patch.object(llm.litellm, "acompletion", fake), pytest.raises(LLMError) as exc_info:
        await acompletion("anthropic/claude-sonnet-4-6", [])
    assert "transport blew up" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, WeirdLeakedException)
    # Unknown errors don't carry status_code/provider — that's fine.
    assert exc_info.value.status_code is None
    assert exc_info.value.provider is None


# ---------------------------------------------------------------------------
# gateway_url helper
# ---------------------------------------------------------------------------


def test_gateway_url_reads_env_each_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    assert gateway_url() is None
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://x.example.com")
    assert gateway_url() == "https://x.example.com"
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    assert gateway_url() is None


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_public_surface() -> None:
    """Lock the public surface so accidental additions show up in review."""
    assert set(llm.__all__) == {
        "LLMError",
        "acompletion",
        "aembedding",
        "gateway_key",
        "gateway_url",
    }


def test_llmerror_default_attrs() -> None:
    err = LLMError("x")
    assert err.status_code is None
    assert err.provider is None
    assert str(err) == "x"


# Importing this in case future call-site tests want to swap in a stub
# `litellm.acompletion`. Marking `Any` here keeps mypy happy.
_unused: Any = None
