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

from engine.shared import llm
from engine.shared.llm import LLMError, acompletion, aembedding, gateway_url

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


@pytest.mark.parametrize(
    "model",
    ["gemini-3.6-flash", "gemini-3.5-flash-lite"],
)
def test_latest_gemini_models_do_not_reinsert_temperature(model: str) -> None:
    """Pin the provider mapping where LiteLLM adds a default temperature."""
    mapped = llm.litellm.get_optional_params(
        model=model,
        custom_llm_provider="gemini",
        max_tokens=600,
        drop_params=True,
    )

    assert mapped["max_output_tokens"] == 600
    assert "temperature" not in mapped


def test_gemini_sampling_shim_is_idempotent_and_model_scoped() -> None:
    """Installing twice must not wrap again or alter older Gemini models."""
    mapper = llm.litellm.GoogleAIStudioGeminiConfig.map_openai_params

    llm._install_gemini_sampling_compatibility_shim()

    assert llm.litellm.GoogleAIStudioGeminiConfig.map_openai_params is mapper
    legacy_mapped = llm.litellm.get_optional_params(
        model="gemini-3-flash-preview",
        custom_llm_provider="gemini",
        max_tokens=600,
        drop_params=True,
    )
    assert legacy_mapped["temperature"] == 1.0


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
async def test_gateway_embedding_forces_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression pin (2026-05-14 cutover): when the gateway is set,
    `aembedding` must force `custom_llm_provider="openai"` regardless of
    the model's provider prefix.

    Without this, LiteLLM SDK picks the wire shape from the prefix —
    `gemini/...` builds the Gemini-native URL
    `/v1beta/models/<m>:batchEmbedContents` against the proxy, which
    answers FastAPI 405 because it only serves `/embeddings`. The proxy
    routes the call to the real Gemini upstream via its own `model_list`,
    so the SDK should speak OpenAI HTTP regardless of the model prefix.
    """
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://customer-proxy.example.com")
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "aembedding", fake):
        await aembedding("gemini/gemini-embedding-2", ["x"])
    assert fake.await_args.kwargs["api_base"] == "https://customer-proxy.example.com"
    assert fake.await_args.kwargs["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_no_gateway_does_not_force_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct-provider mode (no gateway): SDK picks the provider from the
    model prefix. We must NOT inject `custom_llm_provider` — that would
    override the SDK's correct gemini routing and break the dev path."""
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "aembedding", fake):
        await aembedding("gemini/gemini-embedding-2", ["x"])
    assert "custom_llm_provider" not in fake.await_args.kwargs


@pytest.mark.asyncio
async def test_caller_api_base_does_not_force_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider override is scoped to env-injected gateway URLs only.
    A caller passing an explicit `api_base` (e.g. pointing at a Vertex AI
    endpoint or a non-LiteLLM proxy) is trusted to pick its own
    `custom_llm_provider`. Without this scoping, the override would
    silently route caller-supplied endpoints through OpenAI wire shape."""
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "aembedding", fake):
        await aembedding(
            "gemini/gemini-embedding-2",
            ["x"],
            api_base="https://my-custom-gemini-endpoint.example.com",
        )
    assert fake.await_args.kwargs["api_base"] == "https://my-custom-gemini-endpoint.example.com"
    assert "custom_llm_provider" not in fake.await_args.kwargs


@pytest.mark.asyncio
async def test_caller_custom_llm_provider_wins_over_gateway_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a caller explicitly passes `custom_llm_provider`, that wins. The
    gateway injection is a default, not an override (mirrors the api_base
    / api_key precedence rules)."""
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://customer-proxy.example.com")
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "aembedding", fake):
        await aembedding(
            "gemini/gemini-embedding-2", ["x"], custom_llm_provider="gemini"
        )
    assert fake.await_args.kwargs["custom_llm_provider"] == "gemini"


@pytest.mark.asyncio
async def test_gateway_completion_strips_gemini_prefix_and_routes_via_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 405 that kept the team wiki empty.

    `gemini/<m>` makes the LiteLLM SDK build the Gemini-NATIVE URL
    `/v1beta/models/<m>:generateContent` and POST it at our proxy, which
    serves no such path and answers FastAPI 405
    `{"detail":"Method Not Allowed"}`. Every wiki triage batch died on
    exactly that, 13,147 rows deep into the DLQ.

    Two assertions, and BOTH are load-bearing:

      * the prefix is STRIPPED. The proxy's `model_list` registers Gemini
        under the bare `gemini-*`, so `gemini/gemini-3.5-flash` reaches it
        as an unknown model and comes back 400 -- a different failure, not
        a fix. Verified against the live gateway.
      * the provider is `litellm_proxy`, NOT `openai`. Both speak the
        OpenAI wire shape, but `openai` validates params locally and
        rejects the `reasoning_effort` the Gemini synthesis callers pass
        (`UnsupportedParamsError`); `litellm_proxy` forwards it.
    """
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://customer-proxy.example.com")
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "acompletion", fake):
        await acompletion(
            "gemini/gemini-3.5-flash",
            [{"role": "user", "content": "x"}],
        )
    kwargs = fake.await_args.kwargs
    assert kwargs["model"] == "gemini-3.5-flash"
    assert kwargs["custom_llm_provider"] == "litellm_proxy"


@pytest.mark.asyncio
async def test_gateway_completion_preserves_proxy_registered_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`cerebras/` is NOT an SDK routing hint -- it is the model's name on
    the proxy (`model_list` carries `cerebras/*`, verified against the live
    gateway's `GET /v1/models`). Stripping it would turn a working call
    into an unknown-model 400.

    This is the test that stops the fix from being generalised into
    "strip every prefix", which reads tidier and is wrong.
    """
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://customer-proxy.example.com")
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "acompletion", fake):
        await acompletion(
            "cerebras/gpt-oss-120b",
            [{"role": "user", "content": "x"}],
        )
    kwargs = fake.await_args.kwargs
    assert kwargs["model"] == "cerebras/gpt-oss-120b"
    assert "custom_llm_provider" not in kwargs


@pytest.mark.asyncio
async def test_no_gateway_completion_keeps_gemini_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct-provider mode: `gemini/` is exactly right and the SDK's
    native routing is what we want. Rewriting here would break the dev and
    bring-your-own-key paths, where there is no proxy to route through."""
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "acompletion", fake):
        await acompletion(
            "gemini/gemini-3.5-flash",
            [{"role": "user", "content": "x"}],
        )
    kwargs = fake.await_args.kwargs
    assert kwargs["model"] == "gemini/gemini-3.5-flash"
    assert "custom_llm_provider" not in kwargs


@pytest.mark.asyncio
async def test_caller_named_provider_keeps_its_model_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller that named a provider has chosen its own wire shape, and the
    normalization must not second-guess it -- rewriting the model underneath a
    declared provider would silently retarget the call."""
    # THE GATEWAY IS SET. Without it the normalization returns on its first
    # guard ("no proxy in play") and this test passes without ever reaching the
    # branch it claims to cover -- verified: deleting the caller-provider guard
    # left an earlier version of this test green.
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://customer-proxy.example.com")
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "acompletion", fake):
        await acompletion(
            "gemini/gemini-3.5-flash",
            [{"role": "user", "content": "x"}],
            custom_llm_provider="gemini",
        )
    kwargs = fake.await_args.kwargs
    assert kwargs["model"] == "gemini/gemini-3.5-flash"
    assert kwargs["custom_llm_provider"] == "gemini"


@pytest.mark.asyncio
async def test_caller_custom_llm_provider_wins_for_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller that named a provider has already decided its wire shape;
    the rewrite is a default, not an override. The retrieval gatherer
    passes `custom_llm_provider="openai"` deliberately and must keep the
    model string it asked for."""
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://customer-proxy.example.com")
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "acompletion", fake):
        await acompletion(
            "gemini/gemini-3.5-flash",
            [{"role": "user", "content": "x"}],
            custom_llm_provider="openai",
        )
    kwargs = fake.await_args.kwargs
    assert kwargs["model"] == "gemini/gemini-3.5-flash"
    assert kwargs["custom_llm_provider"] == "openai"


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


# ---------------------------------------------------------------------------
# Provider token-rate limiter
# ---------------------------------------------------------------------------
# Cerebras enforces a tokens-per-minute ceiling at the ORGANIZATION level and
# reserves `input + max_completion_tokens` before running a request, so a
# burst of concurrent searches 429s even when average usage is far below the
# limit. The limiter smooths that burst; these tests pin the two properties
# that make it safe to ship enabled-by-config.


def _reset_limiters() -> None:
    llm._TPM_LIMITERS.clear()


def test_limiter_disabled_by_default_is_a_no_op() -> None:
    """Budget 0 means no limiter object is ever created.

    Default-off matters: a self-host install with its own provider account
    has no shared quota to protect, and throttling it would be a pure
    regression.
    """
    _reset_limiters()
    with patch.object(llm, "LLM_TPM_BUDGET", 0):
        assert llm._get_tpm_limiter() is None


def test_estimate_includes_the_reserved_completion_allowance() -> None:
    """The estimate counts max_tokens, not just the prompt.

    The provider reserves the completion allowance UP FRONT — that
    reservation is the entire reason concurrent searches trip the quota, so
    a limiter that ignored it would model the wrong resource.
    """
    messages = [{"role": "user", "content": "x" * 4000}]
    without = llm._estimate_request_tokens(messages, {})
    with_cap = llm._estimate_request_tokens(messages, {"max_tokens": 5000})
    assert with_cap - without == 5000


@pytest.mark.asyncio
async def test_limiter_fails_open_rather_than_outlasting_the_caller() -> None:
    """Budget exhausted → proceed anyway, bounded by max wait.

    THIS IS THE SAFETY PROPERTY. Blocking longer than the caller's own
    deadline converts a fast 429 (which degrades to pre-fan-out evidence)
    into a slow timeout (which returns nothing) — strictly worse for the
    user. The gatherer's turn deadline is 20s and research-os abandons
    /v1/search at 30s, so the limiter must never become the long pole.
    """
    import time

    _reset_limiters()
    messages = [{"role": "user", "content": "x" * 400}]
    with patch.object(llm, "LLM_TPM_BUDGET", 1000), patch.object(
        llm, "LLM_TPM_MAX_WAIT_SECONDS", 0.05
    ):
        # Drain the bucket, then a second acquire cannot be satisfied within
        # the wait window. It must still return, not raise and not hang.
        await llm._acquire_token_budget(messages, {"max_tokens": 800})
        started = time.monotonic()
        await llm._acquire_token_budget(messages, {"max_tokens": 800})
        waited = time.monotonic() - started
    # Bounded by the max wait: it genuinely blocked (so the test is not
    # trivially passing on an un-drained bucket) but gave up promptly rather
    # than queueing for the ~54s the bucket would actually need to refill.
    assert waited >= 0.04, "budget was not exhausted; test proved nothing"
    assert waited < 1.0, f"limiter outlasted its max wait ({waited:.2f}s)"


@pytest.mark.asyncio
async def test_request_larger_than_the_whole_budget_is_let_through() -> None:
    """A single request bigger than the per-minute budget can never be
    satisfied; acquiring would just burn the max wait before proceeding
    anyway. Skip straight through and let the provider decide."""
    _reset_limiters()
    messages = [{"role": "user", "content": "x" * 40000}]
    with patch.object(llm, "LLM_TPM_BUDGET", 100), patch.object(
        llm, "LLM_TPM_MAX_WAIT_SECONDS", 30.0
    ):
        # Would block for 30s if the oversize guard were missing.
        await llm._acquire_token_budget(messages, {})


@pytest.mark.asyncio
async def test_acompletion_still_calls_provider_when_limiter_enabled() -> None:
    """The limiter sits in front of the call without changing its contract."""
    _reset_limiters()
    fake = AsyncMock(return_value="ok")
    with patch.object(llm, "LLM_TPM_BUDGET", 1_000_000), patch(
        "litellm.acompletion", new=fake
    ):
        out = await acompletion(model="m", messages=[{"role": "user", "content": "hi"}])
    assert out == "ok"
    assert fake.await_count == 1
