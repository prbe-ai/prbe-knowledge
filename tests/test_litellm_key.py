"""Tests for the per-tenant LiteLLM virtual-key fetcher.

Covers:
  - control-plane fetch happy path (httpx.MockTransport)
  - in-memory TTL cache (second call doesn't refetch)
  - X-Internal-Backend-Key header is set correctly
  - 404 / 5xx / missing config raise LiteLLMKeyUnavailable
  - tenant_virtual_key_context binds the ContextVar for the block
  - shared.llm.acompletion prefers the contextvar key over LLM_GATEWAY_KEY
  - shared.llm.acompletion falls back to LLM_GATEWAY_KEY when no contextvar
  - shared.llm.acompletion sends NO api_key when LLM_GATEWAY_URL is unset
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from shared import litellm_key as litellm_key_mod
from shared import llm
from shared.litellm_key import (
    LiteLLMKeyUnavailable,
    current_tenant_virtual_key,
    get_tenant_virtual_key,
    invalidate_tenant_virtual_key,
    tenant_virtual_key_context,
)


@pytest.fixture(autouse=True)
def _clear_cache_and_env(monkeypatch: pytest.MonkeyPatch):
    """Reset the in-memory key cache and env between tests."""
    litellm_key_mod._KEY_CACHE.clear()
    monkeypatch.setenv("BACKEND_BASE_URL", "http://prbe-backend.internal:8080")
    monkeypatch.setenv("INTERNAL_BACKEND_API_KEY", "test-internal-key")
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_KEY", raising=False)
    # Clear the pydantic-settings cache so the new env vars are picked up.
    from shared.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_happy_path() -> None:
    """Control plane returns a virtual key; X-Internal-Backend-Key header is set."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"litellm_key": "sk-virtual-cust-abc"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        key = await get_tenant_virtual_key("cust-abc", http=http)

    assert key == "sk-virtual-cust-abc"
    assert captured["url"].endswith("/routing/customer/cust-abc/litellm-key")
    assert captured["headers"]["x-internal-backend-key"] == "test-internal-key"


@pytest.mark.asyncio
async def test_fetch_is_cached_within_ttl() -> None:
    """A second call with same customer_id doesn't re-hit the control plane."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json={"litellm_key": "sk-v1"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        a = await get_tenant_virtual_key("cust-1", http=http)
        b = await get_tenant_virtual_key("cust-1", http=http)

    assert a == b == "sk-v1"
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_fetch_invalidate_forces_refetch() -> None:
    """`invalidate_tenant_virtual_key` clears the cache for one tenant."""
    keys = iter(["sk-v1", "sk-v2"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"litellm_key": next(keys)})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        a = await get_tenant_virtual_key("cust-1", http=http)
        invalidate_tenant_virtual_key("cust-1")
        b = await get_tenant_virtual_key("cust-1", http=http)

    assert a == "sk-v1"
    assert b == "sk-v2"


@pytest.mark.asyncio
async def test_fetch_404_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="no key for customer")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(LiteLLMKeyUnavailable, match="no LiteLLM key"):
            await get_tenant_virtual_key("cust-missing", http=http)


@pytest.mark.asyncio
async def test_fetch_5xx_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="control plane unavailable")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(LiteLLMKeyUnavailable, match="503"):
            await get_tenant_virtual_key("cust-1", http=http)


@pytest.mark.asyncio
async def test_fetch_missing_config_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKEND_BASE_URL", "")
    monkeypatch.setenv("INTERNAL_BACKEND_API_KEY", "")
    from shared.config import get_settings

    get_settings.cache_clear()

    with pytest.raises(LiteLLMKeyUnavailable, match="not configured"):
        await get_tenant_virtual_key("cust-1")


@pytest.mark.asyncio
async def test_fetch_empty_customer_id_raises() -> None:
    with pytest.raises(LiteLLMKeyUnavailable, match="customer_id is required"):
        await get_tenant_virtual_key("")


@pytest.mark.asyncio
async def test_fetch_response_missing_key_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"other_field": "x"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(LiteLLMKeyUnavailable, match="missing 'litellm_key'"):
            await get_tenant_virtual_key("cust-1", http=http)


# ---------------------------------------------------------------------------
# tenant_virtual_key_context — ContextVar plumbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_binds_and_unbinds_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"litellm_key": "sk-cust-xyz"})

    transport = httpx.MockTransport(handler)
    assert current_tenant_virtual_key() is None
    async with (
        httpx.AsyncClient(transport=transport) as http,
        tenant_virtual_key_context("cust-xyz", http=http) as key,
    ):
        assert key == "sk-cust-xyz"
        assert current_tenant_virtual_key() == "sk-cust-xyz"
    # Always unset on block exit.
    assert current_tenant_virtual_key() is None


@pytest.mark.asyncio
async def test_context_unbinds_on_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"litellm_key": "sk-cust-xyz"})

    transport = httpx.MockTransport(handler)
    with pytest.raises(RuntimeError):
        async with httpx.AsyncClient(transport=transport) as http:
            async with tenant_virtual_key_context("cust-xyz", http=http):
                raise RuntimeError("boom")
    assert current_tenant_virtual_key() is None


# ---------------------------------------------------------------------------
# shared.llm integration: contextvar key beats env-var key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acompletion_uses_tenant_key_over_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a tenant key is bound on the ContextVar, it wins over
    ``LLM_GATEWAY_KEY``. Hot-path per-tenant attribution requires this."""
    monkeypatch.setenv("LLM_GATEWAY_URL", "http://litellm.litellm.svc.cluster.local:4000")
    monkeypatch.setenv("LLM_GATEWAY_KEY", "sk-master-fallback")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"litellm_key": "sk-tenant-virtual"})

    fake = AsyncMock(return_value="resp")
    transport = httpx.MockTransport(handler)
    with patch.object(llm.litellm, "acompletion", fake):
        async with httpx.AsyncClient(transport=transport) as http:
            async with tenant_virtual_key_context("cust-abc", http=http):
                await llm.acompletion(
                    "anthropic/claude-sonnet-4-6",
                    [{"role": "user", "content": "x"}],
                )

    kwargs = fake.await_args.kwargs
    assert kwargs["api_base"] == "http://litellm.litellm.svc.cluster.local:4000"
    # Tenant key wins; the env-var master key is shadowed.
    assert kwargs["api_key"] == "sk-tenant-virtual"


@pytest.mark.asyncio
async def test_acompletion_falls_back_to_env_key_without_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside `tenant_virtual_key_context`, the process-wide
    ``LLM_GATEWAY_KEY`` is used (self-host / no-tenant-context path)."""
    monkeypatch.setenv("LLM_GATEWAY_URL", "http://litellm.litellm.svc.cluster.local:4000")
    monkeypatch.setenv("LLM_GATEWAY_KEY", "sk-master-fallback")

    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "acompletion", fake):
        await llm.acompletion("anthropic/claude-sonnet-4-6", [])

    assert fake.await_args.kwargs["api_key"] == "sk-master-fallback"


@pytest.mark.asyncio
async def test_acompletion_no_api_key_without_gateway_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound tenant key with NO ``LLM_GATEWAY_URL`` must NOT leak as
    api_key to direct provider calls — that would break the dev path
    where LiteLLM picks up ANTHROPIC_API_KEY itself."""
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"litellm_key": "sk-tenant-virtual"})

    transport = httpx.MockTransport(handler)
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "acompletion", fake):
        async with httpx.AsyncClient(transport=transport) as http:
            async with tenant_virtual_key_context("cust-abc", http=http):
                await llm.acompletion("anthropic/claude-sonnet-4-6", [])

    # Without LLM_GATEWAY_URL the wrapper sends neither api_base nor api_key.
    assert "api_base" not in fake.await_args.kwargs
    assert "api_key" not in fake.await_args.kwargs


@pytest.mark.asyncio
async def test_caller_api_key_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit per-call api_key override beats both the tenant key
    AND the env-var key — eval / debug pattern."""
    monkeypatch.setenv("LLM_GATEWAY_URL", "http://litellm.litellm.svc.cluster.local:4000")
    monkeypatch.setenv("LLM_GATEWAY_KEY", "sk-master")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"litellm_key": "sk-tenant"})

    transport = httpx.MockTransport(handler)
    fake = AsyncMock(return_value="resp")
    with patch.object(llm.litellm, "acompletion", fake):
        async with httpx.AsyncClient(transport=transport) as http:
            async with tenant_virtual_key_context("cust-abc", http=http):
                await llm.acompletion(
                    "anthropic/claude-sonnet-4-6",
                    [],
                    api_key="sk-explicit-override",
                )

    assert fake.await_args.kwargs["api_key"] == "sk-explicit-override"
