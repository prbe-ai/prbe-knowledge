"""Tests for shared.backend_client.fetch_github_installation_token.

The client wraps prbe-backend's `POST /internal/github/installation_token`
endpoint. Every failure mode must surface as `GitHubAuthError` so call
sites that previously raised against the in-process minter don't need
new except clauses.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx
from pydantic import SecretStr

from shared.backend_client import fetch_github_installation_token
from shared.config import Settings, get_settings
from shared.exceptions import GitHubAuthError


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch):
    """Override get_settings() to return a Settings with backend creds set,
    and reset the lru_cache so each test gets a fresh load."""
    get_settings.cache_clear()

    def _settings() -> Settings:
        return Settings(
            backend_base_url="http://prbe-backend.internal:8080",
            internal_backend_api_key=SecretStr("test-internal-key"),
        )

    monkeypatch.setattr("shared.backend_client.get_settings", _settings)
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_happy_path_returns_token_and_expires_at() -> None:
    expected_url = (
        "http://prbe-backend.internal:8080/internal/github/installation_token"
    )
    with respx.mock(assert_all_called=True) as router:
        route = router.post(expected_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "token": "ghs_abc",
                    "expires_at": "2026-12-31T00:00:00Z",
                    "installation_id": "12345",
                },
            )
        )
        async with httpx.AsyncClient() as http:
            token, expires_at = await fetch_github_installation_token(
                http, customer_id="cust-1"
            )

    assert token == "ghs_abc"
    assert expires_at == datetime(2026, 12, 31, tzinfo=UTC)

    # Verify the request shape: customer_id in body, X-Internal-Backend-Key header.
    request = route.calls[0].request
    assert request.headers["x-internal-backend-key"] == "test-internal-key"
    assert b'"customer_id"' in request.content
    assert b'"cust-1"' in request.content


@pytest.mark.asyncio
async def test_404_raises_no_installation_for_customer() -> None:
    expected_url = (
        "http://prbe-backend.internal:8080/internal/github/installation_token"
    )
    with respx.mock(assert_all_called=True) as router:
        router.post(expected_url).mock(
            return_value=httpx.Response(404, json={"detail": "no install"})
        )
        async with httpx.AsyncClient() as http:
            with pytest.raises(GitHubAuthError) as exc:
                await fetch_github_installation_token(http, customer_id="cust-1")

    assert "no GitHub installation" in str(exc.value)
    assert "cust-1" in str(exc.value)


@pytest.mark.asyncio
async def test_503_raises_app_credentials_not_configured() -> None:
    expected_url = (
        "http://prbe-backend.internal:8080/internal/github/installation_token"
    )
    with respx.mock(assert_all_called=True) as router:
        router.post(expected_url).mock(
            return_value=httpx.Response(503, text="App credentials missing")
        )
        async with httpx.AsyncClient() as http:
            with pytest.raises(GitHubAuthError) as exc:
                await fetch_github_installation_token(http, customer_id="cust-1")

    assert "App credentials not configured" in str(exc.value)


@pytest.mark.asyncio
async def test_5xx_raises_with_status_and_body() -> None:
    expected_url = (
        "http://prbe-backend.internal:8080/internal/github/installation_token"
    )
    with respx.mock(assert_all_called=True) as router:
        router.post(expected_url).mock(
            return_value=httpx.Response(502, text="github mint failed upstream")
        )
        async with httpx.AsyncClient() as http:
            with pytest.raises(GitHubAuthError) as exc:
                await fetch_github_installation_token(http, customer_id="cust-1")

    assert "502" in str(exc.value)
    assert "github mint failed upstream" in str(exc.value)


@pytest.mark.asyncio
async def test_4xx_other_raises_with_status() -> None:
    """401/400 etc. all surface as GitHubAuthError."""
    expected_url = (
        "http://prbe-backend.internal:8080/internal/github/installation_token"
    )
    with respx.mock(assert_all_called=True) as router:
        router.post(expected_url).mock(
            return_value=httpx.Response(401, text="bad internal key")
        )
        async with httpx.AsyncClient() as http:
            with pytest.raises(GitHubAuthError) as exc:
                await fetch_github_installation_token(http, customer_id="cust-1")

    assert "401" in str(exc.value)


@pytest.mark.asyncio
async def test_network_failure_raises_unreachable() -> None:
    expected_url = (
        "http://prbe-backend.internal:8080/internal/github/installation_token"
    )
    with respx.mock(assert_all_called=True) as router:
        router.post(expected_url).mock(side_effect=httpx.ConnectError("nope"))
        async with httpx.AsyncClient() as http:
            with pytest.raises(GitHubAuthError) as exc:
                await fetch_github_installation_token(http, customer_id="cust-1")

    assert "unreachable" in str(exc.value)


@pytest.mark.asyncio
async def test_missing_base_url_raises_before_http_call(monkeypatch) -> None:
    def _settings() -> Settings:
        return Settings(
            backend_base_url="",
            internal_backend_api_key=SecretStr("test-internal-key"),
        )

    monkeypatch.setattr("shared.backend_client.get_settings", _settings)

    with respx.mock(assert_all_called=False) as router:
        # No request should be issued at all.
        route = router.post(
            "http://prbe-backend.internal:8080/internal/github/installation_token"
        )
        async with httpx.AsyncClient() as http:
            with pytest.raises(GitHubAuthError) as exc:
                await fetch_github_installation_token(http, customer_id="cust-1")

    assert "not configured" in str(exc.value)
    assert route.call_count == 0


@pytest.mark.asyncio
async def test_missing_api_key_raises_before_http_call(monkeypatch) -> None:
    def _settings() -> Settings:
        return Settings(
            backend_base_url="http://prbe-backend.internal:8080",
            internal_backend_api_key=SecretStr(""),
        )

    monkeypatch.setattr("shared.backend_client.get_settings", _settings)

    with respx.mock(assert_all_called=False) as router:
        route = router.post(
            "http://prbe-backend.internal:8080/internal/github/installation_token"
        )
        async with httpx.AsyncClient() as http:
            with pytest.raises(GitHubAuthError) as exc:
                await fetch_github_installation_token(http, customer_id="cust-1")

    assert "not configured" in str(exc.value)
    assert route.call_count == 0
