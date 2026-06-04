from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from services.mcp.dependencies import auth_context
from services.mcp.dependencies.jwks import JwtAuthError


class _OauthSettings:
    resolved_auth_mode = "oauth"
    internal_backend_api_key = ""
    mcp_oauth_audience = "https://mcp.example.com/"


class _StaticSettings:
    resolved_auth_mode = "static"
    mcp_api_token = "shared-token"
    default_customer_id = "cust-static"
    mcp_oauth_audience = "https://mcp.example.com/"


def _app(monkeypatch: pytest.MonkeyPatch, settings: object) -> FastAPI:
    monkeypatch.setattr(auth_context, "get_settings", lambda: settings)

    app = FastAPI()
    app.add_middleware(auth_context.McpAuthMiddleware)

    @app.post("/mcp")
    async def mcp() -> dict[str, bool]:
        return {"ok": True}

    return app


async def _post_mcp(
    app: FastAPI, *, headers: dict[str, str] | None = None
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://mcp.example.com",
    ) as client:
        return await client.post("/mcp", headers=headers)


def _challenge(response: httpx.Response) -> str:
    return response.headers["WWW-Authenticate"]


def _assert_oauth_challenge(response: httpx.Response) -> None:
    challenge = _challenge(response)
    assert response.status_code == 401
    assert challenge.startswith("Bearer ")
    assert 'realm="mcp"' in challenge
    assert (
        'resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"'
        in challenge
    )
    assert 'scope="mcp:read"' in challenge


async def test_oauth_missing_mcp_auth_returns_resource_metadata_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = await _post_mcp(_app(monkeypatch, _OauthSettings()))

    _assert_oauth_challenge(response)
    assert 'error="invalid_token"' not in _challenge(response)


async def test_oauth_invalid_mcp_bearer_returns_invalid_token_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def reject_token(_token: str) -> object:
        raise JwtAuthError("invalid jwt")

    monkeypatch.setattr(auth_context, "verify_access_token", reject_token)

    response = await _post_mcp(
        _app(monkeypatch, _OauthSettings()),
        headers={"Authorization": "Bearer stale-token"},
    )

    _assert_oauth_challenge(response)
    assert 'error="invalid_token"' in _challenge(response)


async def test_static_mode_keeps_plain_bearer_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = await _post_mcp(_app(monkeypatch, _StaticSettings()))

    assert response.status_code == 401
    assert _challenge(response) == "Bearer"
