"""assert_session_active — per-request revocation check (fail-closed)."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from engine.mcp.dependencies import revocation
from engine.mcp.dependencies.jwks import AccessClaims, JwtAuthError


def _claims(sid: str | None) -> AccessClaims:
    return AccessClaims(
        customer_id="c",
        user_id="u",
        client_id="cl",
        scope="mcp:read",
        expires_at=None,
        session_id=sid,
    )


def _settings(backend: str = "http://backend:8080", *, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        mcp_revocation_check_enabled=enabled,
        backend_base_url=backend,
        revocation_check_timeout_s=3.0,
        internal_backend_api_key="ik",
    )


def _patch_http(monkeypatch, handler) -> None:
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        return real(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", factory)


async def test_noop_when_no_backend_url():
    # No backend configured -> check disabled; must not raise or call out.
    await revocation.assert_session_active(_claims("s1"), _settings(backend=""))


async def test_noop_when_flag_disabled():
    # Flag off -> dormant even with a sid-less token and a backend configured.
    await revocation.assert_session_active(_claims(None), _settings(enabled=False))


async def test_missing_sid_rejected():
    with pytest.raises(JwtAuthError):
        await revocation.assert_session_active(_claims(None), _settings())


async def test_active_session_passes(monkeypatch):
    _patch_http(monkeypatch, lambda req: httpx.Response(200, json={"active": True}))
    await revocation.assert_session_active(_claims("s1"), _settings())


async def test_revoked_session_rejected(monkeypatch):
    _patch_http(monkeypatch, lambda req: httpx.Response(200, json={"active": False}))
    with pytest.raises(JwtAuthError):
        await revocation.assert_session_active(_claims("s1"), _settings())


async def test_backend_5xx_fails_closed(monkeypatch):
    _patch_http(monkeypatch, lambda req: httpx.Response(500))
    with pytest.raises(JwtAuthError):
        await revocation.assert_session_active(_claims("s1"), _settings())


async def test_backend_unreachable_fails_closed(monkeypatch):
    def boom(req):
        raise httpx.ConnectError("backend down")

    _patch_http(monkeypatch, boom)
    with pytest.raises(JwtAuthError):
        await revocation.assert_session_active(_claims("s1"), _settings())


async def test_sends_internal_key_and_sid(monkeypatch):
    seen: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["key"] = req.headers.get("X-Internal-Backend-Key", "")
        return httpx.Response(200, json={"active": True})

    _patch_http(monkeypatch, handler)
    await revocation.assert_session_active(_claims("sess-9"), _settings())
    assert seen["url"].endswith("/oauth/introspect/sess-9")
    assert seen["key"] == "ik"
