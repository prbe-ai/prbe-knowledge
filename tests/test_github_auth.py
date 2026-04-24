"""Tests for the GitHub App installation-token minter.

Exercises:
  - first mint signs an RS256 JWT + caches the returned token
  - cached call skips the POST
  - 401 from GitHub raises GitHubAuthError with body in the message
  - concurrent mints for one installation_id trigger exactly one POST
"""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from shared.exceptions import GitHubAuthError
from shared.github_auth import _reset_cache_for_tests, mint_installation_token


def _fresh_private_key_pem() -> str:
    """Generate a throwaway 1024-bit RSA key for the test only."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=1024)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem.decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


@pytest.fixture(autouse=True)
def _clear_cache():
    _reset_cache_for_tests()
    yield
    _reset_cache_for_tests()


@pytest.mark.asyncio
async def test_mint_builds_rs256_jwt_and_caches() -> None:
    app_id = "12345"
    pem = _fresh_private_key_pem()
    installation_id = "99"

    captured_auth: list[str] = []
    post_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"/app/installations/{installation_id}/access_tokens"
        captured_auth.append(request.headers["authorization"])
        post_calls["n"] += 1
        return httpx.Response(
            200,
            json={"token": "ghs_abc", "expires_at": "2026-12-31T00:00:00Z"},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        token1, expires1 = await mint_installation_token(http, app_id, pem, installation_id)
        token2, expires2 = await mint_installation_token(http, app_id, pem, installation_id)

    assert token1 == "ghs_abc"
    assert token2 == "ghs_abc"
    assert expires1 == expires2
    assert expires1.year == 2026 and expires1.month == 12

    # Second call hit the cache, not GitHub.
    assert post_calls["n"] == 1

    # Inspect the JWT sent on the first call.
    auth = captured_auth[0]
    assert auth.startswith("Bearer ")
    jwt = auth.removeprefix("Bearer ")
    header_b64, payload_b64, signature_b64 = jwt.split(".")
    header = json.loads(_b64url_decode(header_b64))
    payload = json.loads(_b64url_decode(payload_b64))

    assert header == {"alg": "RS256", "typ": "JWT"}
    assert payload["iss"] == app_id
    assert payload["exp"] - payload["iat"] == 600  # 60s backdate + 540s ttl
    # Signature segment is present and non-empty.
    assert signature_b64


@pytest.mark.asyncio
async def test_mint_accepts_201_response() -> None:
    """GitHub's POST /app/installations/{id}/access_tokens returns 201 Created,
    not 200. We must accept both."""
    app_id = "12345"
    pem = _fresh_private_key_pem()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={"token": "ghs_from_201", "expires_at": "2026-12-31T00:00:00Z"},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        token, _expires = await mint_installation_token(http, app_id, pem, "55")

    assert token == "ghs_from_201"


@pytest.mark.asyncio
async def test_mint_raises_github_auth_error_on_401() -> None:
    app_id = "12345"
    pem = _fresh_private_key_pem()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(GitHubAuthError) as exc:
            await mint_installation_token(http, app_id, pem, "42")

    assert "401" in str(exc.value)
    assert "Bad credentials" in str(exc.value)


@pytest.mark.asyncio
async def test_mint_concurrent_calls_serialize_via_lock() -> None:
    app_id = "12345"
    pem = _fresh_private_key_pem()
    installation_id = "77"
    post_calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        # Small delay to widen the race window between concurrent mints.
        await asyncio.sleep(0.01)
        post_calls["n"] += 1
        return httpx.Response(
            200,
            json={"token": "ghs_xyz", "expires_at": "2026-12-31T00:00:00Z"},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        results = await asyncio.gather(
            mint_installation_token(http, app_id, pem, installation_id),
            mint_installation_token(http, app_id, pem, installation_id),
        )

    # Both callers got the same (cached) token and the endpoint was hit once.
    tokens = {r[0] for r in results}
    assert tokens == {"ghs_xyz"}
    assert post_calls["n"] == 1
