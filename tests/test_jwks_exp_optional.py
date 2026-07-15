"""verify_access_token tolerates a missing `exp` (the issuer's non-expiring
token model) while still rejecting an `exp` that is present and past, and still
enforcing iss/aud/sub. See services/mcp/dependencies/jwks.py.
"""

from __future__ import annotations

import base64
import time
from types import SimpleNamespace

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from jwt import PyJWK

from engine.mcp.dependencies import jwks as jwks_mod
from engine.mcp.dependencies.jwks import JwtAuthError, verify_access_token

ISS = "https://api.knowledge.prbe.ai"
AUD = "https://mcp.knowledge.prbe.ai"
KID = "test-kid"


def _b64u32(n: int) -> str:
    return base64.urlsafe_b64encode(n.to_bytes(32, "big")).rstrip(b"=").decode()


@pytest.fixture
def signing_key(monkeypatch):
    key = ec.generate_private_key(ec.SECP256R1())
    nums = key.public_key().public_numbers()
    jwk = {
        "kty": "EC",
        "crv": "P-256",
        "alg": "ES256",
        "use": "sig",
        "kid": KID,
        "x": _b64u32(nums.x),
        "y": _b64u32(nums.y),
    }
    # Inject the verifying key directly so verify_access_token never hits HTTP.
    monkeypatch.setattr(jwks_mod, "_jwks_cache", {KID: PyJWK(jwk)})
    monkeypatch.setattr(jwks_mod, "_jwks_fetched_at", time.monotonic())
    monkeypatch.setattr(
        jwks_mod,
        "get_settings",
        lambda: SimpleNamespace(
            mcp_oauth_issuer=ISS,
            mcp_oauth_audience=AUD,
            mcp_oauth_jwks_ttl_s=600,
            mcp_oauth_jwks_url="https://issuer/oauth/jwks",
        ),
    )
    return key


def _mint(key, **overrides) -> str:
    payload = {
        "iss": ISS,
        "aud": AUD,
        "sub": "cust-1",
        "user_id": "user-1",
        "client_id": "mcp_client",
        "scope": "mcp:read",
        "iat": int(time.time()),
    }
    payload.update(overrides)
    return pyjwt.encode(payload, key, algorithm="ES256", headers={"kid": KID})


async def test_accepts_token_without_exp(signing_key):
    claims = await verify_access_token(_mint(signing_key))
    assert claims.customer_id == "cust-1"
    assert claims.expires_at is None


async def test_accepts_token_with_future_exp(signing_key):
    exp = int(time.time()) + 3600
    claims = await verify_access_token(_mint(signing_key, exp=exp))
    assert claims.expires_at == exp


async def test_rejects_token_with_past_exp(signing_key):
    with pytest.raises(JwtAuthError):
        await verify_access_token(_mint(signing_key, exp=int(time.time()) - 60))


async def test_rejects_wrong_audience(signing_key):
    with pytest.raises(JwtAuthError):
        await verify_access_token(_mint(signing_key, aud="https://evil.example"))
