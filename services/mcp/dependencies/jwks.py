"""JWKS fetch + JWT verify against the api.knowledge.prbe.ai issuer.

Mirrors prbe-backend's pattern: fetch JWKS once, cache by `kid`, refresh on
unknown-kid or TTL expiry. Validates iss/aud/exp; returns a tiny dataclass
of the claims we care about (sub = customer_id).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
import jwt as pyjwt
from jwt import PyJWK

from services.mcp.config import get_settings


class JwtAuthError(Exception):
    """Raised on any JWT verification failure (401-able)."""


@dataclass(frozen=True)
class AccessClaims:
    customer_id: str
    user_id: str
    client_id: str
    scope: str
    expires_at: int


_jwks_cache: dict[str, PyJWK] = {}
_jwks_fetched_at: float = 0.0


async def _refresh_jwks() -> None:
    global _jwks_cache, _jwks_fetched_at
    settings = get_settings()
    if not settings.mcp_oauth_jwks_url:
        raise JwtAuthError("MCP_OAUTH_JWKS_URL is not configured")
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(settings.mcp_oauth_jwks_url)
        resp.raise_for_status()
    body = resp.json()
    new_cache: dict[str, PyJWK] = {}
    for k in body.get("keys", []):
        kid = k.get("kid")
        if kid:
            new_cache[kid] = PyJWK(k)
    _jwks_cache = new_cache
    _jwks_fetched_at = time.monotonic()


async def _key_for(kid: str) -> PyJWK:
    settings = get_settings()
    ttl = settings.mcp_oauth_jwks_ttl_s
    if not _jwks_cache or (time.monotonic() - _jwks_fetched_at) > ttl:
        await _refresh_jwks()
    key = _jwks_cache.get(kid)
    if key is not None:
        return key
    # Unknown kid — refresh once in case the issuer rotated keys.
    await _refresh_jwks()
    key = _jwks_cache.get(kid)
    if key is None:
        raise JwtAuthError(f"unknown jwt kid: {kid}")
    return key


async def verify_access_token(token: str) -> AccessClaims:
    settings = get_settings()
    try:
        header = pyjwt.get_unverified_header(token)
    except pyjwt.InvalidTokenError as exc:
        raise JwtAuthError(f"malformed jwt: {exc}") from exc
    kid = header.get("kid")
    if not kid:
        raise JwtAuthError("jwt missing kid")
    key = await _key_for(kid)
    try:
        claims: dict[str, Any] = pyjwt.decode(
            token,
            key.key,
            algorithms=["ES256"],
            issuer=settings.mcp_oauth_issuer,
            audience=settings.mcp_oauth_audience,
            options={"require": ["exp", "iss", "aud", "sub"]},
        )
    except pyjwt.InvalidTokenError as exc:
        raise JwtAuthError(f"invalid jwt: {exc}") from exc
    sub = claims.get("sub")
    user_id = claims.get("user_id")
    client_id = claims.get("client_id")
    scope = claims.get("scope") or "mcp:read"
    exp = claims.get("exp")
    if not (sub and user_id and client_id and exp):
        raise JwtAuthError("jwt missing required claims (sub, user_id, client_id, exp)")
    return AccessClaims(
        customer_id=str(sub),
        user_id=str(user_id),
        client_id=str(client_id),
        scope=str(scope),
        expires_at=int(exp),
    )
