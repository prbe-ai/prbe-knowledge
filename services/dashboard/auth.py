"""Neon Auth JWT validation + session resolution.

The dashboard receives `Authorization: Bearer <jwt>` from the Next.js
frontend (the JWT comes from `session.access_token` in the Neon Auth
SDK). We validate it against the JWKS published by Neon Auth, then
resolve the user → active organization → customer chain so downstream
handlers see a single `Session` object.

Layered design:
  * `verify_jwt(token)` — pure: validates signature, exp, returns claims
  * `require_user` (FastAPI dependency) — extracts JWT, validates, returns claims
  * `require_session` (FastAPI dependency) — adds active-org + customer
  * `require_role(min_role)` (factory) — gates by Better Auth role
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Depends, HTTPException, Request
from jose import JWTError, jwk, jwt

from shared.config import get_settings
from shared.db import raw_conn
from shared.logging import get_logger
from shared.provisioning import get_customer_by_organization

log = get_logger(__name__)

_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}

# JWKS cache. Refreshed lazily on signature failure when the kid isn't found.
_jwks_cache: dict[str, Any] = {}
_jwks_fetched_at: float = 0.0
_JWKS_TTL_SECONDS = 600.0


# ---------------------------------------------------------------------------
# Role hierarchy (Better Auth)
# ---------------------------------------------------------------------------

_ROLE_RANK = {"member": 1, "admin": 2, "owner": 3}


def _meets(actual: str, required: str) -> bool:
    return _ROLE_RANK.get(actual, 0) >= _ROLE_RANK.get(required, 0)


# ---------------------------------------------------------------------------
# Session shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserClaims:
    """JWT claims after validation."""

    user_id: str
    email: str
    expires_at: int


@dataclass(frozen=True)
class Session:
    """Resolved session: user + active org + customer + role."""

    user_id: str
    email: str
    organization_id: str | None
    customer_id: str | None
    role: str | None  # Better Auth role: 'owner' | 'admin' | 'member'


# ---------------------------------------------------------------------------
# JWKS fetch + JWT verify
# ---------------------------------------------------------------------------


def _jwks_url() -> str:
    base = get_settings().neon_auth_base_url
    if not base:
        raise HTTPException(
            status_code=503,
            detail="dashboard auth disabled — set NEON_AUTH_BASE_URL",
        )
    return f"{base.rstrip('/')}/.well-known/jwks.json"


async def _fetch_jwks(force: bool = False) -> dict[str, Any]:
    """Fetch (and cache) the Neon Auth JWKS document.

    `force=True` bypasses the cache to refresh after an unknown kid lookup.
    """
    global _jwks_cache, _jwks_fetched_at
    now = time.monotonic()
    if not force and _jwks_cache and (now - _jwks_fetched_at) < _JWKS_TTL_SECONDS:
        return _jwks_cache
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(_jwks_url())
        resp.raise_for_status()
    _jwks_cache = resp.json()
    _jwks_fetched_at = now
    return _jwks_cache


async def _public_key_for(kid: str) -> Any:
    jwks = await _fetch_jwks()
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            return jwk.construct(k)
    # Unknown kid — refresh once in case Neon rotated keys.
    jwks = await _fetch_jwks(force=True)
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            return jwk.construct(k)
    raise HTTPException(
        status_code=401,
        detail=f"unknown jwt key id: {kid}",
        headers=_UNAUTHORIZED_HEADERS,
    )


async def verify_jwt(token: str) -> UserClaims:
    """Verify the JWT signature, expiry, and required claims."""
    try:
        unverified = jwt.get_unverified_header(token)
    except JWTError as exc:
        raise HTTPException(
            status_code=401,
            detail="malformed jwt",
            headers=_UNAUTHORIZED_HEADERS,
        ) from exc

    kid = unverified.get("kid")
    if not kid:
        raise HTTPException(
            status_code=401,
            detail="jwt missing kid",
            headers=_UNAUTHORIZED_HEADERS,
        )
    key = await _public_key_for(kid)

    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=[unverified.get("alg", "RS256")],
            options={"verify_aud": False},
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=401,
            detail=f"invalid jwt: {exc}",
            headers=_UNAUTHORIZED_HEADERS,
        ) from exc

    sub = claims.get("sub")
    email = claims.get("email")
    exp = claims.get("exp")
    if not sub or not email or not exp:
        raise HTTPException(
            status_code=401,
            detail="jwt missing required claims",
            headers=_UNAUTHORIZED_HEADERS,
        )
    return UserClaims(user_id=sub, email=email, expires_at=int(exp))


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


async def require_user(request: Request) -> UserClaims:
    """Validate the bearer JWT and return claims. No org resolution."""
    auth = request.headers.get("authorization")
    if not auth:
        raise HTTPException(
            status_code=401,
            detail="missing authorization",
            headers=_UNAUTHORIZED_HEADERS,
        )
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=401,
            detail="invalid authorization scheme",
            headers=_UNAUTHORIZED_HEADERS,
        )
    return await verify_jwt(token.strip())


async def _active_organization_for_user(user_id: str) -> tuple[str, str] | None:
    """Look up the user's single (active) organization + role.

    Because organization_limit is set to 1, a user has at most one row in
    neon_auth.member. Returns (organization_id, role) or None.
    """
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT "organizationId"::text AS organization_id, role
            FROM neon_auth.member
            WHERE "userId" = $1::uuid
            ORDER BY "createdAt" ASC
            LIMIT 1
            """,
            user_id,
        )
    if row is None:
        return None
    return (row["organization_id"], row["role"])


async def require_session(
    user: UserClaims = Depends(require_user),
) -> Session:
    """Validate JWT + resolve active org + customer."""
    org_role = await _active_organization_for_user(user.user_id)
    if org_role is None:
        return Session(
            user_id=user.user_id,
            email=user.email,
            organization_id=None,
            customer_id=None,
            role=None,
        )
    organization_id, role = org_role
    customer = await get_customer_by_organization(organization_id)
    return Session(
        user_id=user.user_id,
        email=user.email,
        organization_id=organization_id,
        customer_id=customer["customer_id"] if customer else None,
        role=role,
    )


def require_role(min_role: str) -> Any:
    """Dependency factory: requires the user's role on the active org meets min_role.

    Raises 403 on insufficient role, 401 if no session, 409 if no team.
    Returns a callable suitable for `Depends(...)`.
    """

    async def dep(session: Session = Depends(require_session)) -> Session:
        if session.organization_id is None:
            raise HTTPException(
                status_code=409,
                detail="user has no team",
            )
        if session.role is None or not _meets(session.role, min_role):
            raise HTTPException(
                status_code=403,
                detail=f"requires role {min_role}",
            )
        return session

    return dep
