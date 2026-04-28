"""Auth resolution for /retrieve, /query, and /sources.

Three accepted auth shapes:
  1. `Authorization: Bearer <api_key>` — customer-issued, looked up in
     `customers.api_key_hash`. Used by external callers.
  2. `X-Internal-Knowledge-Key: <secret>` + `X-Prbe-Customer: <id>` —
     service-to-service trust. Used by prbe-orchestrator and
     prbe-knowledge-mcp. Internal key gates access; customer header
     sets scope.
  3. Local-dev bypass: `environment=local` + missing headers lets the
     handler fall back to a `customer_id` in the request body. Production
     environments always require auth.
"""

from __future__ import annotations

import hashlib
import hmac

from fastapi import HTTPException, Request

from shared.config import get_settings
from shared.db import raw_conn

UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


class AuthResult:
    """Resolution outcome — `customer_id` if known, plus a flag indicating
    whether the request actually carried an auth header (vs. local-bypass)."""

    __slots__ = ("auth_present", "customer_id")

    def __init__(self, customer_id: str | None, auth_present: bool) -> None:
        self.customer_id = customer_id
        self.auth_present = auth_present


async def _resolve_customer_from_bearer(token: str) -> str:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT customer_id FROM customers WHERE api_key_hash = $1",
            token_hash,
        )
    if row is None:
        raise HTTPException(
            status_code=401,
            detail="invalid api key",
            headers=UNAUTHORIZED_HEADERS,
        )
    return row["customer_id"]


async def authenticate_query(request: Request) -> AuthResult:
    """Derive customer_id from one of the supported auth shapes."""
    settings = get_settings()

    internal_key = request.headers.get("x-internal-knowledge-key")
    if internal_key:
        expected = settings.internal_knowledge_api_key
        if expected is None or not expected.get_secret_value():
            raise HTTPException(
                status_code=503,
                detail="internal API disabled — set INTERNAL_KNOWLEDGE_API_KEY",
            )
        if not hmac.compare_digest(internal_key, expected.get_secret_value()):
            raise HTTPException(
                status_code=401,
                detail="invalid X-Internal-Knowledge-Key",
            )
        customer = request.headers.get("x-prbe-customer")
        if not customer:
            raise HTTPException(
                status_code=400,
                detail="X-Internal-Knowledge-Key requires X-Prbe-Customer",
            )
        return AuthResult(customer_id=customer, auth_present=True)

    authorization = request.headers.get("authorization")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(
                status_code=401,
                detail="invalid authorization scheme",
                headers=UNAUTHORIZED_HEADERS,
            )
        resolved = await _resolve_customer_from_bearer(token.strip())
        return AuthResult(customer_id=resolved, auth_present=True)

    if settings.is_local:
        return AuthResult(customer_id=None, auth_present=False)

    raise HTTPException(
        status_code=401,
        detail="missing bearer token or X-Internal-Knowledge-Key",
        headers=UNAUTHORIZED_HEADERS,
    )


def resolve_customer_id_strict(auth: AuthResult) -> str:
    """Bearer-only resolution for endpoints with no body fallback (/sources).

    /sources is a GET so there's no body where customer_id could come from.
    The local-dev bypass that /retrieve and /query support doesn't apply.
    """
    if not auth.customer_id:
        raise HTTPException(
            status_code=401,
            detail="missing bearer token",
            headers=UNAUTHORIZED_HEADERS,
        )
    return auth.customer_id


def resolve_customer_id_for_body(auth: AuthResult, body_customer_id: str | None) -> str:
    """Resolve customer_id for endpoints that take a body (/retrieve, /query).

    If a header is present, it's authoritative — a mismatching body value is
    a caller bug or a cross-tenant probe and we refuse rather than silently
    shadowing the header.
    """
    if auth.auth_present:
        if body_customer_id and body_customer_id != auth.customer_id:
            raise HTTPException(
                status_code=400,
                detail="customer_id in body does not match authenticated tenant",
            )
        assert auth.customer_id is not None  # type-checker; guaranteed by auth flow
        return auth.customer_id

    # Local dev bypass path.
    if not body_customer_id:
        raise HTTPException(
            status_code=401,
            detail="missing bearer token",
            headers=UNAUTHORIZED_HEADERS,
        )
    return body_customer_id
