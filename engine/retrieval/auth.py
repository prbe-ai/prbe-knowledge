"""Auth resolution for /retrieve, /query, and /sources.

Two accepted auth shapes:
  1. `Authorization: Bearer <api_key>` — customer-issued, looked up in
     `customers.api_key_hash`. Used by external callers.
  2. `X-Internal-Knowledge-Key: <secret>` + `X-Prbe-Customer: <id>` —
     service-to-service trust. Used by prbe-orchestrator and
     prbe-knowledge-mcp. Internal key gates access; customer header
     sets scope. Works in local dev too — INTERNAL_KNOWLEDGE_API_KEY
     is set in `.env` for the docker-compose stack.

Every environment requires one of these. There is no body-fallback.
"""

from __future__ import annotations

import hashlib
import hmac

from fastapi import HTTPException, Request

from shared.config import get_settings
from shared.db import raw_conn

UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


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


async def authenticate_query(request: Request) -> str:
    """Resolve the authenticated customer_id from request headers.

    Returns the customer_id string. Raises HTTPException on any failure;
    callers can rely on the return value being a valid tenant id.
    """
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
        return customer

    authorization = request.headers.get("authorization")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(
                status_code=401,
                detail="invalid authorization scheme",
                headers=UNAUTHORIZED_HEADERS,
            )
        return await _resolve_customer_from_bearer(token.strip())

    raise HTTPException(
        status_code=401,
        detail="missing bearer token or X-Internal-Knowledge-Key",
        headers=UNAUTHORIZED_HEADERS,
    )
