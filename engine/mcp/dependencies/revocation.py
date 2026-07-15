"""Per-request session-liveness check — revocation enforcement.

Access tokens are non-expiring (mcp_oauth_access_token_non_expiring), so the
issuer can't revoke a stateless JWT. The resource server enforces revocation
instead: on each request it asks prbe-backend whether the token's `sid` auth
session is still alive. Revoked / unknown / sid-less -> reject (401 + OAuth
challenge -> the client re-auths and starts fresh).

Fail-CLOSED: any error reaching the issuer also rejects — revocation
correctness is preferred over availability (the operator's choice). Note the
resource server already depends on prbe-backend for JWKS, so this adds no new
hard dependency, only tightens it (per-request vs cached).

No-op unless ``mcp_revocation_check_enabled`` is set AND ``backend_base_url``
is configured. The flag (default off) lets the code deploy dormant and be
switched on deliberately; community/static self-host leaves it off.
"""

from __future__ import annotations

import httpx

from services.mcp.config import Settings
from services.mcp.dependencies.jwks import AccessClaims, JwtAuthError


async def assert_session_active(claims: AccessClaims, settings: Settings) -> None:
    """Raise JwtAuthError (-> 401) unless the token's session is alive."""
    if not (settings.mcp_revocation_check_enabled and settings.backend_base_url):
        return  # revocation enforcement disabled (flag off or no issuer backend)

    sid = claims.session_id
    if not sid:
        # Token minted before `sid` existed (or by an issuer that omits it):
        # it can't be revocation-checked, so force a re-auth to a sid-bearing
        # token rather than letting an un-revocable token through.
        raise JwtAuthError("token missing sid; re-authentication required")

    url = f"{settings.backend_base_url.rstrip('/')}/oauth/introspect/{sid}"
    try:
        async with httpx.AsyncClient(
            timeout=settings.revocation_check_timeout_s
        ) as client:
            resp = await client.get(
                url,
                headers={"X-Internal-Backend-Key": settings.internal_backend_api_key},
            )
        resp.raise_for_status()
        active = bool(resp.json().get("active"))
    except Exception as exc:
        # Fail closed on ANY introspection error (timeout, connID, 5xx, bad
        # JSON): if we can't confirm the session is alive, reject.
        raise JwtAuthError(f"session liveness check failed: {exc}") from exc

    if not active:
        raise JwtAuthError("session revoked")
