"""Per-request auth context for MCP tools.

MCP tools are plain async functions — they don't receive the FastAPI
`Request` object directly. We propagate the customer_id from the HTTP
layer into tool calls via a ContextVar that an ASGI middleware sets on
each request.

Two auth modes (selected by Settings.resolved_auth_mode):

`oauth` (hosted/managed — unchanged behavior). Precedence (first match wins):
  1. Authorization: Bearer <ES256 OAuth jwt>  — Phase G OAuth path. Validates
     against api.knowledge.prbe.ai's JWKS. customer_id from `sub` claim. Used by
     customer AI agents (Claude Desktop, Cursor).
  2. X-Internal-Backend-Key + X-Prbe-Customer  — Phase D internal-only path.
     Disabled when no internal-key env var is set.

`static` (community self-host). A single shared bearer (MCP_API_TOKEN), compared
constant-time, scopes the request to DEFAULT_CUSTOMER_ID. No JWKS, no issuer, no
control-plane callbacks.
"""

from __future__ import annotations

import hmac
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from services.mcp.config import Settings, get_settings
from services.mcp.dependencies.jwks import JwtAuthError, verify_access_token

current_customer: ContextVar[str] = ContextVar("current_customer")


class MissingAuthContext(LookupError):
    """Raised when a tool tries to read auth state outside an auth'd request."""


def get_current_customer() -> str:
    try:
        return current_customer.get()
    except LookupError as exc:
        raise MissingAuthContext(
            "no customer in context — request is missing auth or middleware didn't run"
        ) from exc


def _bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth.split(maxsplit=1)[1].strip()
    return token or None


def _oauth_challenge_headers(
    settings: Settings, *, error: str | None = None
) -> dict[str, str]:
    resource_metadata = (
        f"{settings.mcp_oauth_audience.rstrip('/')}/.well-known/oauth-protected-resource"
    )
    params = [
        'realm="mcp"',
        f'resource_metadata="{resource_metadata}"',
        'scope="mcp:read"',
    ]
    if error:
        params.append(f'error="{error}"')
    return {"WWW-Authenticate": "Bearer " + ", ".join(params)}


class McpAuthMiddleware(BaseHTTPMiddleware):
    """Gates /mcp/* requests. Branches on Settings.resolved_auth_mode:
    `oauth` (JWT primary + internal-key fallback) or `static` (shared bearer)."""

    def __init__(self, app: ASGIApp, *, mount_path: str = "/mcp") -> None:
        super().__init__(app)
        self._mount_path = mount_path

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if not request.url.path.startswith(self._mount_path):
            return await call_next(request)

        settings = get_settings()
        if settings.resolved_auth_mode == "static":
            return await self._dispatch_static(request, call_next, settings)
        return await self._dispatch_oauth(request, call_next, settings)

    async def _dispatch_oauth(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
        settings: Settings,
    ) -> Response:
        bearer = _bearer_token(request)
        if bearer is not None:
            try:
                access_claims = await verify_access_token(bearer)
            except JwtAuthError as exc:
                return JSONResponse(
                    {"detail": str(exc)},
                    status_code=401,
                    headers=_oauth_challenge_headers(settings, error="invalid_token"),
                )
            customer_token = current_customer.set(access_claims.customer_id)
            try:
                return await call_next(request)
            finally:
                current_customer.reset(customer_token)

        # Internal-key fallback (dev/test). HMAC-compared.
        presented = request.headers.get("x-internal-backend-key", "")
        expected = settings.internal_backend_api_key
        if presented and expected and hmac.compare_digest(presented, expected):
            customer = request.headers.get("x-prbe-customer", "")
            if not customer:
                return JSONResponse(
                    {"detail": "missing X-Prbe-Customer"}, status_code=400
                )
            token = current_customer.set(customer)
            try:
                return await call_next(request)
            finally:
                current_customer.reset(token)

        return JSONResponse(
            {"detail": "missing or invalid auth"},
            status_code=401,
            headers=_oauth_challenge_headers(settings),
        )

    async def _dispatch_static(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
        settings: Settings,
    ) -> Response:
        """Community mode: one shared bearer -> DEFAULT_CUSTOMER_ID."""
        bearer = _bearer_token(request)
        expected = settings.mcp_api_token
        if (
            bearer is not None
            and expected
            and hmac.compare_digest(bearer, expected)
        ):
            token = current_customer.set(settings.default_customer_id)
            try:
                return await call_next(request)
            finally:
                current_customer.reset(token)

        return JSONResponse(
            {"detail": "missing or invalid auth"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
