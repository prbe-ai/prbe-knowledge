from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from services.mcp.clients.knowledge import close_client, get_client
from services.mcp.config import get_settings
from services.mcp.dependencies.auth_context import McpAuthMiddleware
from services.mcp.server import mcp

# Read the install script once at startup. Served at /install so the
# one-liner `curl -fsSL https://mcp.knowledge.prbe.ai/install | bash` works.
# Vendored alongside this package at services/mcp/scripts/install.sh.
_INSTALL_SCRIPT_PATH = Path(__file__).resolve().parent / "scripts" / "install.sh"
_INSTALL_SCRIPT = (
    _INSTALL_SCRIPT_PATH.read_text(encoding="utf-8")
    if _INSTALL_SCRIPT_PATH.is_file()
    else "#!/usr/bin/env bash\necho 'install.sh not packaged in this build' >&2\nexit 1\n"
)

log = structlog.get_logger(__name__)


# Build FastMCP's Starlette app at import time so its session_manager is
# eagerly initialized — the lifespan below needs to wrap session_manager.run(),
# which only exists after streamable_http_app() has been called.
_mcp_app = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    # Eagerly construct the knowledge client so missing-config blows up at
    # boot, not on the first tool invocation.
    get_client()
    log.info(
        "mcp.boot",
        environment=settings.environment,
        service=settings.service_name,
    )
    # FastAPI doesn't propagate sub-app lifespans through `mount`, so we run
    # FastMCP's session_manager task group inline here. Without this wrapper
    # the streamable-HTTP transport raises "Task group is not initialized"
    # on the first request.
    async with mcp.session_manager.run():
        try:
            yield
        finally:
            await close_client()


# `redirect_slashes=False` matters: FastAPI's default 307 redirect from
# `/mcp` to `/mcp/` breaks MCP clients that don't replay POST bodies.
# We serve `/mcp` directly via the FastMCP-mounted Starlette app instead.
app = FastAPI(
    title="prbe-knowledge-mcp",
    lifespan=lifespan,
    redirect_slashes=False,
)

# Order matters: ProxyHeaders → McpAuth → MCP transport.
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
app.add_middleware(McpAuthMiddleware)


# Routes defined BEFORE the catch-all mount take precedence over it.
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": get_settings().service_name}


@app.get("/.well-known/oauth-protected-resource")
async def oauth_protected_resource() -> dict[str, object]:
    """RFC 9728 — clients discover the authorization server here after a 401."""
    settings = get_settings()
    return {
        "resource": settings.mcp_oauth_audience,
        "authorization_servers": [settings.mcp_oauth_issuer],
        "scopes_supported": ["mcp:read"],
        "bearer_methods_supported": ["header"],
    }


def _register_test_slow_route(app: FastAPI) -> None:
    """Register `/_test/slow` for graceful-shutdown integration tests.

    Used by `tests/test_graceful_shutdown.py` to prove that an in-flight
    request survives a SIGINT during uvicorn's graceful shutdown window.

    Gated by TWO independent checks (defense in depth):
      1. `MCP_TEST_SLOW_ROUTE_ENABLED=1` env var must be set.
      2. `ENVIRONMENT` must be in the explicit allow-list `{test, dev}`.

    The allow-list (rather than a deny-list of production-ish names) is
    deliberate: any unknown env name (`staging`, `preview`, empty,
    typos, etc.) refuses registration, so a leaked env var into a
    non-prod-but-not-test environment can never expose this unauthenticated
    sleep handler. `_MAX_SLEEP_MS` is also capped at 3000 so even if the
    route ever did slip through both gates, the per-request damage is
    bounded.

    A failed gate just refuses to register — it does NOT raise. Crashing
    the boot would trade one footgun (DoS surface) for a worse one (the
    machine fails to start, hot-restart loop, deploy rollback).
    """
    if os.environ.get("MCP_TEST_SLOW_ROUTE_ENABLED") != "1":
        return
    env = os.environ.get("ENVIRONMENT", "").strip().lower()
    if env not in ("test", "dev"):
        # Allow-list refusal. Silent on purpose: noise here would mask
        # real boot logs. The test fixture sets ENVIRONMENT=test
        # explicitly; any other caller of this app should not have the
        # env var set in the first place.
        return

    _MAX_SLEEP_MS = 3000

    @app.get("/_test/slow", include_in_schema=False)
    async def _test_slow(sleep_ms: int = 2000) -> dict[str, str | int]:
        """Test-only: sleep `sleep_ms` (clamped to [0, 3000]) then return 200."""
        clamped = max(0, min(sleep_ms, _MAX_SLEEP_MS))
        await asyncio.sleep(clamped / 1000)
        return {"status": "ok", "slept_ms": clamped}


_register_test_slow_route(app)


@app.get("/install", include_in_schema=False)
async def install_script() -> PlainTextResponse:
    """One-liner installer: `curl -fsSL https://mcp.knowledge.prbe.ai/install | bash`.

    Adds Probe MCP to Claude Code (`claude mcp add`), Codex
    (`codex mcp add`), and/or Cursor (writes ~/.cursor/mcp.json), and offers
    client-specific behavior snippets so coding agents reach for the server
    proactively.
    """
    return PlainTextResponse(_INSTALL_SCRIPT, media_type="text/x-shellscript")


# Mount the FastMCP Starlette app at `/`. FastMCP exposes the streamable-HTTP
# transport at its internal `streamable_http_path` (default `/mcp`), so the
# external URL is `https://mcp.knowledge.prbe.ai/mcp` — served directly with no redirect.
#
# Mount must be LAST so the explicit `/health` and `/.well-known/...` routes
# win over the catch-all.
#
# Auth precedence (handled by McpAuthMiddleware above):
#   1. Authorization: Bearer <jwt> issued by api.knowledge.prbe.ai (production path)
#   2. X-Internal-Backend-Key + X-Prbe-Customer (dev/test fallback)
# Tool code reads customer_id from the contextvar set by the middleware.
app.mount("/", _mcp_app)
