"""End-to-end coverage of `/oauth/{source}/callback`.

The interesting case is GitHub, whose post-install redirect carries
`installation_id` + `setup_action` + `state` but no `code`. The route
must accept that shape, route it through `GitHubConnector.exchange_oauth_code`,
persist the `IntegrationToken`, and record the workspace mapping.

We drive the ingestion app in-process via ASGITransport, swapping the
app's `ctx.http` for an `httpx.MockTransport` after lifespan startup so
GitHub API calls hit the mock instead of the real network.
"""

from __future__ import annotations

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from httpx import ASGITransport
from pydantic import SecretStr

from services.ingestion.handlers.base import ConnectorContext
from shared.config import Settings, get_settings
from shared.constants import GITHUB_INSTALLATION_SCOPE_PREFIX, SourceSystem
from shared.db import close_pool, init_pool, raw_conn
from shared.embeddings import reset_embedder
from shared.github_auth import _reset_cache_for_tests
from shared.storage import reset_store

CUSTOMER_ID = "cust-1"
INSTALLATION_ID = "99"


def _fresh_private_key_pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=1024)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem.decode("ascii")


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv(
        "TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value()
    )
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("GITHUB_APP_ID", "12345")
    monkeypatch.setenv("GITHUB_APP_SLUG", "prbe-knowledge-dev")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]
    _reset_cache_for_tests()
    yield
    _reset_cache_for_tests()


@pytest.mark.asyncio
async def test_oauth_callback_github_path(live_db, settings, monkeypatch) -> None:
    pem = _fresh_private_key_pem()
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", pem)
    monkeypatch.setenv("DASHBOARD_BASE_URL", "http://dash.local")
    get_settings.cache_clear()  # type: ignore[attr-defined]

    # Customer row required by integration_tokens.customer_id FK.
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'oauth-test', 'dummy')
            ON CONFLICT DO NOTHING
            """,
            CUSTOMER_ID,
        )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/app/installations/{INSTALLATION_ID}/access_tokens":
            return httpx.Response(
                200,
                json={"token": "ghs_abc", "expires_at": "2026-12-31T00:00:00Z"},
            )
        if path == f"/app/installations/{INSTALLATION_ID}":
            return httpx.Response(
                200,
                json={
                    "id": int(INSTALLATION_ID),
                    "account": {"login": "prbe", "type": "Organization"},
                    "target_type": "Organization",
                },
            )
        return httpx.Response(404, json={"message": f"unexpected path {path}"})

    from services.ingestion.main import app as ingestion_app

    await close_pool()
    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        # Swap the app's ctx for one backed by our MockTransport so GitHub
        # API calls hit the mock handler instead of the real network.
        mock_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        original_ctx = ingestion_app.state.ctx
        ingestion_app.state.ctx = ConnectorContext(
            settings=Settings(
                environment="local",
                github_app_id="12345",
                github_app_slug="prbe-knowledge-dev",
                github_app_private_key=SecretStr(pem),
            ),
            http=mock_http,
        )
        try:
            from shared.state_signing import sign_state

            resp = await client.get(
                "/oauth/github/callback",
                params={
                    "installation_id": INSTALLATION_ID,
                    "setup_action": "install",
                    "state": sign_state(CUSTOMER_ID, "github"),
                },
            )
        finally:
            ingestion_app.state.ctx = original_ctx
            await mock_http.aclose()

    assert resp.status_code == 302, resp.text
    location = resp.headers["location"]
    assert location.startswith("http://dash.local/oauth-landed?")
    assert f"customer_id={CUSTOMER_ID}" in location
    assert "ok=1" in location

    await init_pool(settings)
    async with raw_conn() as conn:
        token_row = await conn.fetchrow(
            """
            SELECT scope, status
            FROM integration_tokens
            WHERE customer_id = $1 AND source_system = $2
            """,
            CUSTOMER_ID,
            SourceSystem.GITHUB.value,
        )
        mapping_row = await conn.fetchrow(
            """
            SELECT customer_id, external_id, external_name
            FROM customer_source_mapping
            WHERE source_system = $1 AND external_id = $2
            """,
            SourceSystem.GITHUB.value,
            INSTALLATION_ID,
        )

    assert token_row is not None, "integration_tokens row was not written"
    assert token_row["scope"] == f"{GITHUB_INSTALLATION_SCOPE_PREFIX}{INSTALLATION_ID}"
    assert token_row["status"] == "active"

    assert mapping_row is not None, "customer_source_mapping row was not written"
    assert mapping_row["customer_id"] == CUSTOMER_ID
    assert mapping_row["external_name"] == "prbe"


@pytest.mark.asyncio
async def test_oauth_callback_github_state_with_update_action_still_saves(
    live_db, settings, monkeypatch
) -> None:
    """GitHub stamps setup_action=update when the App was already installed
    at the account level before this customer linked it. `state` is present,
    so this is still a first-time connect — must run the full save path,
    not short-circuit."""
    pem = _fresh_private_key_pem()
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", pem)
    monkeypatch.setenv("DASHBOARD_BASE_URL", "http://dash.local")
    get_settings.cache_clear()  # type: ignore[attr-defined]

    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'update-action-test', 'dummy')
            ON CONFLICT DO NOTHING
            """,
            CUSTOMER_ID,
        )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/app/installations/{INSTALLATION_ID}/access_tokens":
            return httpx.Response(
                200,
                json={"token": "ghs_abc", "expires_at": "2026-12-31T00:00:00Z"},
            )
        if path == f"/app/installations/{INSTALLATION_ID}":
            return httpx.Response(
                200,
                json={
                    "id": int(INSTALLATION_ID),
                    "account": {"login": "prbe", "type": "Organization"},
                    "target_type": "Organization",
                },
            )
        return httpx.Response(404, json={"message": f"unexpected path {path}"})

    from services.ingestion.main import app as ingestion_app

    await close_pool()
    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        mock_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        original_ctx = ingestion_app.state.ctx
        ingestion_app.state.ctx = ConnectorContext(
            settings=Settings(
                environment="local",
                github_app_id="12345",
                github_app_slug="prbe-knowledge-dev",
                github_app_private_key=SecretStr(pem),
            ),
            http=mock_http,
        )
        try:
            from shared.state_signing import sign_state

            resp = await client.get(
                "/oauth/github/callback",
                params={
                    "installation_id": INSTALLATION_ID,
                    "setup_action": "update",
                    "state": sign_state(CUSTOMER_ID, "github"),
                },
            )
        finally:
            ingestion_app.state.ctx = original_ctx
            await mock_http.aclose()

    assert resp.status_code == 302, resp.text

    await init_pool(settings)
    async with raw_conn() as conn:
        token_row = await conn.fetchrow(
            """
            SELECT scope, status
            FROM integration_tokens
            WHERE customer_id = $1 AND source_system = $2
            """,
            CUSTOMER_ID,
            SourceSystem.GITHUB.value,
        )
    assert token_row is not None, (
        "setup_action=update with state present must still save the token "
        "(this is a first-time connect, not a post-install repo update)"
    )
    assert token_row["status"] == "active"


@pytest.mark.asyncio
async def test_oauth_callback_github_update_without_state(
    live_db, settings, monkeypatch
) -> None:
    """GitHub 'Redirect on update' fires with installation_id + setup_action=update
    but no `state` or `code`. We must resolve customer_id via existing mapping,
    skip token re-exchange, and land the user on the dashboard."""
    monkeypatch.setenv("DASHBOARD_BASE_URL", "http://dash.local")
    get_settings.cache_clear()  # type: ignore[attr-defined]

    update_customer = "cust-update"
    update_install = "100"

    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'update-test', 'dummy')
            ON CONFLICT DO NOTHING
            """,
            update_customer,
        )
        await conn.execute(
            """
            INSERT INTO customer_source_mapping
                (source_system, external_id, customer_id, external_name, metadata)
            VALUES ($1, $2, $3, 'prbe', '{}'::jsonb)
            """,
            SourceSystem.GITHUB.value,
            update_install,
            update_customer,
        )

    http_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        http_calls.append(request.url.path)
        return httpx.Response(500, json={"message": "should not be called"})

    from services.ingestion.main import app as ingestion_app

    await close_pool()
    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        mock_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        original_ctx = ingestion_app.state.ctx
        ingestion_app.state.ctx = ConnectorContext(
            settings=Settings(environment="local"),
            http=mock_http,
        )
        try:
            resp = await client.get(
                "/oauth/github/callback",
                params={
                    "installation_id": update_install,
                    "setup_action": "update",
                },
            )
        finally:
            ingestion_app.state.ctx = original_ctx
            await mock_http.aclose()

    assert resp.status_code == 302, resp.text
    location = resp.headers["location"]
    assert location.startswith("http://dash.local/oauth-landed?")
    assert "source=github" in location
    assert f"customer_id={update_customer}" in location
    assert "ok=1" in location
    assert http_calls == [], f"update path should not hit GitHub API, got {http_calls}"

    await init_pool(settings)
    async with raw_conn() as conn:
        token_row = await conn.fetchrow(
            """
            SELECT 1
            FROM integration_tokens
            WHERE customer_id = $1 AND source_system = $2
            """,
            update_customer,
            SourceSystem.GITHUB.value,
        )
    assert token_row is None, "update flow must not write an integration_tokens row"


@pytest.mark.asyncio
async def test_oauth_callback_github_install_without_state_no_mapping(
    live_db, monkeypatch
) -> None:
    """A marketplace install (or any direct install from github.com/apps/<slug>)
    reaches the Setup URL with only installation_id — no `state`, no existing
    mapping. We can't bind this to a tenant, so redirect to the dashboard
    with a recoverable error instead of 422'ing."""
    monkeypatch.setenv("DASHBOARD_BASE_URL", "http://dash.local")
    get_settings.cache_clear()  # type: ignore[attr-defined]

    unknown_install = "200"

    http_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        http_calls.append(request.url.path)
        return httpx.Response(500, json={"message": "should not be called"})

    from services.ingestion.main import app as ingestion_app

    await close_pool()
    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        mock_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        original_ctx = ingestion_app.state.ctx
        ingestion_app.state.ctx = ConnectorContext(
            settings=Settings(environment="local"),
            http=mock_http,
        )
        try:
            resp = await client.get(
                "/oauth/github/callback",
                params={
                    "installation_id": unknown_install,
                    "setup_action": "install",
                },
            )
        finally:
            ingestion_app.state.ctx = original_ctx
            await mock_http.aclose()

    assert resp.status_code == 302, resp.text
    location = resp.headers["location"]
    assert location.startswith("http://dash.local/oauth-landed?")
    assert "ok=0" in location
    assert "error=install_without_state" in location
    assert http_calls == [], "no-mapping path should not hit GitHub API"


# ---------------------------------------------------------------------------
# State-signing rejection paths — generic across all OAuth sources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_redirect_carries_signed_state(live_db) -> None:
    """install must sign customer_id into state — never substitute the
    raw value, since that's the token-attachment CSRF the spec fixes.

    Needs live_db because the ingestion app's lifespan initializes the
    DB pool. The /oauth/slack/install route never touches the DB itself,
    but the lifespan's init_pool() does — same pattern the existing
    GitHub callback tests use.
    """
    from services.ingestion.main import app as ingestion_app

    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        # Swap ctx to a Settings with slack_client_id populated. Matches the
        # pattern used by test_oauth_callback_github_path — the lifespan-built
        # ctx may not see test-injected env vars at the right moment.
        original_ctx = ingestion_app.state.ctx
        ingestion_app.state.ctx = ConnectorContext(
            settings=Settings(
                environment="local",
                slack_client_id="slack-id-test",
            ),
            http=httpx.AsyncClient(),
        )
        try:
            resp = await client.get(
                "/oauth/slack/install",
                params={
                    "customer_id": "cust-1",
                    "redirect_uri": "http://t/oauth/slack/callback",
                },
                follow_redirects=False,
            )
        finally:
            ingestion_app.state.ctx = original_ctx

    assert resp.status_code == 302, resp.text
    location = resp.headers["location"]
    # state must NOT be the raw customer_id.
    assert "state=cust-1" not in location
    # state must look like an itsdangerous signed blob (body.signature).
    from urllib.parse import parse_qs, urlparse

    qs = parse_qs(urlparse(location).query)
    state = qs["state"][0]
    assert "." in state, f"state should be signed body.sig, got: {state!r}"

    # And it should round-trip through verify_state.
    from shared.state_signing import verify_state

    assert verify_state(state, "slack") == "cust-1"


@pytest.mark.asyncio
async def test_callback_rejects_unsigned_state(live_db) -> None:
    """A request that puts the raw customer_id in state must 400."""
    from services.ingestion.main import app as ingestion_app

    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        resp = await client.get(
            "/oauth/slack/callback",
            params={"code": "abc", "state": "cust-1"},
            follow_redirects=False,
        )

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_callback_rejects_state_for_wrong_source(live_db) -> None:
    """State signed for slack must not be accepted by the notion callback."""
    from services.ingestion.main import app as ingestion_app
    from shared.state_signing import sign_state

    slack_state = sign_state("cust-1", "slack")

    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        resp = await client.get(
            "/oauth/notion/callback",
            params={"code": "abc", "state": slack_state},
            follow_redirects=False,
        )

    assert resp.status_code == 400
