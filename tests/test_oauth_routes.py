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
            resp = await client.get(
                "/oauth/github/callback",
                params={
                    "installation_id": INSTALLATION_ID,
                    "setup_action": "install",
                    "state": CUSTOMER_ID,
                },
            )
        finally:
            ingestion_app.state.ctx = original_ctx
            await mock_http.aclose()

    assert resp.status_code == 200, resp.text

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
