"""Admin API gated by X-Internal-Knowledge-Key.

Covers:
  - 503 when INTERNAL_KNOWLEDGE_API_KEY is unset
  - 401 when header missing / wrong
  - create_customer → plaintext key + install URLs
  - 409 on duplicate
  - rotate_key issues a new key (hash changed in DB)
  - list_customers
  - get_integrations reflects integration_tokens + mapping rows
  - get_ingestion_stats reflects documents + queue rows
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import httpx
import pytest
from httpx import ASGITransport

from shared.config import Settings, get_settings
from shared.constants import IntegrationStatus, SourceSystem
from shared.db import close_pool, init_pool, raw_conn
from shared.embeddings import reset_embedder
from shared.storage import reset_store

ADMIN_KEY = "admin-test-key-please"


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv(
        "TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value()
    )
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", ADMIN_KEY)
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


async def _client(app):
    await close_pool()
    transport = ASGITransport(app=app)
    return transport


async def _admin_request(
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    json: dict | None = None,
) -> httpx.Response:
    from services.ingestion.main import app as ingestion_app

    await close_pool()
    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        resp = await client.request(method, path, headers=headers or {}, json=json)
    return resp


def _auth(extra: dict[str, str] | None = None) -> dict[str, str]:
    h = {"X-Internal-Knowledge-Key": ADMIN_KEY}
    if extra:
        h.update(extra)
    return h


# ---------------------------------------------------------------------------
# Auth guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_requires_key_header(live_db, settings) -> None:
    resp = await _admin_request("GET", "/admin/customers")
    await init_pool(settings)
    assert resp.status_code == 401, resp.text

    resp = await _admin_request(
        "GET", "/admin/customers", headers={"X-Internal-Knowledge-Key": "wrong"}
    )
    await init_pool(settings)
    assert resp.status_code == 401, resp.text


@pytest.mark.asyncio
async def test_admin_503_when_key_unset(live_db, settings, monkeypatch) -> None:
    # Setenv to "" — delenv alone leaks through any .env file the developer
    # has on disk (pydantic-settings falls back to those when the env var is
    # absent). verify_internal_knowledge_key treats empty SecretStr as "not set".
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "")
    get_settings.cache_clear()  # type: ignore[attr-defined]

    for method, path in [
        ("GET", "/admin/customers"),
        ("POST", "/admin/customers"),
        ("POST", "/admin/customers/foo/rotate_key"),
        ("GET", "/admin/customers/foo/integrations"),
        ("GET", "/admin/customers/foo/ingestion_stats"),
    ]:
        body = {"customer_id": "x", "display_name": "y", "redirect_uri_base": "z"} if method == "POST" and path == "/admin/customers" else None
        resp = await _admin_request(method, path, headers={"X-Internal-Knowledge-Key": "anything"}, json=body)
        assert resp.status_code == 503, f"{method} {path}: {resp.status_code} {resp.text}"

    await init_pool(settings)


# ---------------------------------------------------------------------------
# Create / rotate / list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_customer_returns_api_key_and_install_urls(live_db, settings) -> None:
    resp = await _admin_request(
        "POST",
        "/admin/customers",
        headers=_auth(),
        json={
            "customer_id": "acme",
            "display_name": "Acme Corp",
            "redirect_uri_base": "https://api.example.com",
        },
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["customer_id"] == "acme"
    assert body["display_name"] == "Acme Corp"
    assert body["api_key"]  # plaintext key returned once
    assert body["bucket"]
    urls = body["install_urls"]
    # No-OAuth sources (Granola — paste-an-API-key flow) are excluded from
    # install_urls. The dashboard renders a custom modal for them instead.
    assert set(urls.keys()) == {
        s.value for s in SourceSystem if s != SourceSystem.GRANOLA
    }
    assert (
        urls["slack"]
        == "https://api.example.com/oauth/slack/install"
        "?customer_id=acme&redirect_uri=https://api.example.com/oauth/slack/callback"
    )

    # Confirm DB stores the hashed key.
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT api_key_hash FROM customers WHERE customer_id='acme'"
        )
    assert (
        row["api_key_hash"]
        == hashlib.sha256(body["api_key"].encode()).hexdigest()
    )


@pytest.mark.asyncio
async def test_create_customer_409_on_duplicate(live_db, settings) -> None:
    payload = {
        "customer_id": "dupe",
        "display_name": "Dupe",
        "redirect_uri_base": "https://api.example.com",
    }
    r1 = await _admin_request("POST", "/admin/customers", headers=_auth(), json=payload)
    assert r1.status_code == 200, r1.text
    r2 = await _admin_request("POST", "/admin/customers", headers=_auth(), json=payload)
    await init_pool(settings)
    assert r2.status_code == 409, r2.text


@pytest.mark.asyncio
async def test_rotate_key_returns_new_key_and_invalidates_old(live_db, settings) -> None:
    create = await _admin_request(
        "POST",
        "/admin/customers",
        headers=_auth(),
        json={
            "customer_id": "rot",
            "display_name": "Rotate",
            "redirect_uri_base": "https://api.example.com",
        },
    )
    assert create.status_code == 200, create.text
    old_key = create.json()["api_key"]
    old_hash = hashlib.sha256(old_key.encode()).hexdigest()

    rot = await _admin_request(
        "POST", "/admin/customers/rot/rotate_key", headers=_auth()
    )
    await init_pool(settings)
    assert rot.status_code == 200, rot.text
    new_key = rot.json()["api_key"]
    assert new_key != old_key

    async with raw_conn() as conn:
        current = await conn.fetchval(
            "SELECT api_key_hash FROM customers WHERE customer_id='rot'"
        )
    assert current == hashlib.sha256(new_key.encode()).hexdigest()
    assert current != old_hash


@pytest.mark.asyncio
async def test_rotate_key_404_for_missing_customer(live_db, settings) -> None:
    resp = await _admin_request(
        "POST", "/admin/customers/ghost/rotate_key", headers=_auth()
    )
    await init_pool(settings)
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_delete_customer_removes_row_and_cascades(live_db, settings) -> None:
    create = await _admin_request(
        "POST",
        "/admin/customers",
        headers=_auth(),
        json={
            "customer_id": "del",
            "display_name": "Delete Me",
            "redirect_uri_base": "https://api.example.com",
        },
    )
    assert create.status_code == 200, create.text

    # Seed a child row in a cascading table and a non-cascading RLS table
    # to prove both get nuked.
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO integration_tokens
                (customer_id, source_system, access_token_encrypted, status)
            VALUES ('del', 'slack', 'encrypted', 'active')
            """
        )
        await conn.execute(
            """
            INSERT INTO audit_log
                (customer_id, actor_id, action, resource_type, resource_id)
            VALUES ('del', 'system', 'test', 'doc', 'doc-1')
            """
        )

    delete = await _admin_request(
        "DELETE", "/admin/customers/del", headers=_auth()
    )
    await init_pool(settings)
    assert delete.status_code == 204, delete.text

    async with raw_conn() as conn:
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM customers WHERE customer_id='del'"
        ) == 0
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM integration_tokens WHERE customer_id='del'"
        ) == 0
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM audit_log WHERE customer_id='del'"
        ) == 0


@pytest.mark.asyncio
async def test_delete_customer_404_for_missing(live_db, settings) -> None:
    resp = await _admin_request(
        "DELETE", "/admin/customers/ghost", headers=_auth()
    )
    await init_pool(settings)
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_list_customers(live_db, settings) -> None:
    for cid in ("one", "two"):
        await _admin_request(
            "POST",
            "/admin/customers",
            headers=_auth(),
            json={
                "customer_id": cid,
                "display_name": cid.upper(),
                "redirect_uri_base": "https://api.example.com",
            },
        )

    resp = await _admin_request("GET", "/admin/customers", headers=_auth())
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    ids = [c["customer_id"] for c in resp.json()["customers"]]
    assert set(ids) >= {"one", "two"}


# ---------------------------------------------------------------------------
# Integrations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_integrations_for_customer(live_db, settings) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ('intg', 'I', 'hashX') ON CONFLICT DO NOTHING"
        )
        # Seed a Slack integration_tokens row.
        await conn.execute(
            """
            INSERT INTO integration_tokens
                (customer_id, source_system, access_token_encrypted, status)
            VALUES ($1, $2, 'enc', $3)
            ON CONFLICT (customer_id, source_system) DO UPDATE
            SET access_token_encrypted = EXCLUDED.access_token_encrypted,
                status = EXCLUDED.status
            """,
            "intg",
            SourceSystem.SLACK.value,
            IntegrationStatus.ACTIVE.value,
        )
        # And a workspace mapping.
        await conn.execute(
            """
            INSERT INTO customer_source_mapping
                (source_system, external_id, customer_id, external_name)
            VALUES ($1, 'T123', 'intg', 'Acme Slack')
            ON CONFLICT DO NOTHING
            """,
            SourceSystem.SLACK.value,
        )

    resp = await _admin_request(
        "GET",
        "/admin/customers/intg/integrations?redirect_uri_base=https://api.example.com",
        headers=_auth(),
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    integrations = {i["source"]: i for i in resp.json()["integrations"]}
    assert set(integrations.keys()) == {s.value for s in SourceSystem}
    slack = integrations["slack"]
    assert slack["connected"] is True
    assert slack["workspaces"] == [
        {"external_id": "T123", "external_name": "Acme Slack"}
    ]
    assert slack["install_url"].startswith("https://api.example.com/oauth/slack/install")
    # Sources without a token row are disconnected with empty workspaces.
    assert integrations["github"]["connected"] is False
    assert integrations["github"]["workspaces"] == []


@pytest.mark.asyncio
async def test_get_integrations_404_for_missing_customer(live_db, settings) -> None:
    resp = await _admin_request(
        "GET", "/admin/customers/nope/integrations", headers=_auth()
    )
    await init_pool(settings)
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# Ingestion stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_ingestion_stats(live_db, settings) -> None:
    now = datetime.now(UTC)
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ('stats', 'S', 'hashS') ON CONFLICT DO NOTHING"
        )
        # One live Slack document.
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id, source_system, source_id, source_url,
                doc_type, content_hash, created_at, updated_at, valid_from, acl, ingested_at
            ) VALUES (
                'doc-1', 1, 'stats', $1, 'src-1', 'https://x/1',
                'slack.message', 'h', $2, $2, $2, '{"principals":[],"captured_at":"2026-04-23T00:00:00+00:00"}'::jsonb, $2
            )
            """,
            SourceSystem.SLACK.value,
            now,
        )
        # One pending queue row.
        await conn.execute(
            """
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id, payload_s3_key, status)
            VALUES ('stats', $1, 'evt-1', 'raw/key', 'pending')
            """,
            SourceSystem.SLACK.value,
        )
        # A backfill_state row in running status.
        await conn.execute(
            """
            INSERT INTO backfill_state
                (customer_id, source_system, status, events_enqueued, started_at)
            VALUES ('stats', $1, 'running', 5, $2)
            """,
            SourceSystem.SLACK.value,
            now,
        )

    resp = await _admin_request(
        "GET", "/admin/customers/stats/ingestion_stats", headers=_auth()
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    per_source = {row["source"]: row for row in body["per_source"]}
    assert set(per_source.keys()) == {s.value for s in SourceSystem}
    slack = per_source["slack"]
    assert slack["documents"] == 1
    assert slack["queue_pending"] == 1
    assert slack["queue_processing"] == 0
    assert slack["queue_dlq"] == 0
    assert slack["last_ingested_at"] is not None

    backfill = {b["source"]: b for b in body["backfill"]}
    assert backfill["slack"]["status"] == "running"
    assert backfill["slack"]["events_enqueued"] == 5

    # Sources without rows still reported with zeroes.
    assert per_source["github"]["documents"] == 0
    assert per_source["github"]["queue_pending"] == 0


@pytest.mark.asyncio
async def test_get_ingestion_stats_404_for_missing_customer(live_db, settings) -> None:
    resp = await _admin_request(
        "GET", "/admin/customers/ghost/ingestion_stats", headers=_auth()
    )
    await init_pool(settings)
    assert resp.status_code == 404, resp.text
