"""Granola admin endpoints + the cursor-preserving re-enqueue helper.

Covers:
  - POST /admin/customers/{id}/integrations/granola
      * 404 when customer doesn't exist
      * 400 when tier is invalid
      * 400 when Granola rejects the key
      * 200 stores encrypted token + enqueues initial backfill
      * upsert: re-connect overwrites existing token
  - DELETE — sets status=revoked
  - POST /refresh
      * 404 when not configured
      * 409 when token revoked
      * 429 when called within debounce window
      * 200 + pg_notify on success
  - re_enqueue_for_polling preserves last_cursor and no-ops on running

All tests use ASGI in-process + live_db. The Granola API validate call is
stubbed via respx so no real network traffic is sent.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx
from httpx import ASGITransport

from services.ingestion.backfill_runner import (
    enqueue_backfill,
    re_enqueue_for_polling,
)
from shared.config import Settings, get_settings
from shared.constants import (
    GRANOLA_REFRESH_CHANNEL,
    BackfillStatus,
    IntegrationStatus,
    SourceSystem,
)
from shared.db import close_pool, raw_conn
from shared.embeddings import reset_embedder
from shared.encryption import decrypt_token
from shared.storage import reset_store

ADMIN_KEY = "admin-test-key-granola"
GRANOLA_NOTES_URL = "https://public-api.granola.ai/v1/notes"


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv(
        "TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value()
    )
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


async def _admin_request(
    method: str,
    path: str,
    *,
    json: dict | None = None,
) -> httpx.Response:
    from services.ingestion.main import app as ingestion_app

    await close_pool()
    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        resp = await client.request(
            method, path, headers={"X-Admin-Key": ADMIN_KEY}, json=json
        )
    return resp


async def _seed_customer(customer_id: str = "cust-1") -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash, status)
            VALUES ($1, $2, $3, 'active')
            """,
            customer_id,
            "Test Customer",
            "stub-hash",
        )


# ---------------------------------------------------------------------------
# CONNECT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_404_when_customer_missing(live_db) -> None:
    resp = await _admin_request(
        "POST",
        "/api/customers/nope/integrations/granola",
        json={"api_key": "grn_test123", "tier": "enterprise"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_connect_400_on_invalid_tier(live_db) -> None:
    await _seed_customer()
    resp = await _admin_request(
        "POST",
        "/api/customers/cust-1/integrations/granola",
        json={"api_key": "grn_test123", "tier": "freebie"},
    )
    assert resp.status_code == 400
    assert "tier" in resp.text


@pytest.mark.asyncio
@respx.mock
async def test_connect_400_when_granola_rejects_key(live_db) -> None:
    await _seed_customer()
    respx.get(GRANOLA_NOTES_URL).mock(
        return_value=httpx.Response(401, json={"message": "bad token"})
    )
    resp = await _admin_request(
        "POST",
        "/api/customers/cust-1/integrations/granola",
        json={"api_key": "grn_test123", "tier": "enterprise"},
    )
    assert resp.status_code == 400
    assert "Granola rejected" in resp.text


@pytest.mark.asyncio
@respx.mock
async def test_connect_stores_encrypted_token_and_enqueues_backfill(live_db) -> None:
    await _seed_customer()
    respx.get(GRANOLA_NOTES_URL).mock(
        return_value=httpx.Response(200, json={"notes": [], "hasMore": False})
    )
    resp = await _admin_request(
        "POST",
        "/api/customers/cust-1/integrations/granola",
        json={"api_key": "grn_secret_001", "tier": "enterprise"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["scope"] == "tier:enterprise"
    assert body["backfill_enqueued"] is True

    async with raw_conn() as conn:
        token_row = await conn.fetchrow(
            "SELECT * FROM integration_tokens WHERE customer_id=$1 AND source_system=$2",
            "cust-1",
            SourceSystem.GRANOLA.value,
        )
        bf_row = await conn.fetchrow(
            "SELECT * FROM backfill_state WHERE customer_id=$1 AND source_system=$2",
            "cust-1",
            SourceSystem.GRANOLA.value,
        )

    assert token_row is not None
    assert token_row["status"] == IntegrationStatus.ACTIVE.value
    assert token_row["scope"] == "tier:enterprise"
    # Token stored encrypted (plaintext should not appear).
    assert "grn_secret_001" not in token_row["access_token_encrypted"]
    assert decrypt_token(token_row["access_token_encrypted"]) == "grn_secret_001"

    assert bf_row is not None
    assert bf_row["status"] == BackfillStatus.PENDING.value
    assert bf_row["last_cursor"] is None  # initial backfill clears cursor


@pytest.mark.asyncio
@respx.mock
async def test_connect_overwrites_existing_token_on_reconnect(live_db) -> None:
    await _seed_customer()
    respx.get(GRANOLA_NOTES_URL).mock(
        return_value=httpx.Response(200, json={"notes": [], "hasMore": False})
    )

    # First connect (personal tier).
    await _admin_request(
        "POST",
        "/api/customers/cust-1/integrations/granola",
        json={"api_key": "grn_first", "tier": "personal"},
    )
    # Reconnect with enterprise tier and a new key.
    await _admin_request(
        "POST",
        "/api/customers/cust-1/integrations/granola",
        json={"api_key": "grn_second", "tier": "enterprise"},
    )

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT scope, access_token_encrypted FROM integration_tokens "
            "WHERE customer_id=$1 AND source_system=$2",
            "cust-1",
            SourceSystem.GRANOLA.value,
        )
    assert row["scope"] == "tier:enterprise"
    assert decrypt_token(row["access_token_encrypted"]) == "grn_second"


# ---------------------------------------------------------------------------
# DISCONNECT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_disconnect_marks_revoked(live_db) -> None:
    await _seed_customer()
    respx.get(GRANOLA_NOTES_URL).mock(
        return_value=httpx.Response(200, json={"notes": [], "hasMore": False})
    )
    await _admin_request(
        "POST",
        "/api/customers/cust-1/integrations/granola",
        json={"api_key": "grn_test123", "tier": "enterprise"},
    )

    resp = await _admin_request("DELETE", "/api/customers/cust-1/integrations/granola")
    assert resp.status_code == 204

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM integration_tokens "
            "WHERE customer_id=$1 AND source_system=$2",
            "cust-1",
            SourceSystem.GRANOLA.value,
        )
    assert row["status"] == IntegrationStatus.REVOKED.value


# ---------------------------------------------------------------------------
# REFRESH
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_404_when_not_configured(live_db) -> None:
    await _seed_customer()
    resp = await _admin_request(
        "POST", "/api/customers/cust-1/integrations/granola/refresh"
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
@respx.mock
async def test_refresh_409_when_revoked(live_db) -> None:
    await _seed_customer()
    respx.get(GRANOLA_NOTES_URL).mock(
        return_value=httpx.Response(200, json={"notes": [], "hasMore": False})
    )
    await _admin_request(
        "POST",
        "/api/customers/cust-1/integrations/granola",
        json={"api_key": "grn_test123", "tier": "enterprise"},
    )
    await _admin_request("DELETE", "/api/customers/cust-1/integrations/granola")

    resp = await _admin_request(
        "POST", "/api/customers/cust-1/integrations/granola/refresh"
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
@respx.mock
async def test_refresh_debounces_within_30s(live_db, monkeypatch) -> None:
    await _seed_customer()
    respx.get(GRANOLA_NOTES_URL).mock(
        return_value=httpx.Response(200, json={"notes": [], "hasMore": False})
    )
    await _admin_request(
        "POST",
        "/api/customers/cust-1/integrations/granola",
        json={"api_key": "grn_test123", "tier": "enterprise"},
    )

    # Mark backfill complete so re_enqueue actually fires (otherwise it's a no-op
    # because status=pending). The debounce check happens BEFORE the re_enqueue,
    # so the second call's 429 doesn't depend on backfill state.
    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE backfill_state SET status=$1, last_progress_at=NOW() "
            "WHERE customer_id=$2 AND source_system=$3",
            BackfillStatus.COMPLETE.value,
            "cust-1",
            SourceSystem.GRANOLA.value,
        )

    first = await _admin_request(
        "POST", "/api/customers/cust-1/integrations/granola/refresh"
    )
    assert first.status_code == 200
    assert first.json()["triggered"] is True

    # Immediate second call: 429 with Retry-After.
    second = await _admin_request(
        "POST", "/api/customers/cust-1/integrations/granola/refresh"
    )
    assert second.status_code == 429
    assert "Retry-After" in second.headers


# ---------------------------------------------------------------------------
# re_enqueue_for_polling helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_re_enqueue_preserves_cursor(live_db) -> None:
    await _seed_customer()
    # Seed a complete backfill_state row with a cursor.
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO backfill_state
                (customer_id, source_system, status, last_cursor,
                 events_enqueued, completed_at, last_progress_at)
            VALUES ($1, $2, $3, $4, 100, NOW(), NOW())
            """,
            "cust-1",
            SourceSystem.GRANOLA.value,
            BackfillStatus.COMPLETE.value,
            '{"watermark": "2026-04-23T00:00:00Z"}',
        )

    triggered = await re_enqueue_for_polling("cust-1", SourceSystem.GRANOLA)
    assert triggered is True

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, last_cursor, events_enqueued FROM backfill_state "
            "WHERE customer_id=$1 AND source_system=$2",
            "cust-1",
            SourceSystem.GRANOLA.value,
        )

    assert row["status"] == BackfillStatus.PENDING.value
    # Cursor MUST be preserved — this is the whole point of the helper.
    assert row["last_cursor"] == '{"watermark": "2026-04-23T00:00:00Z"}'
    # Counter is NOT reset (it'll keep growing on subsequent ticks).
    assert row["events_enqueued"] == 100


@pytest.mark.asyncio
async def test_re_enqueue_noops_on_running(live_db) -> None:
    await _seed_customer()
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO backfill_state
                (customer_id, source_system, status, last_cursor, events_enqueued)
            VALUES ($1, $2, $3, $4, 0)
            """,
            "cust-1",
            SourceSystem.GRANOLA.value,
            BackfillStatus.RUNNING.value,
            '{"watermark": "x"}',
        )

    triggered = await re_enqueue_for_polling("cust-1", SourceSystem.GRANOLA)
    assert triggered is False  # don't restart in-flight work

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM backfill_state "
            "WHERE customer_id=$1 AND source_system=$2",
            "cust-1",
            SourceSystem.GRANOLA.value,
        )
    assert row["status"] == BackfillStatus.RUNNING.value


@pytest.mark.asyncio
async def test_enqueue_backfill_clears_cursor_for_initial_sync(live_db) -> None:
    """Sanity check: enqueue_backfill (initial path) DOES clear cursor — that's
    the whole reason re_enqueue_for_polling exists as a separate helper."""
    await _seed_customer()
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO backfill_state
                (customer_id, source_system, status, last_cursor, events_enqueued)
            VALUES ($1, $2, $3, $4, 100)
            """,
            "cust-1",
            SourceSystem.GRANOLA.value,
            BackfillStatus.COMPLETE.value,
            '{"watermark": "2026-04-23"}',
        )

    await enqueue_backfill("cust-1", SourceSystem.GRANOLA)

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, last_cursor, events_enqueued FROM backfill_state "
            "WHERE customer_id=$1 AND source_system=$2",
            "cust-1",
            SourceSystem.GRANOLA.value,
        )
    assert row["status"] == BackfillStatus.PENDING.value
    assert row["last_cursor"] is None
    assert row["events_enqueued"] == 0


# ---------------------------------------------------------------------------
# pg_notify wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_refresh_fires_pg_notify(live_db, settings) -> None:
    """Manual refresh should send NOTIFY granola_refresh.

    Sets up a LISTEN on the channel before calling refresh; verifies the
    notification arrives. This is the load-bearing test for the manual-refresh
    end-to-end path.
    """
    import asyncpg

    await _seed_customer()
    respx.get(GRANOLA_NOTES_URL).mock(
        return_value=httpx.Response(200, json={"notes": [], "hasMore": False})
    )
    await _admin_request(
        "POST",
        "/api/customers/cust-1/integrations/granola",
        json={"api_key": "grn_test123", "tier": "enterprise"},
    )
    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE backfill_state SET status=$1, last_progress_at=NOW() "
            "WHERE customer_id=$2 AND source_system=$3",
            BackfillStatus.COMPLETE.value,
            "cust-1",
            SourceSystem.GRANOLA.value,
        )

    received: list[str] = []
    notify_event = asyncio.Event()
    listen_conn = await asyncpg.connect(settings.database_url)

    def _on_notify(_c, _pid, _ch, payload) -> None:
        received.append(payload)
        notify_event.set()

    try:
        await listen_conn.add_listener(GRANOLA_REFRESH_CHANNEL, _on_notify)

        resp = await _admin_request(
            "POST", "/api/customers/cust-1/integrations/granola/refresh"
        )
        assert resp.status_code == 200

        # Wait briefly for the notification to round-trip through Postgres.
        try:
            await asyncio.wait_for(notify_event.wait(), timeout=2.0)
        except TimeoutError:
            pytest.fail("did not receive pg_notify within 2s")
    finally:
        await listen_conn.close()

    assert received == ["cust-1"]
