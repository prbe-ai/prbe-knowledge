"""Customer-source mapping: round-trip + resolution precedence.

Covers:
  - record_mapping / resolve_customer roundtrip
  - single_customer_fallback when only one tenant exists
  - Each connector's extract_external_id_from_payload on a realistic fixture
  - The webhook handler's resolution path when no X-Prbe-Customer header is sent
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from shared.config import Settings, get_settings
from shared.constants import SourceSystem
from shared.customer_mapping import (
    record_mapping,
    resolve_customer,
    single_customer_fallback,
)
from shared.db import close_pool, init_pool, raw_conn
from shared.embeddings import reset_embedder
from shared.storage import reset_store


@pytest.fixture(autouse=True)
def _patch(monkeypatch, settings: Settings):
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_record_and_resolve(live_db) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1,'A','x') ON CONFLICT DO NOTHING",
            "cust-map",
        )

    await record_mapping(
        customer_id="cust-map",
        source_system=SourceSystem.SLACK,
        external_id="T_ACME",
        external_name="Acme",
    )
    resolved = await resolve_customer(SourceSystem.SLACK, "T_ACME")
    assert resolved == "cust-map"

    # Unknown external_id returns None.
    assert await resolve_customer(SourceSystem.SLACK, "T_UNKNOWN") is None


@pytest.mark.asyncio
async def test_single_customer_fallback(live_db) -> None:
    # Zero customers → None
    assert await single_customer_fallback() is None

    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ('solo','s','x') ON CONFLICT DO NOTHING"
        )
    assert await single_customer_fallback() == "solo"

    # Two customers → None (ambiguous)
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ('second','t','x') ON CONFLICT DO NOTHING"
        )
    assert await single_customer_fallback() is None


def _dummy_ctx():
    import httpx as _h

    from services.ingestion.handlers.base import ConnectorContext

    return ConnectorContext(settings=Settings(), http=_h.AsyncClient())


def test_slack_extract_external_id() -> None:
    from services.ingestion.handlers.registry import build_connector

    c = build_connector(SourceSystem.SLACK, _dummy_ctx())
    assert c.extract_external_id_from_payload({}, {"team_id": "T_X"}) == "T_X"
    assert c.extract_external_id_from_payload({}, {"team": {"id": "T_Y"}}) == "T_Y"
    assert c.extract_external_id_from_payload({}, {}) is None


def test_linear_extract_external_id() -> None:
    from services.ingestion.handlers.registry import build_connector

    c = build_connector(SourceSystem.LINEAR, _dummy_ctx())
    assert c.extract_external_id_from_payload({}, {"organizationId": "O_1"}) == "O_1"
    assert c.extract_external_id_from_payload({}, {}) is None


def test_github_extract_external_id() -> None:
    from services.ingestion.handlers.registry import build_connector

    c = build_connector(SourceSystem.GITHUB, _dummy_ctx())
    assert (
        c.extract_external_id_from_payload({}, {"installation": {"id": 42}}) == "42"
    )
    assert c.extract_external_id_from_payload({}, {}) is None


def test_notion_extract_external_id() -> None:
    from services.ingestion.handlers.registry import build_connector

    c = build_connector(SourceSystem.NOTION, _dummy_ctx())
    assert (
        c.extract_external_id_from_payload({}, {"workspace_id": "W_1"}) == "W_1"
    )
    assert (
        c.extract_external_id_from_payload({}, {"entity": {"workspace_id": "W_2"}})
        == "W_2"
    )


def test_sentry_extract_external_id() -> None:
    from services.ingestion.handlers.registry import build_connector

    c = build_connector(SourceSystem.SENTRY, _dummy_ctx())
    assert (
        c.extract_external_id_from_payload({}, {"organization": {"slug": "acme"}})
        == "acme"
    )
    assert (
        c.extract_external_id_from_payload(
            {}, {"installation": {"organization": {"slug": "acme2"}}}
        )
        == "acme2"
    )


@pytest.mark.asyncio
async def test_webhook_resolves_via_mapping_without_header(
    live_db, settings: Settings
) -> None:
    """Webhook with no X-Prbe-Customer resolves via customer_source_mapping."""
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1,'x','y') ON CONFLICT DO NOTHING",
            "cust-map",
        )
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1,'z','y') ON CONFLICT DO NOTHING",
            "cust-other",
        )

    # Register Slack team T_TEST against cust-map.
    await record_mapping(
        customer_id="cust-map",
        source_system=SourceSystem.SLACK,
        external_id="T_TEST",
        external_name="Test Workspace",
    )

    from shared.storage import get_store

    store = get_store()
    await store.ensure_bucket(store.bucket_for("cust-map"))

    fixture = json.loads(
        (Path(__file__).parent.parent / "fixtures" / "slack" / "message_simple.json").read_text()
    )
    body = json.dumps(fixture).encode()

    ts = str(int(time.time()))
    sig = (
        "v0="
        + hmac.new(
            b"test-secret", f"v0:{ts}:".encode() + body, hashlib.sha256
        ).hexdigest()
    )
    # No X-Prbe-Customer header — the resolver should use team_id from the payload.
    headers = {
        "content-type": "application/json",
        "x-slack-request-timestamp": ts,
        "x-slack-signature": sig,
    }

    from services.ingestion.main import app as ingestion_app

    await close_pool()
    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        resp = await client.post("/webhooks/slack", content=body, headers=headers)
    await init_pool(settings)

    assert resp.status_code == 200, resp.text
    # The queue row should be attributed to cust-map, not cust-other.
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT customer_id FROM ingestion_queue WHERE source_system='slack'"
        )
    assert row is not None
    assert row["customer_id"] == "cust-map"
