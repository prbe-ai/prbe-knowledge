from __future__ import annotations

import json

import httpx
import pytest

from engine.ingest.handlers.base import ConnectorContext
from engine.shared.config import Settings, get_settings
from engine.shared.constants import BackfillStatus, SourceSystem
from engine.shared.customer_mapping import record_mapping
from engine.shared.db import raw_conn
from engine.shared.embeddings import reset_embedder
from engine.shared.models import IntegrationToken
from engine.shared.storage import reset_store
from engine.shared.tokens import save_token
from kb.slack_lifecycle import handle_slack_lifecycle_event


@pytest.fixture(autouse=True)
def _patch(monkeypatch, settings: Settings):
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


async def _insert_customer(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'x', 'y') ON CONFLICT DO NOTHING",
            customer_id,
        )


async def _save_slack_token(customer_id: str, *, scope: str | None = None) -> None:
    await save_token(
        IntegrationToken(
            customer_id=customer_id,
            source_system=SourceSystem.SLACK,
            access_token="xoxb-test",
            scope=scope or "channels:history,channels:join,channels:read",
        )
    )


async def _cursor_for(customer_id: str) -> dict:
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, last_cursor FROM backfill_state WHERE customer_id=$1",
            customer_id,
        )
    assert row is not None
    return {"status": row["status"], **json.loads(row["last_cursor"])}


@pytest.mark.asyncio
async def test_member_joined_channel_queues_backfill_when_joined_user_is_bot(
    live_db,
) -> None:
    await _insert_customer("cust-slack-life")
    await _save_slack_token("cust-slack-life")
    await record_mapping(
        customer_id="cust-slack-life",
        source_system=SourceSystem.SLACK,
        external_id="T1",
        metadata={"bot_user_id": "UBOT"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected Slack API call: {request.url}")

    ctx = ConnectorContext(
        settings=Settings(),
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        result = await handle_slack_lifecycle_event(
            ctx,
            "cust-slack-life",
            {
                "type": "event_callback",
                "team_id": "T1",
                "event": {
                    "type": "member_joined_channel",
                    "user": "UBOT",
                    "channel": "CNEW",
                    "team": "T1",
                },
            },
        )
    finally:
        await ctx.http.aclose()

    assert result is not None
    assert result["status"] == "accepted"
    assert result["backfill_queued"] is True
    cursor = await _cursor_for("cust-slack-life")
    assert cursor["status"] == BackfillStatus.PENDING.value
    assert cursor["active"] == {"CNEW": None}


@pytest.mark.asyncio
async def test_member_joined_channel_ignores_non_bot_user(live_db) -> None:
    await _insert_customer("cust-slack-nonbot")
    await _save_slack_token("cust-slack-nonbot")
    await record_mapping(
        customer_id="cust-slack-nonbot",
        source_system=SourceSystem.SLACK,
        external_id="T1",
        metadata={"bot_user_id": "UBOT"},
    )

    ctx = ConnectorContext(settings=Settings(), http=httpx.AsyncClient())
    try:
        result = await handle_slack_lifecycle_event(
            ctx,
            "cust-slack-nonbot",
            {
                "type": "event_callback",
                "team_id": "T1",
                "event": {
                    "type": "member_joined_channel",
                    "user": "UOTHER",
                    "channel": "CNEW",
                    "team": "T1",
                },
            },
        )
    finally:
        await ctx.http.aclose()

    assert result is not None
    assert result["status"] == "ignored"
    assert result["reason"] == "non_bot_user_joined"
    async with raw_conn() as conn:
        count = await conn.fetchval(
            "SELECT count(*) FROM backfill_state WHERE customer_id='cust-slack-nonbot'"
        )
    assert count == 0


@pytest.mark.asyncio
async def test_channel_created_joins_public_channel_and_queues_backfill(live_db) -> None:
    await _insert_customer("cust-slack-created")
    await _save_slack_token("cust-slack-created")
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.method == "POST" and request.url.path == "/api/conversations.join":
            return httpx.Response(200, json={"ok": True, "channel": {"id": "CNEW"}})
        return httpx.Response(404, json={"ok": False, "error": "unmocked"})

    ctx = ConnectorContext(
        settings=Settings(),
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        result = await handle_slack_lifecycle_event(
            ctx,
            "cust-slack-created",
            {
                "type": "event_callback",
                "team_id": "T1",
                "event": {
                    "type": "channel_created",
                    "channel": {"id": "CNEW", "name": "new-channel"},
                },
            },
        )
    finally:
        await ctx.http.aclose()

    assert calls == ["POST /api/conversations.join"]
    assert result is not None
    assert result["status"] == "accepted"
    cursor = await _cursor_for("cust-slack-created")
    assert cursor["status"] == BackfillStatus.PENDING.value
    assert cursor["active"] == {"CNEW": None}
