"""Slack URL verification handshake — the webhook route must echo the challenge."""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import httpx
import pytest
from httpx import ASGITransport

from shared.config import Settings, get_settings
from shared.db import close_pool

SECRET = "test-secret"


def _signed(body: bytes) -> dict[str, str]:
    ts = str(int(time.time()))
    sig = (
        "v0="
        + hmac.new(SECRET.encode(), f"v0:{ts}:".encode() + body, hashlib.sha256).hexdigest()
    )
    return {
        "content-type": "application/json",
        "x-slack-request-timestamp": ts,
        "x-slack-signature": sig,
    }


@pytest.fixture(autouse=True)
def _patch(monkeypatch, settings: Settings):
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SECRET)
    monkeypatch.setenv("ENVIRONMENT", "local")
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_url_verification_echoes_challenge() -> None:
    body = json.dumps(
        {
            "token": "ignored-legacy-token",
            "challenge": "abc123challenge",
            "type": "url_verification",
        }
    ).encode()

    from services.ingestion.main import app as ingestion_app

    await close_pool()
    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        resp = await client.post("/webhooks/slack", content=body, headers=_signed(body))

    assert resp.status_code == 200
    assert resp.json() == {"challenge": "abc123challenge"}
