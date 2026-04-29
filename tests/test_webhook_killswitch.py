"""Integration tests for the webhook killswitch short-circuit.

We do NOT spin up a live DB or R2 here — the killswitch check happens
BEFORE any of that work, so mocking just `get_ingestion_killswitch` is
sufficient. If the killswitch path correctly aborts early, the heavy
infra never gets touched.
"""

from __future__ import annotations

from unittest import mock

import httpx
import pytest
from httpx import ASGITransport


@pytest.fixture
def _internal_key(monkeypatch):
    """Match the live config helper — set the shared secret."""
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    from shared.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    yield "test-internal-key"


@pytest.mark.asyncio
async def test_webhook_returns_503_when_killswitch_disabled(_internal_key) -> None:
    """Operator flips the killswitch off → all webhooks 503 with Retry-After.
    R2 is never touched, queue insert is never attempted."""
    from services.ingestion.main import app
    from services.system_settings.store import IngestionKillswitch

    fake_ks = IngestionKillswitch(
        enabled=False, reason="maintenance window", fetched_at=0.0
    )

    async def fake_get(*_args, **_kwargs):
        return fake_ks

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        with mock.patch(
            "services.ingestion.main.get_ingestion_killswitch", side_effect=fake_get
        ):
            resp = await client.post(
                "/webhooks/claude_code",
                content=b'{"foo":"bar"}',
                headers={
                    "X-Internal-Knowledge-Key": _internal_key,
                    "X-Prbe-Customer": "test-customer",
                    "Content-Type": "application/json",
                },
            )

    assert resp.status_code == 503
    assert resp.headers.get("retry-after") == "300"
    body = resp.json()
    assert "maintenance window" in str(body)


@pytest.mark.asyncio
async def test_webhook_503_default_reason_when_none(_internal_key) -> None:
    from services.ingestion.main import app
    from services.system_settings.store import IngestionKillswitch

    fake_ks = IngestionKillswitch(enabled=False, reason=None, fetched_at=0.0)

    async def fake_get(*_args, **_kwargs):
        return fake_ks

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        with mock.patch(
            "services.ingestion.main.get_ingestion_killswitch", side_effect=fake_get
        ):
            resp = await client.post(
                "/webhooks/claude_code",
                content=b'{"foo":"bar"}',
                headers={
                    "X-Internal-Knowledge-Key": _internal_key,
                    "X-Prbe-Customer": "test-customer",
                    "Content-Type": "application/json",
                },
            )

    assert resp.status_code == 503
    assert "ingestion paused" in str(resp.json())


@pytest.mark.asyncio
async def test_webhook_killswitch_runs_before_internal_key_check_NOT(
    _internal_key,
) -> None:
    """Confirms ordering: an unauthorized request still gets 401, NOT 503,
    even when killswitch is off. Auth is the outer ring; killswitch is just
    inside it. (We don't want to leak killswitch state to anonymous callers.)"""
    from services.ingestion.main import app
    from services.system_settings.store import IngestionKillswitch

    fake_ks = IngestionKillswitch(enabled=False, reason="x", fetched_at=0.0)

    async def fake_get(*_args, **_kwargs):
        return fake_ks

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        with mock.patch(
            "services.ingestion.main.get_ingestion_killswitch", side_effect=fake_get
        ):
            resp = await client.post(
                "/webhooks/claude_code",
                content=b"{}",
                headers={
                    # Wrong internal key.
                    "X-Internal-Knowledge-Key": "wrong",
                    "X-Prbe-Customer": "c",
                    "Content-Type": "application/json",
                },
            )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_internal_status_endpoint_returns_killswitch(_internal_key) -> None:
    """Admin/proxy reads via the internal endpoint; force_refresh=True
    bypasses the cache so flips show up immediately."""
    from services.ingestion.main import app
    from services.system_settings.store import IngestionKillswitch

    fake_ks = IngestionKillswitch(
        enabled=False, reason="testing", fetched_at=0.0
    )
    captured = {"force_refresh": None}

    async def fake_get(*, force_refresh: bool = False):
        captured["force_refresh"] = force_refresh
        return fake_ks

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        with mock.patch(
            "services.ingestion.main.get_ingestion_killswitch", side_effect=fake_get
        ):
            resp = await client.get(
                "/api/internal/ingestion-status",
                headers={"X-Internal-Knowledge-Key": _internal_key},
            )

    assert resp.status_code == 200
    assert resp.json() == {"enabled": False, "reason": "testing"}
    assert captured["force_refresh"] is True


@pytest.mark.asyncio
async def test_internal_status_endpoint_requires_internal_key(_internal_key) -> None:
    from services.ingestion.main import app

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/api/internal/ingestion-status")
    assert resp.status_code == 401
