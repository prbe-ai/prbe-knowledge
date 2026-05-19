"""Dispatch client tests: success, retry, 4xx-no-retry, exhausted."""
from __future__ import annotations

import httpx
import pytest

from services.investigation.dispatch import (
    DispatchExhausted,
    dispatch_investigation,
)
from shared.config import get_settings

pytestmark = pytest.mark.asyncio

_OriginalAsyncClient = httpx.AsyncClient


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_BASE_URL", "http://orchestrator-test")
    monkeypatch.setenv("INTERNAL_BACKEND_API_KEY", "test-backend-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _payload(**overrides):
    base = dict(
        customer_id="cust-1",
        source="pagerduty",
        incident_doc_id="pd:incident:T-001",
        source_event_id="pd:incident:T-001:incident.triggered",
        incident_signals={
            "title": "x", "service": "checkout-svc", "triggered_at": "2026-05-17T12:00:00Z",
            "severity": None, "urgency": None, "raw_summary": "", "tags": [],
        },
        version=1,
    )
    base.update(overrides)
    return base


def _make_client_factory(transport: httpx.MockTransport):
    """Return a factory that creates a real AsyncClient with the mock transport."""
    def factory(*a, **kw):
        # Drop any 'transport' kwarg from dispatch.py (there is none, but be safe)
        kw.pop("transport", None)
        return _OriginalAsyncClient(transport=transport, **kw)
    return factory


async def test_dispatch_success_returns_quietly(monkeypatch) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(202, json={"accepted": True})
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "services.investigation.dispatch.httpx.AsyncClient",
        _make_client_factory(transport),
    )
    # Should not raise.
    await dispatch_investigation(_payload())


async def test_dispatch_retries_on_5xx_then_succeeds(monkeypatch) -> None:
    attempts = [0]
    def handler(req: httpx.Request) -> httpx.Response:
        attempts[0] += 1
        if attempts[0] < 3:
            return httpx.Response(503, text="upstream down")
        return httpx.Response(202, json={"accepted": True})
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "services.investigation.dispatch.httpx.AsyncClient",
        _make_client_factory(transport),
    )
    await dispatch_investigation(_payload(), base_delay_s=0.001)
    assert attempts[0] == 3


async def test_dispatch_4xx_does_not_retry(monkeypatch) -> None:
    """A 401 / 422 from orchestrator is a permanent schema/auth error —
    no point retrying."""
    attempts = [0]
    def handler(req: httpx.Request) -> httpx.Response:
        attempts[0] += 1
        return httpx.Response(401, text="bad key")
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "services.investigation.dispatch.httpx.AsyncClient",
        _make_client_factory(transport),
    )
    with pytest.raises(DispatchExhausted, match="401"):
        await dispatch_investigation(_payload(), base_delay_s=0.001)
    assert attempts[0] == 1


async def test_dispatch_raises_after_retry_budget_exhausted(monkeypatch) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "services.investigation.dispatch.httpx.AsyncClient",
        _make_client_factory(transport),
    )
    with pytest.raises(DispatchExhausted, match="503"):
        await dispatch_investigation(_payload(), max_attempts=3, base_delay_s=0.001)


async def test_dispatch_retries_on_transport_error(monkeypatch) -> None:
    attempts = [0]
    def handler(req: httpx.Request) -> httpx.Response:
        attempts[0] += 1
        if attempts[0] < 2:
            raise httpx.ConnectError("network down")
        return httpx.Response(202, json={"accepted": True})
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "services.investigation.dispatch.httpx.AsyncClient",
        _make_client_factory(transport),
    )
    await dispatch_investigation(_payload(), base_delay_s=0.001)
    assert attempts[0] == 2


async def test_dispatch_sets_internal_backend_key_and_customer_header(monkeypatch) -> None:
    captured: dict = {}
    def handler(req: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(req.headers)
        return httpx.Response(202, json={"accepted": True})
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "services.investigation.dispatch.httpx.AsyncClient",
        _make_client_factory(transport),
    )
    await dispatch_investigation(_payload())
    assert captured["headers"]["x-internal-backend-key"] == "test-backend-key"
    assert captured["headers"]["x-prbe-customer"] == "cust-1"
