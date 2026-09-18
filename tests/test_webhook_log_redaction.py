"""Ignored provider events are inspected before their metadata enters logs."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request
from structlog.testing import capture_logs

from kb import ingestion_app
from kb.handlers.notion import NotionConnector

KEY = "ghp_" + hashlib.sha256(b"synthetic ignored webhook regression").hexdigest()[:36]


@pytest.fixture
def ignored_notion(monkeypatch):
    settings = SimpleNamespace(
        internal_knowledge_api_key=SimpleNamespace(get_secret_value=lambda: "synthetic-auth")
    )
    monkeypatch.setattr(ingestion_app, "get_settings", lambda: settings)
    monkeypatch.setattr(ingestion_app, "_verify_internal_key", lambda request: None)
    monkeypatch.setattr(
        ingestion_app,
        "get_ingestion_killswitch",
        AsyncMock(return_value=SimpleNamespace(enabled=True)),
    )
    monkeypatch.setattr(ingestion_app, "get_connector_class", lambda source: NotionConnector)
    # The real parser needs no provider/database state for unknown event types.
    monkeypatch.setattr(
        ingestion_app, "build_connector", lambda *args: object.__new__(NotionConnector)
    )

    async def invoke(event_type):
        raw = json.dumps(
            {"type": event_type, "entity": {"id": "synthetic", "type": "page"}}
        ).encode()

        async def receive():
            return {"type": "http.request", "body": raw, "more_body": False}

        request = Request(
            {
                "type": "http",
                "headers": [],
                "app": SimpleNamespace(state=SimpleNamespace(ctx=None)),
            },
            receive,
        )
        return await ingestion_app.webhook(
            "notion", request, x_trace_id="synthetic-trace", x_prbe_customer="synthetic-tenant"
        )

    return invoke


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", [KEY, "future.benign_event"], ids=["credential", "benign"])
async def test_ignored_notion_event_is_safe_at_structured_log_boundary(ignored_notion, event_type):
    with capture_logs() as logs:
        response = await ignored_notion(event_type)
    assert response.status_code == 200
    assert json.loads(response.body)["status"] == "ignored"
    credential_exposed = KEY in json.dumps(logs)
    assert not credential_exposed
    ignored = [row for row in logs if row.get("event") == "ingestion.ignored"]
    assert len(ignored) == 1
    if event_type != KEY:
        assert ignored[0]["event_type"] == event_type


@pytest.mark.asyncio
async def test_ignored_event_scan_failure_cannot_emit_event_fields(ignored_notion, monkeypatch):
    from engine.shared.exceptions import ScanUnavailable

    original = ingestion_app.redact_payload_async

    async def unavailable(value):
        if value == KEY:
            raise ScanUnavailable("synthetic unavailable")
        return await original(value)

    monkeypatch.setattr(ingestion_app, "redact_payload_async", unavailable)
    with capture_logs() as logs, pytest.raises(ScanUnavailable):
        await ignored_notion(KEY)
    assert not any(row.get("event") == "ingestion.ignored" for row in logs)
    credential_exposed = KEY in json.dumps(logs)
    assert not credential_exposed
