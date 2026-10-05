"""Tests for the Kimi Code connector — sibling of CC, Codex and pi, registered
as a separate SourceSystem so dashboard provenance queries can distinguish
every agent even though the doc shape and unit extraction are shared via
subclassing. research-os's tap sanitizes Kimi Code's wire log into the same
shared session shape before upload, so nothing here parses Kimi natively."""

import pytest

from engine.ingest.handlers import registry
from engine.ingest.handlers.base import make_default_context
from engine.shared.constants import DocType, SourceSystem
from engine.shared.models import WebhookEvent
from kb.handlers.claude_code import (
    ClaudeCodeConnector,
    KimiCodeConnector,
)


def test_kimi_code_connector_is_registered() -> None:
    cls = registry.get_connector_class(SourceSystem.KIMI_CODE)
    assert cls is KimiCodeConnector


def test_kimi_code_connector_can_be_instantiated() -> None:
    ctx = make_default_context()
    c = KimiCodeConnector(ctx)
    assert c.source_system == SourceSystem.KIMI_CODE
    assert c.display_name == "Kimi Code"


def test_kimi_code_subclasses_claude_code() -> None:
    """The shim depends on inherited normalize/parse logic — verify the
    inheritance is intact."""
    assert issubclass(KimiCodeConnector, ClaudeCodeConnector)


def test_kimi_code_class_attrs_distinct_from_cc() -> None:
    assert KimiCodeConnector._doc_id_prefix == "kimi_code"
    assert KimiCodeConnector._agent_label == "kimi_code"
    assert KimiCodeConnector._session_title_prefix == "Kimi Code session"
    # CC class attrs unchanged.
    assert ClaudeCodeConnector._doc_id_prefix == "claude_code"
    assert ClaudeCodeConnector._agent_label == "claude_code"


def _kimi_event(customer_id: str = "cust-1", session_id: str = "s-1") -> WebhookEvent:
    from datetime import UTC, datetime

    return WebhookEvent(
        customer_id=customer_id,
        source_system=SourceSystem.KIMI_CODE,
        source_event_id=f"{session_id}:0",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/kimi_code/cust-1/s-1/0.jsonl",
        raw_payload={
            "device_id": "dev-1",
            "session_id": session_id,
            "batch_seq": 0,
            "cwd": "/tmp/p",
            "events": [],
            "employee_id": "emp-1",
        },
        headers={},
    )


@pytest.mark.asyncio
async def test_normalize_emits_kimi_code_provenance() -> None:
    """Source attribution: Kimi Code sessions get tagged
    source_system=KIMI_CODE and doc_id prefix=kimi_code (vs claude_code:*,
    codex:*, pi:* for the other agents)."""
    c = KimiCodeConnector(make_default_context())
    hydrated = {
        "session_id": "s-1",
        "events": [
            {
                "line_no": 0,
                "raw": {
                    "type": "user",
                    "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                },
            }
        ],
        "session_complete": False,
        "cwd": "/tmp/p",
    }
    result = await c.normalize(_kimi_event(), hydrated)

    assert len(result.documents) == 1
    doc = result.documents[0]
    # Provenance differs from CC, Codex and pi.
    assert doc.source_system == SourceSystem.KIMI_CODE
    assert doc.doc_id.startswith("kimi_code:cust-1:")
    assert doc.title.startswith("Kimi Code session ")
    assert doc.metadata["agent"] == "kimi_code"
    # Doc shape stays CC (we share extraction + UI), exactly as pi and Codex.
    assert doc.doc_type == DocType.CLAUDE_CODE_SESSION
    # ACL row tagged Kimi Code too.
    assert all(row.source_system == SourceSystem.KIMI_CODE for row in result.acl_snapshots)


@pytest.mark.asyncio
async def test_normalize_tolerates_unknown_kimi_keys_on_an_event() -> None:
    """Whatever Kimi-specific fields the sanitizer keeps ride on each event's
    `raw` dict, as `_pi_extras` / `_codex_extras` do. The connector must not
    crash on a key it has never seen."""
    c = KimiCodeConnector(make_default_context())
    hydrated = {
        "session_id": "s-1",
        "events": [
            {
                "line_no": 0,
                "raw": {
                    "type": "system",
                    "subtype": "turn_context",
                    "_kimi_extras": {"model": "kimi-k2", "wire_type": "StatusUpdate"},
                },
            }
        ],
        "session_complete": False,
        "cwd": "/tmp/p",
    }
    result = await c.normalize(_kimi_event(), hydrated)
    assert len(result.documents) == 1


@pytest.fixture
def _internal_key(monkeypatch):
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    from engine.shared.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    yield "test-internal-key"
    monkeypatch.undo()
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize(("source", "expected"), [("kimi_code", 409), ("kimi", 404)])
async def test_webhook_door_resolves_kimi_code(_internal_key, monkeypatch, source, expected):
    """research-os forwards every captured agent's batches to
    `/webhooks/{source_id}`, so Kimi Code arrives at `/webhooks/kimi_code`.
    The route is generic: it resolves the SourceSystem and its registered
    connector, then checks the tenant. A held tenant's 409 therefore proves
    the source resolved -- an unknown one (`kimi`, the CLI alias) 404s first.
    No database or object store is touched on either path."""
    import httpx
    from httpx import ASGITransport

    from engine.system_settings.store import IngestionKillswitch
    from kb import ingestion_app

    async def killswitch(*_args, **_kwargs):
        return IngestionKillswitch(enabled=True, reason=None, fetched_at=0.0)

    async def held(_customer):
        return {"reason": "tenant_held"}

    async def unredacted(value):
        # The trace id is redacted before routing; the scanner is not what
        # this test is about, and it needs the redactd binary.
        return value

    monkeypatch.setattr(ingestion_app, "get_ingestion_killswitch", killswitch)
    monkeypatch.setattr(ingestion_app, "redact_payload_async", unredacted)
    # ASGITransport skips the lifespan that builds the connector context.
    monkeypatch.setattr(ingestion_app.app.state, "ctx", make_default_context(), raising=False)
    monkeypatch.setattr(ingestion_app, "refusal_for", held)
    transport = ASGITransport(app=ingestion_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            f"/webhooks/{source}",
            content=b"{}",
            headers={
                "X-Internal-Knowledge-Key": _internal_key,
                "X-Prbe-Customer": "test-customer",
                "Content-Type": "application/json",
            },
        )
    assert resp.status_code == expected, resp.text
