"""Tests for the Capy connector — sibling of CC, Codex, pi and Kimi Code,
registered as a separate SourceSystem so dashboard provenance queries can
distinguish every agent even though the doc shape and unit extraction are
shared via subclassing. research-os's tap sanitizes Capy (capy.ai) sessions
into the same shared session shape before upload, so nothing here parses Capy
natively."""

import pytest

from engine.ingest.handlers import registry
from engine.ingest.handlers.base import make_default_context
from engine.shared.constants import DocType, SourceSystem
from engine.shared.models import WebhookEvent
from kb.handlers.claude_code import (
    CapyConnector,
    ClaudeCodeConnector,
)


def test_capy_connector_is_registered() -> None:
    cls = registry.get_connector_class(SourceSystem.CAPY)
    assert cls is CapyConnector


def test_capy_connector_can_be_instantiated() -> None:
    ctx = make_default_context()
    c = CapyConnector(ctx)
    assert c.source_system == SourceSystem.CAPY
    assert c.display_name == "Capy"


def test_capy_subclasses_claude_code() -> None:
    """The shim depends on inherited normalize/parse logic — verify the
    inheritance is intact."""
    assert issubclass(CapyConnector, ClaudeCodeConnector)


def test_capy_class_attrs_distinct_from_cc() -> None:
    assert CapyConnector._doc_id_prefix == "capy"
    assert CapyConnector._agent_label == "capy"
    assert CapyConnector._session_title_prefix == "Capy session"
    # CC class attrs unchanged.
    assert ClaudeCodeConnector._doc_id_prefix == "claude_code"
    assert ClaudeCodeConnector._agent_label == "claude_code"


def _capy_event(customer_id: str = "cust-1", session_id: str = "s-1") -> WebhookEvent:
    from datetime import UTC, datetime

    return WebhookEvent(
        customer_id=customer_id,
        source_system=SourceSystem.CAPY,
        source_event_id=f"{session_id}:0",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/capy/cust-1/s-1/0.jsonl",
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
async def test_normalize_emits_capy_provenance() -> None:
    """Source attribution: Capy sessions get tagged source_system=CAPY and
    doc_id prefix=capy (vs claude_code:*, codex:*, pi:*, kimi_code:* for the
    other agents)."""
    c = CapyConnector(make_default_context())
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
    result = await c.normalize(_capy_event(), hydrated)

    assert len(result.documents) == 1
    doc = result.documents[0]
    # Provenance differs from CC, Codex, pi and Kimi Code.
    assert doc.source_system == SourceSystem.CAPY
    assert doc.doc_id.startswith("capy:cust-1:")
    assert doc.title.startswith("Capy session ")
    assert doc.metadata["agent"] == "capy"
    # Doc shape stays CC (we share extraction + UI), exactly as the others.
    assert doc.doc_type == DocType.CLAUDE_CODE_SESSION
    # ACL row tagged Capy too.
    assert all(row.source_system == SourceSystem.CAPY for row in result.acl_snapshots)


@pytest.mark.asyncio
async def test_normalize_takes_a_transcript_of_messages_and_tool_summaries_only() -> None:
    """A Capy transcript is the researcher's messages, the assistant's replies
    and a one-line summary per tool call (probe-events/1 `tool_use` with a
    `summary`): no tool_result, no thinking, no compaction. The connector must
    build a session from that alone."""
    c = CapyConnector(make_default_context())
    hydrated = {
        "session_id": "s-1",
        "events": [
            {
                "line_no": 0,
                "raw": {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "fix the flaky test"}],
                    },
                },
            },
            {
                "line_no": 1,
                "raw": {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Running the suite first."},
                            {
                                "type": "tool_use",
                                "id": "t-1",
                                "name": "Bash",
                                "summary": "pytest tests/test_flaky.py",
                            },
                        ],
                    },
                },
            },
        ],
        "session_complete": False,
        "cwd": "/tmp/p",
    }
    result = await c.normalize(_capy_event(), hydrated)
    assert len(result.documents) == 1
    assert result.documents[0].source_system == SourceSystem.CAPY


@pytest.fixture
def _internal_key(monkeypatch):
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    from engine.shared.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    yield "test-internal-key"
    monkeypatch.undo()
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize(("source", "expected"), [("capy", 409), ("capy_ai", 404)])
async def test_webhook_door_resolves_capy(_internal_key, monkeypatch, source, expected):
    """research-os forwards every captured agent's batches to
    `/webhooks/{source_id}`, so Capy arrives at `/webhooks/capy`. The route is
    generic: it resolves the SourceSystem and its registered connector, then
    checks the tenant. A held tenant's 409 therefore proves the source
    resolved -- an unknown one (`capy_ai`) 404s first. No database or object
    store is touched on either path."""
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
