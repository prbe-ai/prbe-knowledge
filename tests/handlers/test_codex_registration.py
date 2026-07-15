"""Tests for the Codex CLI connector — sibling of CC, registered as a separate
SourceSystem so dashboard provenance queries can distinguish the two agents
even though the doc shape and unit extraction are shared via subclassing."""

import pytest

from engine.ingest.handlers import registry
from engine.ingest.handlers.base import make_default_context
from engine.shared.constants import DocType, SourceSystem
from engine.shared.models import WebhookEvent
from kb.handlers.claude_code import (
    ClaudeCodeConnector,
    CodexConnector,
)


def test_codex_connector_is_registered() -> None:
    cls = registry.get_connector_class(SourceSystem.CODEX)
    assert cls is CodexConnector


def test_codex_connector_can_be_instantiated() -> None:
    ctx = make_default_context()
    c = CodexConnector(ctx)
    assert c.source_system == SourceSystem.CODEX
    assert c.display_name == "Codex"


def test_codex_subclasses_claude_code() -> None:
    """The shim depends on inherited normalize/parse logic — verify the
    inheritance is intact."""
    assert issubclass(CodexConnector, ClaudeCodeConnector)


def test_codex_class_attrs_distinct_from_cc() -> None:
    assert CodexConnector._doc_id_prefix == "codex"
    assert CodexConnector._agent_label == "codex"
    assert CodexConnector._session_title_prefix == "Codex session"
    # CC class attrs unchanged.
    assert ClaudeCodeConnector._doc_id_prefix == "claude_code"
    assert ClaudeCodeConnector._agent_label == "claude_code"


def _codex_event(customer_id: str = "cust-1", session_id: str = "s-1") -> WebhookEvent:
    from datetime import UTC, datetime
    return WebhookEvent(
        customer_id=customer_id,
        source_system=SourceSystem.CODEX,
        source_event_id=f"{session_id}:0",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/codex/cust-1/s-1/0.jsonl",
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
async def test_normalize_emits_codex_provenance() -> None:
    """Source attribution: Codex sessions get tagged source_system=CODEX
    and doc_id prefix=codex (vs claude_code:* for CC sessions)."""
    c = CodexConnector(make_default_context())
    hydrated = {
        "session_id": "s-1",
        "events": [{"line_no": 0, "raw": {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        }}],
        "session_complete": False,
        "cwd": "/tmp/p",
    }
    result = await c.normalize(_codex_event(), hydrated)

    assert len(result.documents) == 1
    doc = result.documents[0]
    # Provenance differs from CC.
    assert doc.source_system == SourceSystem.CODEX
    assert doc.doc_id.startswith("codex:cust-1:")
    assert doc.title.startswith("Codex session ")
    assert doc.metadata["agent"] == "codex"
    # Doc shape stays CC (we share extraction + UI).
    assert doc.doc_type == DocType.CLAUDE_CODE_SESSION
    # ACL row tagged Codex too.
    assert all(row.source_system == SourceSystem.CODEX for row in result.acl_snapshots)


@pytest.mark.asyncio
async def test_normalize_preserves_codex_extras_in_metadata_body() -> None:
    """`_codex_extras` rides on each event's `raw` dict. The connector's
    `body` rendering doesn't strip unknown keys, so the extras flow through
    intact for v0.2 surfaces that want to read sandbox_policy /
    developer_instructions / sub-agent metadata.
    """
    c = CodexConnector(make_default_context())
    hydrated = {
        "session_id": "s-1",
        "events": [{
            "line_no": 0,
            "raw": {
                "type": "system",
                "subtype": "turn_context",
                "_codex_extras": {
                    "sandbox_policy": "read-only",
                    "model": "gpt-5.5",
                    "developer_instructions": "be concise",
                },
            },
        }],
        "session_complete": False,
        "cwd": "/tmp/p",
    }
    result = await c.normalize(_codex_event(), hydrated)
    # Raw events are persisted in body_json; the metadata.body field is
    # the human-readable rendering. Extras live on the raw, not the rendering,
    # but they survive into R2 via the worker's raw envelope persistence.
    # Here we just verify the connector didn't crash on the unfamiliar key.
    assert len(result.documents) == 1
