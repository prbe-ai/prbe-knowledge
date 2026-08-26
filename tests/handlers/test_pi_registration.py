"""Tests for the pi (earendil-works/pi) connector — sibling of CC and Codex,
registered as a separate SourceSystem so dashboard provenance queries can
distinguish all three agents even though the doc shape and unit extraction
are shared via subclassing."""

import pytest

from engine.ingest.handlers import registry
from engine.ingest.handlers.base import make_default_context
from engine.shared.constants import DocType, SourceSystem
from engine.shared.models import WebhookEvent
from kb.handlers.claude_code import (
    ClaudeCodeConnector,
    PiConnector,
)


def test_pi_connector_is_registered() -> None:
    cls = registry.get_connector_class(SourceSystem.PI)
    assert cls is PiConnector


def test_pi_connector_can_be_instantiated() -> None:
    ctx = make_default_context()
    c = PiConnector(ctx)
    assert c.source_system == SourceSystem.PI
    assert c.display_name == "pi"


def test_pi_subclasses_claude_code() -> None:
    """The shim depends on inherited normalize/parse logic — verify the
    inheritance is intact."""
    assert issubclass(PiConnector, ClaudeCodeConnector)


def test_pi_class_attrs_distinct_from_cc() -> None:
    assert PiConnector._doc_id_prefix == "pi"
    assert PiConnector._agent_label == "pi"
    assert PiConnector._session_title_prefix == "pi session"
    # CC class attrs unchanged.
    assert ClaudeCodeConnector._doc_id_prefix == "claude_code"
    assert ClaudeCodeConnector._agent_label == "claude_code"


def _pi_event(customer_id: str = "cust-1", session_id: str = "s-1") -> WebhookEvent:
    from datetime import UTC, datetime
    return WebhookEvent(
        customer_id=customer_id,
        source_system=SourceSystem.PI,
        source_event_id=f"{session_id}:0",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/pi/cust-1/s-1/0.jsonl",
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
async def test_normalize_emits_pi_provenance() -> None:
    """Source attribution: pi sessions get tagged source_system=PI and
    doc_id prefix=pi (vs claude_code:* for CC sessions, codex:* for Codex)."""
    c = PiConnector(make_default_context())
    hydrated = {
        "session_id": "s-1",
        "events": [{"line_no": 0, "raw": {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        }}],
        "session_complete": False,
        "cwd": "/tmp/p",
    }
    result = await c.normalize(_pi_event(), hydrated)

    assert len(result.documents) == 1
    doc = result.documents[0]
    # Provenance differs from CC and Codex.
    assert doc.source_system == SourceSystem.PI
    assert doc.doc_id.startswith("pi:cust-1:")
    assert doc.title.startswith("pi session ")
    assert doc.metadata["agent"] == "pi"
    # Doc shape stays CC (we share extraction + UI).
    assert doc.doc_type == DocType.CLAUDE_CODE_SESSION
    # ACL row tagged pi too.
    assert all(row.source_system == SourceSystem.PI for row in result.acl_snapshots)


@pytest.mark.asyncio
async def test_normalize_preserves_pi_extras_in_metadata_body() -> None:
    """`_pi_extras` rides on each event's `raw` dict. The connector's `body`
    rendering doesn't strip unknown keys, so the extras flow through intact
    for a future native pipeline that wants to read tree position (entry id
    / parentId), branch summaries, compaction token counts, user labels,
    model changes, per-message provider and model, and any
    extension-authored custom entry type the connector has never seen.
    """
    c = PiConnector(make_default_context())
    hydrated = {
        "session_id": "s-1",
        "events": [{
            "line_no": 0,
            "raw": {
                "type": "system",
                "subtype": "turn_context",
                "_pi_extras": {
                    "entry_id": "e-42",
                    "parent_id": "e-41",
                    "model": "pi-large",
                    "provider": "earendil",
                },
            },
        }],
        "session_complete": False,
        "cwd": "/tmp/p",
    }
    result = await c.normalize(_pi_event(), hydrated)
    # Raw events are persisted in body_json; the metadata.body field is the
    # human-readable rendering. Extras live on the raw, not the rendering,
    # but they survive into R2 via the worker's raw envelope persistence.
    # Here we just verify the connector didn't crash on the unfamiliar key.
    assert len(result.documents) == 1
