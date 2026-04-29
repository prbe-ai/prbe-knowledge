import pytest

from services.ingestion.handlers.base import make_default_context
from services.ingestion.handlers.claude_code import ClaudeCodeConnector
from shared.exceptions import InvalidWebhookPayload


def _make() -> ClaudeCodeConnector:
    return ClaudeCodeConnector(make_default_context())


def test_parse_webhook_event_returns_bare_session_id() -> None:
    """Post-migration 0026: source_event_id is the bare session_id (no
    :batch_seq suffix) so the queue UPSERT in main.py:_enqueue can
    coalesce all batches for a session into one row. batch_seq is still
    surfaced via parse_hint so the webhook handler can compose unique
    R2 storage paths per delivery.
    """
    c = _make()
    out = c.parse_webhook_event(
        customer_id="cust-1",
        headers={},
        raw_payload={
            "device_id": "dev-1",
            "session_id": "sess-abc",
            "batch_seq": 12,
            "cwd": "/tmp/p",
            "events": [{"line_no": 47, "raw": {}}],
        },
    )
    assert out is not None
    assert out.source_event_id == "sess-abc"
    assert out.parse_hint == {"session_id": "sess-abc", "batch_seq": 12}


def test_parse_webhook_event_missing_session_raises() -> None:
    c = _make()
    with pytest.raises(InvalidWebhookPayload):
        c.parse_webhook_event(
            customer_id="cust-1",
            headers={},
            raw_payload={"device_id": "dev-1", "batch_seq": 0, "events": []},
        )


def test_parse_webhook_event_empty_events_returns_none() -> None:
    c = _make()
    out = c.parse_webhook_event(
        customer_id="cust-1",
        headers={},
        raw_payload={
            "device_id": "dev-1",
            "session_id": "s",
            "batch_seq": 0,
            "events": [],
        },
    )
    assert out is None  # heartbeat-style empty post; ignore
