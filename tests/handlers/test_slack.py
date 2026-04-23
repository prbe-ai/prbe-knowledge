"""Unit tests for the Slack connector.

Exercises the Connector contract end-to-end on a realistic webhook payload
without needing DB / R2 — proves the base ABC + Slack mapping are wired up.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from datetime import UTC

import httpx
import pytest

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.registry import build_connector
from services.ingestion.handlers.slack import SlackConnector  # noqa: F401 — registers
from shared.config import Settings
from shared.constants import DocType, NodeLabel, SourceSystem


def _make_ctx(*, signing_secret: str | None = None, env: str = "local") -> ConnectorContext:
    from pydantic import SecretStr

    settings = Settings(
        environment=env,
        slack_signing_secret=SecretStr(signing_secret) if signing_secret else None,
    )
    return ConnectorContext(settings=settings, http=httpx.AsyncClient())


SAMPLE_EVENT = {
    "team_id": "T123",
    "type": "event_callback",
    "event": {
        "type": "message",
        "channel": "C456",
        "user": "U789",
        "text": "deploying the payments service now — see <https://example.com/run/42> for logs",
        "ts": "1713628800.000100",
    },
}


def test_parse_webhook_event_valid_message() -> None:
    ctx = _make_ctx()
    slack = build_connector(SourceSystem.SLACK, ctx)

    result = slack.parse_webhook_event("cust-1", {}, SAMPLE_EVENT)

    assert result is not None
    assert result.source_event_id == "C456:1713628800.000100"
    assert result.parse_hint["channel"] == "C456"
    assert result.parse_hint["team_id"] == "T123"
    # Plain messages carry subtype=None in the hint so the normalizer can
    # distinguish them from edits/deletes without re-parsing.
    assert result.parse_hint["subtype"] is None


def test_parse_webhook_event_message_changed_produces_edit_id() -> None:
    ctx = _make_ctx()
    slack = build_connector(SourceSystem.SLACK, ctx)

    edit = {
        "team_id": "T123",
        "type": "event_callback",
        "event": {
            "type": "message",
            "subtype": "message_changed",
            "channel": "C456",
            "event_ts": "1713629000.000400",
            "message": {
                "type": "message",
                "channel": "C456",
                "user": "U789",
                "text": "edited body",
                "ts": "1713628800.000100",
                "edited": {"user": "U789", "ts": "1713629000.000300"},
            },
        },
    }
    result = slack.parse_webhook_event("cust-1", {}, edit)
    assert result is not None
    assert result.source_event_id == "C456:1713628800.000100:edit:1713629000.000400"
    assert result.parse_hint["subtype"] == "message_changed"
    assert result.parse_hint["ts"] == "1713628800.000100"


def test_parse_webhook_event_message_deleted_produces_delete_id() -> None:
    ctx = _make_ctx()
    slack = build_connector(SourceSystem.SLACK, ctx)

    delete = {
        "team_id": "T123",
        "type": "event_callback",
        "event": {
            "type": "message",
            "subtype": "message_deleted",
            "channel": "C456",
            "event_ts": "1713629500.000100",
            "deleted_ts": "1713628800.000100",
            "previous_message": {
                "type": "message",
                "user": "U789",
                "text": "will be gone",
                "ts": "1713628800.000100",
            },
        },
    }
    result = slack.parse_webhook_event("cust-1", {}, delete)
    assert result is not None
    assert result.source_event_id == "C456:1713628800.000100:delete:1713629500.000100"
    assert result.parse_hint["subtype"] == "message_deleted"


def test_parse_webhook_event_ignores_noise() -> None:
    ctx = _make_ctx()
    slack = build_connector(SourceSystem.SLACK, ctx)

    assert slack.parse_webhook_event("cust-1", {}, {"type": "url_verification"}) is None
    assert (
        slack.parse_webhook_event(
            "cust-1",
            {},
            {"event": {"type": "user_typing", "channel": "C1", "user": "U1"}},
        )
        is None
    )


def test_verify_signature_dev_bypass() -> None:
    # Local env with no signing secret → accept (explicit dev bypass)
    ctx = _make_ctx(signing_secret=None, env="local")
    slack = build_connector(SourceSystem.SLACK, ctx)
    assert slack.verify_signature({}, b"{}") is True


def test_verify_signature_prod_rejects_unsigned() -> None:
    ctx = _make_ctx(signing_secret=None, env="main")
    slack = build_connector(SourceSystem.SLACK, ctx)
    assert slack.verify_signature({}, b"{}") is False


def test_verify_signature_valid_hmac() -> None:
    secret = "s3cr3t"
    body = b'{"hello":"world"}'
    ts = str(int(time.time()))
    expected = (
        "v0="
        + hmac.new(
            secret.encode(),
            f"v0:{ts}:".encode() + body,
            hashlib.sha256,
        ).hexdigest()
    )
    ctx = _make_ctx(signing_secret=secret, env="main")
    slack = build_connector(SourceSystem.SLACK, ctx)

    headers = {
        "X-Slack-Request-Timestamp": ts,
        "X-Slack-Signature": expected,
    }
    assert slack.verify_signature(headers, body) is True
    # Tampered body fails
    assert slack.verify_signature(headers, body + b"x") is False


@pytest.mark.asyncio
async def test_normalize_produces_document_and_graph() -> None:
    ctx = _make_ctx()
    slack = build_connector(SourceSystem.SLACK, ctx)

    from datetime import datetime

    from shared.models import WebhookEvent

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.SLACK,
        source_event_id="C456:1713628800.000100",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/slack/cust-1/2026/04/22/test.json",
        raw_payload=SAMPLE_EVENT,
        headers={},
    )

    result = await slack.normalize(event, {})
    assert not result.is_empty
    assert len(result.documents) == 1

    doc = result.documents[0]
    assert doc.source_system == SourceSystem.SLACK
    assert doc.doc_type == DocType.SLACK_MESSAGE
    assert doc.author_id == "U789"
    assert doc.title and "deploying" in doc.title
    assert doc.metadata["body"].startswith("deploying")

    labels = {(n.label, n.canonical_id) for n in result.graph_nodes}
    assert (NodeLabel.CHANNEL, "C456") in labels
    assert (NodeLabel.PERSON, "U789") in labels
    assert (NodeLabel.DOCUMENT, doc.doc_id) in labels

    refs = doc.doc_references
    assert len(refs) == 1
    assert refs[0].external_url == "https://example.com/run/42"

    assert result.acl_snapshots
    assert result.acl_snapshots[0].resource_id == "C456:1713628800.000100"
