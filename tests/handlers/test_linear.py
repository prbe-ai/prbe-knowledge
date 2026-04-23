"""Unit tests for the Linear connector.

Exercises the Connector contract end-to-end on a realistic webhook payload
without needing DB / R2 — proves the Linear mapping produces the right
Document, graph nodes/edges, and ACL snapshot rows.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import httpx
import pytest

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.linear import LinearConnector  # noqa: F401 — registers
from services.ingestion.handlers.registry import build_connector
from shared.config import Settings
from shared.constants import (
    DocType,
    EdgeType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.exceptions import InvalidWebhookPayload

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "linear"


def _load_fixture(name: str) -> dict:
    return json.loads((_FIXTURE_DIR / name).read_text())


def _make_ctx(
    *, webhook_secret: str | None = None, env: str = "local"
) -> ConnectorContext:
    from pydantic import SecretStr

    settings = Settings(
        environment=env,
        linear_webhook_secret=SecretStr(webhook_secret) if webhook_secret else None,
    )
    return ConnectorContext(settings=settings, http=httpx.AsyncClient())


# ---------------------------------------------------------------------------
# parse_webhook_event
# ---------------------------------------------------------------------------


def test_parse_webhook_event_issue_create() -> None:
    ctx = _make_ctx()
    linear = build_connector(SourceSystem.LINEAR, ctx)
    payload = _load_fixture("issue_create.json")

    result = linear.parse_webhook_event("cust-1", {}, payload)

    assert result is not None
    assert result.source_event_id.startswith(
        "issue:11111111-2222-3333-4444-555555555555:create:"
    )
    assert result.parse_hint["type"] == "Issue"
    assert result.parse_hint["action"] == "create"
    assert result.parse_hint["team_id"] == "team_eng"
    assert result.parse_hint["organization_id"] == "org_test"


def test_parse_webhook_event_ignores_other_types() -> None:
    ctx = _make_ctx()
    linear = build_connector(SourceSystem.LINEAR, ctx)

    # Unhandled type.
    assert (
        linear.parse_webhook_event(
            "cust-1",
            {},
            {"type": "Project", "action": "create", "data": {"id": "p1"}},
        )
        is None
    )


def test_parse_webhook_event_remove_action_produces_result() -> None:
    """A `remove` action is no longer dropped — it drives a tombstone write."""
    ctx = _make_ctx()
    linear = build_connector(SourceSystem.LINEAR, ctx)
    result = linear.parse_webhook_event(
        "cust-1",
        {},
        {
            "type": "Issue",
            "action": "remove",
            "createdAt": "2026-04-23T10:00:00.000Z",
            "data": {"id": "i1", "updatedAt": "2026-04-23T10:00:00.000Z"},
        },
    )
    assert result is not None
    assert ":remove:" in result.source_event_id
    assert result.parse_hint["action"] == "remove"


def test_parse_webhook_event_malformed_raises() -> None:
    ctx = _make_ctx()
    linear = build_connector(SourceSystem.LINEAR, ctx)

    with pytest.raises(InvalidWebhookPayload):
        linear.parse_webhook_event(
            "cust-1", {}, {"type": "Issue", "action": "create"}
        )


# ---------------------------------------------------------------------------
# verify_signature
# ---------------------------------------------------------------------------


def test_verify_signature_dev_bypass() -> None:
    ctx = _make_ctx(webhook_secret=None, env="local")
    linear = build_connector(SourceSystem.LINEAR, ctx)
    assert linear.verify_signature({}, b"{}") is True


def test_verify_signature_prod_rejects_unsigned() -> None:
    ctx = _make_ctx(webhook_secret=None, env="main")
    linear = build_connector(SourceSystem.LINEAR, ctx)
    assert linear.verify_signature({}, b"{}") is False


def test_verify_signature_valid_hmac_and_tamper() -> None:
    secret = "linear-s3cr3t"
    body = b'{"hello":"world"}'
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    ctx = _make_ctx(webhook_secret=secret, env="main")
    linear = build_connector(SourceSystem.LINEAR, ctx)

    headers = {"Linear-Signature": expected}
    assert linear.verify_signature(headers, body) is True
    # Tampered body fails.
    assert linear.verify_signature(headers, body + b"x") is False
    # Missing signature fails.
    assert linear.verify_signature({}, body) is False


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_issue_produces_document_and_graph() -> None:
    ctx = _make_ctx()
    linear = build_connector(SourceSystem.LINEAR, ctx)
    payload = _load_fixture("issue_create.json")

    from datetime import UTC, datetime

    from shared.models import WebhookEvent

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.LINEAR,
        source_event_id="issue:11111111-2222-3333-4444-555555555555",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/linear/cust-1/2026/04/22/test.json",
        raw_payload=payload,
        headers={},
    )

    result = await linear.normalize(event, {})
    assert not result.is_empty
    assert len(result.documents) == 1

    doc = result.documents[0]
    assert doc.source_system == SourceSystem.LINEAR
    assert doc.doc_type == DocType.LINEAR_ISSUE
    assert doc.author_id == "user_alice"
    assert doc.title == "Payments service returns 500 after latest deploy"
    assert (
        doc.doc_id
        == "linear:org_test:issue:11111111-2222-3333-4444-555555555555"
    )
    assert doc.metadata["body"].startswith("The payments service")
    assert doc.metadata["team_id"] == "team_eng"
    assert doc.metadata["assignee_id"] == "user_bob"

    # Graph nodes: TICKET + DOCUMENT + two PERSON (creator, assignee).
    labels = {(n.label, n.canonical_id) for n in result.graph_nodes}
    assert (
        NodeLabel.TICKET,
        "11111111-2222-3333-4444-555555555555",
    ) in labels
    assert (NodeLabel.PERSON, "user_alice") in labels
    assert (NodeLabel.PERSON, "user_bob") in labels
    assert (NodeLabel.DOCUMENT, doc.doc_id) in labels

    # Edges: AUTHORED(person→doc) + ASSIGNED_TO(ticket→person) + MENTIONS(doc→OPS-42).
    edge_kinds = {(e.edge_type, e.from_canonical_id, e.to_canonical_id) for e in result.graph_edges}
    assert (EdgeType.AUTHORED, "user_alice", doc.doc_id) in edge_kinds
    assert (
        EdgeType.ASSIGNED_TO,
        "11111111-2222-3333-4444-555555555555",
        "user_bob",
    ) in edge_kinds
    assert (EdgeType.MENTIONS, doc.doc_id, "OPS-42") in edge_kinds

    # doc_references: URL + issue key (but not the issue's own identifier).
    ref_urls = {r.external_url for r in doc.doc_references}
    assert "https://example.com/run/42" in ref_urls
    assert "linear://issue/OPS-42" in ref_urls

    # ACL: workspace + team principals.
    by_principal = {
        (row.principal_type, row.principal_id): row for row in result.acl_snapshots
    }
    ws = by_principal[(PrincipalType.WORKSPACE, "org_test")]
    assert ws.resource_id == "11111111-2222-3333-4444-555555555555"
    assert ws.resource_type == DocType.LINEAR_ISSUE.value
    assert ws.permission == Permission.READ
    assert (PrincipalType.GROUP, "team_eng") in by_principal


@pytest.mark.asyncio
async def test_normalize_comment_links_to_parent_issue() -> None:
    ctx = _make_ctx()
    linear = build_connector(SourceSystem.LINEAR, ctx)

    from datetime import UTC, datetime

    from shared.models import WebhookEvent

    comment_payload = {
        "action": "create",
        "type": "Comment",
        "createdAt": "2026-04-22T10:30:00.000Z",
        "organizationId": "org_test",
        "data": {
            "id": "comment_1",
            "body": "Related: [ENG-99]",
            "url": "https://linear.app/prbe/issue/ENG-123#comment-1",
            "issueId": "11111111-2222-3333-4444-555555555555",
            "userId": "user_bob",
            "user": {"id": "user_bob", "name": "Bob"},
            "createdAt": "2026-04-22T10:30:00.000Z",
            "updatedAt": "2026-04-22T10:30:00.000Z",
        },
    }

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.LINEAR,
        source_event_id="comment:comment_1",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/linear/cust-1/2026/04/22/c.json",
        raw_payload=comment_payload,
        headers={},
    )

    result = await linear.normalize(event, {})
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.doc_type == DocType.LINEAR_COMMENT
    assert doc.parent_doc_id == (
        "linear:org_test:issue:11111111-2222-3333-4444-555555555555"
    )
    assert doc.author_id == "user_bob"

    edge_kinds = {(e.edge_type, e.from_canonical_id, e.to_canonical_id) for e in result.graph_edges}
    assert (EdgeType.AUTHORED, "user_bob", doc.doc_id) in edge_kinds
    assert (EdgeType.MENTIONS, doc.doc_id, "ENG-99") in edge_kinds
