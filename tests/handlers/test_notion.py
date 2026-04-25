"""Unit tests for the Notion connector.

Covers:
- parse_webhook_event on a realistic page.updated payload
- blocks_to_markdown on a mix of block types
- normalize without hydration (no token) → workspace ACL only, empty body
- normalize with hydrated content → body populated, mentions → PERSON nodes,
  permissions → granular ACL rows
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.notion import (  # noqa: F401 — registers
    NotionConnector,
    blocks_to_markdown,
)
from services.ingestion.handlers.registry import build_connector
from shared.config import Settings
from shared.constants import (
    DocType,
    EdgeType,
    IngestionEventType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.models import WebhookEvent

FIXTURES = Path(__file__).resolve().parents[1].parent / "fixtures" / "notion"


def _load(name: str) -> dict:
    with (FIXTURES / name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _make_ctx(env: str = "local") -> ConnectorContext:
    settings = Settings(environment=env)
    return ConnectorContext(settings=settings, http=httpx.AsyncClient())


# ---------------------------------------------------------------------------
# parse_webhook_event
# ---------------------------------------------------------------------------


def test_parse_webhook_event_page_updated() -> None:
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)

    payload = _load("page_updated.json")
    result = notion.parse_webhook_event("cust-1", {}, payload)

    assert result is not None
    assert result.source_event_id == "page:page_abc123:edit:2026-04-22T12:00:00.000Z"
    assert result.event_kind == IngestionEventType.WEBHOOK
    assert result.parse_hint["resource_type"] == "page"
    assert result.parse_hint["resource_id"] == "page_abc123"
    assert result.parse_hint["workspace_id"] == "ws_TEST"
    assert result.parse_hint["is_delete"] is False
    assert result.received_at.tzinfo is not None


def test_parse_webhook_event_page_deleted_produces_tombstone() -> None:
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)
    result = notion.parse_webhook_event(
        "cust-1",
        {},
        {
            "type": "page.deleted",
            "entity": {"type": "page", "id": "x"},
            "timestamp": "2026-04-22T12:00:00.000Z",
        },
    )
    assert result is not None
    assert result.source_event_id.startswith("page:x:delete:")
    assert result.parse_hint["is_delete"] is True


def test_parse_webhook_event_ignores_unknown_type() -> None:
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)
    assert (
        notion.parse_webhook_event(
            "cust-1",
            {},
            {"type": "user.created", "entity": {"type": "user", "id": "x"}},
        )
        is None
    )


def test_parse_synthetic_poll_shape() -> None:
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)

    result = notion.parse_webhook_event(
        "cust-1",
        {},
        {
            "customer_id": "cust-1",
            "resource_type": "page",
            "resource_id": "page_synth",
            "polled_at": "2026-04-22T11:00:00Z",
            "last_edited_time": "2026-04-22T10:59:00Z",
        },
    )

    assert result is not None
    assert result.source_event_id == "page:page_synth:2026-04-22T10:59:00Z"
    assert result.event_kind == IngestionEventType.SYNC


def test_parse_rejects_unknown_shape() -> None:
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)
    from shared.exceptions import InvalidWebhookPayload

    with pytest.raises(InvalidWebhookPayload):
        notion.parse_webhook_event("cust-1", {}, {"foo": "bar"})


# ---------------------------------------------------------------------------
# blocks_to_markdown
# ---------------------------------------------------------------------------


def test_blocks_to_markdown_mixed_types() -> None:
    fx = _load("blocks.json")
    md = blocks_to_markdown(fx["results"])

    # Ordered assertions — the mapping should be stable.
    assert "## Incident runbook" in md
    assert "When payments 500s, page @Bob immediately." in md
    assert "- Check Stripe dashboard" in md
    assert "- Roll back last deploy" in md
    assert "```bash\nfly deploy --image prev\n```" in md
    # Unknown block type → placeholder
    assert "[block:image]" in md


def test_blocks_to_markdown_handles_empty() -> None:
    assert blocks_to_markdown([]) == ""


def test_blocks_to_markdown_numbered_list_restarts() -> None:
    blocks = [
        {"type": "numbered_list_item", "numbered_list_item": {"rich_text": [{"plain_text": "first"}]}},
        {"type": "numbered_list_item", "numbered_list_item": {"rich_text": [{"plain_text": "second"}]}},
        {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "break"}]}},
        {"type": "numbered_list_item", "numbered_list_item": {"rich_text": [{"plain_text": "one again"}]}},
    ]
    md = blocks_to_markdown(blocks)
    lines = md.splitlines()
    assert lines[0] == "1. first"
    assert lines[1] == "2. second"
    assert lines[2] == "break"
    assert lines[3] == "1. one again"


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def _webhook_event(payload: dict) -> WebhookEvent:
    return WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.NOTION,
        source_event_id="page:page_abc123:2026-04-22T12:00:00.000Z",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/notion/cust-1/2026/04/22/test.json",
        raw_payload=payload,
        headers={},
    )


@pytest.mark.asyncio
async def test_normalize_without_token_empty_body() -> None:
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)

    event = _webhook_event(_load("page_updated.json"))
    result = await notion.normalize(event, {})

    assert not result.is_empty
    assert len(result.documents) == 1
    doc = result.documents[0]

    assert doc.doc_id == "notion:page:page_abc123"
    assert doc.doc_type == DocType.NOTION_PAGE
    assert doc.metadata["body"] == ""
    assert doc.metadata["hydrated"] is False
    assert doc.source_system == SourceSystem.NOTION
    assert doc.author_id == "unknown"  # no hydration → no last_edited_by id

    # Workspace-level ACL always present.
    acl_rows = result.acl_snapshots
    assert len(acl_rows) == 1
    row = acl_rows[0]
    assert row.principal_type == PrincipalType.WORKSPACE
    assert row.principal_id == "ws_TEST"
    assert row.resource_type == "notion.page"
    assert row.resource_id == "page_abc123"
    assert row.permission == Permission.READ
    assert row.metadata["inherits"] is True

    labels = {(n.label, n.canonical_id) for n in result.graph_nodes}
    assert (NodeLabel.DOCUMENT, doc.doc_id) in labels
    assert (NodeLabel.PERSON, "unknown") in labels


@pytest.mark.asyncio
async def test_normalize_with_hydrated_content() -> None:
    ctx = _make_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)

    event = _webhook_event(_load("page_updated.json"))
    entity = _load("page_metadata.json")
    blocks = _load("blocks.json")["results"]

    from services.ingestion.handlers.notion import _extract_mentioned_user_ids

    hydrated = {
        "entity": entity,
        "body_markdown": blocks_to_markdown(blocks),
        "mentioned_user_ids": _extract_mentioned_user_ids(blocks),
        "permissions": entity["permissions"],
        "resource_type": "page",
    }

    result = await notion.normalize(event, hydrated)
    assert not result.is_empty
    doc = result.documents[0]

    # Body populated.
    assert "Incident runbook" in doc.metadata["body"]
    assert "Check Stripe dashboard" in doc.metadata["body"]
    assert doc.title == "Payments runbook"
    assert doc.author_id == "user_alice"
    assert doc.source_url == "https://www.notion.so/Payments-runbook-abc123"
    assert doc.body_size_bytes > 0
    assert doc.metadata["hydrated"] is True

    # Mentioned user → PERSON node + MENTIONS edge.
    person_ids = {n.canonical_id for n in result.graph_nodes if n.label == NodeLabel.PERSON}
    assert "user_bob" in person_ids
    assert "user_alice" in person_ids

    mentions_edges = [e for e in result.graph_edges if e.edge_type == EdgeType.MENTIONS]
    assert any(e.to_canonical_id == "user_bob" for e in mentions_edges)

    authored_edges = [e for e in result.graph_edges if e.edge_type == EdgeType.AUTHORED]
    assert any(e.from_canonical_id == "user_alice" for e in authored_edges)

    # ACL: workspace fallback + user + group rows.
    rows = result.acl_snapshots
    assert len(rows) == 3
    principal_pairs = {(r.principal_type, r.principal_id) for r in rows}
    assert (PrincipalType.WORKSPACE, "ws_TEST") in principal_pairs
    assert (PrincipalType.USER, "user_alice") in principal_pairs
    assert (PrincipalType.GROUP, "group_eng") in principal_pairs

    # Editor role → WRITE, reader → READ.
    user_row = next(r for r in rows if r.principal_type == PrincipalType.USER)
    assert user_row.permission == Permission.WRITE
    group_row = next(r for r in rows if r.principal_type == PrincipalType.GROUP)
    assert group_row.permission == Permission.READ

    # Parent chain captured in metadata so Phase 1 enforcement can walk it.
    assert all(r.metadata.get("inherits") is True for r in rows)
    assert all(r.metadata.get("parent_id") == "page_parent_999" for r in rows)


# ---------------------------------------------------------------------------
# signature verification
# ---------------------------------------------------------------------------


def test_verify_signature_dev_bypass() -> None:
    ctx = _make_ctx(env="local")
    notion = build_connector(SourceSystem.NOTION, ctx)
    assert notion.verify_signature({}, b"{}") is True


def test_verify_signature_prod_rejects_unsigned_unknown_caller() -> None:
    ctx = _make_ctx(env="main")
    notion = build_connector(SourceSystem.NOTION, ctx)
    assert notion.verify_signature({}, b"{}") is False


def test_verify_signature_valid_hmac() -> None:
    import hashlib
    import hmac as hmac_mod

    from pydantic import SecretStr

    secret = "notion-secret"
    body = b'{"hello":"notion"}'
    digest = hmac_mod.new(secret.encode(), body, hashlib.sha256).hexdigest()

    settings = Settings(environment="main", notion_client_secret=SecretStr(secret))
    ctx = ConnectorContext(settings=settings, http=httpx.AsyncClient())
    notion = build_connector(SourceSystem.NOTION, ctx)

    assert notion.verify_signature({"X-Notion-Signature": digest}, body) is True
    assert notion.verify_signature({"X-Notion-Signature": digest}, body + b"x") is False
    assert (
        notion.verify_signature({"X-Notion-Signature": f"sha256={digest}"}, body) is True
    )


# ---------------------------------------------------------------------------
# OAuth install URL
# ---------------------------------------------------------------------------


def _make_oauth_ctx(*, client_id: str | None = "ntn_test_client") -> ConnectorContext:
    settings = Settings(
        environment="local",
        notion_client_id=client_id,
    )
    return ConnectorContext(settings=settings, http=httpx.AsyncClient())


def test_oauth_install_url_shape() -> None:
    ctx = _make_oauth_ctx()
    notion = build_connector(SourceSystem.NOTION, ctx)

    url = notion.oauth_install_url(
        "cust-1",
        "https://example.com/oauth/notion/callback",
        "signed-state-token",
    )

    assert url.startswith("https://api.notion.com/v1/oauth/authorize?")
    assert "client_id=ntn_test_client" in url
    assert (
        "redirect_uri=https%3A%2F%2Fexample.com%2Foauth%2Fnotion%2Fcallback"
        in url
    )
    assert "response_type=code" in url
    assert "owner=user" in url
    assert "state=signed-state-token" in url


def test_oauth_install_url_missing_client_id_raises() -> None:
    from shared.exceptions import MissingSecret

    ctx = _make_oauth_ctx(client_id=None)
    notion = build_connector(SourceSystem.NOTION, ctx)

    with pytest.raises(MissingSecret):
        notion.oauth_install_url("cust-1", "https://x/cb", "state")


# ---------------------------------------------------------------------------
# OAuth code exchange
# ---------------------------------------------------------------------------


def _oauth_ctx_with_secret(
    *,
    handler,
    client_id: str | None = "ntn_test_client",
    client_secret: str | None = "ntn_test_secret",
) -> ConnectorContext:
    from pydantic import SecretStr

    settings = Settings(
        environment="local",
        notion_client_id=client_id,
        notion_client_secret=SecretStr(client_secret) if client_secret else None,
    )
    return ConnectorContext(
        settings=settings,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
async def test_exchange_oauth_code_happy() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/oauth/token"
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "access_token": "ntn_access_xyz",
                "token_type": "bearer",
                "refresh_token": "ntn_refresh_abc",
                "bot_id": "bot_42",
                "workspace_id": "ws_alpha",
                "workspace_name": "Acme Eng",
                "workspace_icon": "https://example.com/icon.png",
                "owner": {"user": {"id": "user_alice"}},
                "duplicated_template_id": None,
                "request_id": "req_1",
            },
        )

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)

    token = await notion.exchange_oauth_code(
        code="ntn-code-123",
        redirect_uri="https://example.com/oauth/notion/callback",
    )

    # httpx.BasicAuth puts client_id:client_secret as base64.
    assert seen["auth"].startswith("Basic ")
    assert seen["body"] == {
        "grant_type": "authorization_code",
        "code": "ntn-code-123",
        "redirect_uri": "https://example.com/oauth/notion/callback",
    }

    assert token.access_token == "ntn_access_xyz"
    assert token.refresh_token == "ntn_refresh_abc"
    assert token.source_system == SourceSystem.NOTION
    # connector returns customer_id="" — the route fills it in.
    assert token.customer_id == ""
    # Notion uses capability checkboxes, not OAuth scope strings.
    assert token.scope is None

    # install_metadata carries workspace info into identify_workspaces.
    assert token.install_metadata == {
        "workspace_id": "ws_alpha",
        "workspace_name": "Acme Eng",
        "workspace_icon": "https://example.com/icon.png",
        "bot_id": "bot_42",
        "owner": {"user": {"id": "user_alice"}},
    }


@pytest.mark.asyncio
async def test_exchange_oauth_code_4xx_raises_permanent() -> None:
    from shared.exceptions import PermanentSourceError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)

    with pytest.raises(PermanentSourceError):
        await notion.exchange_oauth_code(
            code="bad", redirect_uri="https://x/cb"
        )


@pytest.mark.asyncio
async def test_exchange_oauth_code_missing_code_raises() -> None:
    from shared.exceptions import InvalidWebhookPayload

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("HTTP should not be called when code is None")

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)

    with pytest.raises(InvalidWebhookPayload):
        await notion.exchange_oauth_code(
            code=None, redirect_uri="https://x/cb"
        )


@pytest.mark.asyncio
async def test_exchange_oauth_code_missing_secret_raises() -> None:
    from shared.exceptions import MissingSecret

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("HTTP should not be called when secret is missing")

    ctx = _oauth_ctx_with_secret(handler=handler, client_secret=None)
    notion = build_connector(SourceSystem.NOTION, ctx)

    with pytest.raises(MissingSecret):
        await notion.exchange_oauth_code(
            code="x", redirect_uri="https://x/cb"
        )


@pytest.mark.asyncio
async def test_exchange_oauth_code_no_refresh_token() -> None:
    """Notion's docs say refresh_token is nullable — connector must not
    crash when it's absent (older public integrations)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "ntn_access_xyz",
                "token_type": "bearer",
                "bot_id": "bot_42",
                "workspace_id": "ws_alpha",
                "workspace_name": "Acme Eng",
                "owner": {"user": {"id": "user_alice"}},
                # no refresh_token / workspace_icon
            },
        )

    ctx = _oauth_ctx_with_secret(handler=handler)
    notion = build_connector(SourceSystem.NOTION, ctx)
    token = await notion.exchange_oauth_code(
        code="x", redirect_uri="https://x/cb"
    )
    assert token.access_token == "ntn_access_xyz"
    assert token.refresh_token is None
    assert token.install_metadata["workspace_id"] == "ws_alpha"
    assert token.install_metadata["workspace_icon"] is None
