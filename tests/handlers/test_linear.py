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
    assert doc.body.startswith("The payments service")
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


# ---------------------------------------------------------------------------
# exchange_oauth_code — error mapping
#
# Linear's /oauth/token can return a non-2xx for many reasons (transient
# 503 from CF edge, 400 invalid_grant for a spent code, 401 for revoked
# credentials, etc.). Before this fix, `raise_for_status()` raised
# `httpx.HTTPStatusError` which is NOT a `PrbeError`, so the admin route's
# `except PrbeError` block missed it and FastAPI returned an opaque 500
# with no body. Now: 5xx → TransientSourceError, 4xx → PermanentSourceError,
# 200-without-access_token → PermanentSourceError. All three carry the
# upstream response body in `.context["body"]` so the dashboard can render
# something useful.
# ---------------------------------------------------------------------------


def _make_oauth_ctx(status_code: int, body: str | dict) -> ConnectorContext:
    """Build a connector context whose http client returns a fixed Linear
    /oauth/token response."""
    from pydantic import SecretStr

    settings = Settings(
        environment="local",
        linear_client_id="cid_test",
        linear_client_secret=SecretStr("secret_test"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.linear.app/oauth/token"
        if isinstance(body, str):
            return httpx.Response(status_code, text=body)
        return httpx.Response(status_code, json=body)

    transport = httpx.MockTransport(handler)
    return ConnectorContext(
        settings=settings, http=httpx.AsyncClient(transport=transport)
    )


@pytest.mark.asyncio
async def test_exchange_oauth_code_5xx_raises_transient() -> None:
    from shared.exceptions import TransientSourceError

    ctx = _make_oauth_ctx(503, "upstream temporarily unavailable")
    linear = build_connector(SourceSystem.LINEAR, ctx)

    with pytest.raises(TransientSourceError) as exc_info:
        await linear.exchange_oauth_code(code="abc", redirect_uri="https://x/cb")

    assert exc_info.value.context["status"] == 503
    assert "upstream temporarily unavailable" in exc_info.value.context["body"]


@pytest.mark.asyncio
async def test_exchange_oauth_code_4xx_raises_permanent() -> None:
    from shared.exceptions import PermanentSourceError

    ctx = _make_oauth_ctx(
        400, {"error": "invalid_grant", "error_description": "code expired"}
    )
    linear = build_connector(SourceSystem.LINEAR, ctx)

    with pytest.raises(PermanentSourceError) as exc_info:
        await linear.exchange_oauth_code(code="abc", redirect_uri="https://x/cb")

    assert exc_info.value.context["status"] == 400
    assert "invalid_grant" in exc_info.value.context["body"]


@pytest.mark.asyncio
async def test_exchange_oauth_code_missing_access_token_raises_permanent() -> None:
    from shared.exceptions import PermanentSourceError

    ctx = _make_oauth_ctx(200, {"scope": "read,write"})  # no access_token
    linear = build_connector(SourceSystem.LINEAR, ctx)

    with pytest.raises(PermanentSourceError) as exc_info:
        await linear.exchange_oauth_code(code="abc", redirect_uri="https://x/cb")

    assert "missing access_token" in str(exc_info.value)


@pytest.mark.asyncio
async def test_exchange_oauth_code_success() -> None:
    ctx = _make_oauth_ctx(
        200, {"access_token": "lin_oauth_xxx", "scope": "read,write"}
    )
    linear = build_connector(SourceSystem.LINEAR, ctx)

    token = await linear.exchange_oauth_code(code="abc", redirect_uri="https://x/cb")
    assert token.access_token == "lin_oauth_xxx"
    assert token.scope == "read,write"
    assert token.source_system == SourceSystem.LINEAR


# ---------------------------------------------------------------------------
# OAuth install URL — must include prompt=consent so Linear issues a
# refresh_token alongside access_token. Without it, Linear hands back a
# long-lived (10-year) access_token only — when revoked server-side the
# connector has no path back without a fresh user-driven OAuth.
# ---------------------------------------------------------------------------


def test_oauth_install_url_includes_prompt_consent() -> None:
    from pydantic import SecretStr

    settings = Settings(environment="local", linear_client_id="cid_test", linear_client_secret=SecretStr("s"))
    ctx = ConnectorContext(settings=settings, http=httpx.AsyncClient())
    linear = build_connector(SourceSystem.LINEAR, ctx)
    url = linear.oauth_install_url(customer_id="cust-1", redirect_uri="https://x/cb")
    assert "prompt=consent" in url
    # Sanity: the other required params are still there
    assert "client_id=cid_test" in url
    assert "response_type=code" in url
    assert "state=cust-1" in url
    assert "redirect_uri=https%3A%2F%2Fx%2Fcb" in url or "redirect_uri=https://x/cb" in url


# ---------------------------------------------------------------------------
# exchange_oauth_code — refresh_token + expires_at persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exchange_oauth_code_persists_refresh_token_and_expires_at() -> None:
    """When `prompt=consent` is on the authorize URL Linear's response
    carries `refresh_token` + `expires_in`. The exchange handler must
    propagate both onto the returned IntegrationToken so save_token
    persists them — without this, refresh is impossible and a revoked
    access_token strands the integration permanently."""
    from datetime import UTC, datetime

    ctx = _make_oauth_ctx(
        200,
        {
            "access_token": "lin_oauth_xxx",
            "refresh_token": "lin_refresh_yyy",
            "expires_in": 3600,
            "scope": "read,write",
        },
    )
    linear = build_connector(SourceSystem.LINEAR, ctx)

    before = datetime.now(UTC)
    token = await linear.exchange_oauth_code(code="abc", redirect_uri="https://x/cb")

    assert token.access_token == "lin_oauth_xxx"
    assert token.refresh_token == "lin_refresh_yyy"
    assert token.expires_at is not None
    delta = (token.expires_at - before).total_seconds()
    # Should land in [3590, 3610] — the 3600-second window plus the tiny
    # latency of the call.
    assert 3590 <= delta <= 3610, f"expires_at delta out of range: {delta}"


@pytest.mark.asyncio
async def test_exchange_oauth_code_handles_missing_refresh_fields() -> None:
    """Older OAuth flow (no prompt=consent) returns access_token alone.
    The exchange handler must NOT crash; refresh_token + expires_at land
    as None, and refresh attempts later raise a clear PermanentSourceError."""
    ctx = _make_oauth_ctx(200, {"access_token": "lin_oauth_xxx", "scope": "read,write"})
    linear = build_connector(SourceSystem.LINEAR, ctx)

    token = await linear.exchange_oauth_code(code="abc", redirect_uri="https://x/cb")
    assert token.access_token == "lin_oauth_xxx"
    assert token.refresh_token is None
    assert token.expires_at is None


# ---------------------------------------------------------------------------
# exchange_refresh_token — refresh flow + error mapping
# ---------------------------------------------------------------------------


def _make_refresh_ctx(status_code: int, body: str | dict) -> ConnectorContext:
    """Connector ctx whose http client returns a fixed Linear /oauth/token
    response specifically for refresh_token grants."""
    from pydantic import SecretStr

    settings = Settings(
        environment="local",
        linear_client_id="cid_test",
        linear_client_secret=SecretStr("secret_test"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.linear.app/oauth/token"
        # We don't introspect request body here; the connector's grant_type
        # routing is tested implicitly via the integration shape.
        if isinstance(body, str):
            return httpx.Response(status_code, text=body)
        return httpx.Response(status_code, json=body)

    transport = httpx.MockTransport(handler)
    return ConnectorContext(
        settings=settings, http=httpx.AsyncClient(transport=transport)
    )


@pytest.mark.asyncio
async def test_exchange_refresh_token_no_refresh_token_raises_permanent() -> None:
    """Tokens minted before prompt=consent has no refresh_token. Caller is
    expected to flag the row auth_failed and prompt re-OAuth."""
    from shared.exceptions import PermanentSourceError
    from shared.models import IntegrationToken

    ctx = _make_refresh_ctx(200, {})
    linear = build_connector(SourceSystem.LINEAR, ctx)

    token = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.LINEAR,
        access_token="lin_oauth_old",
        refresh_token=None,
    )

    with pytest.raises(PermanentSourceError, match="without a stored refresh_token"):
        await linear.exchange_refresh_token(token)


@pytest.mark.asyncio
async def test_exchange_refresh_token_success_rotates_credential() -> None:
    """Linear may issue a new refresh_token on each refresh (rotation). The
    handler must surface the new one and preserve scope when the response
    omits it."""
    from shared.models import IntegrationToken

    ctx = _make_refresh_ctx(
        200,
        {
            "access_token": "lin_oauth_new",
            "refresh_token": "lin_refresh_new",
            "expires_in": 7200,
        },
    )
    linear = build_connector(SourceSystem.LINEAR, ctx)

    old = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.LINEAR,
        access_token="lin_oauth_old",
        refresh_token="lin_refresh_old",
        scope="read,write",
    )
    new = await linear.exchange_refresh_token(old)

    assert new.access_token == "lin_oauth_new"
    assert new.refresh_token == "lin_refresh_new"  # rotated
    assert new.scope == "read,write"  # preserved from old when response omits
    assert new.expires_at is not None  # populated from expires_in
    assert new.customer_id == "cust-1"  # preserved


@pytest.mark.asyncio
async def test_exchange_refresh_token_echoes_existing_refresh_when_omitted() -> None:
    """Some OAuth providers don't rotate refresh_tokens; they echo back the
    existing one or omit it entirely. The handler must keep using the
    persisted refresh_token in the latter case."""
    from shared.models import IntegrationToken

    ctx = _make_refresh_ctx(200, {"access_token": "lin_oauth_new", "expires_in": 3600})
    linear = build_connector(SourceSystem.LINEAR, ctx)

    old = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.LINEAR,
        access_token="lin_oauth_old",
        refresh_token="lin_refresh_unchanged",
    )
    new = await linear.exchange_refresh_token(old)

    assert new.refresh_token == "lin_refresh_unchanged"


@pytest.mark.asyncio
async def test_exchange_refresh_token_4xx_raises_permanent() -> None:
    """Linear returns 400/401 when the refresh_token has been invalidated
    server-side (user revoked grant, secret rotated, etc.). No retry will
    help; cron should mark the row auth_failed."""
    from shared.exceptions import PermanentSourceError
    from shared.models import IntegrationToken

    ctx = _make_refresh_ctx(400, {"error": "invalid_grant"})
    linear = build_connector(SourceSystem.LINEAR, ctx)
    old = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.LINEAR,
        access_token="x",
        refresh_token="rotten",
    )

    with pytest.raises(PermanentSourceError):
        await linear.exchange_refresh_token(old)


@pytest.mark.asyncio
async def test_exchange_refresh_token_5xx_raises_transient() -> None:
    from shared.exceptions import TransientSourceError
    from shared.models import IntegrationToken

    ctx = _make_refresh_ctx(503, "down")
    linear = build_connector(SourceSystem.LINEAR, ctx)
    old = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.LINEAR,
        access_token="x",
        refresh_token="r",
    )

    with pytest.raises(TransientSourceError):
        await linear.exchange_refresh_token(old)


# ---------------------------------------------------------------------------
# verify_token_health — periodic liveness probe
# ---------------------------------------------------------------------------


def _make_viewer_ctx(status_code: int, body: dict) -> ConnectorContext:
    """Connector ctx whose http client returns a fixed Linear GraphQL
    response for the viewer probe."""
    from pydantic import SecretStr

    settings = Settings(
        environment="local",
        linear_client_id="cid_test",
        linear_client_secret=SecretStr("s"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.linear.app/graphql"
        return httpx.Response(status_code, json=body)

    transport = httpx.MockTransport(handler)
    return ConnectorContext(
        settings=settings, http=httpx.AsyncClient(transport=transport)
    )


@pytest.mark.asyncio
async def test_verify_token_health_returns_true_for_healthy_viewer() -> None:
    from shared.models import IntegrationToken

    ctx = _make_viewer_ctx(200, {"data": {"viewer": {"id": "uuid-x"}}})
    linear = build_connector(SourceSystem.LINEAR, ctx)
    token = IntegrationToken(
        customer_id="cust-1", source_system=SourceSystem.LINEAR, access_token="x"
    )

    assert await linear.verify_token_health(token) is True


@pytest.mark.asyncio
async def test_verify_token_health_returns_false_on_401() -> None:
    """HTTP 401 is the definitive "this token is dead" signal. The cron
    flips the row to auth_failed on this return value."""
    from shared.models import IntegrationToken

    ctx = _make_viewer_ctx(
        401,
        {"errors": [{"message": "Authentication required"}]},
    )
    linear = build_connector(SourceSystem.LINEAR, ctx)
    token = IntegrationToken(
        customer_id="cust-1", source_system=SourceSystem.LINEAR, access_token="x"
    )

    assert await linear.verify_token_health(token) is False


@pytest.mark.asyncio
async def test_verify_token_health_returns_false_on_graphql_auth_error_200() -> None:
    """Linear sometimes returns HTTP 200 with an `AUTHENTICATION_ERROR`
    in the GraphQL errors array (observed when the token is partially
    valid — e.g. expired but not yet purged). Treat that as a definite
    negative health signal too."""
    from shared.models import IntegrationToken

    ctx = _make_viewer_ctx(
        200,
        {
            "errors": [
                {
                    "message": "Authentication required",
                    "extensions": {"code": "AUTHENTICATION_ERROR"},
                }
            ]
        },
    )
    linear = build_connector(SourceSystem.LINEAR, ctx)
    token = IntegrationToken(
        customer_id="cust-1", source_system=SourceSystem.LINEAR, access_token="x"
    )

    assert await linear.verify_token_health(token) is False


@pytest.mark.asyncio
async def test_verify_token_health_5xx_raises_transient() -> None:
    """5xx is inconclusive — don't poison the row over a Linear edge blip."""
    from shared.exceptions import TransientSourceError
    from shared.models import IntegrationToken

    ctx = _make_viewer_ctx(503, {})
    linear = build_connector(SourceSystem.LINEAR, ctx)
    token = IntegrationToken(
        customer_id="cust-1", source_system=SourceSystem.LINEAR, access_token="x"
    )

    with pytest.raises(TransientSourceError):
        await linear.verify_token_health(token)
