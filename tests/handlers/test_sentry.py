"""Unit tests for the Sentry connector.

Exercises parse_webhook_event for both the issue and installation lifecycle
branches, verify_signature across dev / prod / valid / tampered flows, and
normalize on a realistic issue.created payload (Document + graph + ACL).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import SecretStr

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.registry import build_connector
from services.ingestion.handlers.sentry import SentryConnector
from shared.config import Settings
from shared.constants import (
    DocType,
    EdgeType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.exceptions import (
    InvalidWebhookPayload,
    PermanentSourceError,
    TransientSourceError,
)
from shared.models import IntegrationToken, WebhookEvent

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "sentry"


def _make_ctx(
    *, webhook_secret: str | None = None, env: str = "local"
) -> ConnectorContext:
    settings = Settings(
        environment=env,
        sentry_webhook_secret=SecretStr(webhook_secret) if webhook_secret else None,
    )
    return ConnectorContext(settings=settings, http=httpx.AsyncClient())


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _build() -> SentryConnector:
    ctx = _make_ctx()
    return build_connector(SourceSystem.SENTRY, ctx)  # type: ignore[return-value]


def _make_oauth_ctx(
    status_code: int,
    body: str | dict,
    *,
    requests: list[httpx.Request] | None = None,
) -> ConnectorContext:
    settings = Settings(
        environment="local",
        sentry_client_id="sentry-cid",
        sentry_client_secret=SecretStr("sentry-secret"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        assert (
            str(request.url)
            == "https://sentry.io/api/0/sentry-app-installations/install-uuid/authorizations/"
        )
        if isinstance(body, str):
            return httpx.Response(status_code, text=body)
        return httpx.Response(status_code, json=body)

    return ConnectorContext(
        settings=settings,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


# ---------------------------------------------------------------------------
# parse_webhook_event
# ---------------------------------------------------------------------------


def test_parse_issue_created_produces_lifecycle_scoped_id() -> None:
    sentry = _build()
    payload = _load("issue_created.json")
    headers = {"Sentry-Hook-Resource": "issue"}

    result = sentry.parse_webhook_event("cust-1", headers, payload)

    assert result is not None
    assert result.source_event_id == "issue:1234567890:created"
    assert result.parse_hint["resource"] == "issue"
    assert result.parse_hint["action"] == "created"
    assert result.parse_hint["issue_id"] == "1234567890"
    assert result.parse_hint["project_slug"] == "payments-api"
    # lastSeen in fixture is 2026-04-22T14:10:02Z
    assert result.received_at == datetime(2026, 4, 22, 14, 10, 2, tzinfo=UTC)


def test_parse_installation_hook_returns_none() -> None:
    sentry = _build()
    payload = {
        "action": "created",
        "installation": {"uuid": "inst-xyz-0001"},
        "data": {"installation": {"uuid": "inst-xyz-0001"}},
    }
    headers = {"Sentry-Hook-Resource": "installation"}

    assert sentry.parse_webhook_event("cust-1", headers, payload) is None


def test_parse_event_alert_produces_event_id() -> None:
    sentry = _build()
    payload = _load("event_alert.json")
    headers = {"Sentry-Hook-Resource": "event_alert"}

    result = sentry.parse_webhook_event("cust-1", headers, payload)

    assert result is not None
    assert result.source_event_id == "event:abcdef0123456789abcdef0123456789"
    assert result.parse_hint["group_id"] == "1234567890"


def test_parse_issue_with_ignored_action_returns_none() -> None:
    sentry = _build()
    payload = _load("issue_created.json")
    payload["action"] = "ignored"
    headers = {"Sentry-Hook-Resource": "issue"}

    assert sentry.parse_webhook_event("cust-1", headers, payload) is None


def test_parse_missing_data_raises() -> None:
    sentry = _build()
    headers = {"Sentry-Hook-Resource": "issue"}
    with pytest.raises(InvalidWebhookPayload):
        sentry.parse_webhook_event("cust-1", headers, {"action": "created"})


# ---------------------------------------------------------------------------
# verify_signature
# ---------------------------------------------------------------------------


def test_verify_signature_dev_bypass() -> None:
    ctx = _make_ctx(webhook_secret=None, env="local")
    sentry = build_connector(SourceSystem.SENTRY, ctx)
    assert sentry.verify_signature({}, b"{}") is True


def test_verify_signature_prod_rejects_unsigned() -> None:
    ctx = _make_ctx(webhook_secret=None, env="main")
    sentry = build_connector(SourceSystem.SENTRY, ctx)
    assert sentry.verify_signature({}, b"{}") is False


def test_verify_signature_valid_and_tampered() -> None:
    secret = "shh-sentry"
    body = b'{"action":"created","data":{"issue":{"id":"1"}}}'
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    ctx = _make_ctx(webhook_secret=secret, env="main")
    sentry = build_connector(SourceSystem.SENTRY, ctx)

    headers = {"Sentry-Hook-Signature": expected}
    assert sentry.verify_signature(headers, body) is True
    # Tampered body fails.
    assert sentry.verify_signature(headers, body + b"x") is False
    # Missing header fails.
    assert sentry.verify_signature({}, body) is False


# ---------------------------------------------------------------------------
# exchange_oauth_code
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exchange_oauth_code_success() -> None:
    requests: list[httpx.Request] = []
    ctx = _make_oauth_ctx(
        200,
        {
            "token": "sntrys_access",
            "refreshToken": "sntrys_refresh",
            "expiresAt": "2026-05-03T10:30:00Z",
            "scope": "org:read project:read event:read project:releases",
        },
        requests=requests,
    )
    sentry = build_connector(SourceSystem.SENTRY, ctx)

    token = await sentry.exchange_oauth_code(
        code="auth-code",
        redirect_uri="https://api.prbe.ai/oauth/sentry/callback",
        extra_params={"installation_id": "install-uuid"},
    )

    assert token.source_system == SourceSystem.SENTRY
    assert token.access_token == "sntrys_access"
    assert token.refresh_token == "sntrys_refresh"
    assert token.scope == "org:read project:read event:read project:releases"
    assert token.expires_at == datetime(2026, 5, 3, 10, 30, tzinfo=UTC)

    assert len(requests) == 1
    request_body = json.loads(requests[0].content.decode())
    assert request_body["grant_type"] == "authorization_code"
    assert request_body["client_id"] == "sentry-cid"
    assert request_body["client_secret"] == "sentry-secret"
    assert request_body["code"] == "auth-code"


@pytest.mark.asyncio
async def test_exchange_oauth_code_5xx_raises_transient() -> None:
    ctx = _make_oauth_ctx(503, "temporarily unavailable")
    sentry = build_connector(SourceSystem.SENTRY, ctx)

    with pytest.raises(TransientSourceError) as exc_info:
        await sentry.exchange_oauth_code(
            code="auth-code",
            redirect_uri="https://api.prbe.ai/oauth/sentry/callback",
            extra_params={"installation_id": "install-uuid"},
        )

    assert exc_info.value.context["status"] == 503
    assert "temporarily unavailable" in exc_info.value.context["body"]


@pytest.mark.asyncio
async def test_exchange_oauth_code_4xx_raises_permanent() -> None:
    ctx = _make_oauth_ctx(400, {"error": "invalid_grant"})
    sentry = build_connector(SourceSystem.SENTRY, ctx)

    with pytest.raises(PermanentSourceError) as exc_info:
        await sentry.exchange_oauth_code(
            code="auth-code",
            redirect_uri="https://api.prbe.ai/oauth/sentry/callback",
            extra_params={"installation_id": "install-uuid"},
        )

    assert exc_info.value.context["status"] == 400
    assert "invalid_grant" in exc_info.value.context["body"]


@pytest.mark.asyncio
async def test_exchange_oauth_code_missing_access_token_raises_permanent() -> None:
    ctx = _make_oauth_ctx(200, {"refreshToken": "refresh-only"})
    sentry = build_connector(SourceSystem.SENTRY, ctx)

    with pytest.raises(PermanentSourceError) as exc_info:
        await sentry.exchange_oauth_code(
            code="auth-code",
            redirect_uri="https://api.prbe.ai/oauth/sentry/callback",
            extra_params={"installation_id": "install-uuid"},
        )

    assert "missing token" in str(exc_info.value)


@pytest.mark.asyncio
async def test_exchange_oauth_code_requires_installation_id() -> None:
    ctx = _make_oauth_ctx(200, {"token": "sntrys_access"})
    sentry = build_connector(SourceSystem.SENTRY, ctx)

    with pytest.raises(InvalidWebhookPayload) as exc_info:
        await sentry.exchange_oauth_code(
            code="auth-code",
            redirect_uri="https://api.prbe.ai/oauth/sentry/callback",
        )

    assert "missing installationId" in str(exc_info.value)


# ---------------------------------------------------------------------------
# identify_workspaces
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identify_workspaces_records_org_slug_and_id() -> None:
    settings = Settings(environment="local")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert str(request.url) == "https://sentry.io/api/0/organizations/"
        return httpx.Response(
            200,
            json=[
                {"id": "12345", "slug": "acme", "name": "Acme"},
                {"id": "67890", "slug": "beta", "name": "Beta"},
            ],
        )

    ctx = ConnectorContext(
        settings=settings,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    sentry = build_connector(SourceSystem.SENTRY, ctx)
    token = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.SENTRY,
        access_token="sntrys_access",
    )

    refs = await sentry.identify_workspaces(token)

    by_id = {ref.external_id: ref for ref in refs}
    assert set(by_id) == {"acme", "12345", "beta", "67890"}
    assert by_id["acme"].external_name == "Acme"
    assert by_id["acme"].metadata["mapping_kind"] == "org_slug"
    assert by_id["12345"].external_name == "Acme"
    assert by_id["12345"].metadata["mapping_kind"] == "org_id"
    assert requests[0].headers["authorization"] == "Bearer sntrys_access"


@pytest.mark.asyncio
async def test_backfill_starts_from_connected_org_mapping() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert str(request.url) == "https://sentry.io/api/0/organizations/acme/issues/"
        return httpx.Response(
            200,
            json=[
                {
                    "id": "1234567890",
                    "title": "TypeError: boom",
                    "lastSeen": "2026-04-22T14:10:02Z",
                    "project": {"slug": "payments-api"},
                }
            ],
        )

    ctx = ConnectorContext(
        settings=Settings(environment="local"),
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    sentry = build_connector(SourceSystem.SENTRY, ctx)
    token = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.SENTRY,
        access_token="sntrys_access",
    )

    with patch(
        "services.ingestion.handlers.sentry._connected_sentry_org_slug",
        new_callable=AsyncMock,
        return_value="acme",
    ):
        events = [event async for event in sentry.backfill("cust-1", token)]

    assert len(events) == 1
    assert events[0].source_event_id == "issue:1234567890:backfill"
    assert events[0].raw_payload["organization"]["slug"] == "acme"
    assert requests[0].headers["authorization"] == "Bearer sntrys_access"


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_issue_emits_document_graph_and_acl() -> None:
    sentry = _build()
    payload = _load("issue_created.json")

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.SENTRY,
        source_event_id="issue:1234567890:created",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/sentry/cust-1/2026/04/22/test.json",
        raw_payload=payload,
        headers={"Sentry-Hook-Resource": "issue"},
    )

    result = await sentry.normalize(event, {})
    assert not result.is_empty
    assert len(result.documents) == 1

    doc = result.documents[0]
    assert doc.source_system == SourceSystem.SENTRY
    assert doc.doc_type == DocType.SENTRY_ISSUE
    assert doc.doc_id == "sentry:issue:1234567890"
    assert doc.source_id == "1234567890"
    assert doc.title and "NoneType" in doc.title
    assert doc.author_id == "alice"
    assert doc.metadata["project_slug"] == "payments-api"
    assert doc.metadata["platform"] == "python"
    assert doc.metadata["action"] == "created"
    assert "payments.charges.handle_webhook" in doc.metadata["body"]
    # ACL embedded on the doc includes both workspace and project principals.
    principal_types = {p.principal_type for p in doc.acl.principals}
    assert PrincipalType.WORKSPACE in principal_types
    assert PrincipalType.GROUP in principal_types

    # Graph nodes: ERROR_GROUP, SERVICE, DOCUMENT, PERSON (assignee).
    node_keys = {(n.label, n.canonical_id) for n in result.graph_nodes}
    assert (NodeLabel.ERROR_GROUP, "1234567890") in node_keys
    assert (NodeLabel.SERVICE, "payments-api") in node_keys
    assert (NodeLabel.DOCUMENT, doc.doc_id) in node_keys
    assert (NodeLabel.PERSON, "alice") in node_keys

    # FIRES_IN edge ERROR_GROUP -> SERVICE.
    fires_in = [e for e in result.graph_edges if e.edge_type == EdgeType.FIRES_IN]
    assert len(fires_in) == 1
    assert fires_in[0].from_canonical_id == "1234567890"
    assert fires_in[0].to_canonical_id == "payments-api"

    # ASSIGNED_TO because the fixture has an assignee.
    assigned = [e for e in result.graph_edges if e.edge_type == EdgeType.ASSIGNED_TO]
    assert len(assigned) == 1
    assert assigned[0].to_canonical_id == "alice"

    # ACL snapshot rows: workspace + project-scoped group.
    acl_principal_types = {
        (r.principal_type, r.principal_id) for r in result.acl_snapshots
    }
    assert (PrincipalType.WORKSPACE, "acme") in acl_principal_types
    assert (PrincipalType.GROUP, "sentry-project:payments-api") in acl_principal_types
    for row in result.acl_snapshots:
        assert row.source_system == SourceSystem.SENTRY
        assert row.resource_type == "sentry.issue"
        assert row.resource_id == "1234567890"
        assert row.permission == Permission.READ


@pytest.mark.asyncio
async def test_normalize_event_alert_produces_issue_sample_doc() -> None:
    """Event webhooks produce a deterministic sample doc scoped to the issue.

    Identity is `sentry:issue:{group_id}:sample` with a fixed content_hash,
    so subsequent events for the same issue are no-oped by the normalizer's
    content-hash dedup. Fresh event data comes from the live Sentry tool,
    not the index.
    """
    sentry = _build()
    payload = _load("event_alert.json")

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.SENTRY,
        source_event_id="event:abcdef0123456789abcdef0123456789",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/sentry/cust-1/2026/04/22/event.json",
        raw_payload=payload,
        headers={"Sentry-Hook-Resource": "event_alert"},
    )

    result = await sentry.normalize(event, {})
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.doc_id == "sentry:issue:1234567890:sample"
    assert doc.doc_type == DocType.SENTRY_EVENT
    assert doc.parent_doc_id == "sentry:issue:1234567890"
    # Never attribute a sample to an actor — it's not an authored artifact.
    assert doc.author_id is None
    assert not any(e.edge_type == EdgeType.AUTHORED for e in result.graph_edges)
    # Body contains the exception header + a top stack frame.
    assert "TypeError" in doc.metadata["body"]
    assert "handle_webhook" in doc.metadata["body"]
    # Sample strategy is recorded in metadata so downstream callers / agents
    # know this is one-per-issue, not per-event.
    assert doc.metadata["sample_strategy"] == "first_event_per_issue"


@pytest.mark.asyncio
async def test_normalize_event_deterministic_identity_across_events() -> None:
    """Two different events for the same issue → identical doc_id + content_hash.

    This is what makes the index collapse the firehose to one row per issue:
    the normalizer's content_hash dedup no-ops every subsequent event.
    """
    sentry = _build()

    base_payload = _load("event_alert.json")
    # Clone and mutate one field in the inner event object to simulate a
    # different underlying occurrence (different event_id + stacktrace frame).
    second_payload = json.loads(json.dumps(base_payload))
    second_payload["data"]["event"]["event_id"] = "deadbeef" * 4
    # Add an extra stacktrace frame so the body genuinely differs.
    values = second_payload["data"]["event"]["exception"]["values"]
    if values and isinstance(values[0].get("stacktrace", {}).get("frames"), list):
        values[0]["stacktrace"]["frames"].append(
            {"filename": "other.py", "function": "other_fn", "lineno": 99}
        )

    def _event_for(p: dict) -> WebhookEvent:
        return WebhookEvent(
            customer_id="cust-1",
            source_system=SourceSystem.SENTRY,
            source_event_id=f"event:{p['data']['event']['event_id']}",
            received_at=datetime.now(UTC),
            payload_s3_key="raw/sentry/cust-1/t.json",
            raw_payload=p,
            headers={"Sentry-Hook-Resource": "event_alert"},
        )

    first = await sentry.normalize(_event_for(base_payload), {})
    second = await sentry.normalize(_event_for(second_payload), {})

    assert first.documents[0].doc_id == second.documents[0].doc_id
    assert first.documents[0].content_hash == second.documents[0].content_hash


@pytest.mark.asyncio
async def test_normalize_event_skips_when_group_id_missing() -> None:
    """Without a groupID there's no issue to anchor the sample to — skip."""
    sentry = _build()
    payload = _load("event_alert.json")
    payload["data"]["event"]["groupID"] = None

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.SENTRY,
        source_event_id="event:no_group",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/sentry/cust-1/t.json",
        raw_payload=payload,
        headers={"Sentry-Hook-Resource": "event_alert"},
    )

    result = await sentry.normalize(event, {})
    assert result.is_empty
    assert result.skipped_reason and "groupID" in result.skipped_reason
