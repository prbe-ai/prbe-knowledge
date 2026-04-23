"""Unit tests for the GitHub connector.

Exercises the Connector contract on realistic GitHub webhook payloads without
needing DB / R2. Also unit-tests the CODEOWNERS parser directly — it's the
most failure-prone piece and deserves standalone coverage.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.github import (
    GitHubConnector,
    parse_codeowners,
)
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
from shared.models import WebhookEvent

FIXTURES = Path(__file__).resolve().parents[1].parent / "fixtures" / "github"


def _make_ctx(
    *, webhook_secret: str | None = None, env: str = "local"
) -> ConnectorContext:
    settings = Settings(
        environment=env,
        github_webhook_secret=SecretStr(webhook_secret) if webhook_secret else None,
    )
    return ConnectorContext(settings=settings, http=httpx.AsyncClient())


def _load(name: str) -> dict:
    with (FIXTURES / name).open() as fh:
        return json.load(fh)


def _build() -> GitHubConnector:
    ctx = _make_ctx()
    connector = build_connector(SourceSystem.GITHUB, ctx)
    assert isinstance(connector, GitHubConnector)
    return connector


# ---------------------------------------------------------------------------
# parse_webhook_event
# ---------------------------------------------------------------------------


def test_parse_pull_request_opened() -> None:
    connector = _build()
    payload = _load("pr_opened.json")
    result = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "pull_request"}, payload
    )
    assert result is not None
    assert result.source_event_id == "pr:prbe/payments:42:opened:2026-04-22T10:00:00Z"
    assert result.parse_hint["repo"] == "prbe/payments"
    assert result.parse_hint["number"] == 42


def test_parse_issue_opened() -> None:
    connector = _build()
    payload = _load("issue_opened.json")
    result = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "issues"}, payload
    )
    assert result is not None
    assert result.source_event_id == "issue:prbe/payments:17:opened:2026-04-22T09:00:00Z"
    assert result.parse_hint["number"] == 17


def test_parse_push_head_commit() -> None:
    connector = _build()
    payload = _load("push_with_codeowners.json")
    result = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "push"}, payload
    )
    assert result is not None
    assert result.source_event_id.startswith("push:prbe/payments:")
    assert result.parse_hint["touches_codeowners"] is True


def test_parse_pr_review_submitted() -> None:
    connector = _build()
    payload = _load("pr_review.json")
    result = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "pull_request_review"}, payload
    )
    assert result is not None
    assert result.source_event_id == "review:prbe/payments:42:3003"
    assert result.parse_hint["pr_number"] == 42
    assert result.parse_hint["review_id"] == 3003


def test_parse_returns_none_for_irrelevant_events() -> None:
    connector = _build()
    payload_star = {
        "action": "created",
        "repository": {"full_name": "prbe/payments", "owner": {"login": "prbe"}},
    }
    # `watch` and `star` events aren't in our allowlist.
    assert (
        connector.parse_webhook_event(
            "cust-1", {"X-GitHub-Event": "watch"}, payload_star
        )
        is None
    )
    assert (
        connector.parse_webhook_event(
            "cust-1", {"X-GitHub-Event": "star"}, payload_star
        )
        is None
    )

    # `pull_request` with an unhandled action (e.g. `assigned`) is also None.
    pr_payload = _load("pr_opened.json")
    pr_payload["action"] = "assigned"
    assert (
        connector.parse_webhook_event(
            "cust-1", {"X-GitHub-Event": "pull_request"}, pr_payload
        )
        is None
    )


def test_parse_raises_on_missing_github_event_header() -> None:
    connector = _build()
    with pytest.raises(InvalidWebhookPayload):
        connector.parse_webhook_event("cust-1", {}, {"repository": {}})


# ---------------------------------------------------------------------------
# verify_signature
# ---------------------------------------------------------------------------


def test_verify_signature_valid_hmac() -> None:
    secret = "s3cr3t"
    body = b'{"hello":"world"}'
    expected_sig = (
        "sha256="
        + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    )
    ctx = _make_ctx(webhook_secret=secret, env="main")
    connector = build_connector(SourceSystem.GITHUB, ctx)

    headers = {"X-Hub-Signature-256": expected_sig}
    assert connector.verify_signature(headers, body) is True
    # Tampered body fails.
    assert connector.verify_signature(headers, body + b"x") is False


def test_verify_signature_dev_bypass() -> None:
    ctx = _make_ctx(webhook_secret=None, env="local")
    connector = build_connector(SourceSystem.GITHUB, ctx)
    assert connector.verify_signature({}, b"{}") is True


def test_verify_signature_prod_rejects_unsigned() -> None:
    ctx = _make_ctx(webhook_secret=None, env="main")
    connector = build_connector(SourceSystem.GITHUB, ctx)
    assert connector.verify_signature({}, b"{}") is False


def test_verify_signature_rejects_malformed_header() -> None:
    ctx = _make_ctx(webhook_secret="s3cr3t", env="main")
    connector = build_connector(SourceSystem.GITHUB, ctx)
    # Header present but wrong prefix.
    assert connector.verify_signature({"X-Hub-Signature-256": "md5=abc"}, b"{}") is False
    # Header missing entirely.
    assert connector.verify_signature({}, b"{}") is False


# ---------------------------------------------------------------------------
# normalize — PR
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_pr_produces_document_and_graph() -> None:
    connector = _build()
    payload = _load("pr_opened.json")

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="pr:prbe/payments:42:opened:2026-04-22T10:00:00Z",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/pr.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "pull_request"},
    )

    result = await connector.normalize(event, {})
    assert not result.is_empty
    assert len(result.documents) == 1

    doc = result.documents[0]
    assert doc.source_system == SourceSystem.GITHUB
    assert doc.doc_type == DocType.GITHUB_PULL_REQUEST
    assert doc.author_id == "alice"
    assert doc.doc_id == "github:prbe/payments:pr:42"
    assert doc.source_url == "https://github.com/prbe/payments/pull/42"
    assert doc.metadata["base_ref"] == "main"
    assert doc.metadata["head_ref"] == "fix/payments-retry"
    assert doc.metadata["changed_files"] == 3

    labels = {(n.label, n.canonical_id) for n in result.graph_nodes}
    assert (NodeLabel.REPO, "prbe/payments") in labels
    assert (NodeLabel.PR, "prbe/payments#42") in labels
    assert (NodeLabel.PERSON, "alice") in labels
    assert (NodeLabel.DOCUMENT, doc.doc_id) in labels

    edge_types = {e.edge_type for e in result.graph_edges}
    assert EdgeType.AUTHORED in edge_types
    assert EdgeType.TOUCHES in edge_types
    # Body references #17 (same-repo) and prbe/other#8 (cross-repo) → MENTIONS.
    mentions = [e for e in result.graph_edges if e.edge_type == EdgeType.MENTIONS]
    mention_targets = {e.to_canonical_id for e in mentions}
    assert "prbe/payments#17" in mention_targets
    assert "prbe/other#8" in mention_targets

    # ACL snapshot captures the workspace-scoped READ permission.
    assert result.acl_snapshots
    acl = result.acl_snapshots[0]
    assert acl.principal_type == PrincipalType.WORKSPACE
    assert acl.principal_id == "prbe"
    assert acl.permission == Permission.READ
    assert acl.resource_id == "prbe/payments"


# ---------------------------------------------------------------------------
# normalize — push with CODEOWNERS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_push_with_codeowners_emits_owns_edges() -> None:
    connector = _build()
    payload = _load("push_with_codeowners.json")
    codeowners_text = (FIXTURES / "codeowners.txt").read_text()

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="push:prbe/payments:deadbeefcafe",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/push.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "push"},
    )

    result = await connector.normalize(
        event, {"codeowners_content": codeowners_text, "codeowners_path": ".github/CODEOWNERS"}
    )

    doc_types = [d.doc_type for d in result.documents]
    assert DocType.GITHUB_COMMIT in doc_types
    assert DocType.GITHUB_CODEOWNERS in doc_types

    co_doc = next(d for d in result.documents if d.doc_type == DocType.GITHUB_CODEOWNERS)
    ownership = co_doc.metadata["ownership_map"]
    assert "*" in ownership
    assert "@prbe/payments-squad" in ownership["*"]
    assert co_doc.metadata["codeowners_fetch_skipped"] is False

    # One OWNS edge per (pattern, owner) — verify a couple.
    owns_edges = [e for e in result.graph_edges if e.edge_type == EdgeType.OWNS]
    assert owns_edges, "expected at least one OWNS edge"
    team_edge = next(
        e for e in owns_edges if e.properties.get("path_pattern") == "*"
    )
    assert team_edge.properties["is_team"] is True
    assert team_edge.from_canonical_id == "prbe/payments-squad"
    assert team_edge.to_canonical_id == "prbe/payments"


@pytest.mark.asyncio
async def test_normalize_push_without_token_skips_ownership() -> None:
    connector = _build()
    payload = _load("push_with_codeowners.json")

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="push:prbe/payments:deadbeefcafe",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/push.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "push"},
    )

    # No hydrated content — simulates missing installation token.
    result = await connector.normalize(event, {})
    co_doc = next(d for d in result.documents if d.doc_type == DocType.GITHUB_CODEOWNERS)
    assert co_doc.metadata["codeowners_fetch_skipped"] is True
    assert co_doc.metadata["ownership_map"] == {}
    owns_edges = [e for e in result.graph_edges if e.edge_type == EdgeType.OWNS]
    assert owns_edges == []


# ---------------------------------------------------------------------------
# parse_codeowners unit test
# ---------------------------------------------------------------------------


def test_parse_codeowners_handles_comments_blanks_and_teams() -> None:
    text = (FIXTURES / "codeowners.txt").read_text()
    result = parse_codeowners(text)

    assert result["*"] == ["@prbe/payments-squad"]
    assert result["docs/"] == ["@prbe/tech-writers", "@alice"]
    assert result["services/*.py"] == ["@prbe/backend-team"]
    assert result["/infra/terraform/"] == ["@ops-lead"]

    # Comment/blank lines must not become entries.
    assert "#" not in result
    assert "nothing-here" not in result


def test_parse_codeowners_empty_input() -> None:
    assert parse_codeowners("") == {}
    assert parse_codeowners("# only a comment\n\n   # indented\n") == {}


def test_parse_codeowners_skips_pattern_without_owner() -> None:
    result = parse_codeowners("docs/    # owner missing — skip\n*.py @alice\n")
    assert "docs/" not in result
    assert result["*.py"] == ["@alice"]


def test_parse_codeowners_last_match_wins_on_duplicate_pattern() -> None:
    text = "*.py @alice\n*.py @bob @carol\n"
    result = parse_codeowners(text)
    assert result["*.py"] == ["@bob", "@carol"]
