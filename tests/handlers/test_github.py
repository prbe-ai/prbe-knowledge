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
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.github import (
    GitHubConnector,
    _parse_co_authors,
    parse_codeowners,
)
from services.ingestion.handlers.registry import build_connector
from shared.config import Settings
from shared.constants import (
    GITHUB_INSTALLATION_SCOPE_PREFIX,
    DocType,
    EdgeType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.exceptions import InvalidWebhookPayload, NotSupportedByConnector
from shared.models import IntegrationToken, WebhookEvent

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
    # Trailing :<16-hex> is a stable payload-fingerprint suffix that disambiguates
    # rapid same-second updates (GitHub timestamps are per-second).
    assert result.source_event_id.startswith(
        "pr:prbe/payments:42:opened:2026-04-22T10:00:00Z:"
    )
    assert len(result.source_event_id.rsplit(":", 1)[-1]) == 16
    assert result.parse_hint["repo"] == "prbe/payments"
    assert result.parse_hint["number"] == 42


def test_parse_issue_opened() -> None:
    connector = _build()
    payload = _load("issue_opened.json")
    result = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "issues"}, payload
    )
    assert result is not None
    assert result.source_event_id.startswith(
        "issue:prbe/payments:17:opened:2026-04-22T09:00:00Z:"
    )
    assert len(result.source_event_id.rsplit(":", 1)[-1]) == 16
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
# Co-authored-by trailer parsing
# ---------------------------------------------------------------------------


def test_parse_co_authors_basic() -> None:
    """Standard `Co-authored-by:` trailer with name and email."""
    msg = (
        "feat: ship the thing\n"
        "\n"
        "Co-authored-by: Mahit Singh <mahit@prbe.ai>\n"
    )
    assert _parse_co_authors(msg) == [{"name": "Mahit Singh", "email": "mahit@prbe.ai"}]


def test_parse_co_authors_multiple_dedupes_by_email() -> None:
    msg = (
        "feat: ship the thing\n"
        "\n"
        "Co-authored-by: Mahit Singh <mahit@prbe.ai>\n"
        "Co-authored-by: Bob Jones <bob@example.com>\n"
        "Co-authored-by: M Singh <mahit@prbe.ai>\n"  # dup email, different name
    )
    out = _parse_co_authors(msg)
    assert len(out) == 2
    assert out[0]["email"] == "mahit@prbe.ai"
    assert out[1]["email"] == "bob@example.com"


def test_parse_co_authors_case_insensitive_trailer_key() -> None:
    msg = "fix\n\nCO-AUTHORED-BY: Alice <alice@example.com>\n"
    assert _parse_co_authors(msg) == [{"name": "Alice", "email": "alice@example.com"}]


def test_parse_co_authors_lowercases_email_only() -> None:
    msg = "fix\n\nCo-authored-by: Mahit Singh <Mahit@PRBE.AI>\n"
    assert _parse_co_authors(msg) == [{"name": "Mahit Singh", "email": "mahit@prbe.ai"}]


def test_parse_co_authors_ignores_non_trailer_text() -> None:
    """An inline mention or prose mention of co-authored-by must not match."""
    msg = "fix\n\nThis was originally co-authored-by alice but landed solo.\n"
    assert _parse_co_authors(msg) == []


def test_parse_co_authors_empty_message() -> None:
    assert _parse_co_authors("") == []


@pytest.mark.asyncio
async def test_normalize_push_with_co_authored_by_emits_person_and_edges() -> None:
    """A commit with `Co-authored-by:` trailers must:
      1. Stash the parsed co-authors in `Document.metadata["co_authors"]`.
      2. Emit a Person node per unique co-author email.
      3. Emit an AUTHORED edge from each co-author Person to the commit doc.
      4. NOT touch `Document.author_id` (that field stays single-valued
         and equals the primary committer).
    """
    connector = _build()
    payload = _load("push_with_codeowners.json")

    # Inject Co-authored-by trailers into the head commit's message and the
    # mirrored commits[0] entry. Mahit (distinct email) should produce a
    # Person + edge; Alice (matching primary committer's email) should
    # be skipped as a duplicate of the primary author.
    extended_message = (
        payload["head_commit"]["message"]
        + "\n\n"
        + "Co-authored-by: Mahit Singh <mahit@prbe.ai>\n"
        + "Co-authored-by: Alice Dev <alice@example.com>\n"  # primary, dedup
    )
    payload["head_commit"]["message"] = extended_message
    payload["commits"][0]["message"] = extended_message

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="push:prbe/payments:deadbeefcafe-coauth",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/push.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "push"},
    )

    result = await connector.normalize(event, {})

    commit_doc = next(d for d in result.documents if d.doc_type == DocType.GITHUB_COMMIT)
    assert commit_doc.author_id == "alice"  # primary author unchanged
    assert commit_doc.metadata["co_authors"] == [
        {"name": "Mahit Singh", "email": "mahit@prbe.ai"}
    ]

    # Mahit's Person node exists (keyed by email).
    person_ids = {n.canonical_id for n in result.graph_nodes if n.label == NodeLabel.PERSON}
    assert "mahit@prbe.ai" in person_ids
    assert "alice" in person_ids  # primary still there

    # AUTHORED edges from BOTH alice AND mahit point to the commit doc.
    authored_to_commit = [
        e
        for e in result.graph_edges
        if e.edge_type == EdgeType.AUTHORED and e.to_canonical_id == commit_doc.doc_id
    ]
    authored_from = {e.from_canonical_id for e in authored_to_commit}
    assert authored_from == {"alice", "mahit@prbe.ai"}


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


# ---------------------------------------------------------------------------
# oauth_install_url / exchange_oauth_code / identify_workspaces
# ---------------------------------------------------------------------------


def _make_github_ctx(
    *,
    app_slug: str | None = None,
    http: httpx.AsyncClient | None = None,
) -> ConnectorContext:
    settings = Settings(
        environment="local",
        github_app_slug=app_slug,
    )
    return ConnectorContext(
        settings=settings,
        http=http if http is not None else httpx.AsyncClient(),
    )


def test_oauth_install_url_uses_app_slug() -> None:
    ctx = _make_github_ctx(app_slug="prbe-knowledge-dev")
    connector = build_connector(SourceSystem.GITHUB, ctx)
    url = connector.oauth_install_url(
        "cust-1", "https://api.example.com/oauth/github/callback"
    )
    assert "/apps/prbe-knowledge-dev/installations/new" in url
    assert "state=cust-1" in url


def test_oauth_install_url_raises_without_slug() -> None:
    ctx = _make_github_ctx(app_slug=None)
    connector = build_connector(SourceSystem.GITHUB, ctx)
    with pytest.raises(NotSupportedByConnector):
        connector.oauth_install_url(
            "cust-1", "https://api.example.com/oauth/github/callback"
        )


@pytest.mark.asyncio
async def test_exchange_oauth_code_without_installation_id_raises() -> None:
    ctx = _make_github_ctx()
    connector = build_connector(SourceSystem.GITHUB, ctx)
    with pytest.raises(InvalidWebhookPayload):
        await connector.exchange_oauth_code(
            code=None,
            redirect_uri="https://api.example.com/oauth/github/callback",
            extra_params={},
        )


@pytest.mark.asyncio
async def test_exchange_oauth_code_with_installation_id_returns_token_scoped_to_installation() -> None:
    """After PR B the connector no longer mints to validate — the token is
    constructed directly. The first real fetch via prbe-backend's
    /internal/github/installation_token endpoint surfaces any failure.
    """
    installation_id = "99"
    ctx = _make_github_ctx()
    connector = build_connector(SourceSystem.GITHUB, ctx)

    token = await connector.exchange_oauth_code(
        code=None,
        redirect_uri="https://api.example.com/oauth/github/callback",
        extra_params={"installation_id": installation_id, "setup_action": "install"},
    )

    assert token.source_system is SourceSystem.GITHUB
    assert token.scope == f"{GITHUB_INSTALLATION_SCOPE_PREFIX}{installation_id}"
    assert token.access_token  # non-empty placeholder


@pytest.mark.asyncio
async def test_identify_workspaces_returns_installation_id() -> None:
    """Without the App private key (now in prbe-backend) we can't fetch
    the account login from GitHub — return only the installation_id.
    Webhook routing still works because extract_external_id_from_payload
    keys on the same id.
    """
    installation_id = "99"
    ctx = _make_github_ctx()
    connector = build_connector(SourceSystem.GITHUB, ctx)

    token = IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        access_token="installation-minted-on-demand",
        scope=f"{GITHUB_INSTALLATION_SCOPE_PREFIX}{installation_id}",
    )
    refs = await connector.identify_workspaces(token)

    assert len(refs) == 1
    ref = refs[0]
    assert ref.external_id == installation_id
    assert ref.external_name is None
    assert ref.metadata["installation_id"] == installation_id


# ---------------------------------------------------------------------------
# parse — release / commit_comment / repository
# ---------------------------------------------------------------------------


def test_parse_release_published() -> None:
    connector = _build()
    payload = _load("release_published.json")
    result = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "release"}, payload
    )
    assert result is not None
    assert result.source_event_id.startswith(
        "release:prbe/payments:7001:published:2026-04-22T11:05:00Z:"
    )
    # Trailing :<16-hex> is the payload fingerprint.
    assert len(result.source_event_id.rsplit(":", 1)[-1]) == 16
    assert result.parse_hint["repo"] == "prbe/payments"
    assert result.parse_hint["release_id"] == 7001
    assert result.parse_hint["action"] == "published"


def test_parse_release_deleted_passes_through_action() -> None:
    connector = _build()
    payload = _load("release_deleted.json")
    result = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "release"}, payload
    )
    assert result is not None
    assert result.parse_hint["action"] == "deleted"


def test_parse_release_unhandled_action_returns_none() -> None:
    connector = _build()
    payload = _load("release_published.json")
    payload["action"] = "assigned"
    assert (
        connector.parse_webhook_event(
            "cust-1", {"X-GitHub-Event": "release"}, payload
        )
        is None
    )


def test_parse_commit_comment_created() -> None:
    connector = _build()
    payload = _load("commit_comment_created.json")
    result = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "commit_comment"}, payload
    )
    assert result is not None
    assert result.source_event_id.startswith(
        "commit_comment:prbe/payments:8001:created:2026-04-22T13:00:00Z:"
    )
    assert result.parse_hint["comment_id"] == 8001
    assert result.parse_hint["commit_id"] == "deadbeefcafe1234567890abcdef0011223344"


def test_parse_repository_created_passes_through() -> None:
    connector = _build()
    payload = _load("repository_created.json")
    result = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "repository"}, payload
    )
    assert result is not None
    assert result.parse_hint["action"] == "created"
    assert result.parse_hint["repo"] == "prbe/billing"
    assert "old_full_name" not in result.parse_hint


def test_parse_repository_renamed_extracts_old_name() -> None:
    connector = _build()
    payload = _load("repository_renamed.json")
    result = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "repository"}, payload
    )
    assert result is not None
    assert result.parse_hint["action"] == "renamed"
    assert result.parse_hint["repo"] == "prbe/payments"
    assert result.parse_hint["old_full_name"] == "prbe/payments-old"


def test_parse_repository_transferred_extracts_old_owner() -> None:
    connector = _build()
    payload = _load("repository_transferred.json")
    result = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "repository"}, payload
    )
    assert result is not None
    assert result.parse_hint["action"] == "transferred"
    assert result.parse_hint["old_full_name"] == "prbe-old/payments"


def test_parse_repository_edited_returns_none() -> None:
    """`edited`, `publicized`, `privatized` don't change the repo registry —
    drop them at parse so they don't burn an ingestion_queue row."""
    connector = _build()
    payload = _load("repository_created.json")
    payload["action"] = "edited"
    assert (
        connector.parse_webhook_event(
            "cust-1", {"X-GitHub-Event": "repository"}, payload
        )
        is None
    )


def test_parse_repository_renamed_missing_changes_raises() -> None:
    connector = _build()
    payload = _load("repository_renamed.json")
    payload["changes"] = {}
    with pytest.raises(InvalidWebhookPayload):
        connector.parse_webhook_event(
            "cust-1", {"X-GitHub-Event": "repository"}, payload
        )


# ---------------------------------------------------------------------------
# normalize — release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_release_produces_document_and_graph() -> None:
    connector = _build()
    payload = _load("release_published.json")

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="release:prbe/payments:7001:published:2026-04-22T11:05:00Z",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/release.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "release"},
    )

    result = await connector.normalize(event, {})
    assert not result.is_empty
    assert len(result.documents) == 1

    doc = result.documents[0]
    assert doc.doc_type == DocType.GITHUB_RELEASE
    assert doc.author_id == "alice"
    assert doc.doc_id == "github:prbe/payments:release:7001"
    assert doc.source_url == "https://github.com/prbe/payments/releases/tag/v1.4.0"
    assert doc.title is not None and doc.title.startswith("v1.4.0")
    assert doc.deleted_at is None
    assert doc.metadata["tag_name"] == "v1.4.0"
    assert doc.metadata["prerelease"] is False
    assert doc.metadata["draft"] is False
    assert doc.metadata["release_id"] == 7001

    labels = {(n.label, n.canonical_id) for n in result.graph_nodes}
    assert (NodeLabel.REPO, "prbe/payments") in labels
    assert (NodeLabel.PERSON, "alice") in labels
    assert (NodeLabel.DOCUMENT, doc.doc_id) in labels

    edge_types = {e.edge_type for e in result.graph_edges}
    assert EdgeType.AUTHORED in edge_types
    assert EdgeType.TOUCHES in edge_types

    assert result.acl_snapshots
    acl = result.acl_snapshots[0]
    assert acl.principal_type == PrincipalType.WORKSPACE
    assert acl.permission == Permission.READ


@pytest.mark.asyncio
async def test_normalize_release_deleted_produces_tombstone() -> None:
    connector = _build()
    payload = _load("release_deleted.json")

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="release:prbe/payments:7001:deleted:2026-04-22T12:00:00Z",
        received_at=datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC),
        payload_s3_key="raw/github/cust-1/release.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "release"},
    )

    result = await connector.normalize(event, {})
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.deleted_at is not None
    assert doc.body == ""
    assert "__deleted__" in doc.content_hash or doc.content_hash != ""
    # Confirms tombstone hash path was taken (mirrors PR/issue tombstone test).
    expected_hash = hashlib.sha256(
        f"{doc.doc_id}|__deleted__|{event.received_at.isoformat()}".encode()
    ).hexdigest()
    assert doc.content_hash == expected_hash


# ---------------------------------------------------------------------------
# normalize — commit_comment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_commit_comment_produces_document_and_graph() -> None:
    connector = _build()
    payload = _load("commit_comment_created.json")

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="commit_comment:prbe/payments:8001:created:2026-04-22T13:00:00Z",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/commit_comment.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "commit_comment"},
    )

    result = await connector.normalize(event, {})
    assert not result.is_empty
    assert len(result.documents) == 1

    doc = result.documents[0]
    assert doc.doc_type == DocType.GITHUB_COMMIT_COMMENT
    assert doc.author_id == "alice"
    assert doc.doc_id == "github:prbe/payments:commit_comment:8001"
    assert doc.metadata["commit_id"] == "deadbeefcafe1234567890abcdef0011223344"
    assert doc.metadata["path"] == "services/payments/retry.py"
    assert doc.metadata["position"] == 42
    assert doc.body.startswith("Did we mean to remove")

    labels = {(n.label, n.canonical_id) for n in result.graph_nodes}
    assert (NodeLabel.REPO, "prbe/payments") in labels
    assert (NodeLabel.PERSON, "alice") in labels
    assert (NodeLabel.DOCUMENT, doc.doc_id) in labels

    edge_types = {e.edge_type for e in result.graph_edges}
    assert EdgeType.AUTHORED in edge_types
    assert EdgeType.TOUCHES in edge_types


# ---------------------------------------------------------------------------
# normalize — repository (codegraph bridge fan-out)
# ---------------------------------------------------------------------------


def _make_bridge_mocks(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """Patch the code-graph bridge functions on the github handler module
    so we can assert they were called without a real DB / R2 / queue."""
    from unittest.mock import AsyncMock

    backfill = AsyncMock(return_value=True)
    disconnect = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "services.ingestion.handlers.github.code_graph_bridge.enqueue_initial_backfill",
        backfill,
    )
    monkeypatch.setattr(
        "services.ingestion.handlers.github.code_graph_bridge.enqueue_disconnect",
        disconnect,
    )
    return backfill, disconnect


@pytest.mark.asyncio
async def test_normalize_repository_created_calls_backfill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backfill, disconnect = _make_bridge_mocks(monkeypatch)
    connector = _build()
    payload = _load("repository_created.json")

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="repository:prbe/billing:created:2026-04-22T14:00:00Z",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/repository.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "repository"},
    )

    result = await connector.normalize(event, {})
    assert result.documents == []  # registry signal, no Document
    assert backfill.await_count == 1
    kwargs = backfill.await_args.kwargs
    assert kwargs["repo"] == "prbe/billing"
    assert kwargs["head_sha"] == "HEAD"
    assert kwargs["originating_source"] == SourceSystem.GITHUB
    assert disconnect.await_count == 0


@pytest.mark.asyncio
async def test_normalize_repository_deleted_calls_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backfill, disconnect = _make_bridge_mocks(monkeypatch)
    connector = _build()
    payload = _load("repository_deleted.json")

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="repository:prbe/billing:deleted:2026-04-22T15:00:00Z",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/repository.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "repository"},
    )

    await connector.normalize(event, {})
    assert backfill.await_count == 0
    assert disconnect.await_count == 1
    assert disconnect.await_args.kwargs["repos"] == ["prbe/billing"]


@pytest.mark.asyncio
async def test_normalize_repository_renamed_disconnects_old_and_backfills_new(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backfill, disconnect = _make_bridge_mocks(monkeypatch)
    connector = _build()
    payload = _load("repository_renamed.json")

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="repository:prbe/payments:renamed:2026-04-22T16:00:00Z",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/repository.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "repository"},
    )

    await connector.normalize(event, {})
    assert disconnect.await_count == 1
    assert disconnect.await_args.kwargs["repos"] == ["prbe/payments-old"]
    assert backfill.await_count == 1
    assert backfill.await_args.kwargs["repo"] == "prbe/payments"


@pytest.mark.asyncio
async def test_normalize_repository_transferred_disconnects_old_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backfill, disconnect = _make_bridge_mocks(monkeypatch)
    connector = _build()
    payload = _load("repository_transferred.json")

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="repository:prbe/payments:transferred:2026-04-22T17:00:00Z",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/repository.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "repository"},
    )

    await connector.normalize(event, {})
    assert backfill.await_count == 0
    assert disconnect.await_count == 1
    # Pre-transfer org was prbe-old; current full_name is prbe/payments.
    assert disconnect.await_args.kwargs["repos"] == ["prbe-old/payments"]


# ---------------------------------------------------------------------------
# Regression tests for review findings
# ---------------------------------------------------------------------------


def test_parse_repository_source_event_id_stable_across_retries() -> None:
    """GitHub retries deliver identical bytes; source_event_id must collide
    so the ingestion_queue UNIQUE constraint dedupes them. Earlier code put
    `datetime.now()` in the ID and made every retry a fresh row."""
    connector = _build()
    payload = _load("repository_created.json")
    first = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "repository"}, payload
    )
    second = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "repository"}, payload
    )
    assert first is not None and second is not None
    assert first.source_event_id == second.source_event_id


def test_parse_repository_archived_returns_none() -> None:
    """`archived` is a no-op: bridge enqueue_initial_backfill dedupes on
    {repo}:HEAD so an archive→disconnect→unarchive cycle would tombstone
    permanently. Drop archive at parse so we don't burn a queue row."""
    connector = _build()
    payload = _load("repository_created.json")
    payload["action"] = "archived"
    assert (
        connector.parse_webhook_event(
            "cust-1", {"X-GitHub-Event": "repository"}, payload
        )
        is None
    )


def test_parse_repository_unarchived_returns_none() -> None:
    connector = _build()
    payload = _load("repository_created.json")
    payload["action"] = "unarchived"
    assert (
        connector.parse_webhook_event(
            "cust-1", {"X-GitHub-Event": "repository"}, payload
        )
        is None
    )


def test_parse_repository_transferred_with_rename() -> None:
    """GitHub can transfer-and-rename in one event (admin moves
    old-org/foo to new-org/bar atomically). The OLD path uses
    changes.repository.name.from for the trailing segment, not
    repository.name (which holds the new name)."""
    connector = _build()
    payload = _load("repository_transferred.json")
    payload["changes"]["repository"] = {"name": {"from": "payments-old"}}
    result = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "repository"}, payload
    )
    assert result is not None
    assert result.parse_hint["old_full_name"] == "prbe-old/payments-old"


@pytest.mark.asyncio
async def test_normalize_repository_renamed_backfills_before_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If disconnect lands first and backfill fails, the customer is left
    with no code-graph state until the next signal. Ordering: backfill
    new path first, then disconnect old."""
    from unittest.mock import AsyncMock

    call_order: list[str] = []

    async def _track_backfill(**kwargs: Any) -> bool:
        call_order.append(f"backfill:{kwargs['repo']}")
        return True

    async def _track_disconnect(**kwargs: Any) -> bool:
        call_order.append(f"disconnect:{','.join(kwargs['repos'])}")
        return True

    monkeypatch.setattr(
        "services.ingestion.handlers.github.code_graph_bridge.enqueue_initial_backfill",
        AsyncMock(side_effect=_track_backfill),
    )
    monkeypatch.setattr(
        "services.ingestion.handlers.github.code_graph_bridge.enqueue_disconnect",
        AsyncMock(side_effect=_track_disconnect),
    )

    connector = _build()
    payload = _load("repository_renamed.json")
    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="repository:prbe/payments:renamed:abc",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/repository.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "repository"},
    )
    await connector.normalize(event, {})
    assert call_order == ["backfill:prbe/payments", "disconnect:prbe/payments-old"]


@pytest.mark.asyncio
async def test_normalize_repository_transferred_with_rename_disconnects_old_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, disconnect = _make_bridge_mocks(monkeypatch)
    connector = _build()
    payload = _load("repository_transferred.json")
    payload["changes"]["repository"] = {"name": {"from": "payments-old"}}

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="repository:prbe/payments:transferred:abc",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/repository.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "repository"},
    )

    await connector.normalize(event, {})
    assert disconnect.await_count == 1
    assert disconnect.await_args.kwargs["repos"] == ["prbe-old/payments-old"]


def test_parse_release_drafts_filtered() -> None:
    """Drafts shouldn't appear in search until publication. The follow-up
    `published` event re-fires with the same release.id."""
    connector = _build()
    payload = _load("release_published.json")
    payload["release"]["draft"] = True
    assert (
        connector.parse_webhook_event(
            "cust-1", {"X-GitHub-Event": "release"}, payload
        )
        is None
    )


def test_parse_release_released_action_dropped() -> None:
    """`released` fires alongside `published` for the same publish event.
    Including both would write the release twice (different actions in
    source_event_id, neither deduping)."""
    connector = _build()
    payload = _load("release_published.json")
    payload["action"] = "released"
    assert (
        connector.parse_webhook_event(
            "cust-1", {"X-GitHub-Event": "release"}, payload
        )
        is None
    )


def test_parse_release_created_action_dropped() -> None:
    """GitHub fires `created` on draft save. Drop at parse — drafts are
    filtered defensively at normalize too."""
    connector = _build()
    payload = _load("release_published.json")
    payload["action"] = "created"
    assert (
        connector.parse_webhook_event(
            "cust-1", {"X-GitHub-Event": "release"}, payload
        )
        is None
    )


@pytest.mark.asyncio
async def test_normalize_release_null_author_skips_person_node() -> None:
    """Releases by GitHub Actions / GraphQL can carry null author.
    Don't collapse all null-author releases under a (PERSON, "unknown")
    node with bogus AUTHORED edges — leave author_id null and omit the
    PERSON node + AUTHORED edge entirely."""
    connector = _build()
    payload = _load("release_published.json")
    payload["release"]["author"] = None

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="release:prbe/payments:7001:published:abc",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/release.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "release"},
    )

    result = await connector.normalize(event, {})
    assert len(result.documents) == 1
    assert result.documents[0].author_id is None

    labels = {n.label for n in result.graph_nodes}
    assert NodeLabel.PERSON not in labels
    edge_types = {e.edge_type for e in result.graph_edges}
    assert EdgeType.AUTHORED not in edge_types


@pytest.mark.asyncio
async def test_normalize_release_caps_oversized_body() -> None:
    """Auto-generated changelogs can hit MB+. Cap at 256 KiB and flag
    body_truncated so retrieval can surface the partial-document state."""
    connector = _build()
    payload = _load("release_published.json")
    # 300 KB of ASCII; well above the 256 KiB cap.
    payload["release"]["body"] = "x" * (300 * 1024)

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="release:prbe/payments:7001:published:big",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/release.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "release"},
    )
    result = await connector.normalize(event, {})
    doc = result.documents[0]
    assert doc.metadata["body_truncated"] is True
    assert doc.body_size_bytes == 256 * 1024
    assert len(doc.body.encode("utf-8")) == 256 * 1024


@pytest.mark.asyncio
async def test_normalize_commit_comment_null_user_skips_person_node() -> None:
    connector = _build()
    payload = _load("commit_comment_created.json")
    payload["comment"]["user"] = None

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="commit_comment:prbe/payments:8001:created:abc",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/commit_comment.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "commit_comment"},
    )

    result = await connector.normalize(event, {})
    assert result.documents[0].author_id is None
    labels = {n.label for n in result.graph_nodes}
    assert NodeLabel.PERSON not in labels


def test_parse_repository_edited_default_branch_passes_through() -> None:
    connector = _build()
    payload = _load("repository_default_branch_changed.json")
    result = connector.parse_webhook_event(
        "cust-1", {"X-GitHub-Event": "repository"}, payload
    )
    assert result is not None
    assert result.parse_hint["action"] == "edited"


def test_parse_repository_edited_without_default_branch_returns_none() -> None:
    """Cosmetic `edited` events (description, topics, homepage) shouldn't
    burn an ingestion_queue row."""
    connector = _build()
    payload = _load("repository_default_branch_changed.json")
    payload["changes"] = {"description": {"from": "old"}}
    assert (
        connector.parse_webhook_event(
            "cust-1", {"X-GitHub-Event": "repository"}, payload
        )
        is None
    )


@pytest.mark.asyncio
async def test_normalize_repository_default_branch_change_disconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default-branch flip means prior code-graph state is stale.
    Disconnect to clear it; next push to the new default repopulates."""
    backfill, disconnect = _make_bridge_mocks(monkeypatch)
    connector = _build()
    payload = _load("repository_default_branch_changed.json")

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GITHUB,
        source_event_id="repository:prbe/payments:edited:abc",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/cust-1/repository.json",
        raw_payload=payload,
        headers={"X-GitHub-Event": "repository"},
    )

    await connector.normalize(event, {})
    assert backfill.await_count == 0
    assert disconnect.await_count == 1
    assert disconnect.await_args.kwargs["repos"] == ["prbe/payments"]
