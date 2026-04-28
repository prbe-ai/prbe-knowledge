"""Unit tests for the edge-backfill `_edges_for_doc` parser.

DB-backed integration is exercised via the existing entity_filter
test suite — once the script runs, those tests' fixture seeders are
the same INSERTs the script does on real data.
"""

from __future__ import annotations

import pytest

from scripts.backfill_entity_edges import _edges_for_doc
from shared.constants import EdgeType, NodeLabel, SourceSystem


def test_github_commit_yields_doc_to_repo() -> None:
    out = _edges_for_doc(
        SourceSystem.GITHUB.value,
        "github:prbe-ai/prbe-backend:commit:abc1234",
        {},
    )
    assert len(out) == 1
    assert out[0].edge_type == EdgeType.TOUCHES.value
    assert out[0].target_label == NodeLabel.REPO.value
    assert out[0].target_canonical_id == "prbe-ai/prbe-backend"


@pytest.mark.parametrize(
    "doc_id",
    [
        "github:prbe-ai/prbe-backend:pr:42",
        "github:prbe-ai/prbe-backend:issue:7",
        "github:prbe-ai/prbe-backend:review:99",
        "github:prbe-ai/prbe-backend:codeowners:abc",
    ],
)
def test_github_all_doc_types_yield_repo_edge(doc_id: str) -> None:
    out = _edges_for_doc(SourceSystem.GITHUB.value, doc_id, {})
    assert len(out) == 1
    assert out[0].target_canonical_id == "prbe-ai/prbe-backend"


def test_slack_message_yields_doc_to_channel() -> None:
    out = _edges_for_doc(
        SourceSystem.SLACK.value,
        "slack:T123:C456:1234567890.567890",
        {},
    )
    assert len(out) == 1
    assert out[0].edge_type == EdgeType.MEMBER_OF.value
    assert out[0].target_label == NodeLabel.CHANNEL.value
    assert out[0].target_canonical_id == "C456"


def test_slack_falls_back_to_metadata_channel_id() -> None:
    """If doc_id parsing fails (legacy format) but channel_id is in
    metadata, use that."""
    out = _edges_for_doc(
        SourceSystem.SLACK.value,
        "slack:legacy_format",  # not the canonical 4-part shape
        {"channel_id": "C-from-meta"},
    )
    assert len(out) == 1
    assert out[0].target_canonical_id == "C-from-meta"


def test_linear_issue_yields_doc_to_ticket() -> None:
    out = _edges_for_doc(
        SourceSystem.LINEAR.value,
        "linear:org-uuid:issue:PROJ-456",
        {},
    )
    assert len(out) == 1
    assert out[0].edge_type == EdgeType.LINKED_FROM.value
    assert out[0].target_label == NodeLabel.TICKET.value
    assert out[0].target_canonical_id == "PROJ-456"


def test_linear_comment_uses_metadata_issue_id() -> None:
    out = _edges_for_doc(
        SourceSystem.LINEAR.value,
        "linear:org-uuid:comment:comment-abc",
        {"issue_id": "PROJ-789"},
    )
    assert len(out) == 1
    assert out[0].target_canonical_id == "PROJ-789"


def test_linear_comment_no_metadata_yields_no_edge() -> None:
    out = _edges_for_doc(
        SourceSystem.LINEAR.value,
        "linear:org-uuid:comment:comment-abc",
        {},
    )
    assert out == []


def test_sentry_issue_yields_error_group_and_service() -> None:
    out = _edges_for_doc(
        SourceSystem.SENTRY.value,
        "sentry:issue:1234567890",
        {"project_slug": "payments-api"},
    )
    assert len(out) == 2
    labels = {e.target_label for e in out}
    assert NodeLabel.ERROR_GROUP.value in labels
    assert NodeLabel.SERVICE.value in labels


def test_sentry_event_sample_still_yields_edges() -> None:
    out = _edges_for_doc(
        SourceSystem.SENTRY.value,
        "sentry:issue:1234567890:sample",
        {"project_slug": "api-svc"},
    )
    assert len(out) == 2


def test_sentry_no_project_slug_yields_only_error_group() -> None:
    """Defensive — a sentry doc with missing project_slug metadata
    still gets the ErrorGroup edge from doc_id parsing."""
    out = _edges_for_doc(SourceSystem.SENTRY.value, "sentry:issue:gid", {})
    assert len(out) == 1
    assert out[0].target_label == NodeLabel.ERROR_GROUP.value


def test_notion_yields_no_edges() -> None:
    out = _edges_for_doc(
        SourceSystem.NOTION.value,
        "notion:workspace:page-abc",
        {},
    )
    assert out == []


def test_granola_yields_no_edges() -> None:
    out = _edges_for_doc(
        SourceSystem.GRANOLA.value,
        "granola:meeting:m-123",
        {},
    )
    assert out == []


def test_claude_code_yields_no_edges() -> None:
    out = _edges_for_doc(
        SourceSystem.CLAUDE_CODE.value,
        "claude_code:session:abc",
        {},
    )
    assert out == []


def test_malformed_github_doc_id_yields_no_edge() -> None:
    """Defensive: a doc_id that doesn't match the expected shape
    yields no edges rather than crashing."""
    out = _edges_for_doc(SourceSystem.GITHUB.value, "github:", {})
    assert out == []


def test_malformed_slack_doc_id_falls_back_to_no_edge() -> None:
    out = _edges_for_doc(SourceSystem.SLACK.value, "slack", {})
    assert out == []
