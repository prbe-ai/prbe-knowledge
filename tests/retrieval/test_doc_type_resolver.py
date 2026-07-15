"""Doc-type token resolver: maps Haiku's unqualified tokens (e.g. "commit")
to the dotted DocType strings the documents table uses (e.g. "github.commit"),
narrowing by source filter when one is supplied.
"""

from __future__ import annotations

import pytest

from engine.retrieval.doc_type_resolver import resolve_doc_type_token
from engine.shared.constants import SourceSystem


@pytest.mark.parametrize(
    ("token", "expected_first"),
    [
        ("commit", "github.commit"),
        ("pr", "github.pull_request"),
        ("review", "github.review"),
        ("message", "slack.message"),
        ("thread", "slack.thread"),
        ("page", "notion.page"),
        ("ticket", "linear.issue"),
        ("comment", "linear.comment"),
        ("session", "claude_code.session"),
        ("meeting", "granola.meeting"),
    ],
)
def test_known_tokens_resolve(token: str, expected_first: str) -> None:
    out = resolve_doc_type_token(token)
    assert out is not None
    assert expected_first in out


def test_issue_token_fans_out_to_three_sources() -> None:
    """'issue' alone matches Linear, GitHub, and Sentry without source narrowing."""
    out = resolve_doc_type_token("issue")
    assert out is not None
    assert set(out) == {"linear.issue", "github.issue", "sentry.issue"}


def test_issue_token_narrowed_by_github_source() -> None:
    out = resolve_doc_type_token("issue", sources=[SourceSystem.GITHUB])
    assert out == ["github.issue"]


def test_issue_token_narrowed_by_linear_source() -> None:
    out = resolve_doc_type_token("issue", sources=[SourceSystem.LINEAR])
    assert out == ["linear.issue"]


def test_unknown_token_returns_none() -> None:
    assert resolve_doc_type_token("frobnication") is None


def test_none_or_empty_returns_none() -> None:
    assert resolve_doc_type_token(None) is None
    assert resolve_doc_type_token("") is None


def test_inconsistent_token_and_source_returns_none() -> None:
    """User asks for 'message' (Slack only) but filters to GitHub —
    inconsistent. Resolver returns None rather than falling back to the
    wider unfiltered set, which would silently override the source filter."""
    out = resolve_doc_type_token("message", sources=[SourceSystem.GITHUB])
    assert out is None


def test_token_is_case_insensitive() -> None:
    assert resolve_doc_type_token("COMMIT") == ["github.commit"]
    assert resolve_doc_type_token("Commit") == ["github.commit"]
