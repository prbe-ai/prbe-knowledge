"""Tests for the class-body wiki-link parser.

Covers the kind=class / kind=source split, multi-link extraction, the
empty-target / unclosed-bracket skip rule, and line/col tracking. ``col``
is 0-indexed throughout (chars from start of line to the opening ``[``).

Refs: docs/superpowers/specs/2026-04-29-debugging-knowledge-graph-design.md §5.3.
"""

from __future__ import annotations

from services.kg.wiki_links import WikiLink, parse_wiki_links


def test_parse_class_link() -> None:
    body = "See [[auth-403-rbac]] for the alternative."
    links = parse_wiki_links(body)
    assert links == [WikiLink(kind="class", target="auth-403-rbac", line=1, col=4)]


def test_parse_source_code_link() -> None:
    body = "Defined in [[src/auth/jwt.ts#refreshToken]]."
    links = parse_wiki_links(body)
    assert links == [
        WikiLink(
            kind="source",
            target="src/auth/jwt.ts#refreshToken",
            line=1,
            col=11,
        )
    ]


def test_parse_multiple() -> None:
    body = "[[a-b]] and [[c-d]] and [[src/x.py#fn]]"
    links = parse_wiki_links(body)
    assert [link.target for link in links] == ["a-b", "c-d", "src/x.py#fn"]


def test_ignores_unmatched_brackets_and_empty_target() -> None:
    body = "[[unclosed and [single] and [[]]"
    assert parse_wiki_links(body) == []


def test_line_and_column_tracking() -> None:
    body = "line one\nthen [[target]]"
    links = parse_wiki_links(body)
    assert links[0].line == 2
    assert links[0].col == 5
