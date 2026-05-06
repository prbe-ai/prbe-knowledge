"""Unit tests for the typed wiki-link parser.

No DB, no I/O — just exercises the regex grammar + frontmatter walker
+ context window + dedup. Tests live alongside the rest of the
services/synthesis unit tests.
"""

from __future__ import annotations

import pytest

from services.synthesis.wiki_links import (
    ExtractedLink,
    extract_links,
    extract_links_from_frontmatter,
    extract_links_from_markdown,
)

# ---------------------------------------------------------------------------
# Markdown grammar
# ---------------------------------------------------------------------------


def test_bare_link_has_empty_link_type() -> None:
    """`[[type:slug]]` -> one ExtractedLink with link_type=''."""
    body = "see [[person:maison]] for context."
    links = extract_links_from_markdown(body)
    assert len(links) == 1
    assert links[0].dst_wiki_type == "person"
    assert links[0].dst_slug == "maison"
    assert links[0].link_type == ""
    assert links[0].link_source == "markdown"


def test_single_pipe_is_display_label_not_verb() -> None:
    """`[[type:slug|display]]` -> link_type='' (single optional = display)."""
    body = "Maison [[person:maison|Maison]] joined."
    links = extract_links_from_markdown(body)
    assert len(links) == 1
    assert links[0].link_type == ""


def test_double_pipe_is_verb_then_display() -> None:
    """`[[type:slug|verb|display]]` -> link_type='verb'."""
    body = "Maison [[person:maison|works_at|Maison]] Probe."
    links = extract_links_from_markdown(body)
    assert len(links) == 1
    assert links[0].link_type == "works_at"


def test_multiple_links_in_one_body() -> None:
    body = (
        "[[person:maison|works_at|Maison]] drove "
        "[[decision:auth-rollback]] in [[event:2026-05-05-1on1|the 1:1]]."
    )
    links = extract_links_from_markdown(body)
    assert len(links) == 3
    by_dst = {(link.dst_wiki_type, link.dst_slug): link for link in links}
    assert by_dst[("person", "maison")].link_type == "works_at"
    assert by_dst[("decision", "auth-rollback")].link_type == ""
    assert by_dst[("event", "2026-05-05-1on1")].link_type == ""


def test_no_links_returns_empty_list() -> None:
    assert extract_links_from_markdown("plain markdown with no refs") == []
    assert extract_links_from_markdown("") == []


def test_invalid_wiki_type_dropped_with_warning(capsys: pytest.CaptureFixture[str]) -> None:
    body = "see [[bogus:foo]] and [[person:maison]]"
    links = extract_links_from_markdown(body)
    assert len(links) == 1
    assert links[0].dst_wiki_type == "person"
    # structlog renders to stdout in this codebase (not stdlib logging),
    # so we read captured stdout for the event name.
    assert "wiki_links.invalid_type" in capsys.readouterr().out


def test_context_window_strips_newlines_and_caps_at_200() -> None:
    """80 before + 80 after, newlines -> spaces, hard cap at 200 chars."""
    body = ("a" * 200) + "\n[[person:maison]]\n" + ("b" * 200)
    links = extract_links_from_markdown(body)
    assert len(links) == 1
    ctx = links[0].context
    assert "\n" not in ctx
    assert "\r" not in ctx
    assert len(ctx) <= 200
    # The match itself ([[person:maison]]) should appear in the context.
    assert "[[person:maison]]" in ctx


# ---------------------------------------------------------------------------
# Frontmatter walker
# ---------------------------------------------------------------------------


def test_frontmatter_scalar_string() -> None:
    """`works_at: company:probe` -> one link, link_type='works_at'."""
    fm = {"works_at": "company:probe"}
    links = extract_links_from_frontmatter(fm)
    assert links == [
        ExtractedLink(
            dst_wiki_type="company",
            dst_slug="probe",
            link_type="works_at",
            context="",
            link_source="frontmatter",
        )
    ]


def test_frontmatter_list_of_strings() -> None:
    """`owns: [service_card:auth, service_card:wiki]` -> two links."""
    fm = {"owns": ["service_card:auth", "service_card:wiki"]}
    links = extract_links_from_frontmatter(fm)
    assert len(links) == 2
    assert all(link.link_type == "owns" for link in links)
    assert {link.dst_slug for link in links} == {"auth", "wiki"}


def test_frontmatter_non_string_or_dict_ignored() -> None:
    """Ints, dicts, mixed-type lists, plain strings without `:` -> skipped."""
    fm = {
        "count": 42,
        "config": {"nested": "company:probe"},
        "mixed": ["company:probe", 7],
        "free_text": "not a link reference",
    }
    assert extract_links_from_frontmatter(fm) == []


def test_frontmatter_invalid_wiki_type_dropped(capsys: pytest.CaptureFixture[str]) -> None:
    fm = {"works_at": "bogus:foo"}
    links = extract_links_from_frontmatter(fm)
    assert links == []
    assert "wiki_links.invalid_type" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Combined extractor — dedup
# ---------------------------------------------------------------------------


def test_combined_dedup_collapses_identical_markdown_links() -> None:
    """Same dst+link_type+source listed twice -> one ExtractedLink."""
    body = "[[person:maison]] and again [[person:maison]]"
    links = extract_links(body, {})
    assert len(links) == 1


def test_combined_keeps_markdown_and_frontmatter_separate() -> None:
    """A markdown link and a frontmatter link with the same dst still produce
    two rows because link_source differs."""
    body = "see [[company:probe]]"
    fm = {"works_at": "company:probe"}
    links = extract_links(body, fm)
    sources = {link.link_source for link in links}
    assert sources == {"markdown", "frontmatter"}
    assert len(links) == 2
