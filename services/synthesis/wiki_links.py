"""Typed wiki-link extractor + persistence helper.

Two responsibilities, kept in one module:

  (a) Pure parser — extract `[[type:slug]]` references from a wiki
      page's body markdown and from its YAML frontmatter. No DB, no
      I/O. The parser is intentionally lenient: invalid wiki_types are
      dropped with a warning rather than raising, because agents
      occasionally emit typos and a single bad reference must not fail
      a whole page persist.

  (b) `persist_links_for_page(conn, ...)` — atomic delete-then-insert
      inside the caller's transaction. Replaces all `link_source IN
      ('markdown','frontmatter')` links for the source page; manual
      links are preserved. The caller owns the transaction (so link
      writes can commit/rollback alongside the page persist when the
      caller threads a single connection through both).

Design notes (per docs/wiki-bootstrap-plan.md "Zero-LLM-call link
extraction"):

  - Markdown grammar:
      `[[type:slug]]`              -> bare mention, link_type=""
      `[[type:slug|display]]`      -> display label, link_type=""
      `[[type:slug|verb|display]]` -> relation verb + display, link_type=verb
    Display label is captured for completeness but never persisted —
    it's a render-time concern.

  - Frontmatter grammar:
      `<field>: <type>:<slug>`               -> one link, link_type=<field>
      `<field>: [<type>:<slug>, <type>:<slug>]` -> N links, link_type=<field>
    Non-string / non-list-of-string values are silently skipped.

  - Context window: 80 chars before + 80 chars after, newlines
    stripped, hard-capped at 200 (matches the migration 0045 CHECK).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, get_args

from services.synthesis.models import WikiType
from shared.logging import get_logger

log = get_logger(__name__)

# Allowed wiki_types — single source of truth lives in models.py.
_WIKI_TYPES: frozenset[str] = frozenset(get_args(WikiType))

# Markdown link grammar. The body uses `[[type:slug]]`, optionally with
# one or two `|`-separated suffixes. Both `type` and `slug` use a
# restrictive character class (lowercase ASCII + digits + dashes /
# underscores) to avoid matching arbitrary text inside `[[ ... ]]`.
# `agent_tools` accepts arbitrary slug chars up to 64; we keep the
# parser's class narrow but include underscore so legitimate slugs
# like `auth_rollback` extract instead of silently dropping.
_MARKDOWN_LINK_RE = re.compile(
    r"\[\[(?P<type>[a-z_]+):(?P<slug>[a-z0-9_-]+)"
    r"(?:\|(?P<a>[^\]|]+))?"
    r"(?:\|(?P<b>[^\]|]+))?\]\]"
)

# Frontmatter scalar grammar. A value is treated as a link iff it
# matches `<wiki_type>:<slug>` exactly. Same character classes as the
# markdown matcher.
_FRONTMATTER_REF_RE = re.compile(r"^([a-z_]+):([a-z0-9_-]+)$")

_CONTEXT_WINDOW = 80
_CONTEXT_MAX = 200


LinkSource = Literal["markdown", "frontmatter"]


@dataclass(frozen=True)
class ExtractedLink:
    """One typed link extracted from a wiki page.

    `dst_wiki_type` is NOT validated against `WikiType` at the dataclass
    level — the parser stays lenient and the writer (or an explicit
    filter) drops invalid types. This keeps the parser pure: garbage in
    -> garbage out + a warning, no exceptions.
    """

    dst_wiki_type: str
    dst_slug: str
    link_type: str
    context: str
    link_source: LinkSource


# ---------------------------------------------------------------------------
# Pure parsers
# ---------------------------------------------------------------------------


def _build_context(body: str, start: int, end: int) -> str:
    """Take 80 chars before + 80 chars after the match, strip newlines, cap 200."""
    pre = body[max(0, start - _CONTEXT_WINDOW) : start]
    post = body[end : end + _CONTEXT_WINDOW]
    snippet = (pre + body[start:end] + post).replace("\n", " ").replace("\r", " ")
    if len(snippet) > _CONTEXT_MAX:
        snippet = snippet[:_CONTEXT_MAX]
    return snippet


def _validate_type(dst_wiki_type: str, *, where: str, slug: str) -> bool:
    """Return True iff `dst_wiki_type` is in the WikiType allowlist.

    Logs a warning for the caller (synthesis worker / writer) when an
    agent emits a typo'd wiki_type. We never raise — the page is more
    important than the link graph.
    """
    if dst_wiki_type in _WIKI_TYPES:
        return True
    log.warning(
        "wiki_links.invalid_type",
        where=where,
        dst_wiki_type=dst_wiki_type,
        dst_slug=slug,
    )
    return False


def extract_links_from_markdown(body: str) -> list[ExtractedLink]:
    """Pull `[[type:slug...]]` references out of a markdown body.

    Three accepted shapes (see module docstring): bare, with display
    label, or with relation verb + display label. Invalid wiki_types
    are dropped with a warning.
    """
    links: list[ExtractedLink] = []
    for m in _MARKDOWN_LINK_RE.finditer(body):
        dst_wiki_type = m.group("type")
        dst_slug = m.group("slug")
        a = m.group("a")
        b = m.group("b")

        # 1 optional group  -> display label only, link_type=""
        # 2 optional groups -> first is the relation verb, second is the display
        link_type = a.strip() if a is not None and b is not None else ""

        if not _validate_type(dst_wiki_type, where="markdown", slug=dst_slug):
            continue

        links.append(
            ExtractedLink(
                dst_wiki_type=dst_wiki_type,
                dst_slug=dst_slug,
                link_type=link_type,
                context=_build_context(body, m.start(), m.end()),
                link_source="markdown",
            )
        )
    return links


def extract_links_from_frontmatter(frontmatter: dict[str, Any]) -> list[ExtractedLink]:
    """Pull `type:slug` references out of an already-parsed frontmatter dict.

    Field name becomes `link_type`. Skip fields whose value isn't a
    string or list-of-strings; skip individual list items that aren't
    strings or don't match the `type:slug` shape.
    """
    links: list[ExtractedLink] = []
    for field, value in frontmatter.items():
        if isinstance(value, str):
            candidates: list[str] = [value]
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            candidates = list(value)
        else:
            # Ints, dicts, mixed lists, etc. — not a link field.
            continue

        for raw in candidates:
            match = _FRONTMATTER_REF_RE.match(raw.strip())
            if match is None:
                continue
            dst_wiki_type, dst_slug = match.group(1), match.group(2)
            if not _validate_type(dst_wiki_type, where="frontmatter", slug=dst_slug):
                continue
            links.append(
                ExtractedLink(
                    dst_wiki_type=dst_wiki_type,
                    dst_slug=dst_slug,
                    link_type=str(field),
                    context="",
                    link_source="frontmatter",
                )
            )
    return links


def extract_links(body_markdown: str, frontmatter: dict[str, Any]) -> list[ExtractedLink]:
    """Combined extractor — markdown + frontmatter, deduped.

    Dedup key matches the wiki_links uniqueness constraint shape:
    `(dst_wiki_type, dst_slug, link_type, link_source)`. Two markdown
    occurrences of the same `[[type:slug]]` collapse to one row; a
    markdown link AND a frontmatter link with the same dst+type still
    produce two rows (different link_source).
    """
    seen: set[tuple[str, str, str, str]] = set()
    out: list[ExtractedLink] = []
    for link in [
        *extract_links_from_markdown(body_markdown),
        *extract_links_from_frontmatter(frontmatter),
    ]:
        key = (link.dst_wiki_type, link.dst_slug, link.link_type, link.link_source)
        if key in seen:
            continue
        seen.add(key)
        out.append(link)
    return out


# ---------------------------------------------------------------------------
# Persistence — caller owns the transaction
# ---------------------------------------------------------------------------


async def persist_links_for_page(
    conn: Any,
    *,
    customer_id: str,
    src_wiki_type: str,
    src_slug: str,
    extracted: list[ExtractedLink],
) -> None:
    """Replace markdown + frontmatter links for the given source page.

    The caller MUST have already opened a transaction on `conn` (e.g.,
    via `with_tenant(customer_id)`). This helper does plain SQL only:
    no BEGIN, no COMMIT. Manual links (`link_source = 'manual'`) are
    preserved across calls.

    Steps inside the caller's tx:
      1. DELETE existing markdown + frontmatter links for this src.
      2. Bulk INSERT extracted links via `executemany`, ON CONFLICT DO
         NOTHING (the unique constraint absorbs concurrent re-inserts).
    """
    await conn.execute(
        """
        DELETE FROM wiki_links
        WHERE customer_id = $1
          AND src_wiki_type = $2
          AND src_slug = $3
          AND link_source IN ('markdown', 'frontmatter')
        """,
        customer_id,
        src_wiki_type,
        src_slug,
    )

    if not extracted:
        return

    rows = [
        (
            customer_id,
            src_wiki_type,
            src_slug,
            link.dst_wiki_type,
            link.dst_slug,
            link.link_type,
            link.context,
            link.link_source,
        )
        for link in extracted
    ]
    await conn.executemany(
        """
        INSERT INTO wiki_links (
            customer_id, src_wiki_type, src_slug,
            dst_wiki_type, dst_slug, link_type, context, link_source
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT ON CONSTRAINT uq_wiki_links DO NOTHING
        """,
        rows,
    )
