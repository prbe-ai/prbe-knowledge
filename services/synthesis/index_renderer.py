"""LLM-driven wiki index renderer.

Replaces the old deterministic ``render_index_markdown`` function (which
grouped pages by ``wiki_type`` under fixed section headers) with a
Gemini-Pro call that:

  1. Writes a short intro paragraph describing what the company is about,
     synthesized from the page titles + summaries it sees.
  2. Emits a ``mermaid`` ``graph TD`` block showing repo<->repo /
     repo<->service relationships and how each repo contributes to the
     overall product.
  3. Organizes the page list however makes sense for THIS corpus —
     no hardcoded "## Service Cards / ## Decisions" sections.

The motivation is that wiki taxonomies are wildly different per
company; a fixed grouping by ``wiki_type`` was both noisy (one page
per section) and uninformative (the user wanted to see structure, not
a bucket label).

Falls back to a flat alphabetical list when the LLM call fails or the
``GOOGLE_API_KEY`` is unset, so the index page always renders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import asyncpg

from shared.constants import WIKI_AGENT_MODEL
from shared.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class _PageRow:
    wiki_type: str
    slug: str
    title: str
    summary: str


_INDEX_SYSTEM_PROMPT = (
    "You are writing the front page of an engineering wiki. The wiki "
    "covers one company; you have the full list of pages it contains "
    "(title + 1-line summary + type). Produce a Markdown body that "
    "feels like a thoughtful overview, NOT a table of contents.\n\n"
    "**Do NOT emit a top-level `# Wiki` heading.** The dashboard "
    "already renders the page title above your body. That said, you "
    "SHOULD lead with a `# {Company name}` heading where you infer the "
    "company / product name from the corpus (typical signals: a repo "
    "or service named `<name>-something`, a project page, recurring "
    "mentions). One H1 per page.\n\n"
    "Required structure:\n\n"
    "  1. **Intro** (~3-5 sentences) under that H1. What is this "
    "company about? Mention the main product / surface area / what's "
    "getting built. Do NOT list pages here.\n\n"
    "  2. **Architecture diagram** — a fenced ```mermaid block`` with "
    "`graph TD` (top-down). Nodes = repos and services. Use the actual "
    "repo / service names as node labels. Group related nodes with "
    "subgraphs when the structure is obvious. Aim for 5-15 nodes.\n\n"
    "    **Edge discipline (CRITICAL — read carefully):**\n"
    "    Only emit an edge between two nodes when there is BIDIRECTIONAL "
    "evidence in the corpus that the connection is real:\n"
    "      - Page A's summary explicitly mentions B (e.g. 'depends on B', "
    "'feeds B', 'reads from B'), AND\n"
    "      - Page B's summary explicitly mentions A (or B is documented "
    "as a thing A would talk to).\n\n"
    "    Naming similarity (`prbe-foo` and `prbe-bar` share a prefix) "
    "is NOT evidence. Past association (a repo that USED to be part of "
    "the product but isn't referenced by current core services) is NOT "
    "evidence — drop those edges. **Sparse accurate beats dense "
    "speculative**: when in doubt, omit. A 5-node graph with 4 verified "
    "edges is better than a 15-node graph where half the edges are "
    "guesses. This diagram is REQUIRED — even a 3-node graph beats no "
    "graph.\n\n"
    "  3. **Pages** — list every page with a wiki link. Organize them "
    "however makes sense for THIS corpus (group by product line, by "
    "team, by service, by type — your call). **Lead with the most "
    "load-bearing pages first**: typically the company's repos / "
    "services come first (those are what the company actually builds), "
    "then runbooks, then people / customers / projects / events. Use "
    "`[[Title]]` syntax so the dashboard rewrites them into routed "
    "links. Include the 1-line summary after each link. **Never emit "
    "a bullet with no content** (`- ` on its own line) — every bullet "
    "must have a page link AND a summary or be omitted.\n\n"
    "Tone: direct, builder-to-builder. No corporate language. Don't "
    "narrate ('Below you will find...'). Just write the page.\n\n"
    "Output ONLY the Markdown body — no `# Wiki` heading at the top, "
    "no ```markdown fences around the whole thing."
)


def _rows_to_pages(rows: list[asyncpg.Record]) -> list[_PageRow]:
    """Normalize asyncpg rows into the typed page list the LLM sees.

    Falls back to ``body_preview`` when ``metadata.summary`` is absent
    (manual uploads can omit it). Mirrors the precedent the deterministic
    renderer set so output equivalence with the fallback path holds.
    """
    pages: list[_PageRow] = []
    for row in rows:
        meta = row["metadata"] or {}
        if isinstance(meta, (str, bytes, bytearray)):
            import orjson

            meta = orjson.loads(meta)
        if not isinstance(meta, dict):
            meta = {}
        wiki_type = meta.get("wiki_type") or row["source_id"].split(":", 1)[0]
        slug = meta.get("slug") or row["source_id"].split(":", 1)[-1]
        title = row["title"] or slug
        summary = meta.get("summary") or row["body_preview"] or ""
        if isinstance(summary, str):
            summary = summary.strip().splitlines()[0] if summary.strip() else ""
        else:
            summary = ""
        pages.append(
            _PageRow(wiki_type=str(wiki_type), slug=str(slug), title=str(title), summary=summary)
        )
    return pages


def _fallback_flat_list(pages: list[_PageRow]) -> str:
    """Plain alphabetical list rendered when the LLM path is unavailable.

    No hardcoded section headers — matches the new no-grouping intent.
    """
    if not pages:
        return "# Wiki\n\nNo pages yet.\n"
    sorted_pages = sorted(pages, key=lambda p: p.title.lower())
    parts = ["# Wiki", ""]
    for page in sorted_pages:
        line = f"- [[{page.title}]]"
        if page.summary:
            line += f" — {page.summary}"
        parts.append(line)
    return "\n".join(parts).rstrip() + "\n"


def _format_pages_for_prompt(pages: list[_PageRow]) -> str:
    """Compact YAML-ish render the LLM consumes as user content."""
    lines: list[str] = []
    for page in pages:
        lines.append(
            f"- type: {page.wiki_type}\n"
            f"  slug: {page.slug}\n"
            f"  title: {page.title}\n"
            f"  summary: {page.summary or '(none)'}"
        )
    return "\n".join(lines)


async def render_index_via_llm(
    rows: list[asyncpg.Record],
    *,
    client: Any | None = None,
    model: str = WIKI_AGENT_MODEL,
) -> str:
    """Produce the wiki index body via Gemini Pro.

    Returns the markdown body verbatim. Falls back to ``_fallback_flat_list``
    when the LLM call fails or ``GOOGLE_API_KEY`` is unset — the index
    page must always render.
    """
    pages = _rows_to_pages(rows)
    if not pages:
        return "# Wiki\n\nNo pages yet.\n"

    if client is None:
        try:
            from google import genai

            from shared.config import get_settings

            api_key = get_settings().google_api_key.get_secret_value()
            if not api_key:
                log.warning("index_renderer.no_google_api_key_falling_back")
                return _fallback_flat_list(pages)
            client = genai.Client(api_key=api_key)
        except ImportError as exc:
            log.warning(
                "index_renderer.google_genai_missing_falling_back", error=str(exc)
            )
            return _fallback_flat_list(pages)

    user_prompt = (
        f"Wiki page corpus ({len(pages)} pages):\n\n"
        f"{_format_pages_for_prompt(pages)}\n"
    )

    try:
        resp = await client.aio.models.generate_content(
            model=model,
            contents=user_prompt,
            config={
                "system_instruction": _INDEX_SYSTEM_PROMPT,
                "max_output_tokens": 4096,
            },
        )
    except Exception as exc:
        log.warning(
            "index_renderer.gemini_failed_falling_back",
            error=str(exc),
            error_class=type(exc).__name__,
            page_count=len(pages),
        )
        return _fallback_flat_list(pages)

    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        log.warning("index_renderer.empty_response_falling_back", page_count=len(pages))
        return _fallback_flat_list(pages)

    text = _strip_leading_wiki_heading(text)
    text = _strip_empty_bullets(text)
    return text + "\n" if not text.endswith("\n") else text


# The dashboard renders its own page title above the body, so a leading
# `# Wiki` line in the LLM output produces a duplicate "Wiki / Wiki"
# stack. The system prompt forbids it but cheap belt-and-braces defence
# beats trusting the model on every drain.
_LEADING_WIKI_HEADING_RE = re.compile(r"^\s*#\s+Wiki\s*\n+", re.IGNORECASE)

# Empty bullet lines — `- ` on its own with nothing after the dash, or
# the same with a `[[]]` skeleton the model sometimes emits when it
# loses track. Stripping these is preferable to rendering a phantom
# bullet in the UI.
_EMPTY_BULLET_RE = re.compile(
    r"^[ \t]*[-*+][ \t]*(?:\[\[\s*\]\])?[ \t]*$\n?", re.MULTILINE
)


def _strip_leading_wiki_heading(text: str) -> str:
    return _LEADING_WIKI_HEADING_RE.sub("", text, count=1)


def _strip_empty_bullets(text: str) -> str:
    return _EMPTY_BULLET_RE.sub("", text)
