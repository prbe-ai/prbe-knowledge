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
from shared.db import with_tenant
from shared.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class _PageRow:
    wiki_type: str
    slug: str
    title: str
    summary: str


@dataclass(frozen=True)
class _RepoEdge:
    """Verified cross-repo edge for the architecture diagram."""

    source: str  # owner/name
    target: str  # owner/name
    bidirectional: bool


_INDEX_SYSTEM_PROMPT = (
    "You are writing the front page of an engineering wiki. The wiki "
    "covers one company; you have the full list of pages it contains "
    "(title + 1-line summary + type). Produce a Markdown body that "
    "feels like a thoughtful overview, NOT a table of contents.\n\n"
    "Required structure:\n\n"
    "  1. **`# {Company}` H1 + intro** (~3-5 sentences). Infer the "
    "company / product name from the corpus (typical signals: a repo "
    "named `<name>-something`, a project page, recurring mentions). "
    "Use that as the H1 — never the literal word `Wiki`, since the "
    "dashboard already shows the page title above your body. The "
    "intro: what is this company about, what's the main product, "
    "what's getting built? Do NOT list pages here.\n\n"
    "  2. **Pages** — list every page with a wiki link. Organize them "
    "however makes sense for THIS corpus (group by product line, by "
    "team, by service, by type — your call). **Lead with the most "
    "load-bearing pages first**: typically the company's repos / "
    "services come first (those are what the company actually builds), "
    "then runbooks, then people / customers / projects / events. Use "
    "`[[Title]]` syntax so the dashboard rewrites them into routed "
    "links. Include the 1-line summary after each link. **Never emit "
    "a bullet with no content** (`- ` on its own line) — every bullet "
    "must have a page link AND a summary or be omitted.\n\n"
    "**Do NOT emit a ```mermaid``` block.** The architecture diagram "
    "is generated deterministically by the system from verified "
    "code-graph edges and spliced into your output between the intro "
    "and the Pages section. Anything you write inside a fenced "
    "mermaid block will be replaced — and risks producing a "
    "malformed diagram if the syntax doesn't parse. Just write the "
    "intro, then go straight to the Pages section.\n\n"
    "Tone: direct, builder-to-builder. No corporate language. Don't "
    "narrate ('Below you will find...'). Just write the page.\n\n"
    "Output ONLY the Markdown body — no `# Wiki` heading at the top, "
    "no ```markdown fences around the whole thing, no ```mermaid "
    "block (the system handles the diagram)."
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


async def fetch_verified_repo_edges(customer_id: str) -> list[_RepoEdge]:
    """Read code-graph-extracted DEPENDS_ON edges for the customer's repos.

    Bidirectionality is computed at READ time: an edge is bidirectional
    iff the reverse edge (``B → A``) also exists in the result set. The
    extractor side (services/ingestion/code_graph/cross_repo_deps.py)
    persists each direction independently as repos finish their backfill,
    so the read derives the pairing without needing a "wait for all
    repos" coordinator.
    """
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT n_from.canonical_id AS source,
                   n_to.canonical_id   AS target
            FROM graph_edges e
            JOIN graph_nodes n_from
                 ON n_from.node_id = e.from_node_id
                AND n_from.customer_id = e.customer_id
            JOIN graph_nodes n_to
                 ON n_to.node_id = e.to_node_id
                AND n_to.customer_id = e.customer_id
            WHERE e.customer_id = $1
              AND e.edge_type = 'DEPENDS_ON'
              AND n_from.label = 'Repo'
              AND n_to.label = 'Repo'
              AND e.valid_to IS NULL
            """,
            customer_id,
        )
    pairs: set[tuple[str, str]] = {(r["source"], r["target"]) for r in rows}
    edges: list[_RepoEdge] = []
    seen: set[frozenset[str]] = set()
    for source, target in pairs:
        key = frozenset((source, target))
        if key in seen:
            continue
        seen.add(key)
        bidirectional = (target, source) in pairs
        edges.append(_RepoEdge(source=source, target=target, bidirectional=bidirectional))
    # Stable ordering: bidirectional first, then alpha by source then target,
    # so prompt input is deterministic and re-renders are diff-friendly.
    edges.sort(key=lambda e: (not e.bidirectional, e.source, e.target))
    return edges


def _format_edges_for_prompt(edges: list[_RepoEdge]) -> str:
    """Render the verified edges block.

    Empty edge set → a directive to SKIP the architecture diagram
    entirely. We'd rather show no diagram than a misleading "isolated
    nodes" placeholder that suggests we know the repos don't relate
    when in reality the code-graph extraction may simply not have run
    yet.
    """
    if not edges:
        return (
            "Verified architecture edges: NONE.\n"
            "Code-graph extraction has not produced any cross-repo edges "
            "for this customer (either it has not run yet, or no inter-"
            "repo references have been verified in the corpus).\n\n"
            "**SKIP the architecture diagram entirely.** Do NOT emit the "
            "```mermaid ``` block from step 2 of the structure. Move "
            "directly from the intro to the **Pages** section. Do NOT "
            "invent edges from page summaries; do NOT render isolated "
            "nodes as a placeholder. Showing no diagram is honest; "
            "showing a fake one is misleading."
        )
    lines = [
        "Verified architecture edges (USE ONLY THESE — do NOT invent more):",
        "",
    ]
    for edge in edges:
        marker = "<-->" if edge.bidirectional else "--->"
        note = "" if edge.bidirectional else "  (one-way; only the source side has evidence)"
        lines.append(f"  {edge.source} {marker} {edge.target}{note}")
    lines.append("")
    lines.append(
        "These are facts the page list / intro can reference. Do NOT "
        "emit a Mermaid diagram yourself — the system splices one in "
        "deterministically after your output."
    )
    return "\n".join(lines)


async def render_index_via_llm(
    rows: list[asyncpg.Record],
    *,
    customer_id: str | None = None,
    client: Any | None = None,
    model: str = WIKI_AGENT_MODEL,
) -> str:
    """Produce the wiki index body via Gemini Pro.

    Returns the markdown body verbatim. Falls back to ``_fallback_flat_list``
    when the LLM call fails or ``GOOGLE_API_KEY`` is unset — the index
    page must always render.

    When ``customer_id`` is supplied the function fetches the verified
    cross-repo edges for that customer (extracted by the code-graph
    pipeline) and passes them to the LLM as facts. The prompt then
    requires the architecture diagram to use ONLY these edges. Without
    a customer_id (older callers / tests) the LLM falls back to the
    looser "infer from page summaries" path.
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

    # DIAGRAM DISABLED — edges are no longer fetched or fed to the LLM
    # since the wiki index doesn't render an architecture diagram.
    # See the splice block at the end of this function.
    # edges: list[_RepoEdge] = []
    # if customer_id:
    #     try:
    #         edges = await fetch_verified_repo_edges(customer_id)
    #     except Exception as exc:
    #         log.warning(
    #             "index_renderer.edge_fetch_failed",
    #             error=str(exc),
    #             error_class=type(exc).__name__,
    #         )
    # edges_block = _format_edges_for_prompt(edges)

    user_prompt = (
        f"Wiki page corpus ({len(pages)} pages):\n\n"
        f"{_format_pages_for_prompt(pages)}"
    )

    try:
        resp = await client.aio.models.generate_content(
            model=model,
            contents=user_prompt,
            config={
                "system_instruction": _INDEX_SYSTEM_PROMPT,
                # 16k accommodates a 50-page index with summaries
                # comfortably. Bumped from 4096 after a probe-founders
                # run truncated mid-block — page list never landed in
                # the body, leaving only the intro + a half-written
                # Mermaid attempt.
                "max_output_tokens": 16384,
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

    # DIAGRAM DISABLED — paused 2026-05-08 (PR #192 paused cross-repo
    # edge extraction; this commit pauses the rendering side). The
    # wiki index no longer includes a mermaid architecture diagram.
    # We still strip any pre-existing mermaid block the LLM might
    # accidentally emit despite the system-prompt forbid (defense in
    # depth) — but we no longer rebuild and splice in a new one.
    #
    # To revive: uncomment the _build_mermaid_block + splice insertion
    # below, and re-enable cross-repo edge extraction (see CROSS-REPO
    # DEPS DISABLED markers in codegraph.py and nightly_trigger.py).
    from services.synthesis.diagram_renderer import splice_mermaid_block
    text = splice_mermaid_block(text, "")
    # from services.synthesis.diagram_renderer import (
    #     _build_mermaid_block,
    #     splice_mermaid_block,
    # )
    # new_block = _build_mermaid_block(edges)
    # text = splice_mermaid_block(text, new_block)

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
