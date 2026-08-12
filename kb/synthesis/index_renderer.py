"""LLM-driven wiki index renderer: the front page is an OVERVIEW, not an index.

What this used to produce, and why it changed. The prompt asked for "a
thoughtful overview, NOT a table of contents" and then instructed the model to
list every page with its summary. With 37 pages that arithmetic only has one
outcome: four sentences of prose on top of a 37-line directory. Worse, the
model was handed page titles and one-line summaries ONLY -- it had never read a
single page it was describing -- so re-grouping the list it was given was the
most it could physically do.

Both halves are fixed here:

  1. The renderer reads page BODIES (``_PER_PAGE_BODY_CHARS`` of each), so the
     overview can make concrete claims about the work.
  2. The prompt forbids enumerating pages. Discovery lives where directories
     belong -- the dashboard's page browser, ``probe wiki list``, and the MCP's
     ``view="pages"`` -- and the front page spends itself on meaning instead.

The architecture diagram (item 2 of the old docstring) has been disabled since
2026-05-08; see the splice block at the end of ``render_index_via_llm``.

FAILURE IS A NO-OP, NOT A SUBSTITUTE. When the LLM is unavailable the renderer
returns ``None`` and the caller leaves the existing page alone. The old code
substituted a flat alphabetical page list, which under this design would mean
one Gemini hiccup silently reverts the front page to the directory this change
exists to remove.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import asyncpg

from engine.shared.constants import WIKI_AGENT_MODEL
from engine.shared.db import with_tenant
from engine.shared.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class _PageRow:
    wiki_type: str
    slug: str
    title: str
    summary: str
    #: The page's actual prose, trimmed to `_PER_PAGE_BODY_CHARS`. Empty when
    #: the caller did not fetch bodies (older callers, and the tests that
    #: predate them).
    body: str = ""


# How much of each page the renderer reads. The front page is a synthesis of
# what the wiki KNOWS, and a model given only titles and one-line summaries
# cannot write one -- it can only re-list what it was handed, which is how the
# old front page ended up as a directory with a paragraph on top.
#
# 4k chars is roughly the first half of a substantial page: enough to carry the
# claims and the shape, cheap enough that a 100-page wiki still fits in one
# request with room to spare.
_PER_PAGE_BODY_CHARS = 4000

# Ceiling across ALL pages, so a wiki that grows to hundreds of pages degrades
# by dropping the tail rather than by failing the request. Pages arrive
# newest-updated first, so the tail is the stalest material.
_TOTAL_BODY_CHARS = 250_000


@dataclass(frozen=True)
class _RepoEdge:
    """Verified cross-repo edge for the architecture diagram."""

    source: str  # owner/name
    target: str  # owner/name
    bidirectional: bool


_INDEX_SYSTEM_PROMPT = (
    "You are writing the front page of an engineering wiki. You have the "
    "wiki's pages, each with its type, title, one-line summary, and the "
    "opening of its actual text. Write a HIGH-LEVEL OVERVIEW of what this "
    "company is and what it is working on.\n\n"
    "**DO NOT LIST THE PAGES.** This is the instruction most likely to be "
    "ignored, so read it twice. No 'Pages' section, no directory, no "
    "grouped bullet list of every page with its summary after it. A reader "
    "who wants the full list has a page browser and a `wiki list` command; "
    "the front page is the one surface that can tell them what it all MEANS, "
    "and spending it on an index they can get elsewhere wastes it.\n\n"
    "What to write:\n\n"
    "  1. **`# {Company}` H1.** Infer the company / product name from the "
    "corpus (typical signals: a repo named `<name>-something`, a project "
    "page, recurring mentions). Never the literal word `Wiki` — the "
    "dashboard already shows the page title above your body.\n\n"
    "  2. **The overview itself, in prose** — a handful of short sections "
    "with `##` headings, chosen to fit THIS company. Aim for something a "
    "new engineer could read in two minutes and come away knowing what is "
    "being built, how the main pieces fit together, what the team has "
    "learned, and what is actively moving. Draw on the page BODIES you have "
    "been given, not just their titles: specifics are the whole point. "
    "Concrete claims, named systems, real decisions and their reasons.\n\n"
    "Linking: mention a page with `[[Title]]` where it genuinely helps a "
    "reader go deeper on something you just said. That is a handful of "
    "links inside prose, not a bullet per page. A paragraph that is mostly "
    "links has become the directory this page must not be.\n\n"
    "Do NOT write a section that merely names the categories of pages that "
    "exist ('the wiki also covers several runbooks and people'). Say "
    "something true about the work or leave it out.\n\n"
    "**Do NOT emit a ```mermaid``` block.** Diagram rendering is disabled "
    "and anything inside a fenced mermaid block is stripped before the page "
    "is stored.\n\n"
    "Tone: direct, builder-to-builder. No corporate language. Don't narrate "
    "('Below you will find...'). Just write the page.\n\n"
    "Output ONLY the Markdown body — no ```markdown fences around the whole "
    "thing."
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
        try:
            body = row["body"]
        except (KeyError, IndexError):
            body = ""
        pages.append(
            _PageRow(
                wiki_type=str(wiki_type),
                slug=str(slug),
                title=str(title),
                summary=summary,
                body=str(body or ""),
            )
        )
    return pages


def _format_pages_for_prompt(pages: list[_PageRow]) -> tuple[str, int]:
    """Render the corpus the LLM reads. Returns `(text, pages_without_body)`.

    Carries each page's TEXT, not just its title and summary. The old version
    sent metadata only, which meant the model writing the front page had never
    read a single page it was describing -- so the best it could do was
    reorganise the list it was handed. An overview needs the prose.

    Trimming is reported, never silent: the caller logs how many pages arrived
    with no body and how much text was dropped, because a front page that
    quietly stopped seeing half the wiki would read as the wiki having gotten
    less interesting.
    """
    lines: list[str] = []
    spent = 0
    missing_body = 0
    for page in pages:
        body = page.body.strip()
        if not body:
            missing_body += 1
        excerpt = body[:_PER_PAGE_BODY_CHARS]
        if spent + len(excerpt) > _TOTAL_BODY_CHARS:
            excerpt = excerpt[: max(0, _TOTAL_BODY_CHARS - spent)]
        spent += len(excerpt)
        truncated = len(body) > len(excerpt)
        lines.append(
            f"- type: {page.wiki_type}\n"
            f"  slug: {page.slug}\n"
            f"  title: {page.title}\n"
            f"  summary: {page.summary or '(none)'}\n"
            f"  text: |\n"
            + (
                "\n".join(f"    {line}" for line in excerpt.splitlines())
                if excerpt
                else "    (no text)"
            )
            + ("\n    [...truncated]" if truncated else "")
        )
    return "\n".join(lines), missing_body


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
              AND n_from.label = 'Document'
              AND n_to.label = 'Document'
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
) -> str | None:
    """Produce the wiki index body via Gemini Pro.

    Returns the markdown body, or ``None`` when no overview could be written
    (LLM unavailable, errored, or empty). ``None`` means LEAVE THE PAGE ALONE:
    the caller keeps whatever overview is already published rather than
    replacing it, because there is no useful thing to substitute. A page list
    would be exactly the directory this page stopped being.

    ``customer_id`` is accepted for the disabled architecture-diagram path and
    is otherwise unused; see the splice block at the end.
    """
    pages = _rows_to_pages(rows)
    if not pages:
        return "# Wiki\n\nNo pages yet.\n"

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

    corpus, missing_body = _format_pages_for_prompt(pages)
    if missing_body:
        # Not fatal -- the model still has titles and summaries for those
        # pages -- but it is the difference between an overview and a
        # re-listing, so it is said out loud rather than absorbed.
        log.info(
            "index_renderer.pages_without_body",
            page_count=len(pages),
            missing_body=missing_body,
        )
    user_prompt = f"Wiki page corpus ({len(pages)} pages):\n\n{corpus}"

    # Phase-0b chunk B: the production call routes through
    # shared.llm.acompletion so the call honors LLM_GATEWAY_URL for
    # gateway-routed tenants (the central LiteLLM proxy supplies the
    # provider key). Without the gateway, LiteLLM falls back to the
    # direct provider call using GOOGLE_API_KEY. The ``client`` kwarg
    # is preserved for tests that inject a stub mimicking the google-
    # genai surface (`client.aio.models.generate_content`) — when
    # supplied we drive that legacy path verbatim so existing fixtures
    # keep working. The no-gateway-and-no-key codepath now declines to
    # write rather than substituting a page list.
    if client is not None:
        try:
            resp = await client.aio.models.generate_content(
                model=model,
                contents=user_prompt,
                config={
                    "system_instruction": _INDEX_SYSTEM_PROMPT,
                    # 4k fits a two-minute-read overview with room to
                    # spare. It was 16384 to accommodate a 50-page list
                    # with summaries; the page list is gone, and a tighter
                    # ceiling is also a cheap brake on the model drifting
                    # back into enumerating pages.
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
            return None
        text = (getattr(resp, "text", None) or "").strip()
    else:
        from engine.shared import llm as shared_llm
        from engine.shared.config import get_settings

        # Preserve the "no key + no gateway → deterministic fallback"
        # contract: the index page must always render.
        if not (
            shared_llm.gateway_url()
            or get_settings().google_api_key.get_secret_value()
        ):
            log.warning("index_renderer.no_google_api_key_falling_back")
            return None

        try:
            resp = await shared_llm.acompletion(
                model=f"gemini/{model}",
                messages=[
                    {"role": "system", "content": _INDEX_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                # 4k fits a two-minute-read overview with room to spare.
                # It was 16384 to accommodate a 50-page list with
                # summaries; the page list is gone, and a tighter ceiling
                # is also a cheap brake on the model drifting back into
                # enumerating pages.
                max_tokens=4096,
            )
        except shared_llm.LLMError as exc:
            log.warning(
                "index_renderer.gemini_failed_falling_back",
                error=str(exc),
                error_class=type(exc).__name__,
                status_code=exc.status_code,
                provider=exc.provider,
                page_count=len(pages),
            )
            return None

        try:
            text = (resp.choices[0].message.content or "").strip()
        except (AttributeError, IndexError) as exc:
            log.warning(
                "index_renderer.malformed_response_falling_back",
                error=str(exc),
                page_count=len(pages),
            )
            return None

    if not text:
        log.warning("index_renderer.empty_response_falling_back", page_count=len(pages))
        return None

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
    from kb.synthesis.diagram_renderer import splice_mermaid_block
    text = splice_mermaid_block(text, "")
    # from kb.synthesis.diagram_renderer import (
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
