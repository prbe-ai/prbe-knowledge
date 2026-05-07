"""Surgical wiki-diagram regeneration — replace just the Mermaid block.

Runs on every nightly cross-repo refresh + on demand. Unlike
``regenerate_wiki_index`` (which rewrites the whole index page —
intro paragraph, mermaid block, page list, all of it), this function
preserves every byte of the existing index body except the
``mermaid`` fenced block. Reasoning: the intro prose and page-list
organization shouldn't shift day-to-day from edge changes; only the
diagram should reflect new evidence.

Behavior matrix when called:

  - body has a ```mermaid block, new edges exist:
      → replace the block in place.
  - body has a ```mermaid block, no edges exist:
      → remove the block (drop the surrounding blank lines too).
  - body has NO ```mermaid block, new edges exist:
      → insert a block after the first paragraph (or before the
        first `##` heading, whichever comes first).
  - body has NO ```mermaid block, no edges exist:
      → no-op. Don't write a new index document at all.

The Mermaid block itself is generated deterministically from the
verified edges so we don't pay an LLM per nightly run. Future
enhancement: use the LLM to group nodes into subgraphs by inferred
purpose. Skip for now — bare alphabetical edge list is honest and
matches what the LLM would output absent any grouping signal.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from services.ingestion.handlers.base import make_default_context
from services.ingestion.handlers.wiki import (
    INDEX_SLUG,
    WIKI_PAYLOAD_KEY,
    build_normalization_result,
)
from services.ingestion.normalizer import (
    Normalizer,
    fetch_live_body_from_chunks,
)
from services.synthesis.index_renderer import (
    _RepoEdge,
    fetch_verified_repo_edges,
)
from shared.constants import DocClass, SourceSystem
from shared.db import with_tenant
from shared.embeddings import Embedder
from shared.logging import get_logger
from shared.models import NormalizationResult, WebhookEvent
from shared.storage import get_store

log = get_logger(__name__)


# Match a fenced ```mermaid block, including the surrounding blank
# lines so a removal leaves the body cleanly. Uses non-greedy `.*?`
# so adjacent blocks aren't merged.
_MERMAID_BLOCK_RE = re.compile(
    r"\n*```mermaid\s*\n.*?```\n*",
    re.DOTALL,
)


def _build_mermaid_block(edges: list[_RepoEdge]) -> str:
    """Render verified edges as a Mermaid `graph TD` fenced block.

    Bidirectional edges use a normal arrow rendered ONCE per pair.
    One-way edges use the same arrow style with a `|one-way|` label.
    Node labels strip the `owner/` prefix so the diagram is readable.
    """
    if not edges:
        return ""
    lines = ["```mermaid", "graph TD"]
    seen_nodes: set[str] = set()
    seen_pairs: set[frozenset[str]] = set()
    # Emit nodes first so isolated repos still appear if present in
    # any edge endpoint. Sort for deterministic output.
    endpoints = sorted({e.source for e in edges} | {e.target for e in edges})
    for ep in endpoints:
        node_id = _node_id(ep)
        label = ep.rsplit("/", 1)[-1]
        if node_id not in seen_nodes:
            seen_nodes.add(node_id)
            lines.append(f"  {node_id}[{label}]")
    for edge in edges:
        if edge.bidirectional:
            key = frozenset((edge.source, edge.target))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            lines.append(f"  {_node_id(edge.source)} --- {_node_id(edge.target)}")
        else:
            lines.append(
                f"  {_node_id(edge.source)} -->|one-way| {_node_id(edge.target)}"
            )
    lines.append("```")
    return "\n".join(lines)


def _node_id(repo_full_name: str) -> str:
    """Mermaid-safe node id: alphanumerics + underscores only."""
    short = repo_full_name.rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_]", "_", short)


def splice_mermaid_block(body: str, new_block: str) -> str:
    """Replace, insert, or remove the Mermaid block per the matrix in the
    module docstring. Strips ALL existing mermaid blocks first so a body
    that somehow accumulated multiple (e.g. an older LLM-emitted block
    plus a manual splice) collapses to exactly one. Idempotent: calling
    with the same `new_block` twice produces the same body the second
    time.
    """
    stripped = _MERMAID_BLOCK_RE.sub("\n\n", body)
    # Collapse any 3+ consecutive newlines that the strip may have left
    # behind so the "Architecture\n\n\n## Pages" case doesn't leave
    # gappy whitespace if multiple blocks lived back-to-back.
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)

    if not new_block:
        return stripped
    return _insert_mermaid_after_intro(stripped, new_block)


def _insert_mermaid_after_intro(body: str, new_block: str) -> str:
    """Insert a new Mermaid block between the intro paragraph and the
    first ``##`` section heading. Falls back to appending at the very
    end when no such heading exists (rare).
    """
    section_match = re.search(r"\n##\s+", body)
    if section_match is None:
        return body.rstrip() + f"\n\n{new_block}\n"
    insertion_point = section_match.start()
    return body[:insertion_point] + f"\n\n{new_block}\n" + body[insertion_point:]


async def regenerate_wiki_diagram(
    *,
    customer_id: str,
    normalizer: Normalizer | None = None,
) -> bool:
    """Surgically refresh the Mermaid block on the index page.

    Returns True when the index document was written (the body
    actually changed), False when no change was needed (no existing
    index, or new diagram identical to current).
    """
    index_doc_id = f"wiki:index:{INDEX_SLUG}"
    edges = await fetch_verified_repo_edges(customer_id)
    new_block = _build_mermaid_block(edges)

    async with with_tenant(customer_id) as conn:
        existing = await conn.fetchrow(
            """
            SELECT version, updated_at, metadata
            FROM documents
            WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL
            """,
            customer_id,
            index_doc_id,
        )
        if existing is None:
            log.info(
                "diagram_renderer.no_index",
                customer=customer_id,
                edges=len(edges),
            )
            return False
        body = await fetch_live_body_from_chunks(conn, customer_id, index_doc_id)

    if not body.strip():
        log.info(
            "diagram_renderer.empty_body",
            customer=customer_id,
            edges=len(edges),
        )
        return False

    new_body = splice_mermaid_block(body, new_block)
    if new_body == body:
        log.info(
            "diagram_renderer.no_change",
            customer=customer_id,
            edges=len(edges),
        )
        return False

    if normalizer is None:
        ctx = make_default_context()
        normalizer = Normalizer(ctx, store=get_store(), embedder=Embedder())

    received_at = datetime.now(UTC)
    raw_payload: dict[str, Any] = {
        WIKI_PAYLOAD_KEY: {
            "wiki_type": "index",
            "slug": INDEX_SLUG,
            "title": "Wiki",
            "body": new_body,
            "frontmatter": existing["metadata"].get("frontmatter", {})
            if isinstance(existing["metadata"], dict)
            else {},
            "doc_class": DocClass.AGENT_ARTIFACT.value,
            "is_delete": False,
            "updated_at": received_at.isoformat(),
            "summary": f"Architecture diagram refresh ({len(edges)} edges).",
            "commit_message": (
                f"Refresh architecture diagram ({len(edges)} edges)"
            ),
            "commit_author": "system:diagram_refresh",
            "commit_run_id": None,
            "author_id": "system:diagram_refresh",
        }
    }
    event = WebhookEvent(
        customer_id=customer_id,
        source_system=SourceSystem.WIKI,
        source_event_id=f"index:{INDEX_SLUG}:diagram:{received_at.isoformat()}",
        received_at=received_at,
        payload_s3_key="",
        payload_s3_keys=[],
        raw_payload=raw_payload,
        headers={},
    )
    norm: NormalizationResult = build_normalization_result(event)
    await normalizer._persist(customer_id, SourceSystem.WIKI, norm)
    log.info(
        "diagram_renderer.persisted",
        customer=customer_id,
        edges=len(edges),
    )
    return True


__all__ = [
    "regenerate_wiki_diagram",
    "splice_mermaid_block",
]
