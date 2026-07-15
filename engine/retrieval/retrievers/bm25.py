"""BM25-ish retriever via Postgres `ts_rank_cd`.

Postgres doesn't have true BM25 out of the box — `ts_rank_cd` (cover-density
ranking) is a reasonable stand-in and runs on the `idx_chunks_content_tsv`
GIN index over the stored `chunks.content_tsv` column (migration 0062).
The column is `GENERATED ALWAYS AS (to_tsvector('english', content)) STORED`,
so the bitmap-heap recheck and `ts_rank_cd` both read the precomputed
lexeme array off the heap instead of re-tokenizing `content` on every one
of the ~10k+ candidate rows. EXPLAIN ANALYZE on acme showed the
old expression-based path spent ~5.7s of a 5.9s query in per-row
tokenization; the materialized column reduces that to score math + heap
reads. For Phase 1 we can still swap this to pg_bm25 or a real BM25 lib
if ranking quality matters enough.

Query parsing: we OR the user's tokens via `to_tsquery` (built from a
simple word-split) instead of relying on `plainto_tsquery`'s implicit
AND. AND-strictness silently zero-matches realistic queries: "agent
session 3c325e11-2008-46a9-..." had no chunk that contained every word
(metadata chunks have "session" + the UUID prefix, transcripts have
neither), so BM25 returned zero hits. OR-of-tokens lets partial matches
contribute; `ts_rank_cd` then ranks by how many of the query's terms hit
and how densely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from engine.retrieval.temporal import build_predicate
from engine.shared.constants import TOP_K_BM25
from engine.shared.db import with_tenant
from engine.shared.models import TemporalSpec, normalize_author_id

# Pull alphanumeric/underscore runs as tokens. Hyphens split — Postgres'
# `english` parser already produces the individual hex parts of a UUID
# as separate lexemes on the index side, so splitting the query the same
# way keeps token alignment.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(slots=True)
class BM25Hit:
    chunk_id: str
    doc_id: str
    doc_version: int
    source_system: str
    source_url: str
    title: str | None
    content: str
    created_at: datetime
    updated_at: datetime
    score: float
    author_id: str | None = None
    kind: str = "content"


def _build_or_tsquery_string(query_text: str) -> str:
    """Build a `to_tsquery` input that ORs every >=2-char token in the
    user's query. Returns "" when the query has no usable tokens — caller
    skips the SQL pass.
    """
    tokens = [t for t in _TOKEN_RE.findall(query_text) if len(t) >= 2]
    if not tokens:
        return ""
    return " | ".join(tokens)


# Identifier-frame descriptor words. These accompany a stable identifier
# in queries like "agent session <uuid>", "ticket PRB-17", "pr <repo>#49"
# to tell the reader what kind of thing the id refers to. When at least
# one stable identifier is in the query they add zero topical signal and
# balloon BM25 selectivity — every claude_code transcript chunk contains
# "agent" and "session", so OR'ing them in drags 10k+ unrelated chunks
# through the heap recheck + ts_rank_cd. Stripped only when an identifier
# is present; a bare "session timeout" remains a valid topical query.
_BM25_IDENTIFIER_DESCRIPTORS: frozenset[str] = frozenset(
    {
        "agent",
        "session",
        "ticket",
        "issue",
        "pr",
        "prs",
        "pull",
        "commit",
        "sha",
    }
)


def residualize_for_bm25(
    query_text: str, identifier_canonical_ids: list[str]
) -> str | None:
    """Return the topical residual of `query_text` once identifier tokens and
    identifier-frame descriptors are stripped, or None when nothing useful
    remains.

    id_lookup pins docs by exact identifier match; for queries that consist
    entirely of "<descriptor> <identifier>" (e.g. "agent session
    3c325e11-2008-46a9-83f7-fc40d11eaf82" or "ticket PRB-17"), BM25 has no
    recall to add — every token left in the OR'd tsquery is either the
    identifier itself (id_lookup already handles it) or a high-DF
    descriptor that matches tens of thousands of unrelated chunks. Skipping
    BM25 in that case removes seconds of pure noise work without losing
    recall (vector + graph still run, id_lookup pins the doc).

    When the user adds genuine topical tokens (e.g. "<uuid> auth refactor"),
    the residual "auth refactor" is returned so BM25 still contributes —
    now selective enough to be cheap.
    """
    if not identifier_canonical_ids:
        return query_text or None

    stops: set[str] = set(_BM25_IDENTIFIER_DESCRIPTORS)
    for cid in identifier_canonical_ids:
        for tok in _TOKEN_RE.findall(cid):
            stops.add(tok.lower())

    residual = [
        tok
        for tok in _TOKEN_RE.findall(query_text)
        if len(tok) >= 2 and tok.lower() not in stops
    ]
    if not residual:
        return None
    return " ".join(residual)


async def bm25_search(
    customer_id: str,
    query_text: str,
    top_k: int = TOP_K_BM25,
    sources: list[str] | None = None,
    doc_types: list[str] | None = None,
    temporal: TemporalSpec | None = None,
    include_drafts: bool = False,
    author_ids: list[str] | None = None,
    sort_by: Literal["relevance", "recency"] = "relevance",
    source_keys: list[str] | None = None,
) -> list[BM25Hit]:
    """`include_drafts` defaults to False — retrieval hides ``visibility='draft'``
    rows (see migration 0082 + Plan A Component 6). Reviewer surfaces pass
    True after role-checking; API-key callers cannot bypass.

    `author_ids`, when set, hard-filters by
    `documents.author_id = ANY(...)`. Mirrors `sql_list`'s author filter.

    `sort_by="recency"` swaps `ORDER BY ts_rank_cd DESC` for
    `ORDER BY d.updated_at DESC`. The `content_tsv @@ to_tsquery` filter
    still narrows the pool to query-token matches; only the final ordering
    flips. Used by the gatherer when the extractor flagged temporal intent.

    `source_keys`, when set, hard-filters by
    `documents.metadata->>'source_key' = ANY(...)` (custom-ingest scope
    key). Applied BEFORE the LIMIT -- mirrors vector_search.
    """
    spec = temporal or TemporalSpec()
    or_query = _build_or_tsquery_string(query_text)
    if not or_query:
        return []

    async with with_tenant(customer_id) as conn:
        params: list = [customer_id, or_query, top_k]
        source_filter = ""
        if sources:
            params.append(sources)
            source_filter = f"AND d.source_system = ANY(${len(params)}::text[])"

        doc_type_filter = ""
        if doc_types:
            params.append(doc_types)
            doc_type_filter = f"AND d.doc_type = ANY(${len(params)}::text[])"

        author_filter = ""
        if author_ids:
            params.append(author_ids)
            author_filter = f"AND d.author_id = ANY(${len(params)}::text[])"

        source_key_filter = ""
        if source_keys:
            params.append(source_keys)
            source_key_filter = (
                f"AND d.metadata->>'source_key' = ANY(${len(params)}::text[])"
            )

        pred = build_predicate(
            spec, doc_alias="d", chunk_alias="c", next_param_index=len(params) + 1
        )
        params.extend(pred.params)

        # Hide drafts unless the reviewer surface explicitly opts in.
        visibility_filter = (
            ""
            if include_drafts
            else "AND c.visibility = 'approved' AND d.visibility = 'approved'"
        )

        order_by_sql = (
            "d.updated_at DESC, c.chunk_id"
            if sort_by == "recency"
            else "score DESC, c.chunk_id"
        )

        rows = await conn.fetch(
            f"""
            SELECT c.chunk_id,
                   c.doc_id,
                   d.version AS doc_version,
                   d.source_system,
                   d.source_url,
                   d.title,
                   d.author_id,
                   c.content,
                   c.kind,
                   d.created_at,
                   d.updated_at,
                   ts_rank_cd(c.content_tsv,
                              to_tsquery('english', $2)) AS score
            FROM chunks c
            JOIN documents d
              ON c.doc_id = d.doc_id
             AND d.customer_id = c.customer_id
             AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
            WHERE c.customer_id = $1
              AND c.content_tsv @@ to_tsquery('english', $2)
              {pred.chunk_sql}
              {pred.doc_sql}
              {source_filter}
              {doc_type_filter}
              {visibility_filter}
              {author_filter}
              {source_key_filter}
            ORDER BY {order_by_sql}
            LIMIT $3
            """,
            *params,
        )

    return [
        BM25Hit(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            doc_version=r["doc_version"],
            source_system=r["source_system"],
            source_url=r["source_url"],
            title=r["title"],
            content=r["content"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            score=float(r["score"]),
            author_id=normalize_author_id(r["author_id"]),
            kind=r["kind"],
        )
        for r in rows
    ]
