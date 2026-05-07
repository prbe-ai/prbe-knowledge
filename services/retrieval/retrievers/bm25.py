"""BM25-ish retriever via Postgres `ts_rank_cd`.

Postgres doesn't have true BM25 out of the box — `ts_rank_cd` (cover-density
ranking) is a reasonable stand-in and runs on the `idx_chunks_fts_content`
GIN index we built in schema.sql. For Phase 1 we can swap this to pg_bm25
or a real BM25 lib if ranking quality matters enough.

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

from services.retrieval.temporal import build_predicate
from shared.constants import TOP_K_BM25
from shared.db import with_tenant
from shared.models import TemporalSpec, normalize_author_id

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


async def bm25_search(
    customer_id: str,
    query_text: str,
    top_k: int = TOP_K_BM25,
    sources: list[str] | None = None,
    doc_types: list[str] | None = None,
    temporal: TemporalSpec | None = None,
) -> list[BM25Hit]:
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

        pred = build_predicate(
            spec, doc_alias="d", chunk_alias="c", next_param_index=len(params) + 1
        )
        params.extend(pred.params)

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
                   ts_rank_cd(to_tsvector('english', c.content),
                              to_tsquery('english', $2)) AS score
            FROM chunks c
            JOIN documents d
              ON c.doc_id = d.doc_id
             AND d.customer_id = c.customer_id
             AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
            WHERE c.customer_id = $1
              AND to_tsvector('english', c.content) @@ to_tsquery('english', $2)
              {pred.chunk_sql}
              {pred.doc_sql}
              {source_filter}
              {doc_type_filter}
            ORDER BY score DESC
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
