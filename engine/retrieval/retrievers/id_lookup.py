"""Exact-id retriever — pins docs whose `source_id`/`doc_id` matches a
router-extracted canonical_id.

Vector and BM25 both fail on UUID-precise queries: embeddings of random
hex are noise (every session metadata chunk lands at ~0.50 cosine), and
`plainto_tsquery` ANDs hyphenated UUID parts plus the surrounding query
words, so a query like "agent session 3c325e11-2008-46a9-..." misses the
metadata chunk that has only "session 3c325e11" in the title.

This retriever sidesteps relevance entirely: when the router extracts a
high-confidence entity whose canonical_id looks like a stable identifier
(UUID, ticket code, PR ref, etc.), look up matching docs by exact equality
on `documents.source_id` (uses idx_documents_customer_source) plus a
suffix match on `doc_id` so docs whose source_id was prefixed at ingest
(e.g. `claude_code:<customer>:<uuid>`) still match the bare canonical_id.

Returned hits enter fusion at rank 1 with a flat unit score; RRF then
dominates the ranking for the matched docs without disturbing relevance
ordering of unrelated candidates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from engine.retrieval.temporal import build_predicate
from engine.shared.db import with_tenant
from engine.shared.models import TemporalSpec, normalize_author_id

# A canonical_id qualifies for exact-id lookup when it looks like a stable
# identifier: a UUID, a ticket code (LETTERS-DIGITS), a #-prefixed issue/PR
# number, or a long alphanumeric token. Plain words like "auth" or
# "prbe-backend" are intentionally rejected — they belong to vector/BM25/graph.
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_TICKET_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}-\d{1,6}$")
_HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{12,40}$")
_ISSUE_REF_RE = re.compile(r"^[a-zA-Z0-9_./-]+#\d{1,6}$")


def is_lookup_candidate(canonical_id: str) -> bool:
    """Return True when `canonical_id` should drive an exact-id lookup.

    Conservative on purpose: false positives here run an extra SQL pass
    that almost never matches; false negatives demote a precise query
    back to vector/BM25 noise.
    """
    if not canonical_id:
        return False
    if _UUID_RE.match(canonical_id):
        return True
    if _TICKET_RE.match(canonical_id):
        return True
    if _ISSUE_REF_RE.match(canonical_id):
        return True
    return bool(_HASH_PREFIX_RE.match(canonical_id))


@dataclass(slots=True)
class IdLookupHit:
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


async def id_lookup_search(
    customer_id: str,
    canonical_ids: list[str],
    temporal: TemporalSpec | None = None,
    include_drafts: bool = False,
) -> list[IdLookupHit]:
    """Return one content chunk per doc whose source_id/doc_id/source_url
    matches any of `canonical_ids`.

    Match shape:
      - `documents.source_id = ANY($canonical_ids)` — direct hit on the
        ingested identifier (handler-supplied; e.g. session UUID for
        claude_code, `issue:<uuid>` for Linear when the router already
        knows the prefix).
      - `documents.source_id LIKE '%:<canonical_id>'` — handlers that
        encode a kind prefix in source_id (per memory
        `feedback_documents_source_id_format.md`) still match when the
        router only emits the bare UUID.
      - `documents.doc_id LIKE '%:<canonical_id>'` — fallback for docs
        whose doc_id terminator equals the canonical_id (covers GitHub
        PR refs, Linear ticket codes after coalescing, etc.).
      - `documents.source_url LIKE '%/<canonical_id>...'` — Linear stores
        tickets keyed by an internal UUID (source_id = `issue:<uuid>`,
        doc_id ends in `:<uuid>`) but the URL carries the human handle
        (`/issue/PRB-17/...`). Patterns anchor on a path separator so
        `/PRB-17/` matches but `/PRB-170/` does not. Until we backfill an
        identifier alias for tickets, URL match is the only signal that
        connects extractor-emitted `PRB-17` to the issue's doc row.

    Temporal applies the same predicate as the other retrievers so a
    historical-version lookup goes against the right SCD2 row. Returns one
    chunk per doc (chunk_index ASC) — fusion only needs anchor signal.
    """
    ids = [c for c in canonical_ids if is_lookup_candidate(c)]
    if not ids:
        return []

    spec = temporal or TemporalSpec()
    suffixes = [f"%:{c}" for c in ids]
    # URL path-segment patterns. The four variants cover the boundary the
    # ticket code can sit against in a real URL:
    #   /PRB-17/  in the middle of the path
    #   /PRB-17   at the very end (no trailing slash)
    #   /PRB-17?  immediately before a query string
    #   /PRB-17#  immediately before a fragment
    # `%PRB-17%` would over-match (`/PRB-170/`, `?prb-17-attached`); the
    # leading `/` plus a terminator on the trailing side keeps matches
    # to whole path segments.
    url_patterns: list[str] = []
    for c in ids:
        url_patterns.append(f"%/{c}/%")
        url_patterns.append(f"%/{c}")
        url_patterns.append(f"%/{c}?%")
        url_patterns.append(f"%/{c}#%")

    async with with_tenant(customer_id) as conn:
        params: list[Any] = [customer_id, ids, suffixes, url_patterns]
        pred = build_predicate(
            spec, doc_alias="d", chunk_alias="c", next_param_index=len(params) + 1
        )
        params.extend(pred.params)

        # Default branch hides drafts (Plan A Component 6); reviewer
        # surfaces pass include_drafts=True to bypass.
        visibility_filter = (
            ""
            if include_drafts
            else "AND c.visibility = 'approved' AND d.visibility = 'approved'"
        )

        rows = await conn.fetch(
            f"""
            SELECT DISTINCT ON (c.doc_id)
                   c.chunk_id,
                   c.doc_id,
                   d.version AS doc_version,
                   d.source_system,
                   d.source_url,
                   d.title,
                   d.author_id,
                   c.content,
                   d.created_at,
                   d.updated_at
            FROM chunks c
            JOIN documents d
              ON c.doc_id = d.doc_id
             AND d.customer_id = c.customer_id
             AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
            WHERE c.customer_id = $1
              AND COALESCE(c.kind, 'content') = 'content'
              AND (
                d.source_id = ANY($2::text[])
                OR d.source_id LIKE ANY($3::text[])
                OR d.doc_id LIKE ANY($3::text[])
                OR d.source_url LIKE ANY($4::text[])
              )
              {pred.chunk_sql}
              {pred.doc_sql}
              {visibility_filter}
            ORDER BY c.doc_id, c.chunk_index ASC
            """,
            *params,
        )

    return [
        IdLookupHit(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            doc_version=r["doc_version"],
            source_system=r["source_system"],
            source_url=r["source_url"],
            title=r["title"],
            content=r["content"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            score=1.0,
            author_id=normalize_author_id(r["author_id"]),
        )
        for r in rows
    ]
