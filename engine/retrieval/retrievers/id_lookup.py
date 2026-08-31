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

from engine.retrieval.helpers import source_key_predicate
from engine.retrieval.temporal import build_predicate
from engine.shared.db import with_tenant
from engine.shared.identifiers import DetectedIdentifier
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
    # Which detected identifier this doc matched -- the id-pins lane needs
    # the mapping (one pin slot per identifier, best doc each), and a flat
    # hit list cannot carry it (outside-voice F4).
    matched_canonical_id: str = ""


async def id_lookup_search(
    customer_id: str,
    canonical_ids: list[str],
    temporal: TemporalSpec | None = None,
    include_drafts: bool = False,
    sources: list[str] | None = None,
    doc_types: list[str] | None = None,
    author_ids: list[str] | None = None,
    source_keys: list[str] | None = None,
    source_keys_include_keyless: bool = False,
) -> list[IdLookupHit]:
    """Return one content chunk per doc whose source_id/doc_id/source_url
    matches any of `canonical_ids`, carrying WHICH id matched.

    Match shape (per id, via a lateral unnest so the mapping survives):
      - `documents.source_id = <id>` -- direct hit on the ingested
        identifier (indexed equality).
      - `documents.source_id LIKE '%:<id>'` / `doc_id LIKE '%:<id>'` --
        handlers that prefix a kind into the stored id still match a bare
        one.
      - `documents.source_url LIKE` path-segment forms -- Linear stores
        tickets keyed by an internal UUID; the human handle (`PRB-17`)
        appears only in the URL. Anchored on `/<id>` with a segment
        terminator so `/PRB-170/` cannot match `/PRB-17`. This arm is a
        tenant-scoped scan (no usable index under FORCE RLS -- the same
        leakproof wall every trgm consumer here hits), which is WHY the
        id-pins lane calls this exactly ONCE per request, never per
        sub-query. Measured cost class: one bounded documents scan.

    Scope filters (sources / doc_types / author_ids / source_keys /
    temporal / visibility) are enforced IN THIS QUERY, not downstream:
    a pinned doc bypassing the workspace lens would be a scope leak with
    a certainty label on it (outside-voice F3).

    Returns rows ordered so the caller's per-id best-doc pick is
    deterministic: matched id, then updated_at DESC, then doc_id.
    """
    ids = [c for c in canonical_ids if is_lookup_candidate(c)]
    if not ids:
        return []

    spec = temporal or TemporalSpec()

    async with with_tenant(customer_id) as conn:
        params: list[Any] = [customer_id, ids]

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
        source_key_filter = source_key_predicate(
            params, source_keys, alias="d",
            include_keyless=source_keys_include_keyless,
        )
        pred = build_predicate(
            spec, doc_alias="d", chunk_alias="c", next_param_index=len(params) + 1
        )
        params.extend(pred.params)

        visibility_filter = (
            ""
            if include_drafts
            else "AND c.visibility = 'approved' AND d.visibility = 'approved'"
        )

        rows = await conn.fetch(
            f"""
            SELECT DISTINCT ON (c.doc_id)
                   m.cid AS matched_canonical_id,
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
            FROM documents d
            JOIN unnest($2::text[]) AS m(cid)
              ON (
                d.source_id = m.cid
                OR d.source_id LIKE '%:' || m.cid
                OR d.doc_id LIKE '%:' || m.cid
                OR d.source_url LIKE '%/' || m.cid || '/%'
                OR d.source_url LIKE '%/' || m.cid
                OR d.source_url LIKE '%/' || m.cid || '?%'
                OR d.source_url LIKE '%/' || m.cid || '#%'
              )
            JOIN chunks c
              ON c.doc_id = d.doc_id
             AND c.customer_id = d.customer_id
             AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
            WHERE d.customer_id = $1
              AND COALESCE(c.kind, 'content') = 'content'
              {source_filter}
              {doc_type_filter}
              {author_filter}
              {source_key_filter}
              {pred.chunk_sql}
              {pred.doc_sql}
              {visibility_filter}
            ORDER BY c.doc_id, c.chunk_index ASC
            """,
            *params,
        )

    hits = [
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
            matched_canonical_id=r["matched_canonical_id"],
        )
        for r in rows
    ]
    # Deterministic order for per-id best-doc selection downstream.
    hits.sort(key=lambda h: (h.matched_canonical_id, _neg_ts(h.updated_at), h.doc_id))
    return hits


def _neg_ts(ts: datetime) -> float:
    """Sort key helper: newest first without reverse-sorting the whole tuple."""
    return -ts.timestamp()


def resolve_pins(
    detected: list[DetectedIdentifier],
    hits: list[IdLookupHit],
    top_k: int,
) -> tuple[list[IdLookupHit], set[str]]:
    """Best doc per detected identifier, capped so ranking keeps room.

    Returns (pins, unresolved_canonical_ids). One pin slot per identifier
    -- the BEST matching doc by the deterministic order id_lookup_search
    established -- deduped by doc_id (two ids resolving to the same doc
    pin it once), capped at max(1, top_k // 2) so a query quoting ten ids
    cannot return a phone book, and safe at top_k=1 (outside-voice F4).
    """
    by_cid: dict[str, IdLookupHit] = {}
    for h in hits:
        by_cid.setdefault(h.matched_canonical_id, h)
    pins: list[IdLookupHit] = []
    seen_docs: set[str] = set()
    unresolved: set[str] = set()
    cap = max(1, top_k // 2)
    for d in detected:
        h = by_cid.get(d.canonical_id)
        if h is None:
            unresolved.add(d.canonical_id)
            continue
        if h.doc_id in seen_docs:
            continue
        if len(pins) < cap:
            seen_docs.add(h.doc_id)
            pins.append(h)
    return pins, unresolved
