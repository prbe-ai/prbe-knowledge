"""Identifier retriever — resolves typed identifiers to documents, for
the id-pins lane.

Two lanes, split by what the user typed (identifiers.EXACT_KINDS vs
INFERRED_KINDS): EXACT lookups (`id_lookup_search`) match a whole typed
identifier by equality/suffix — whatever matches is right. INFERRED
lookups (`_prefix_lookup`, `_number_ref_lookup`) EXPAND a partial
reference (a short sha, a bare '#N') and may pin only when the expansion
is unique; anything else is reported ambiguous, never guessed.

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

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from engine.retrieval.helpers import source_key_predicate
from engine.retrieval.temporal import build_predicate
from engine.shared.constants import SourceSystem
from engine.shared.db import with_tenant
from engine.shared.identifiers import EXACT_KINDS, DetectedIdentifier
from engine.shared.models import TemporalSpec, normalize_author_id

# A canonical_id qualifies for exact-id lookup when it looks like a stable
# identifier: a UUID, a ticket code (LETTERS-DIGITS), a #-prefixed issue/PR
# number, or a long alphanumeric token. Plain words like "auth" or
# "prbe-backend" are intentionally rejected — they belong to vector/BM25/graph.
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_TICKET_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}-\d{1,6}$")
_HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{12,40}$")
_ISSUE_REF_RE = re.compile(r"^[a-zA-Z0-9_./-]+#\d{1,6}$")
_PD_ID_RE = re.compile(r"^Q[A-Z0-9]{12,15}$")


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
    if _PD_ID_RE.match(canonical_id):
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
    # INFERRED resolutions (prefix / number ref) say HOW they resolved, so
    # consumers never present an expansion as an exact match the user typed
    # (review: cross-file). Empty for exact hits.
    resolution_note: str = ""


def _append_scope_filters(
    params: list[Any],
    *,
    spec: TemporalSpec,
    include_drafts: bool,
    sources: list[str] | None,
    doc_types: list[str] | None,
    author_ids: list[str] | None,
    source_keys: list[str] | None,
    source_keys_include_keyless: bool,
    doc_only: bool = False,
) -> str:
    """The one scope gate every lookup in this module shares.

    Appends parameters to `params` and returns the `AND ...` SQL tail
    (source/doc_type/author/source_key/temporal/visibility). ONE builder on
    purpose: several lookup queries each re-implementing the workspace lens
    is several chances for a pinned doc to leak past it.

    `doc_only=True` omits the chunk-alias fragments (chunk temporal +
    chunk visibility) for the inferred lookups' documents-only phase 1;
    their phase 2 chunk fetch applies the full gate, so a doc whose chunks
    are all filtered still never pins.
    """
    out: list[str] = []
    if sources:
        params.append(sources)
        out.append(f"AND d.source_system = ANY(${len(params)}::text[])")
    if doc_types:
        params.append(doc_types)
        out.append(f"AND d.doc_type = ANY(${len(params)}::text[])")
    if author_ids:
        params.append(author_ids)
        out.append(f"AND d.author_id = ANY(${len(params)}::text[])")
    out.append(
        source_key_predicate(
            params, source_keys, alias="d",
            include_keyless=source_keys_include_keyless,
        )
    )
    pred = build_predicate(
        spec, doc_alias="d", chunk_alias="c", next_param_index=len(params) + 1
    )
    params.extend(pred.params)
    if not doc_only:
        out.append(pred.chunk_sql)
    out.append(pred.doc_sql)
    if not include_drafts:
        if doc_only:
            out.append("AND d.visibility = 'approved'")
        else:
            out.append("AND c.visibility = 'approved' AND d.visibility = 'approved'")
    return "\n              ".join(x for x in out if x)


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

    One row per (matched id, doc) pair — DISTINCT ON (m.cid, c.doc_id),
    NOT plain doc_id: a doc matching two typed ids must count as resolving
    BOTH, or the loser is falsely reported unresolved and the mapping goes
    nondeterministic (review F4). Rows come back ordered so the caller's
    per-id best-doc pick is deterministic: matched id, then updated_at
    DESC, then doc_id.
    """
    ids = [c for c in canonical_ids if is_lookup_candidate(c)]
    if not ids:
        return []

    # URL arms only make sense for ticket-shaped ids: Linear keys the doc by
    # an internal UUID and puts the human handle (PRB-17) in the URL path.
    # UUIDs and shas resolve via the indexed source_id/doc_id arms; running
    # the leading-wildcard URL LIKEs for them turns an indexable lookup into
    # a tenant-wide documents scan (review F10). Derived here, not threaded:
    # detect_identifiers canonicalizes tickets to uppercase, so the anchored
    # ticket regex is an exact kind test on the canonical form.
    url_flags = [bool(_TICKET_RE.match(c)) for c in ids]

    spec = temporal or TemporalSpec()

    async with with_tenant(customer_id) as conn:
        params: list[Any] = [customer_id, ids, url_flags]
        scope_sql = _append_scope_filters(
            params,
            spec=spec,
            include_drafts=include_drafts,
            sources=sources,
            doc_types=doc_types,
            author_ids=author_ids,
            source_keys=source_keys,
            source_keys_include_keyless=source_keys_include_keyless,
        )

        rows = await conn.fetch(
            f"""
            SELECT DISTINCT ON (m.cid, c.doc_id)
                   m.cid AS matched_canonical_id,
                   {_HIT_COLUMNS}
            FROM documents d
            JOIN unnest($2::text[], $3::bool[]) AS m(cid, url_ok)
              ON (
                d.source_id = m.cid
                OR d.source_id LIKE '%:' || m.cid
                OR d.doc_id LIKE '%:' || m.cid
                OR (m.url_ok AND (
                    d.source_url LIKE '%/' || m.cid || '/%'
                    OR d.source_url LIKE '%/' || m.cid
                    OR d.source_url LIKE '%/' || m.cid || '?%'
                    OR d.source_url LIKE '%/' || m.cid || '#%'
                ))
              )
            {_CHUNK_JOIN}
            WHERE d.customer_id = $1
              AND COALESCE(c.kind, 'content') = 'content'
              {scope_sql}
            ORDER BY m.cid, c.doc_id, c.chunk_index ASC
            """,
            *params,
        )

    hits = [_row_to_hit(r, r["matched_canonical_id"]) for r in rows]
    # Deterministic order for per-id best-doc selection downstream.
    hits.sort(key=lambda h: (h.matched_canonical_id, _neg_ts(h.updated_at), h.doc_id))
    return hits



_HIT_COLUMNS = """
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
"""

_CHUNK_JOIN = """
            JOIN chunks c
              ON c.doc_id = d.doc_id
             AND c.customer_id = d.customer_id
             AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
"""


def _row_to_hit(r: Any, canonical_id: str, note: str = "") -> IdLookupHit:
    return IdLookupHit(
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
        matched_canonical_id=canonical_id,
        resolution_note=note,
    )


# SQL-safety gate, NOT a kind re-derivation: prefixes are concatenated into
# LIKE patterns, so they must be provably free of LIKE metacharacters. The
# kind routing already guarantees hex; this makes the guarantee local.
_HEX_ONLY_RE = re.compile(r"^[0-9a-f]{7,11}$")

# Phase-1 row cap for the inferred lookups. Hitting it means the match set
# was pathological; everything in that fetch is then treated as AMBIGUOUS
# (fail-CLOSED) — a truncated competitor set must never turn "ambiguous"
# into "unique" (review: efficiency).
_INFERRED_MATCH_CAP = 500


async def _fetch_first_chunks(
    customer_id: str,
    doc_ids: list[str],
    *,
    temporal: TemporalSpec | None,
    include_drafts: bool,
    sources: list[str] | None,
    doc_types: list[str] | None,
    author_ids: list[str] | None,
    source_keys: list[str] | None,
    source_keys_include_keyless: bool,
) -> dict[str, Any]:
    """First content chunk per doc, full scope enforced — phase 2 of the
    inferred lookups. Docs whose chunks are all filtered out simply drop
    (no pin), which the caller reports as unresolved."""
    if not doc_ids:
        return {}
    async with with_tenant(customer_id) as conn:
        params: list[Any] = [customer_id, doc_ids]
        scope_sql = _append_scope_filters(
            params,
            spec=temporal or TemporalSpec(),
            include_drafts=include_drafts,
            sources=sources,
            doc_types=doc_types,
            author_ids=author_ids,
            source_keys=source_keys,
            source_keys_include_keyless=source_keys_include_keyless,
        )
        rows = await conn.fetch(
            f"""
            SELECT DISTINCT ON (c.doc_id)
                   {_HIT_COLUMNS}
            FROM documents d
            {_CHUNK_JOIN}
            WHERE d.customer_id = $1
              AND d.doc_id = ANY($2::text[])
              AND COALESCE(c.kind, 'content') = 'content'
              {scope_sql}
            ORDER BY c.doc_id, c.chunk_index ASC
            """,
            *params,
        )
    return {r["doc_id"]: r for r in rows}


async def _prefix_lookup(
    customer_id: str,
    prefixes: list[str],
    *,
    temporal: TemporalSpec | None = None,
    include_drafts: bool = False,
    sources: list[str] | None = None,
    doc_types: list[str] | None = None,
    author_ids: list[str] | None = None,
    source_keys: list[str] | None = None,
    source_keys_include_keyless: bool = False,
) -> tuple[list[IdLookupHit], set[str]]:
    """Resolve bare hex prefixes (short sha / uuid first-segment) to full
    stored identifiers. Returns (hits, ambiguous_prefixes).

    Git's short-hash rule, enforced on the IDENTIFIER, not the doc count:
    the match arms are ANCHORED at the start of a stored identifier
    (start-of-string, after '@' in `owner/repo@<sha>`, after ':' in
    `issue:<uuid>`) — never a bare substring, which would match mid-sha —
    and matches group by the FULL identifier the prefix expanded to
    (extracted in SQL). One distinct identifier resolves, even when several
    docs carry it; two distinct identifiers -> ambiguous, no pin, reported.
    A guessed expansion pinned with a certainty label is the one outcome
    this module exists to prevent.

    Uniqueness is judged within the request's scope (sources/doc_types/
    source_keys/visibility): "unique among the documents this caller may
    see" is the deliberate contract — the workspace lens defines the
    caller's world, exactly as git resolves within one repository.

    Two phases so the cost matches the answer: phase 1 scans DOCUMENTS
    only (no chunk join, no content) to find and group matches; phase 2
    fetches one content chunk for the few unique winners. Leading-wildcard
    arms are the same bounded tenant-scan cost class as the ticket URL
    arms, and run once per request.
    """
    clean = [c for c in prefixes if _HEX_ONLY_RE.match(c)]
    if not clean:
        return [], set()

    async with with_tenant(customer_id) as conn:
        params: list[Any] = [customer_id, clean]
        scope_sql = _append_scope_filters(
            params,
            spec=temporal or TemporalSpec(),
            include_drafts=include_drafts,
            sources=sources,
            doc_types=doc_types,
            author_ids=author_ids,
            source_keys=source_keys,
            source_keys_include_keyless=source_keys_include_keyless,
            doc_only=True,
        )
        params.append(_INFERRED_MATCH_CAP + 1)
        rows = await conn.fetch(
            f"""
            SELECT DISTINCT
                   m.cid AS matched_cid,
                   d.doc_id,
                   d.source_id,
                   d.updated_at,
                   lower(substring(
                       d.source_id from '(?:^|[@:])(' || m.cid || '[0-9a-fA-F-]*)'
                   )) AS resolved_id
            FROM documents d
            JOIN unnest($2::text[]) AS m(cid)
              ON (
                d.source_id LIKE m.cid || '%'
                OR d.source_id LIKE '%@' || m.cid || '%'
                OR d.source_id LIKE '%:' || m.cid || '%'
              )
            WHERE d.customer_id = $1
              {scope_sql}
            LIMIT ${{}}
            """.replace("${}", f"${len(params)}"),
            *params,
        )

    capped = len(rows) > _INFERRED_MATCH_CAP
    by_cid: dict[str, list[Any]] = {}
    for r in rows:
        by_cid.setdefault(r["matched_cid"], []).append(r)

    ambiguous: set[str] = set()
    winners: dict[str, str] = {}  # cid -> doc_id
    for cid, group in by_cid.items():
        if capped:
            ambiguous.add(cid)
            continue
        resolved = {r["resolved_id"] or r["doc_id"] for r in group}
        if len(resolved) > 1:
            ambiguous.add(cid)
            continue
        # One identifier, possibly several docs carrying it (a parent doc
        # plus derived children): the primary doc is the one whose
        # source_id IS the identifier (or the shortest doc_id), newest
        # first as the tiebreak.
        best = min(
            group,
            key=lambda r: (
                0 if (r["resolved_id"] or "") == r["source_id"].lower() else 1,
                len(r["doc_id"]),
                _neg_ts(r["updated_at"]),
                r["doc_id"],
            ),
        )
        winners[cid] = best["doc_id"]

    chunk_rows = await _fetch_first_chunks(
        customer_id,
        sorted(set(winners.values())),
        temporal=temporal,
        include_drafts=include_drafts,
        sources=sources,
        doc_types=doc_types,
        author_ids=author_ids,
        source_keys=source_keys,
        source_keys_include_keyless=source_keys_include_keyless,
    )
    hits = [
        _row_to_hit(
            chunk_rows[doc_id],
            cid,
            note=f"Uniquely resolved partial identifier: {cid}",
        )
        for cid, doc_id in winners.items()
        if doc_id in chunk_rows
    ]
    return hits, ambiguous


async def _number_ref_lookup(
    customer_id: str,
    refs: list[DetectedIdentifier],
    *,
    temporal: TemporalSpec | None = None,
    include_drafts: bool = False,
    sources: list[str] | None = None,
    doc_types: list[str] | None = None,
    author_ids: list[str] | None = None,
    source_keys: list[str] | None = None,
    source_keys_include_keyless: bool = False,
) -> tuple[list[IdLookupHit], set[str]]:
    """Resolve PR/issue number refs ('#383', 'research-os#539') to a repo's
    PR, issue, or squash-merge commit doc. Returns (hits, ambiguous_refs).

    The number is matched in SQL; the repo is resolved in PYTHON, because
    the qualifier is a SOFT filter: 'the PR #232' carries qualifier 'the',
    and demanding a repo literally named 'the' would lose a resolvable
    number. Rules, in order, per ref:

      1. Group matching docs by repo (`split_part(doc_id, ':', 2)`, guarded
         to github-shaped doc_ids so a non-github doc can never mint a
         phantom repo group).
      2. Qualifier matching exactly one group (full name or tail) resolves
         there; matching two -> ambiguous.
      3. Otherwise: exactly one group total resolves; two or more ->
         ambiguous (picking a repo would be a guess).

    Within the resolved repo, prefer the PR/issue doc over the commit doc
    whose title carries '(#N)', newest first. Numbers are digits-only by
    regex, so LIKE injection is impossible; qualifiers never reach SQL.
    Uniqueness is scope-relative, same contract as _prefix_lookup. Two
    phases, same cost shape as _prefix_lookup.
    """
    parsed = [r for r in refs if r.number.isdigit()]
    if not parsed:
        return [], set()
    nums = sorted({r.number for r in parsed})

    async with with_tenant(customer_id) as conn:
        params: list[Any] = [customer_id, nums, SourceSystem.GITHUB.value]
        scope_sql = _append_scope_filters(
            params,
            spec=temporal or TemporalSpec(),
            include_drafts=include_drafts,
            sources=sources,
            doc_types=doc_types,
            author_ids=author_ids,
            source_keys=source_keys,
            source_keys_include_keyless=source_keys_include_keyless,
            doc_only=True,
        )
        params.append(_INFERRED_MATCH_CAP + 1)
        rows = await conn.fetch(
            f"""
            SELECT DISTINCT
                   m.num AS matched_num,
                   d.doc_id,
                   d.updated_at,
                   split_part(d.doc_id, ':', 2) AS repo_group,
                   CASE
                     WHEN d.doc_id LIKE '%:pr:' || m.num THEN 0
                     WHEN d.source_id LIKE '%#' || m.num THEN 0
                     WHEN d.doc_id LIKE '%:issue:' || m.num THEN 1
                     ELSE 2
                   END AS pref
            FROM documents d
            JOIN unnest($2::text[]) AS m(num)
              ON (
                d.source_id LIKE '%#' || m.num
                OR d.doc_id LIKE '%:pr:' || m.num
                OR d.doc_id LIKE '%:issue:' || m.num
                OR d.title LIKE '%(#' || m.num || ')%'
              )
            WHERE d.customer_id = $1
              AND d.source_system = $3
              AND d.doc_id LIKE $3 || ':%'
              {scope_sql}
            LIMIT ${{}}
            """.replace("${}", f"${len(params)}"),
            *params,
        )

    capped = len(rows) > _INFERRED_MATCH_CAP
    by_num: dict[str, list[Any]] = {}
    for r in rows:
        by_num.setdefault(r["matched_num"], []).append(r)

    ambiguous: set[str] = set()
    winners: dict[str, tuple[str, str]] = {}  # canonical -> (doc_id, note)
    for ref in parsed:
        group_rows = by_num.get(ref.number)
        if not group_rows:
            continue  # plain unresolved — no signal either way
        if capped:
            ambiguous.add(ref.canonical_id)
            continue
        groups: dict[str, list[Any]] = {}
        for r in group_rows:
            groups.setdefault(r["repo_group"], []).append(r)
        chosen_repo: str | None = None
        if ref.qualifier:
            q = ref.qualifier.lower()
            matched = [
                g for g in groups
                if g.lower() == q or g.lower().endswith("/" + q)
            ]
            if len(matched) == 1:
                chosen_repo = matched[0]
            elif len(matched) > 1:
                ambiguous.add(ref.canonical_id)
                continue
            # 0 matched: junk qualifier — fall through to the bare rule.
        if chosen_repo is None:
            if len(groups) == 1:
                chosen_repo = next(iter(groups))
            else:
                ambiguous.add(ref.canonical_id)
                continue
        best = min(
            groups[chosen_repo],
            key=lambda r: (r["pref"], _neg_ts(r["updated_at"]), r["doc_id"]),
        )
        winners[ref.canonical_id] = (
            best["doc_id"],
            f"Resolved reference #{ref.number} to {chosen_repo}#{ref.number}",
        )

    chunk_rows = await _fetch_first_chunks(
        customer_id,
        sorted({doc_id for doc_id, _ in winners.values()}),
        temporal=temporal,
        include_drafts=include_drafts,
        sources=sources,
        doc_types=doc_types,
        author_ids=author_ids,
        source_keys=source_keys,
        source_keys_include_keyless=source_keys_include_keyless,
    )
    hits = [
        _row_to_hit(chunk_rows[doc_id], canonical, note=note)
        for canonical, (doc_id, note) in winners.items()
        if doc_id in chunk_rows
    ]
    return hits, ambiguous


async def lookup_identifiers(
    customer_id: str,
    detected: list[DetectedIdentifier],
    temporal: TemporalSpec | None = None,
    include_drafts: bool = False,
    sources: list[str] | None = None,
    doc_types: list[str] | None = None,
    author_ids: list[str] | None = None,
    source_keys: list[str] | None = None,
    source_keys_include_keyless: bool = False,
) -> tuple[list[IdLookupHit], set[str]]:
    """Route detected identifiers to the lookup their kind requires.

    EXACT kinds (identifiers.EXACT_KINDS) go through `id_lookup_search`
    unchanged — the user typed the whole id, so whatever the equality/
    suffix arms match is right. INFERRED kinds (hex_prefix / number_ref)
    go through the expansion lookups above, which pin only on a unique
    resolution and report the ambiguous rest. An issue_ref additionally
    rides the number lane: 'research-os#539' stores as
    'acme/research-os#539', so equality alone cannot resolve a repo typed
    without its owner — the repo-tail rules there can.

    The three lookups are independent (own pooled connections) and run
    concurrently. Returns (hits, ambiguous_canonical_ids), hits in the
    deterministic order resolve_pins documents; exact hits come first, so
    a canonical resolved by both lanes keeps its exact resolution.
    Ambiguous ids carry no hits, so resolve_pins naturally reports them
    unresolved and the pure-lookup short-circuit stays off — the full
    ranked loop is the honest answer for a reference this lane cannot
    uniquely resolve.
    """
    exact = [d.canonical_id for d in detected if d.kind in EXACT_KINDS]
    prefixes = [d.canonical_id for d in detected if d.kind == "hex_prefix"]
    numbers = [d for d in detected if d.kind == "number_ref"]
    for d in detected:
        if d.kind == "issue_ref":
            qualifier, _, num = d.canonical_id.rpartition("#")
            if num.isdigit():
                numbers.append(
                    DetectedIdentifier(
                        kind="number_ref",
                        canonical_id=d.canonical_id,
                        qualifier=qualifier,
                        number=num,
                    )
                )

    async def _exact() -> tuple[list[IdLookupHit], set[str]]:
        if not exact:
            return [], set()
        found = await id_lookup_search(
            customer_id,
            exact,
            temporal=temporal,
            include_drafts=include_drafts,
            sources=sources,
            doc_types=doc_types,
            author_ids=author_ids,
            source_keys=source_keys,
            source_keys_include_keyless=source_keys_include_keyless,
        )
        return found, set()

    async def _prefixes() -> tuple[list[IdLookupHit], set[str]]:
        if not prefixes:
            return [], set()
        return await _prefix_lookup(
            customer_id,
            prefixes,
            temporal=temporal,
            include_drafts=include_drafts,
            sources=sources,
            doc_types=doc_types,
            author_ids=author_ids,
            source_keys=source_keys,
            source_keys_include_keyless=source_keys_include_keyless,
        )

    async def _numbers() -> tuple[list[IdLookupHit], set[str]]:
        if not numbers:
            return [], set()
        return await _number_ref_lookup(
            customer_id,
            numbers,
            temporal=temporal,
            include_drafts=include_drafts,
            sources=sources,
            doc_types=doc_types,
            author_ids=author_ids,
            source_keys=source_keys,
            source_keys_include_keyless=source_keys_include_keyless,
        )

    results = await asyncio.gather(_exact(), _prefixes(), _numbers())
    hits: list[IdLookupHit] = []
    ambiguous: set[str] = set()
    for h, a in results:
        hits.extend(h)
        ambiguous |= a
    # ONE deterministic order for the merged list — resolve_pins' per-id
    # best-doc pick depends on it. Stable sort: exact-lane hits were
    # extended first, so equal keys keep exact ahead of inferred.
    hits.sort(key=lambda h: (h.matched_canonical_id, _neg_ts(h.updated_at), h.doc_id))
    return hits, ambiguous

def _neg_ts(ts: datetime) -> float:
    """Sort key helper: newest first without reverse-sorting the whole tuple."""
    return -ts.timestamp()


def resolve_pins(
    detected: list[DetectedIdentifier],
    hits: list[IdLookupHit],
    top_k: int,
) -> tuple[list[IdLookupHit], set[str], set[str]]:
    """Best doc per detected identifier, capped so ranking keeps room.

    Returns (pins, unresolved_canonical_ids, overflow_canonical_ids).
    One pin slot per identifier -- the BEST matching doc by the
    deterministic order id_lookup_search established -- deduped by doc_id
    (two ids resolving to the same doc pin it once), capped at
    max(1, top_k // 2) so a query quoting ten ids cannot return a phone
    book, and safe at top_k=1 (outside-voice F4).

    `overflow` is the third category the cap creates: ids that RESOLVED
    but lost their pin slot to the cap. They are not unresolved (the doc
    exists) and not pinned (no slot), so the caller must treat them as
    blocking the pure-lookup short-circuit -- otherwise a response could
    silently omit documents the user typed the exact id of (review F5).
    """
    by_cid: dict[str, IdLookupHit] = {}
    for h in hits:
        by_cid.setdefault(h.matched_canonical_id, h)
    pins: list[IdLookupHit] = []
    seen_docs: set[str] = set()
    unresolved: set[str] = set()
    overflow: set[str] = set()
    cap = max(1, top_k // 2)
    for d in detected:
        hit = by_cid.get(d.canonical_id)
        if hit is None:
            unresolved.add(d.canonical_id)
            continue
        if hit.doc_id in seen_docs:
            # Resolved AND represented: another id already pinned this doc.
            continue
        if len(pins) >= cap:
            overflow.add(d.canonical_id)
            continue
        seen_docs.add(hit.doc_id)
        pins.append(hit)
    return pins, unresolved, overflow
