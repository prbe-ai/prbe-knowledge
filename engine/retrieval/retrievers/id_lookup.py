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
) -> str:
    """The one scope gate every lookup in this module shares.

    Appends parameters to `params` and returns the `AND ...` SQL tail
    (source/doc_type/author/source_key/temporal/visibility). ONE builder on
    purpose: three lookup queries each re-implementing the workspace lens
    is three chances for a pinned doc to leak past it.
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
    out.append(pred.chunk_sql)
    out.append(pred.doc_sql)
    if not include_drafts:
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
            JOIN chunks c
              ON c.doc_id = d.doc_id
             AND c.customer_id = d.customer_id
             AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
            WHERE d.customer_id = $1
              AND COALESCE(c.kind, 'content') = 'content'
              {scope_sql}
            ORDER BY m.cid, c.doc_id, c.chunk_index ASC
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


def _row_to_hit(r: Any, canonical_id: str) -> IdLookupHit:
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
    )


_HEX_ONLY_RE = re.compile(r"^[0-9a-f]{7,11}$")


async def _prefix_lookup(
    customer_id: str,
    prefixes: list[str],
    *,
    spec: TemporalSpec,
    include_drafts: bool,
    sources: list[str] | None,
    doc_types: list[str] | None,
    author_ids: list[str] | None,
    source_keys: list[str] | None,
    source_keys_include_keyless: bool,
) -> tuple[list[IdLookupHit], set[str]]:
    """Resolve bare hex prefixes (short sha / uuid first-segment) to full
    stored identifiers. Returns (hits, ambiguous_prefixes).

    A prefix is an INFERRED identifier: the user typed part of it, and the
    lookup fills in the rest. That expansion may pin ONLY when it is
    unique — one matching document. Two matches means guessing, and a
    guessed doc pinned with a certainty label is the one outcome this
    module exists to prevent (same rule as git's short-hash resolution).
    Ambiguous prefixes are reported so the caller can log them and keep
    the full ranked loop.

    Match anchors, per the stored shapes:
      - `source_id LIKE <p> || '%'`        — session uuids (bare uuid)
      - `source_id LIKE '%@' || <p> || '%'` — commits (`owner/repo@<sha>`)
      - `source_id LIKE '%:' || <p> || '%'` — linear (`issue:<uuid>`)
      - `doc_id    LIKE '%:' || <p> || '%'` — any tail-identifier doc_id
    Leading-wildcard arms are the same bounded tenant-scan cost class as
    the ticket URL arms, and run once per request. Prefixes are regex-
    constrained to hex, so LIKE metacharacters cannot be injected.
    """
    clean = [c for c in prefixes if _HEX_ONLY_RE.match(c)]
    if not clean:
        return [], set()

    async with with_tenant(customer_id) as conn:
        params: list[Any] = [customer_id, clean]
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
            JOIN unnest($2::text[]) AS m(cid)
              ON (
                d.source_id LIKE m.cid || '%'
                OR d.source_id LIKE '%@' || m.cid || '%'
                OR d.source_id LIKE '%:' || m.cid || '%'
                OR d.doc_id LIKE '%:' || m.cid || '%'
              )
            {_CHUNK_JOIN}
            WHERE d.customer_id = $1
              AND COALESCE(c.kind, 'content') = 'content'
              {scope_sql}
            ORDER BY m.cid, c.doc_id, c.chunk_index ASC
            """,
            *params,
        )

    by_cid: dict[str, list[Any]] = {}
    for r in rows:
        by_cid.setdefault(r["matched_canonical_id"], []).append(r)
    hits: list[IdLookupHit] = []
    ambiguous: set[str] = set()
    for cid, group in by_cid.items():
        docs = {r["doc_id"] for r in group}
        if len(docs) > 1:
            ambiguous.add(cid)
            continue
        hits.append(_row_to_hit(group[0], cid))
    hits.sort(key=lambda h: (h.matched_canonical_id, _neg_ts(h.updated_at), h.doc_id))
    return hits, ambiguous


async def _number_ref_lookup(
    customer_id: str,
    refs: list[str],
    *,
    spec: TemporalSpec,
    include_drafts: bool,
    sources: list[str] | None,
    doc_types: list[str] | None,
    author_ids: list[str] | None,
    source_keys: list[str] | None,
    source_keys_include_keyless: bool,
) -> tuple[list[IdLookupHit], set[str]]:
    """Resolve PR/issue number refs ('#383', 'research-os#539') to a repo's
    PR, issue, or squash-merge commit doc. Returns (hits, ambiguous_refs).

    The number is matched in SQL; the repo is resolved in PYTHON, because
    the qualifier is a SOFT filter: 'the PR #232' captures 'the' as its
    qualifier, and demanding a repo literally named 'the' would silently
    lose a resolvable number. Rules, in order, per ref:

      1. Group all matching docs by repo (`split_part(doc_id, ':', 2)`,
         github-only docs — a Slack message quoting '(#383)' in its title
         must not create a phantom repo group).
      2. If the qualifier matches exactly one group (repo tail-match),
         resolve there. Qualifier matching two groups -> ambiguous.
      3. No qualifier match (or no qualifier): exactly one group total
         resolves; two or more -> ambiguous (the same number exists in
         several repos, and picking one would be a guess).

    Within the resolved repo, prefer the PR/issue doc over the commit doc
    whose title carries '(#N)' (pref rank in SQL), newest first. The
    number is digits-only by regex, so LIKE injection is impossible; the
    qualifier never reaches SQL at all.
    """
    parsed: list[tuple[str, str, str]] = []  # (canonical, qualifier, num)
    for ref in refs:
        qualifier, _, num = ref.rpartition("#")
        if num.isdigit():
            parsed.append((ref, qualifier, num))
    if not parsed:
        return [], set()
    nums = sorted({num for _, _, num in parsed})

    async with with_tenant(customer_id) as conn:
        params: list[Any] = [customer_id, nums]
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
            SELECT DISTINCT ON (m.num, c.doc_id)
                   m.num AS matched_num,
                   split_part(d.doc_id, ':', 2) AS repo_group,
                   CASE
                     WHEN d.doc_id LIKE '%:pr:' || m.num THEN 0
                     WHEN d.source_id LIKE '%#' || m.num THEN 0
                     WHEN d.doc_id LIKE '%:issue:' || m.num THEN 1
                     WHEN d.doc_id LIKE '%:issues:' || m.num THEN 1
                     ELSE 2
                   END AS pref,
                   {_HIT_COLUMNS}
            FROM documents d
            JOIN unnest($2::text[]) AS m(num)
              ON (
                d.source_id LIKE '%#' || m.num
                OR d.doc_id LIKE '%:pr:' || m.num
                OR d.doc_id LIKE '%:issue:' || m.num
                OR d.doc_id LIKE '%:issues:' || m.num
                OR d.title LIKE '%(#' || m.num || ')%'
              )
            {_CHUNK_JOIN}
            WHERE d.customer_id = $1
              AND d.source_system = 'github'
              AND COALESCE(c.kind, 'content') = 'content'
              {scope_sql}
            ORDER BY m.num, c.doc_id, c.chunk_index ASC
            """,
            *params,
        )

    by_num: dict[str, list[Any]] = {}
    for r in rows:
        by_num.setdefault(r["matched_num"], []).append(r)

    hits: list[IdLookupHit] = []
    ambiguous: set[str] = set()
    for canonical, qualifier, num in parsed:
        group_rows = by_num.get(num)
        if not group_rows:
            continue  # plain unresolved — no signal either way
        groups: dict[str, list[Any]] = {}
        for r in group_rows:
            groups.setdefault(r["repo_group"], []).append(r)
        chosen: list[Any] | None = None
        if qualifier:
            q = qualifier.lower()
            matched = [
                g for g in groups
                if g.lower() == q or g.lower().endswith("/" + q)
            ]
            if len(matched) == 1:
                chosen = groups[matched[0]]
            elif len(matched) > 1:
                ambiguous.add(canonical)
                continue
            # 0 matched: junk qualifier — fall through to the bare rule.
        if chosen is None:
            if len(groups) == 1:
                chosen = next(iter(groups.values()))
            else:
                ambiguous.add(canonical)
                continue
        best = min(
            chosen,
            key=lambda r: (r["pref"], _neg_ts(r["updated_at"]), r["doc_id"]),
        )
        hits.append(_row_to_hit(best, canonical))
    return hits, ambiguous


_EXACT_KINDS = frozenset({"uuid", "ticket", "issue_ref", "commit_sha", "pd_incident"})


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

    EXACT kinds (uuid / ticket / issue_ref / commit_sha / pd_incident) go
    through `id_lookup_search` unchanged — the user typed the whole id, so
    whatever the equality/suffix arms match is right. INFERRED kinds
    (hex_prefix / number_ref) go through the expansion lookups above,
    which pin only on a unique resolution and report the ambiguous rest.

    Returns (hits, ambiguous_canonical_ids). Ambiguous ids carry no hits,
    so `resolve_pins` naturally reports them unresolved and the pure-
    lookup short-circuit stays off — the full ranked loop is the honest
    answer for a reference this lane cannot uniquely resolve.
    """
    spec = temporal or TemporalSpec()
    scope = dict(
        spec=spec,
        include_drafts=include_drafts,
        sources=sources,
        doc_types=doc_types,
        author_ids=author_ids,
        source_keys=source_keys,
        source_keys_include_keyless=source_keys_include_keyless,
    )
    hits: list[IdLookupHit] = []
    ambiguous: set[str] = set()

    exact = [d.canonical_id for d in detected if d.kind in _EXACT_KINDS]
    if exact:
        hits.extend(
            await id_lookup_search(
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
        )
    prefixes = [d.canonical_id for d in detected if d.kind == "hex_prefix"]
    if prefixes:
        h, a = await _prefix_lookup(customer_id, prefixes, **scope)
        hits.extend(h)
        ambiguous |= a
    numbers = [d.canonical_id for d in detected if d.kind == "number_ref"]
    if numbers:
        h, a = await _number_ref_lookup(customer_id, numbers, **scope)
        hits.extend(h)
        ambiguous |= a
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
        h = by_cid.get(d.canonical_id)
        if h is None:
            unresolved.add(d.canonical_id)
            continue
        if h.doc_id in seen_docs:
            # Resolved AND represented: another id already pinned this doc.
            continue
        if len(pins) >= cap:
            overflow.add(d.canonical_id)
            continue
        seen_docs.add(h.doc_id)
        pins.append(h)
    return pins, unresolved, overflow
