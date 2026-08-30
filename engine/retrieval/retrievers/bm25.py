"""BM25-ish retriever via Postgres `ts_rank_cd`.

STALE PREMISE — READ THIS FIRST. The next paragraph's opening claim is no
longer true of this database, and the retriever has not caught up:

    pg_search 0.23.4 IS installed, and `idx_chunks_bm25` (174 MB, ParadeDB,
    key_field=chunk_id) already exists over chunks. `pg_stat_user_indexes`
    reports scans = 0. It has never been used, because this module ranks with
    `ts_rank_cd` instead.

Measured against that index on the managed data plane, same corpus:

    ts_rank_cd, cross-table OR (current)   30,000 ms (statement timeout)
    ts_rank_cd, OR rewritten as UNION       2,997 ms
    pg_search `@@@`, TopKScanExecState        263 ms, 77 heap fetches

The gap is structural, not tuning. `ts_rank_cd` must find every matching row
(the OR-ed tsquery matches 99,670 of 204,637 chunks, 48.7% of the corpus),
score each, sort them all, and keep 120. A real BM25 index keeps ranking
inside the index and returns top-K directly.

Migrating is Phase 2 of the retrieval-latency work and is deliberately NOT
bundled with the Phase 1 fixes: it needs a title denormalization, a 206k-row
backfill, and a search-quality evaluation, because BM25 ranking is genuinely
different from cover-density ranking. Do not "just switch the operator".

Everything below describes the CURRENT implementation and remains accurate.

Postgres' built-in text search has no true BM25 — `ts_rank_cd` (cover-density
ranking) is the stand-in used here, and runs on the `idx_chunks_content_tsv`
GIN index over the stored `chunks.content_tsv` column (migration 0062).
The column is `GENERATED ALWAYS AS (to_tsvector('english', content)) STORED`,
so the bitmap-heap recheck and `ts_rank_cd` both read the precomputed
lexeme array off the heap instead of re-tokenizing `content` on every one
of the ~10k+ candidate rows. EXPLAIN ANALYZE on acme showed the
old expression-based path spent ~5.7s of a 5.9s query in per-row
tokenization; the materialized column reduces that to score math + heap
reads. For Phase 1 we can still swap this to pg_bm25 or a real BM25 lib
if ranking quality matters enough.

Titles are searchable too (migration 0099), and they could not be before.
`chunks.content_tsv` is a GENERATED column, and a generation expression may
only reference its own row, so it can never reach `documents.title` -- the two
are in different tables. Nothing in the query read the title, so a file named
`model.ckpt` or a PR titled "Fix the retry loop" was findable by keyword ONLY
if those words also appeared in the body. Vector search did see titles (chunks
embed as `title: {title} | text: {content}`), but an exact filename is exactly
the query where semantic similarity is weakest and lexical matching should win.

Two things follow from adding `documents.title_tsv` to the WHERE:

  * RANKING. Scores are the SUM of the content rank and the title rank.
    `title_tsv` is setweight'd 'A' while `content_tsv` is unweighted ('D'), so
    under ts_rank_cd's default weights a title hit is worth 1.0 against a body
    hit's 0.1. A document matching in both outranks one matching in either.

  * WHICH CHUNK REPRESENTS A TITLE-ONLY HIT. A title match belongs to the
    DOCUMENT, not to any one chunk, so matching on it alone would surface every
    chunk of that document -- all with identical scores, and a long document
    would bury everything else. Title-only matches are therefore restricted to
    `chunk_index = 0`: one representative chunk per document. Chunks that match
    on their own content are unaffected and still surface individually.

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

from engine.retrieval.helpers import source_key_predicate
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

    Still used, but no longer for RANKING. pg_search ranks now; this only
    answers "did this chunk's own content match?" so the title-only cap
    below can tell a chunk that earned its slot from one that rode in on
    its document's title. Evaluated on the bounded pool (~1200 rows), never
    on the table, so it costs a stored-column check and no index probe.
    """
    tokens = [t for t in _TOKEN_RE.findall(query_text) if len(t) >= 2]
    if not tokens:
        return ""
    return " | ".join(tokens)


def _build_pg_search_query(tokens: list[str]) -> str:
    """Space-joined tokens for pg_search's `@@@ 'text'` form.

    pg_search parses this into its own OR-of-terms and scores with BM25, so
    unlike `to_tsquery` it does NOT need explicit `|` operators and does not
    reject punctuation. Tokens are already stripped to [A-Za-z0-9_] runs by
    `_TOKEN_RE`, which also removes the quote characters that would otherwise
    let a query escape into pg_search's query-string syntax.
    """
    return " ".join(tokens)


# How many chunks ONE document may contribute on the strength of its TITLE
# alone. Chunks whose own content matched are never capped.
#
# 1 reproduces the pre-Phase-2 rule exactly. `bm25.py` used to express this
# structurally as `AND c.chunk_index = 0` -- "a title match contributes the
# document's first chunk" -- which is a cap of one that also picked the chunk
# arbitrarily rather than by relevance. Making it a number means the value is
# visible and tunable instead of implied by a predicate, and BM25 now picks
# WHICH chunk by score.
#
# Raise deliberately and behind a search-quality eval: the failure mode the old
# rule was guarding is one long document filling the result set, and that risk
# grows linearly with this number.
BM25_TITLE_ONLY_PER_DOC = 1

# The per-document cap is a window function, and a window over the full match
# set defeats pg_search's TopK execution -- measured on production, 204,637
# chunks:
#
#     no cap                        TopKScanExecState   216 ms
#     cap over the full match set   NormalScanExecState 533 ms   (53,646 rows sorted)
#     cap over an over-fetched pool TopKScanExecState   227 ms
#
# So the cap runs INSIDE a bounded pool. This multiplier sizes that pool. A
# top-1200 pool held 253 distinct documents on this corpus, against the 40 that
# a cap of 1 needs to fill 120 slots, so the margin is ~6x. If a caller ever
# reports short result sets on a corpus dominated by one document, this is the
# knob.
_BM25_POOL_MULTIPLIER = 10

# Weight on a title match relative to a content match.
#
# The old ranker got a ~10x title advantage for free: migration 0099 stores
# `documents.title_tsv` with setweight('A') while `chunks.content_tsv` is
# unweighted ('D'), and ts_rank_cd's default weights are {D:0.1, A:1.0}. BM25
# has no equivalent, so the intent has to be restated as a number.
#
# 10.0 reproduces the OLD ratio exactly. That is the point: ts_rank_cd's default
# weights are {D: 0.1, A: 1.0}, so the previous ranker gave a title match
# precisely 10x a body match, and this is a performance change, not a decision
# to re-rank titles.
#
# This was 2.5 on the reasoning that BM25 term scores already scale with rarity
# so a large multiplier double-boosts. Measured, that reasoning was wrong in the
# direction that matters: at 2.5 a title-only hit on `checkpoints/model.ckpt`
# scored 3.44 against 4.37 for a document merely MENTIONING the file in its body
# -- the exact inversion `test_title_match_outranks_a_body_only_match` exists to
# prevent. Worse, the winner flipped with corpus statistics: the same comparison
# on a differently-populated index put title ahead at 2.5. A value that depends
# on the corpus is not a guarantee.
#
# 10.0 restores a ratio that does not depend on which documents happen to be
# indexed, and it is the ratio this system already shipped for months.
_BM25_TITLE_BOOST = 10.0


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
    source_keys_include_keyless: bool = False,
    per_source_top_k: int | None = None,
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
    pg_query = _build_pg_search_query(
        [t for t in _TOKEN_RE.findall(query_text) if len(t) >= 2]
    )

    async with with_tenant(customer_id) as conn:
        # $2 is the pg_search query (ranking); $4 is the tsquery form, used
        # ONLY to decide whether a chunk's own content matched.
        params: list = [customer_id, pg_query, top_k, or_query]
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

        # Drafts are hidden on BOTH sides, but the two halves are applied at
        # different depths now: the chunk half inside the pg_search pool (so a
        # draft never consumes a pool slot) and the document half at the join.
        # Both are inlined at their callsites below.

        # Column references are unqualified here: this orders the OUTER select,
        # whose columns come from the pool/join projection, not from `c`/`d`.
        order_by_sql = (
            "updated_at DESC, chunk_id"
            if sort_by == "recency"
            else "score DESC, chunk_id"
        )

        # ---- 1. ANN-equivalent for keywords: a bounded, index-ranked pool ----
        # Single table. `title` lives on the chunk (migration 0100), so the
        # cross-table OR that made this un-indexable is now unwriteable: there
        # is no second table in this scan to OR against.
        #
        # `title` is boosted rather than restricted to chunk_index=0. A title
        # match is evidence the DOCUMENT is relevant, so it should raise that
        # document's chunks in the ranking; picking chunk 0 was expressing a
        # ranking idea as a join predicate. The cap below keeps the guarantee
        # the old predicate was really providing.
        params.append(top_k * _BM25_POOL_MULTIPLIER)
        pool_idx = len(params)
        # The tenant and visibility filters appear TWICE below, and both
        # copies are load-bearing.
        #
        # As plain SQL predicates (`c.customer_id = $1`, `c.visibility =
        # 'approved'`) pg_search cannot see them: the ParadeDB scan applies
        # them as per-candidate HEAP filters, so TopK walks the heap for every
        # chunk in the OR match set before the LIMIT ever binds. Measured on
        # the research primary (2026-08-30, probe tenant, a real 17-token
        # query): 403,559 buffer touches and 5,814 ms to return 300 rows --
        # against 407 ms for the identical ranking with no filter at all. The
        # entire gap is heap checks, and it scales with the match set and the
        # cold cache, which is what produced the 13-21 s storm numbers.
        #
        # Restating them INSIDE the boolean as `must` clauses keeps the
        # filtering index-side and TopK execution intact: same query, same
        # tenant, 1,269 ms under FORCE RLS as the app role. The nested
        # `should` boolean must ride inside `must`, not beside it -- Tantivy
        # treats bare `should` legs as optional once any `must` exists, which
        # would silently widen the match set to the whole tenant.
        #
        # `match(..., conjunction_mode => true)`, NOT `term()`, for
        # customer_id: the field is indexed under the default tokenizer, which
        # splits on hyphens, so term('customer_id', 'bucket-robotics') looks
        # up a token that does not exist and returns ZERO rows for every
        # hyphenated tenant (verified live). conjunction_mode requires every
        # token of the id, which is correct and cheap.
        #
        # The SQL predicates STAY, because the index-side clause is a
        # pre-filter, not the correctness filter: tokenized ids overlap
        # ('probe' matches probe-demo's first token), and under FORCE RLS the
        # policy qual re-applies the tenant check regardless. Belt and braces,
        # in that order.
        tenant_must = "paradedb.match('customer_id', $1, conjunction_mode => true)"
        visibility_must = (
            "" if include_drafts else "paradedb.term('visibility', 'approved'),"
        )
        pool_sql = f"""
            SELECT c.chunk_id,
                   c.doc_id,
                   c.content,
                   c.kind,
                   c.chunk_index,
                   c.first_seen_version,
                   c.last_seen_version,
                   paradedb.score(c.chunk_id) AS score,
                   (c.content_tsv @@ to_tsquery('english', $4)) AS content_hit
            FROM chunks c
            WHERE c.customer_id = $1
              AND c.chunk_id @@@ paradedb.boolean(must => ARRAY[
                    {tenant_must},
                    {visibility_must}
                    paradedb.boolean(should => ARRAY[
                      paradedb.boost({_BM25_TITLE_BOOST}, paradedb.match('title', $2)),
                      paradedb.match('content', $2)
                    ])
                  ])
              {"" if include_drafts else "AND c.visibility = 'approved'"}
              {pred.chunk_sql}
            ORDER BY paradedb.score(c.chunk_id) DESC
            LIMIT ${pool_idx}
        """

        # ---- 2. cap title-only contributions, INSIDE the pool ----
        # `content_hit` is the whole point: a chunk that matched on its own
        # content earned its slot and is never capped, however many its
        # document contributes. Only free riders are limited.
        params.append(BM25_TITLE_ONLY_PER_DOC)
        cap_idx = len(params)
        capped_sql = f"""
            SELECT * FROM (
                SELECT p.*,
                       CASE WHEN p.content_hit THEN 0
                            ELSE ROW_NUMBER() OVER (
                                PARTITION BY p.doc_id, p.content_hit
                                ORDER BY p.score DESC, p.chunk_id
                            )
                       END AS _title_rn
                FROM ({pool_sql}) p
            ) q
            WHERE q.content_hit OR q._title_rn <= ${cap_idx}
        """

        # ---- 3. join documents for the doc-level filters and projection ----
        # Doc-level predicates stay a join because they filter on columns that
        # belong to the document, not the chunk. They run against the capped
        # pool (hundreds of rows), not the table.
        inner_sql = f"""
            SELECT k.chunk_id,
                   k.doc_id,
                   d.version AS doc_version,
                   d.source_system,
                   d.source_url,
                   d.title,
                   d.author_id,
                   k.content,
                   k.kind,
                   d.created_at,
                   d.updated_at,
                   k.score
            FROM ({capped_sql}) k
            JOIN documents d
              ON k.doc_id = d.doc_id
             AND d.customer_id = $1
             AND d.version BETWEEN k.first_seen_version AND k.last_seen_version
            WHERE TRUE
              {pred.doc_sql}
              {source_filter}
              {doc_type_filter}
              {"" if include_drafts else "AND d.visibility = 'approved'"}
              {author_filter}
              {source_key_filter}
        """
        if per_source_top_k is not None:
            # Per-source top-K slotting (PR#78 recall guarantee, server-side).
            # Window orders by SELECTED columns, not the raw ts_rank expr.
            partition_order = (
                "updated_at DESC, chunk_id"
                if sort_by == "recency"
                else "score DESC, chunk_id"
            )
            params.append(per_source_top_k)
            ps_idx = len(params)
            sql = f"""
            SELECT chunk_id, doc_id, doc_version, source_system, source_url,
                   title, author_id, content, kind, created_at, updated_at, score
            FROM (
                SELECT sub.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY sub.source_system ORDER BY {partition_order}
                       ) AS _ps_rn
                FROM ({inner_sql}) sub
            ) ranked
            WHERE _ps_rn <= ${ps_idx}
            -- Interleave sources; see the same ORDER BY in vector.py. Ordering
            -- by score would undo the PARTITION one line above and let the
            -- highest-scoring source take every slot under the LIMIT.
            ORDER BY _ps_rn, {partition_order}
            LIMIT $3
            """
        else:
            sql = f"{inner_sql}\n            ORDER BY {order_by_sql}\n            LIMIT $3"

        rows = await conn.fetch(sql, *params)

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
