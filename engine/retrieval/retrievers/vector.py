"""Vector retriever — pgvector HNSW top-k with temporal filtering."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import asyncpg

from engine.retrieval.helpers import source_key_predicate
from engine.retrieval.temporal import build_predicate
from engine.shared.constants import TOP_K_VECTOR
from engine.shared.db import with_tenant
from engine.shared.embeddings import get_embedder_v2
from engine.shared.models import TemporalSpec, normalize_author_id

# Global ANN pool size for the per-source path's first phase. Sized from a live
# measurement on the research plane (792k chunks, 2026-08-26): LIMIT 400 ran in
# 456 ms through the HNSW index, and 400 comfortably covers every source's
# quota (per_source_top_k caps at 50, and tenants carry a handful of sources).
# The floor exists so a small top_k cannot shrink the pool below usefulness --
# the pool's whole job is to satisfy MOST sources' quotas so the per-source
# top-up phase has little or nothing to do.
PER_SOURCE_ANN_POOL = 400


@dataclass(slots=True)
class VectorHit:
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
    # 'content' (default for legacy rows) or 'metadata'. The fusion layer
    # uses kind to combine per-doc scores (metadata signal boosts the doc's
    # best content chunk's ranking) and to drop synthetic key:value text from
    # the response.
    kind: str = "content"


async def vector_search(
    customer_id: str,
    query_text: str,
    top_k: int = TOP_K_VECTOR,
    sources: list[str] | None = None,
    doc_types: list[str] | None = None,
    temporal: TemporalSpec | None = None,
    include_drafts: bool = False,
    author_ids: list[str] | None = None,
    sort_by: Literal["relevance", "recency"] = "relevance",
    source_keys: list[str] | None = None,
    source_keys_include_keyless: bool = False,
    per_source_top_k: int | None = None,
) -> list[VectorHit]:
    """Embed `query_text`, ANN-search against chunks, return top_k hits.

    Score is cosine similarity (1 - cosine distance) so higher is better.

    `temporal` controls which versions of each doc are considered. Defaults
    to TemporalSpec() = latest-live.

    `doc_types`, when set, hard-filters by `documents.doc_type` (dotted form,
    e.g. ['github.commit', 'github.pull_request']). The search pipeline
    passes None and uses doc_type as a soft RRF boost; the list pipeline
    passes the resolved set as a hard filter — same retriever, two callers.

    `include_drafts` defaults to False so retrieval returns only rows with
    ``visibility = 'approved'`` (the partial indexes from migration 0082
    keep this cheap). Reviewer-scoped BFF surfaces flip this to True;
    API-key callers never bypass.

    `author_ids`, when set, hard-filters by `documents.author_id = ANY(...)`.
    Mirrors `sql_list`'s author filter (services/retrieval/retrievers/sql.py:246).
    The gatherer's extractor populates this list from `person` entities when
    the query asks "what did <person> do" / "PRs by <person>" / etc.

    `sort_by="recency"` swaps the SQL `ORDER BY` from cosine-distance to
    `d.updated_at DESC, c.chunk_id`. The ANN filter (chunks with embeddings
    matching the query) still narrows the pool, but final order is by
    recency. Used by the gatherer when the extractor flagged temporal
    intent.

    `source_keys`, when set, hard-filters by
    `documents.metadata->>'source_key' = ANY(...)` -- the key the
    custom-ingest door stamps per document. Docs without a source_key
    (connector-ingested) drop out. The predicate applies BEFORE the LIMIT
    (never post-trim), but note the HNSW caveat: pgvector evaluates
    filters on rows the ANN scan visits, so a highly selective scope can
    UNDER-RETURN (fewer than top_k in-scope hits exist among the scanned
    candidates even though more exist in the table). We mitigate by
    enabling pgvector's iterative scan (`hnsw.iterative_scan =
    relaxed_order`, pgvector >= 0.8) so the scan keeps widening until
    enough in-scope rows are found; on older pgvector builds the SET
    fails softly (savepoint rollback) and the pre-mitigation under-return
    behavior remains. relaxed_order may return near-ties slightly out of
    distance order -- acceptable for a fused retrieval channel.

    That mitigation covers EVERY filter on the ANN path, not just
    source_keys: the visibility predicate is unconditional, so pgvector is
    always post-filtering something.

    `per_source_top_k`, when set (unified search sends it on EVERY request),
    guarantees each source_system its own top-K slot -- the PR#78 recall
    guarantee, server-side. See `_per_source_ann_search` for how that
    guarantee is kept WITHOUT abandoning the ANN index. The first
    implementation kept it by skipping the ANN LIMIT and windowing the full
    matching set, which planned as a Parallel Seq Scan + a 626k-row Sort and
    ran 37-52 SECONDS on the research plane -- ~97% of the retrieval stage's
    budget, and the actual cause of the 2026-08-26 search timeouts.

    Result ordering is deterministic. The ANN pool is ordered by distance
    ALONE (the only shape HNSW can serve), then the outer query applies the
    `chunk_id` tiebreak over that bounded pool. Putting the tiebreak in the
    ANN ORDER BY is what turned this into a 3,355 ms seq scan; see the
    comment at the ordering callsite.
    """
    embedder = get_embedder_v2()
    query_vec = await embedder.embed_query(query_text)
    literal = "[" + ",".join(f"{x:.7f}" for x in query_vec) + "]"

    spec = temporal or TemporalSpec()

    inner_sql, params, ann_order_sql, outer_order_sql = _build_inner_query(
        customer_id=customer_id,
        literal=literal,
        top_k=top_k,
        sources=sources,
        doc_types=doc_types,
        author_ids=author_ids,
        source_keys=source_keys,
        source_keys_include_keyless=source_keys_include_keyless,
        spec=spec,
        include_drafts=include_drafts,
        sort_by=sort_by,
    )

    # The per-source guarantee on the relevance path gets its own strategy:
    # a bounded ANN pool plus per-source ANN top-ups, all through the index.
    # recency keeps the windowed full-scan below -- it cannot use the ANN
    # index by construction, and its filters prune the pool first.
    if per_source_top_k is not None and sort_by != "recency":
        rows = await _per_source_ann_search(
            customer_id=customer_id,
            inner_sql=inner_sql,
            params=params,
            ann_order_sql=ann_order_sql,
            top_k=top_k,
            per_source_top_k=per_source_top_k,
            sources=sources,
        )
        return _to_hits(rows)

    async with with_tenant(customer_id) as conn:
        # Selective post-filter mitigation (see docstring). This used to be
        # gated on `source_keys`, which under-scoped it: pgvector applies
        # EVERY filter after the ANN scan, and the visibility filter below is
        # unconditional (include_drafts defaults False). So the under-return
        # this guards against applies to essentially every ANN query, not just
        # keyed ones -- a doc_type or author filter under-returns exactly the
        # same way. Gate on the ANN path itself instead.
        if sort_by != "recency":
            await _enable_iterative_scan(conn)

        # ANN candidate pool. The index returns its best N by distance; every
        # later step (per-source windowing, deterministic tiebreak) runs over
        # that bounded pool rather than the table.
        #
        # recency cannot take the ANN LIMIT: ordering by `updated_at` is not a
        # shape the HNSW index can serve, so it keeps the narrowed full scan
        # and pays a sort over the filtered pool (author/doc_type/temporal
        # filters prune before it matters).
        if sort_by == "recency":
            candidate_sql = inner_sql
        else:
            params.append(top_k)
            candidate_sql = (
                f"{inner_sql}\n            ORDER BY {ann_order_sql}"
                f"\n            LIMIT ${len(params)}"
            )

        if per_source_top_k is not None:
            # recency + per-source: the original windowed shape, unchanged.
            # Give each source_system its own top-K slot instead of one global
            # budget, so a loud source can't bury a quiet one in a mixed-source
            # request (the PR#78 recall guarantee, moved server-side). LIMIT $3
            # stays an overall safety cap. The window orders by the SELECTED
            # columns (score / updated_at), not the raw distance expression,
            # which isn't in scope at the wrapping layer.
            partition_order = "updated_at DESC, chunk_id"
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
                FROM ({candidate_sql}) sub
            ) ranked
            WHERE _ps_rn <= ${ps_idx}
            -- Interleave sources: each source's rank-1 before any source's
            -- rank-2. Ordering by score here instead would re-impose exactly
            -- the cross-source competition the PARTITION just prevented --
            -- cosine scores are NOT comparable across sources (terse
            -- structured projections always lose to chatty transcripts), so
            -- rank, not score, is the only fair cross-source currency.
            ORDER BY _ps_rn, {partition_order}
            LIMIT $3
            """
        else:
            # Deterministic tiebreak lives HERE, outside the ANN pool, so it
            # sorts at most `pool_size` rows instead of defeating the index.
            sql = f"""
            SELECT chunk_id, doc_id, doc_version, source_system, source_url,
                   title, author_id, content, kind, created_at, updated_at, score
            FROM ({candidate_sql}) pool
            ORDER BY {outer_order_sql}
            LIMIT $3
            """

        rows = await conn.fetch(sql, *params)

    return _to_hits(rows)


def _build_inner_query(
    *,
    customer_id: str,
    literal: str,
    top_k: int,
    sources: list[str] | None,
    doc_types: list[str] | None,
    author_ids: list[str] | None,
    source_keys: list[str] | None,
    source_keys_include_keyless: bool,
    spec: TemporalSpec,
    include_drafts: bool,
    sort_by: str,
) -> tuple[str, list, str, str]:
    """The filtered candidate SELECT shared by every path.

    Returns (inner_sql, params, ann_order_sql, outer_order_sql). `params`
    starts [customer_id, literal, top_k] so $3 stays the overall LIMIT in
    every consumer -- both existing paths and the per-source strategy append
    their own parameters after the shared tail.
    """
    params: list = [customer_id, literal, top_k]
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

    # Default branch hides drafts; visibility filter is a sibling of
    # the existing valid_to predicate. Reviewer surfaces pass
    # include_drafts=True to bypass.
    visibility_filter = (
        ""
        if include_drafts
        else "AND c.visibility = 'approved' AND d.visibility = 'approved'"
    )

    # ANN ordering MUST be `ORDER BY <distance>` and nothing else, or the
    # HNSW index cannot serve it. This previously read
    # `c.embedding_v2 <=> $2::halfvec, c.chunk_id`; the `chunk_id`
    # tiebreaker forced exact distances for every candidate row and the
    # planner fell back to a Parallel Seq Scan + Sort. Measured on the
    # managed plane against 203,454 rows:
    #
    #     ORDER BY dist, chunk_id  ->  Parallel Seq Scan + Sort   3,355 ms
    #     ORDER BY dist            ->  Index Scan (hnsw)             12.7 ms
    #
    # This is the documented pgvector behaviour (pgvector#760), not a
    # planner quirk. Determinism is NOT dropped -- it moves to the outer
    # query, which tiebreaks a bounded pool instead of the table.
    ann_order_sql = "c.embedding_v2 <=> $2::halfvec"
    outer_order_sql = (
        "updated_at DESC, chunk_id"
        if sort_by == "recency"
        else "score DESC, chunk_id"
    )

    inner_sql = f"""
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
                   1 - (c.embedding_v2 <=> $2::halfvec) AS score
            FROM chunks c
            JOIN documents d
              ON c.doc_id = d.doc_id
             AND d.customer_id = c.customer_id
             AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
            WHERE c.customer_id = $1
              AND c.embedding_v2 IS NOT NULL
              {pred.chunk_sql}
              {pred.doc_sql}
              {source_filter}
              {doc_type_filter}
              {visibility_filter}
              {author_filter}
              {source_key_filter}
        """
    return inner_sql, params, ann_order_sql, outer_order_sql


async def _enable_iterative_scan(conn: asyncpg.Connection) -> None:
    """Let a filtered ANN scan widen until the LIMIT is satisfied.

    with_tenant runs inside a transaction, so SET LOCAL scopes to this
    query and cannot leak across a pgbouncer-pooled connection. The
    savepoint makes the missing-GUC case (pgvector < 0.8) a soft no-op
    instead of poisoning the transaction.
    """
    await conn.execute("SAVEPOINT iterscan")
    try:
        await conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
        await conn.execute("RELEASE SAVEPOINT iterscan")
    except asyncpg.PostgresError:
        await conn.execute("ROLLBACK TO SAVEPOINT iterscan")


async def _per_source_ann_search(
    *,
    customer_id: str,
    inner_sql: str,
    params: list,
    ann_order_sql: str,
    top_k: int,
    per_source_top_k: int,
    sources: list[str] | None,
) -> list[Any]:
    """The per-source recall guarantee, kept ON the ANN index.

    HISTORY, because the previous shape looked reasonable and cost 37-52
    seconds. The guarantee (PR#78): every source_system gets its own top-K
    slots, because cosine scores are not comparable across sources and a
    global budget hands every slot to the chattiest corpus --
    `custom_ingest`'s first hit once ranked 61st globally and a LIMIT of 30
    cut that corpus entirely. The first server-side implementation kept the
    guarantee by SKIPPING the ANN LIMIT and windowing the FULL matching set.
    Correct, and catastrophically slow: on the research plane that planned as
    a Parallel Seq Scan over 626k joined rows + Sort + WindowAgg, 37-52s per
    query, ~97% of the retrieval stage -- the 2026-08-26 search timeouts.
    Its comment claimed production used the fast default path; unified
    search sends per_source_top_k on every request, so production ALWAYS
    took the slow one.

    The replacement keeps both properties -- per-source recall AND the index
    -- by decomposing:

      1. POOL: one global ANN query, `ORDER BY distance LIMIT pool_size`
         (index scan, ~456ms measured at 400 on 792k chunks). Because the
         pool is globally distance-ordered, any source with >= K rows in it
         has its true per-source top-K there already.
      2. TOP-UP: only for sources the pool left short, one ANN query each
         with `d.source_system = $s`, again `ORDER BY distance LIMIT K`.
         pgvector's iterative scan widens each scan until the quota is
         found, which is precisely the rank-61 case done correctly: walk
         deeper for the quiet source, but through the index, bounded by
         `hnsw.max_scan_tuples` (default 20k visited tuples) instead of by
         the table.
      3. The top-ups run CONCURRENTLY (each on its own pooled connection),
         so wall clock is pool + max(top-up) -- measured ~600ms per quiet
         source -- not pool + sum. Serial SQL for the same decomposition
         measured 2.9s; concurrent Python measures ~1.1s wall.

    Failure honesty: iterative scan gives up after max_scan_tuples, so an
    ultra-rare source inside a huge corpus can still under-return. The old
    full scan would have found it, 40 seconds late; the budget upstream
    (`ENGINE_TIMEOUT_SECONDS` = 30s < the old path's floor) means those
    results were never actually delivered to anyone. Bounded-but-fast is the
    honest trade, and it is the same one the default ANN path already makes.

    `sources`, when the caller set it, is both a hard filter (already inside
    `inner_sql`) and the quota list -- no discovery query needed. Otherwise
    the tenant's live source list comes from a skip-scan on
    `idx_documents_customer_source` (~1ms), never a DISTINCT seq scan.
    """
    pool_limit = max(top_k, PER_SOURCE_ANN_POOL)

    async def _fetch_pool() -> list[Any]:
        pool_params = [*params, pool_limit]
        sql = (
            f"{inner_sql}\n            ORDER BY {ann_order_sql}"
            f"\n            LIMIT ${len(pool_params)}"
        )
        async with with_tenant(customer_id) as conn:
            await _enable_iterative_scan(conn)
            return await conn.fetch(sql, *pool_params)

    async def _fetch_sources() -> list[str]:
        if sources:
            return list(sources)
        # Loose index scan over (customer_id, source_system, ...): each
        # recursion hops to the next distinct source via the btree, so cost
        # is O(distinct sources), not O(documents). A plain DISTINCT here
        # seq-scanned ~211k rows.
        sql = """
            WITH RECURSIVE r AS (
                (SELECT source_system FROM documents
                 WHERE customer_id = $1
                 ORDER BY source_system LIMIT 1)
                UNION ALL
                SELECT (SELECT d2.source_system FROM documents d2
                        WHERE d2.customer_id = $1
                          AND d2.source_system > r.source_system
                        ORDER BY d2.source_system LIMIT 1)
                FROM r WHERE r.source_system IS NOT NULL
            )
            SELECT source_system FROM r WHERE source_system IS NOT NULL
        """
        async with with_tenant(customer_id) as conn:
            rows = await conn.fetch(sql, customer_id)
        return [r["source_system"] for r in rows]

    async def _fetch_topup(source: str) -> list[Any]:
        topup_params = [*params, source, per_source_top_k]
        sql = (
            f"{inner_sql}\n              AND d.source_system = ${len(topup_params) - 1}"
            f"\n            ORDER BY {ann_order_sql}"
            f"\n            LIMIT ${len(topup_params)}"
        )
        async with with_tenant(customer_id) as conn:
            await _enable_iterative_scan(conn)
            return await conn.fetch(sql, *topup_params)

    pool_rows, src_list = await asyncio.gather(_fetch_pool(), _fetch_sources())

    counts: dict[str, int] = defaultdict(int)
    for r in pool_rows:
        counts[r["source_system"]] += 1
    short = [s for s in src_list if counts[s] < per_source_top_k]

    topup_rows: list[Any] = []
    if short:
        for batch in await asyncio.gather(*(_fetch_topup(s) for s in short)):
            topup_rows.extend(batch)

    # Merge in Python, mirroring the SQL window this replaces exactly:
    # rank rows within each source by (score DESC, chunk_id), keep at most K
    # per source, then interleave by rank (every source's rank-1 before any
    # source's rank-2 -- see the interleave rationale above), cap at top_k.
    seen: set[str] = set()
    by_source: dict[str, list[Any]] = defaultdict(list)
    for r in [*pool_rows, *topup_rows]:
        # A short source's pool rows reappear in its top-up (the top-up is a
        # superset by construction); first occurrence wins, rows identical.
        if r["chunk_id"] in seen:
            continue
        seen.add(r["chunk_id"])
        by_source[r["source_system"]].append(r)

    ranked: list[tuple[int, float, str, Any]] = []
    for rows_for_source in by_source.values():
        rows_for_source.sort(key=lambda r: (-r["score"], r["chunk_id"]))
        for rank, r in enumerate(rows_for_source[:per_source_top_k], start=1):
            ranked.append((rank, -r["score"], r["chunk_id"], r))
    ranked.sort(key=lambda t: t[:3])
    return [r for _, _, _, r in ranked[:top_k]]


def _to_hits(rows: list[Any]) -> list[VectorHit]:
    return [
        VectorHit(
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
