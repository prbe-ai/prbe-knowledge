"""Vector retriever — pgvector HNSW top-k with temporal filtering."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import asyncpg

from engine.retrieval.helpers import origin_of, project_scope_predicate, source_key_predicate
from engine.retrieval.temporal import build_predicate, live_version_join
from engine.shared.constants import TOP_K_VECTOR, VECTOR_RECENCY_POOL_MULTIPLIER
from engine.shared.db import with_tenant
from engine.shared.embeddings import get_embedder_v2
from engine.shared.logging import get_logger
from engine.shared.models import TemporalMode, TemporalSpec, normalize_author_id

log = get_logger(__name__)

# Global ANN pool size for the per-source path's first phase. Sized from a live
# measurement on the research plane (792k chunks, 2026-08-26): LIMIT 400 ran in
# 456 ms through the HNSW index, and 400 comfortably covers every source's
# quota (per_source_top_k caps at 50, and tenants carry a handful of sources).
# The floor exists so a small top_k cannot shrink the pool below usefulness --
# the pool's whole job is to satisfy MOST sources' quotas so the per-source
# top-up phase has little or nothing to do.
PER_SOURCE_ANN_POOL = 400

# Process-wide ceiling on concurrent ANN statements from the per-source path.
#
# Why it exists, measured on the research plane 2026-08-26: the pre-fan-out
# runs up to 4 reformulated sub-queries concurrently, each sub-query's vector
# channel now issues 1 pool + up to N top-up statements, and the engine's two
# replicas double that again -- ~20+ simultaneous ANN scans against a Postgres
# pod with 2 vCPUs and a 6.2GB HNSW index over 1GB of shared_buffers. The
# per-channel timing shipped alongside this path made the effect legible:
# channel_total (summed work) 214s vs channel_max (wall) 74s on one request --
# every statement individually index-shaped and individually fast, all of them
# thrashing the same cache and cores.
#
# Four keeps a single sub-query's pool + a few top-ups flowing while forcing
# the cross-sub-query storm to queue. Queueing is the point: on an
# oversubscribed database, admission control beats parallelism -- the same
# statements complete sooner in fours than in twenty-fours.
#
# Module-level (per process, not per request) because the storm IS
# cross-request: the four sub-queries arrive as concurrent tasks in one
# process, and bounding each request separately would bound nothing.
#
# SIX since 2026-08-30, re-priced for a database that no longer matches the
# one the four was measured on. The 2026-08-26 sizing was against 2 vCPUs,
# 1GB shared_buffers and a 6.2GB HNSW index mostly on disk -- admission
# control was standing in for capacity. The database now runs 3.5 vCPUs
# (research-os #1175) with the index resident on a 24Gi node (#1143), so
# statements are CPU-shaped and short, and width 4 left cores idle while
# top-ups queued: the vector channel measured ~5.5s under a 4-sub-query
# burst with individual statements at ~1-2s. Six admits one more pool + a
# top-up concurrently without re-creating the 24-way storm the four was
# built against. Re-measure, not raise, if the index outgrows memory again
# -- on a disk-bound database four was the better number. COUPLED KNOB:
# SEARCH_AGENT_PREFANOUT_MAX_SUBQUERIES (constants.py) sets how many
# sub-queries feed this gate; its documented rollback to 4 restores the old
# fan-out volume through THIS wider gate, so on a disk-bound database the
# two must be considered together, not flipped independently.
_ANN_STATEMENT_SEMAPHORE = asyncio.Semaphore(6)

# Per-source TOP-UP routing: a short source whose tenant holds at most this
# many live, embedded chunks is answered EXACTLY (distance computed for every
# one of its chunks, no HNSW); a bigger one keeps the HNSW top-up.
#
# Why, measured on the research plane 2026-10-02 (EXPLAIN ANALYZE as `app`,
# cold cache): the source filter lives on the JOINed documents row, so an
# HNSW top-up walks the tenant's whole graph until enough of ITS rows survive
# the join, and a source is only ever topped up because it was rare near the
# query. So the walk runs long and finds the wrong rows:
#   - new-workspace/codex, a source that tenant does not have (the planner
#     priced it at 2,076 docs): 18,935 tuples, 22.4 s, 0 rows. Exact: 25 ms.
#   - probe/custom_ingest (30,431 chunks): 6,416-20,096 tuples, 8.1-26.0 s
#     cold (0.31 s when the same walk is repeated warm); of the 20 rows one
#     walk returned, 3 were in the source's true top 20. Exact: 9.1-10.9 s
#     cold, 0.77-0.92 s warm (372k buffer hits), and it IS the true top 20.
#     Exact reads the same pages every time, so in steady state it is warm.
#   - bucket-robotics/codex (10,956): 10.4 s vs exact 2.4 s, both cold.
# On a probe-sized rig (443k chunks) the custom_ingest top-up was p50 5.5 s
# with recall@20 0.05; exact, p50 0.4 s with recall 1.0.
#
# 40,000 covers every short source measured above. Above it exact stops
# paying even warm (~12 buffer hits a row, TOAST-bound): probe/github (58k)
# answered one walk in 130 ms, though another took 10.7 s -- the walk bound
# below is what handles those.
#
# The planner cannot make this call itself: `documents` is not partitioned,
# so it multiplies the tenant's and the source's selectivities independently
# and prices an absent source as thousands of rows.
PER_SOURCE_EXACT_MAX_CHUNKS = 40_000

# Bound on one HNSW top-up's walk (`hnsw.max_scan_tuples`, pgvector default
# 20,000), so one statement cannot run for tens of seconds: 5,000 measured
# 7.6-9.4 s cold where 20,000 measured 25-26 s (probe/custom_ingest). Only a
# source above PER_SOURCE_EXACT_MAX_CHUNKS that is sparse around the query
# reaches it. A capped walk returns a PREFIX of what the uncapped walk would
# have found, and gives up the rows found last -- rows that were rarely in
# the source's true top-K anyway. On the rig, recall@20 against exact for
# the capped sources: codex 0.20 -> 0.09, claude_code 0.01 -> 0.00, github
# 0.00 -> 0.00; across every short source, with the exact route above,
# 0.589 today -> 0.699.
PER_SOURCE_TOPUP_MAX_SCAN_TUPLES = 5_000

# How long a tenant's per-source live-chunk count is trusted before it is
# recounted. Routing only ever picks between "exact" (always correct) and
# today's HNSW statement, so a stale count can cost speed, never correctness.
SOURCE_SIZE_TTL_SECONDS = 600.0

#: (customer_id, source_system) -> (expires_at monotonic, live chunk count,
#: capped at PER_SOURCE_EXACT_MAX_CHUNKS + 1). Kept after it expires: a stale
#: count still routes while its recount runs.
_SOURCE_SIZE_CACHE: dict[tuple[str, str], tuple[float, int]] = {}
#: Counts in flight, so concurrent sub-queries do not start the same one twice.
_SOURCE_SIZE_INFLIGHT: set[tuple[str, str]] = set()
#: Strong references to the background count tasks (the event loop keeps only
#: weak ones); tests await them.
_SOURCE_SIZE_TASKS: set[asyncio.Task[None]] = set()


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
    # Who wrote the TEXT: "human", "generated", or None when the document does
    # not say. Selected here rather than derived downstream because only the
    # retriever has the document row -- a consumer holding a hit has no way to
    # look it up, and guessing is the failure this field exists to prevent.
    # None renders as ABSENT, never as "human": claiming a person wrote
    # something we cannot attribute is worse than admitting we do not know.
    origin: str | None = None


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
    project_id: str | None = None,
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

    `sort_by="recency"` keeps the ANN candidate pool (the index's best
    `top_k * VECTOR_RECENCY_POOL_MULTIPLIER` by distance) and orders THAT
    pool by `updated_at DESC, chunk_id`. Before this the recency path took
    no ANN LIMIT at all: with no distance predicate in the inner query it
    returned the newest `top_k` chunks in scope regardless of the query --
    "latest X" became "latest anything". Used by the gatherer when the
    extractor flagged temporal intent. `vector.recency_pool_short` logs when
    the pool held fewer rows than requested.

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
        project_id=project_id,
        spec=spec,
        include_drafts=include_drafts,
        sort_by=sort_by,
    )

    # The per-source guarantee gets its own strategy on BOTH sorts: a
    # bounded ANN pool plus per-source ANN top-ups, all through the index.
    # For recency the pool and top-ups are still distance-ordered (the only
    # order the HNSW index serves); only the per-source ranking that hands
    # out the K slots switches to updated_at. A global recency pool alone
    # would let one loud source's 180 nearest chunks starve a quiet one.
    if per_source_top_k is not None:
        rows = await _per_source_ann_search(
            customer_id=customer_id,
            inner_sql=inner_sql,
            params=params,
            ann_order_sql=ann_order_sql,
            top_k=top_k,
            per_source_top_k=per_source_top_k,
            sources=sources,
            rank_by=sort_by,
            # The size count reads LIVE chunks of LIVE documents, which bounds
            # what these two modes can match. AS_OF / ALL also match closed
            # chunks, which the count does not see -- and have no ANN index
            # to escape from anyway (see the HNSW index's comment in schema.sql).
            exact_eligible=spec.mode in (TemporalMode.LATEST, TemporalMode.CHANGED_BETWEEN),
        )
        return _to_hits(rows)

    async with _ANN_STATEMENT_SEMAPHORE, with_tenant(customer_id) as conn:
        # Selective post-filter mitigation (see docstring). This used to be
        # gated on `source_keys`, which under-scoped it: pgvector applies
        # EVERY filter after the ANN scan, and the visibility filter below is
        # unconditional (include_drafts defaults False). So the under-return
        # this guards against applies to essentially every ANN query, not just
        # keyed ones -- a doc_type or author filter under-returns exactly the
        # same way. Gate on the ANN path itself instead.
        await _enable_iterative_scan(conn)

        # ANN candidate pool. The index returns its best N by distance; every
        # later step (per-source windowing, deterministic tiebreak) runs over
        # that bounded pool rather than the table.
        #
        # recency ALSO takes the ANN LIMIT, just a wider one: ordering by
        # `updated_at` is not a shape the HNSW index can serve, so the pool
        # is the index's best N by distance and the OUTER query sorts that
        # bounded pool by recency. Without the LIMIT the inner query had no
        # distance predicate at all and "latest X" returned the newest
        # chunks in scope whatever X was.
        pool_size = (
            top_k * VECTOR_RECENCY_POOL_MULTIPLIER if sort_by == "recency" else top_k
        )
        params.append(pool_size)
        candidate_sql = (
            f"{inner_sql}\n            ORDER BY {ann_order_sql}"
            f"\n            LIMIT ${len(params)}"
        )

        # Deterministic tiebreak lives HERE, outside the ANN pool, so it
        # sorts at most `pool_size` rows instead of defeating the index.
        sql = f"""
        SELECT chunk_id, doc_id, doc_version, source_system, source_url,
               title, author_id, content, kind, created_at, updated_at, score,
               origin
        FROM ({candidate_sql}) pool
        ORDER BY {outer_order_sql}
        LIMIT $3
        """

        rows = await conn.fetch(sql, *params)

    if sort_by == "recency" and len(rows) < top_k:
        # The ANN pool ran dry before top_k: either the scope is small or a
        # relevant-but-old chunk sat outside the pool. Counted, not guessed.
        log.info(
            "vector.recency_pool_short",
            customer_id=customer_id,
            requested=top_k,
            pool_size=pool_size,
            returned=len(rows),
        )
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
    project_id: str | None,
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
    project_filter = project_scope_predicate(params, project_id, alias="d")

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
                   d.metadata->>'origin' AS origin,
                   1 - (c.embedding_v2 <=> $2::halfvec) AS score
            FROM chunks c
            JOIN documents d
              ON c.doc_id = d.doc_id
             AND d.customer_id = c.customer_id
             {live_version_join("d", "c")}
            WHERE c.customer_id = $1
              AND c.embedding_v2 IS NOT NULL
              {pred.chunk_sql}
              {pred.doc_sql}
              {source_filter}
              {doc_type_filter}
              {visibility_filter}
              {author_filter}
              {source_key_filter}
              {project_filter}
        """
    return inner_sql, params, ann_order_sql, outer_order_sql


#: Set once per process, after the first SET is proven to have taken effect.
#: `None` = not yet checked; the verification costs two extra round trips and
#: runs exactly once, never on the steady-state hot path.
_ITERSCAN_VERIFIED: bool | None = None

_ITERSCAN_GUC = "hnsw.iterative_scan"
_ITERSCAN_WANT = "relaxed_order"


async def _enable_iterative_scan(conn: asyncpg.Connection) -> None:
    """Let a filtered ANN scan widen until the LIMIT is satisfied.

    with_tenant runs inside a transaction, so SET LOCAL scopes to this
    query and cannot leak across a pgbouncer-pooled connection.

    THE SAVEPOINT USED TO BE THE WHOLE GUARD, AND IT GUARDED NOTHING. Its old
    docstring claimed it made "the missing-GUC case (pgvector < 0.8) a soft
    no-op". In fact PostgreSQL accepts ANY assignment to a namespaced
    `foo.bar` setting as a placeholder when the owning library is not loaded --
    proven live against the kb database:

        SET LOCAL hnsw.totally_made_up_guc = 'banana'   -> succeeds, SHOW returns 'banana'
        SET LOCAL hnsw.iterative_scan = 'not_a_real_mode' -> succeeds

    So the except branch could never fire, and a typo, a rename upstream, or a
    value pgvector stopped accepting would silently disable iterative scan --
    turning filtered ANN searches into quiet under-returns with no error and no
    log line anywhere.

    The placeholder IS reconciled once pgvector loads (verified: production
    order yields `relaxed_order`), so the mechanism works. It just was not
    checked. This reads the value back after forcing the library to load, once
    per process, and says so loudly if it did not take.
    """
    global _ITERSCAN_VERIFIED

    await conn.execute("SAVEPOINT iterscan")
    try:
        await conn.execute(f"SET LOCAL {_ITERSCAN_GUC} = '{_ITERSCAN_WANT}'")
        await conn.execute("RELEASE SAVEPOINT iterscan")
    except asyncpg.PostgresError:
        await conn.execute("ROLLBACK TO SAVEPOINT iterscan")
        _ITERSCAN_VERIFIED = False
        log.error(
            "vector.iterative_scan_rejected",
            guc=_ITERSCAN_GUC,
            reason="SET was refused outright; filtered ANN scans will under-return",
        )
        return

    if _ITERSCAN_VERIFIED is False:
        # Keep saying so. The verification itself runs once per process (two
        # round trips), but a single startup log line is the wrong signal for a
        # fault whose entire character is that it is invisible: a pod that came
        # up mis-set would log once and then under-return on every filtered ANN
        # search for days. Re-logging costs nothing -- no query, no round trip --
        # and puts the line next to the searches it is degrading.
        log.error(
            "vector.iterative_scan_not_applied",
            guc=_ITERSCAN_GUC,
            expected=_ITERSCAN_WANT,
            reason="filtered ANN scans are silently under-returning",
        )
        return
    if _ITERSCAN_VERIFIED is True:
        return

    # A namespaced GUC only resolves to its real definition once the owning
    # library is in the session, so touching a vector operator first is what
    # makes the read-back meaningful rather than a second look at the
    # placeholder. Cheap: a literal-on-literal distance, no table involved.
    await conn.execute("SAVEPOINT iterscan_verify")
    try:
        await conn.fetchval("SELECT '[1,0]'::vector <=> '[0,1]'::vector")
        actual = await conn.fetchval(f"SHOW {_ITERSCAN_GUC}")
        await conn.execute("RELEASE SAVEPOINT iterscan_verify")
    except asyncpg.PostgresError as exc:
        await conn.execute("ROLLBACK TO SAVEPOINT iterscan_verify")
        _ITERSCAN_VERIFIED = False
        log.warning(
            "vector.iterative_scan_unverified",
            guc=_ITERSCAN_GUC,
            error=str(exc),
            reason="could not read the setting back; proceeding, but filtered "
            "ANN scans may under-return without saying so",
        )
        return

    _ITERSCAN_VERIFIED = actual == _ITERSCAN_WANT
    if not _ITERSCAN_VERIFIED:
        log.error(
            "vector.iterative_scan_not_applied",
            guc=_ITERSCAN_GUC,
            expected=_ITERSCAN_WANT,
            actual=actual,
            reason="the SET was accepted as a placeholder but pgvector did not "
            "adopt it; filtered ANN scans will silently under-return",
        )
    else:
        log.info("vector.iterative_scan_verified", guc=_ITERSCAN_GUC, value=actual)


_MAX_SCAN_TUPLES_GUC = "hnsw.max_scan_tuples"


async def _bound_topup_walk(conn: asyncpg.Connection) -> None:
    """Cap this transaction's HNSW walk at PER_SOURCE_TOPUP_MAX_SCAN_TUPLES.

    `set_config(..., true)` is SET LOCAL: it ends with with_tenant's
    transaction and cannot leak through the pool. Like iterative_scan this is
    a namespaced pgvector GUC, accepted as a placeholder until pgvector loads
    in the session and adopted when it does -- which happens before any walk,
    because the walk IS pgvector. The live test reads it back as pgvector
    sees it (tests/retrieval/test_vector_per_source_routing_live.py).
    """
    await conn.execute(
        "SELECT set_config($1, $2, true)",
        _MAX_SCAN_TUPLES_GUC,
        str(PER_SOURCE_TOPUP_MAX_SCAN_TUPLES),
    )


def _routes_exact(live_chunks: int | None) -> bool:
    """True when a short source should be answered exactly. `None` (no
    count) keeps the HNSW top-up, which is what ran before this routing."""
    return live_chunks is not None and live_chunks <= PER_SOURCE_EXACT_MAX_CHUNKS


#: Live, embedded chunks per source for one tenant, each count stopped at $3
#: (= PER_SOURCE_EXACT_MAX_CHUNKS + 1) so its cost is bounded by the threshold,
#: not by the source. The docs gate first: a source with >= $3 live documents
#: has at least that many chunks (and is big), and saying so reads only the
#: documents index -- the chunk join runs for the small ones alone. CASE
#: evaluates the chunk subquery lazily, only for sources under the gate.
#: Cost is O(min(source, threshold)) index probes: 5.0 s / 192k buffers for
#: probe's custom_ingest + pi + github on a cold research-plane cache, which
#: is why it runs in the background (`_source_sizes`).
#:
#: Counts LIVE chunks of LIVE documents and ignores the request's other
#: filters (visibility, source_keys, project, doc_type, author): every one of
#: those only removes rows, so the count is an upper bound on what the exact
#: statement will touch, and it is the same for every request, so it caches.
_SOURCE_SIZE_SQL = f"""
    SELECT s AS source_system,
           CASE
             WHEN (SELECT count(*) FROM (
                     SELECT 1 FROM documents d
                     WHERE d.customer_id = $1 AND d.source_system = s
                       AND d.valid_to IS NULL
                     LIMIT $3) docs) >= $3
             THEN $3
             ELSE (SELECT count(*) FROM (
                     SELECT 1 FROM documents d
                     JOIN chunks c
                       ON c.customer_id = d.customer_id
                      AND c.doc_id = d.doc_id
                      {live_version_join("d", "c")}
                     WHERE d.customer_id = $1 AND c.customer_id = $1
                       AND d.source_system = s
                       AND d.valid_to IS NULL AND c.valid_to IS NULL
                       AND c.embedding_v2 IS NOT NULL
                     LIMIT $3) live)
           END AS live_chunks
    FROM unnest($2::text[]) AS s
"""


async def _count_source_sizes(customer_id: str, sources: list[str]) -> dict[str, int]:
    """Run _SOURCE_SIZE_SQL: each source's live, embedded chunk count for this
    tenant, capped at PER_SOURCE_EXACT_MAX_CHUNKS + 1."""
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            _SOURCE_SIZE_SQL, customer_id, sources, PER_SOURCE_EXACT_MAX_CHUNKS + 1
        )
    return {r["source_system"]: int(r["live_chunks"]) for r in rows}


async def _refresh_source_sizes(customer_id: str, sources: list[str]) -> None:
    """Count `sources` and cache the result. Never raises: a failed count
    leaves the old entries (or none), and those sources keep the route they
    had -- for a source never counted, the HNSW top-up that ran before this
    routing existed."""
    try:
        sizes = await _count_source_sizes(customer_id, sources)
    except Exception as exc:  # a background task: nothing above it would see the error
        log.warning(
            "vector.source_sizes_failed",
            customer_id=customer_id,
            sources=sources,
            error=type(exc).__name__,
            reason="routing stays on the previous count, or the HNSW top-up",
        )
        return
    finally:
        for s in sources:
            _SOURCE_SIZE_INFLIGHT.discard((customer_id, s))
    expires = time.monotonic() + SOURCE_SIZE_TTL_SECONDS
    for s, n in sizes.items():
        _SOURCE_SIZE_CACHE[(customer_id, s)] = (expires, n)


def _source_sizes(customer_id: str, sources: list[str]) -> dict[str, int]:
    """The cached live-chunk counts for `sources`, WITHOUT waiting for a count.

    A source with no entry, or an expired one, is (re)counted in the
    background -- one statement for all of them, at most one in flight per
    (tenant, source) -- and an expired entry is still returned meanwhile. The
    count is never on the request path because it is not always cheap: at the
    40k threshold it read 192k buffers / 5.0 s for probe's custom_ingest +
    pi + github on a cold research-plane cache (2026-10-02). A source never
    counted yet is simply absent from the result, which routes it to the HNSW
    top-up -- what every top-up did before this routing.
    """
    out: dict[str, int] = {}
    due: list[str] = []
    now = time.monotonic()
    for s in sources:
        hit = _SOURCE_SIZE_CACHE.get((customer_id, s))
        if hit is not None:
            out[s] = hit[1]
        if (hit is None or hit[0] <= now) and (customer_id, s) not in _SOURCE_SIZE_INFLIGHT:
            due.append(s)
    if due:
        _SOURCE_SIZE_INFLIGHT.update((customer_id, s) for s in due)
        task = asyncio.get_running_loop().create_task(_refresh_source_sizes(customer_id, due))
        _SOURCE_SIZE_TASKS.add(task)
        task.add_done_callback(_SOURCE_SIZE_TASKS.discard)
    return out


async def _per_source_ann_search(
    *,
    customer_id: str,
    inner_sql: str,
    params: list,
    ann_order_sql: str,
    top_k: int,
    per_source_top_k: int,
    sources: list[str] | None,
    rank_by: str = "relevance",
    exact_eligible: bool = False,
) -> list[Any]:
    """The per-source recall guarantee, kept ON the ANN index.

    `rank_by="recency"`: the pool and the top-ups are unchanged (distance-
    ordered, index-served); only the per-source ranking below hands out the
    K slots by `updated_at DESC, chunk_id` instead of score. That keeps the
    quiet-source guarantee on the recency path, which a single global
    recency pool cannot give.

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

    WHICH top-up a short source gets (2026-10-02). Step 2's walk is only
    cheap when the source is common near the query. When it is not, the walk
    runs to `hnsw.max_scan_tuples`: 18,935 tuples / 22.4 s for a source the
    tenant does not even have, 20,096 tuples / 25.3 s for 3 rows of a 30k-
    chunk source (research plane, EXPLAIN ANALYZE as `app`). So each short
    source is routed on its tenant's live-chunk count (`_source_sizes`:
    counted for the short sources only, off the request path, cached per
    process):

      - count <= PER_SOURCE_EXACT_MAX_CHUNKS: EXACT. Same filtered SELECT,
        `AND d.source_system = $s`, ordered by `score DESC, chunk_id` -- an
        order HNSW cannot serve, so the planner walks the documents
        (customer_id, source_system) index into the chunk partition's
        (customer_id, doc_id) index and sorts that source's rows. It returns
        the TRUE top-K for the source: recall can only go up.
      - bigger: the HNSW top-up of step 2, with its walk capped at
        PER_SOURCE_TOPUP_MAX_SCAN_TUPLES instead of 20,000.
      - no count (not counted yet in this process, the count failed, or a
        temporal mode it cannot bound): the HNSW top-up, as before this
        routing existed.

    Failure honesty: iterative scan gives up after max_scan_tuples, so an
    ultra-rare source inside a huge corpus can still under-return -- now only
    for sources above the exact threshold, and at the lower cap. The old
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

    # Parameter slot 3 is each query's own LIMIT. `_build_inner_query` binds
    # $3 to top_k for the single-query paths; here every ANN query has a
    # DIFFERENT limit (pool size, per-source quota), so each swaps its own
    # value into the slot instead of appending a new parameter and leaving $3
    # dangling. Postgres infers a statement's parameter list from the highest
    # $n it references, and a $3 that appears in no expression has no
    # inferable type: the bind fails with `could not determine data type of
    # parameter $3`. That is not hypothetical -- the first deploy of this
    # path did exactly that on every request, and the fake-connection unit
    # tests could not see it because bind-time errors only exist on a real
    # protocol. The live-binding test exists because of this.

    async def _fetch_pool() -> list[Any]:
        pool_params = list(params)
        pool_params[2] = pool_limit
        sql = (
            f"{inner_sql}\n            ORDER BY {ann_order_sql}"
            f"\n            LIMIT $3"
        )
        async with _ANN_STATEMENT_SEMAPHORE, with_tenant(customer_id) as conn:
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
        topup_params = [*params, source]
        topup_params[2] = per_source_top_k
        sql = (
            f"{inner_sql}\n              AND d.source_system = ${len(topup_params)}"
            f"\n            ORDER BY {ann_order_sql}"
            f"\n            LIMIT $3"
        )
        async with _ANN_STATEMENT_SEMAPHORE, with_tenant(customer_id) as conn:
            await _enable_iterative_scan(conn)
            await _bound_topup_walk(conn)
            return await conn.fetch(sql, *topup_params)

    async def _fetch_exact(source: str) -> list[Any]:
        # Same filters as the top-up; only the ORDER BY differs, and that is
        # the point: `score DESC` is not a shape the HNSW index can serve, so
        # no plan can take the walk. Every row of this one small source gets
        # its exact distance and the true top-K comes back, chunk_id breaking
        # ties exactly as the outer query of the other paths does.
        exact_params = [*params, source]
        exact_params[2] = per_source_top_k
        sql = (
            f"{inner_sql}\n              AND d.source_system = ${len(exact_params)}"
            f"\n            ORDER BY score DESC, c.chunk_id"
            f"\n            LIMIT $3"
        )
        async with _ANN_STATEMENT_SEMAPHORE, with_tenant(customer_id) as conn:
            rows: list[Any] = await conn.fetch(sql, *exact_params)
        return rows

    pool_rows, src_list = await asyncio.gather(_fetch_pool(), _fetch_sources())

    counts: dict[str, int] = defaultdict(int)
    for r in pool_rows:
        counts[r["source_system"]] += 1
    short = [s for s in src_list if counts[s] < per_source_top_k]

    topup_rows: list[Any] = []
    if short:
        # Counted only for the sources that need a top-up, in the background,
        # and cached: routing reads whatever count is already there.
        sizes = _source_sizes(customer_id, short) if exact_eligible else {}
        exact = [s for s in short if _routes_exact(sizes.get(s))]
        log.info(
            "vector.per_source_topups",
            customer_id=customer_id,
            exact=exact,
            ann=[s for s in short if s not in exact],
            sizes={s: sizes.get(s) for s in short},
        )
        for batch in await asyncio.gather(
            *(_fetch_exact(s) if s in exact else _fetch_topup(s) for s in short)
        ):
            topup_rows.extend(batch)

    # Merge in Python, mirroring the SQL window this replaces exactly:
    # rank rows within each source by (score DESC, chunk_id), keep at most K
    # per source, then interleave by rank (every source's rank-1 before any
    # source's rank-2 -- see the interleave rationale above), cap at top_k.
    return _rank_per_source(
        [*pool_rows, *topup_rows],
        per_source_top_k=per_source_top_k,
        top_k=top_k,
        rank_by=rank_by,
    )


def _rank_per_source(
    rows: list[Any],
    *,
    per_source_top_k: int,
    top_k: int,
    rank_by: str = "relevance",
) -> list[Any]:
    """Merge pool + top-up rows in Python, mirroring the SQL window this
    replaced: dedupe by chunk_id (a short source's pool rows reappear in its
    top-up -- first occurrence wins, rows identical), rank rows WITHIN each
    source, keep at most K per source, then interleave by rank (every
    source's rank-1 before any source's rank-2 -- see the interleave
    rationale in the per-source docstring), cap at top_k.

    `rank_by="recency"` ranks within a source by `updated_at DESC, chunk_id`
    instead of `score DESC, chunk_id`; the pool that fed it stays distance-
    ordered either way, which is what keeps the quiet-source guarantee on
    the recency path.
    """
    if rank_by == "recency":
        def _key(r: Any) -> tuple[Any, ...]:
            # Newest first; `updated_at` is never NULL on documents.
            return (-r["updated_at"].timestamp(), r["chunk_id"])
    else:
        def _key(r: Any) -> tuple[Any, ...]:
            return (-r["score"], r["chunk_id"])

    seen: set[str] = set()
    by_source: dict[str, list[Any]] = defaultdict(list)
    for r in rows:
        if r["chunk_id"] in seen:
            continue
        seen.add(r["chunk_id"])
        by_source[r["source_system"]].append(r)

    ranked: list[tuple[int, tuple[Any, ...], Any]] = []
    for rows_for_source in by_source.values():
        rows_for_source.sort(key=_key)
        for rank, r in enumerate(rows_for_source[:per_source_top_k], start=1):
            ranked.append((rank, _key(r), r))
    ranked.sort(key=lambda t: t[:2])
    return [r for _, _, r in ranked[:top_k]]


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
            # `.get`-shaped: an inner query that predates this column still
            # returns rows, and a missing key must read as "unknown origin"
            # rather than raising mid-search.
            origin=origin_of(r),
        )
        for r in rows
    ]

