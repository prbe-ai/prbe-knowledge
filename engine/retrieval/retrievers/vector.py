"""Vector retriever — pgvector HNSW top-k with temporal filtering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import asyncpg

from engine.retrieval.temporal import build_predicate
from engine.shared.constants import TOP_K_VECTOR
from engine.shared.db import with_tenant
from engine.shared.embeddings import get_embedder_v2
from engine.shared.models import TemporalSpec, normalize_author_id


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
    keep this cheap). Reviewer-scoped BFF surfaces flip this to True after
    role-checking ``wiki_reviewer``; API-key callers never bypass.

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
    relaxed_order`, pgvector >= 0.8) for source_keys queries so the scan
    keeps widening until enough in-scope rows are found; on older
    pgvector builds the SET fails softly (savepoint rollback) and the
    pre-mitigation under-return behavior remains. relaxed_order may
    return near-ties slightly out of distance order -- acceptable for a
    fused retrieval channel.
    """
    embedder = get_embedder_v2()
    query_vec = await embedder.embed_query(query_text)
    literal = "[" + ",".join(f"{x:.7f}" for x in query_vec) + "]"

    spec = temporal or TemporalSpec()

    async with with_tenant(customer_id) as conn:
        if source_keys:
            # Selective post-filter mitigation (see docstring). with_tenant
            # runs inside a transaction, so SET LOCAL scopes to this query
            # and the savepoint makes the missing-GUC case (pgvector < 0.8)
            # a soft no-op instead of poisoning the transaction.
            await conn.execute("SAVEPOINT iterscan")
            try:
                await conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
                await conn.execute("RELEASE SAVEPOINT iterscan")
            except asyncpg.PostgresError:
                await conn.execute("ROLLBACK TO SAVEPOINT iterscan")

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

        # Default branch hides drafts; visibility filter is a sibling of
        # the existing valid_to predicate. Reviewer surfaces pass
        # include_drafts=True to bypass.
        visibility_filter = (
            ""
            if include_drafts
            else "AND c.visibility = 'approved' AND d.visibility = 'approved'"
        )

        # Default: ANN distance ordering (HNSW-indexed, fast). recency: re-order
        # by updated_at DESC across the narrowed pool (slower than the HNSW path
        # but bounded — author/doc_type/temporal filters typically prune by
        # orders of magnitude before this row count matters).
        order_by_sql = (
            "d.updated_at DESC, c.chunk_id"
            if sort_by == "recency"
            else "c.embedding_v2 <=> $2::halfvec, c.chunk_id"
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
            ORDER BY {order_by_sql}
            LIMIT $3
            """,
            *params,
        )

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
