"""Vector retriever — pgvector HNSW top-k with temporal filtering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from services.retrieval.temporal import build_predicate
from shared.constants import TOP_K_VECTOR
from shared.db import with_tenant
from shared.embeddings import get_embedder_v2
from shared.models import TemporalSpec, normalize_author_id


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
    """
    embedder = get_embedder_v2()
    query_vec = await embedder.embed_query(query_text)
    literal = "[" + ",".join(f"{x:.7f}" for x in query_vec) + "]"

    spec = temporal or TemporalSpec()

    async with with_tenant(customer_id) as conn:
        params: list = [customer_id, literal, top_k]
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

        # Default branch hides drafts; visibility filter is a sibling of
        # the existing valid_to predicate. Reviewer surfaces pass
        # include_drafts=True to bypass.
        visibility_filter = (
            ""
            if include_drafts
            else "AND c.visibility = 'approved' AND d.visibility = 'approved'"
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
            ORDER BY c.embedding_v2 <=> $2::halfvec, c.chunk_id
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
