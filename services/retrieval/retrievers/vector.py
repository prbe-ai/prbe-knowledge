"""Vector retriever — pgvector HNSW top-k with temporal filtering."""

from __future__ import annotations

from dataclasses import dataclass

from services.retrieval.temporal import build_predicate
from shared.constants import TOP_K_VECTOR
from shared.db import with_tenant
from shared.embeddings import get_embedder
from shared.models import TemporalSpec


@dataclass(slots=True)
class VectorHit:
    chunk_id: str
    doc_id: str
    doc_version: int
    source_system: str
    source_url: str
    title: str | None
    content: str
    score: float


async def vector_search(
    customer_id: str,
    query_text: str,
    top_k: int = TOP_K_VECTOR,
    sources: list[str] | None = None,
    temporal: TemporalSpec | None = None,
) -> list[VectorHit]:
    """Embed `query_text`, ANN-search against chunks, return top_k hits.

    Score is cosine similarity (1 - cosine distance) so higher is better.

    `temporal` controls which versions of each doc are considered. Defaults
    to TemporalSpec() = latest-live.
    """
    embedder = get_embedder()
    query_vec = await embedder.embed_query(query_text)
    literal = "[" + ",".join(f"{x:.7f}" for x in query_vec) + "]"

    spec = temporal or TemporalSpec()

    async with with_tenant(customer_id) as conn:
        params: list = [customer_id, literal, top_k]
        source_filter = ""
        if sources:
            params.append(sources)
            source_filter = f"AND d.source_system = ANY(${len(params)}::text[])"

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
                   c.content,
                   1 - (c.embedding <=> $2::halfvec) AS score
            FROM chunks c
            JOIN documents d
              ON c.doc_id = d.doc_id
             AND d.customer_id = c.customer_id
             AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
            WHERE c.customer_id = $1
              {pred.chunk_sql}
              {pred.doc_sql}
              {source_filter}
            ORDER BY c.embedding <=> $2::halfvec
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
            score=float(r["score"]),
        )
        for r in rows
    ]
