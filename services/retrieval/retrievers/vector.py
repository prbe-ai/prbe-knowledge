"""Vector retriever — pgvector HNSW top-k.

Phase 0 minimum. BM25 + graph + RRF land in Tier 5.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.constants import TOP_K_VECTOR
from shared.db import with_tenant
from shared.embeddings import get_embedder


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
) -> list[VectorHit]:
    """Embed `query_text`, ANN-search against chunks, return top_k hits.

    Score is cosine similarity (1 - cosine distance) so higher is better.
    """
    embedder = get_embedder()
    query_vec = await embedder.embed_query(query_text)
    literal = "[" + ",".join(f"{x:.7f}" for x in query_vec) + "]"

    async with with_tenant(customer_id) as conn:
        source_filter = ""
        params: list = [customer_id, literal, top_k]
        if sources:
            source_filter = "AND d.source_system = ANY($4::text[])"
            params.append(sources)

        rows = await conn.fetch(
            f"""
            SELECT c.chunk_id,
                   c.doc_id,
                   c.doc_version,
                   d.source_system,
                   d.source_url,
                   d.title,
                   c.content,
                   1 - (c.embedding <=> $2::halfvec) AS score
            FROM chunks c
            JOIN documents d
              ON c.doc_id = d.doc_id AND c.doc_version = d.version
            WHERE c.customer_id = $1
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
