"""Directed-vector retriever — looks up per-doc trigger phrases.

Trigger phrases live in the `directed_vectors` table (one row per phrase
per doc) and are short snippets describing what a wiki page should be
retrievable for. Engineer-pinned (source='human') and LLM-generated
(source='llm') phrases coexist in the same table.

This retriever runs ONE pgvector HNSW lookup against directed_vectors,
collapses to the best (lowest cosine distance) row per doc via DISTINCT
ON, and returns one DirectedHit per matched document. The hits are NOT
chunk-level — fusion treats them as a doc-level booster (analogous to
metadata_score_for_doc) and the phrase text NEVER enters
QueryDocument.chunks. The agent only sees the owning doc's content
chunks; the trigger phrase that surfaced it is reflected via
QueryDocument.retriever_scores['directed'].

A doc that surfaces ONLY via directed (no content chunk in the
candidate pool from vector/bm25/graph) is silently dropped by fusion —
that's the existing rule for any doc-level signal. In practice wiki
pages always have content chunks, so directed-only-misses are not a
concern; the design is to amplify ranking, not to be a sole-source
retriever.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from services.retrieval.temporal import build_predicate
from shared.constants import TOP_K_DIRECTED
from shared.db import with_tenant
from shared.embeddings import get_embedder
from shared.models import TemporalSpec, normalize_author_id


@dataclass(slots=True)
class DirectedHit:
    """One matched document via its best directed-vector trigger phrase.

    `score` is cosine SIMILARITY (1 - cosine distance) so higher is
    better — same convention as VectorHit. `matched_text` is the
    triggering phrase; surfaced for telemetry / debugging only — the
    fusion + response layer never returns it to the caller.
    """

    doc_id: str
    doc_version: int
    source_system: str
    source_url: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    score: float
    matched_text: str
    vector_id: str
    author_id: str | None = None


async def directed_search(
    customer_id: str,
    query_text: str,
    top_k: int = TOP_K_DIRECTED,
    temporal: TemporalSpec | None = None,
) -> list[DirectedHit]:
    """Embed `query_text`, ANN-search directed_vectors, return one hit per doc.

    Multiple directed_vectors rows can match for a single doc; DISTINCT ON
    collapses them to the best (lowest cosine distance) match per doc.
    The retriever returns at most `top_k` documents.

    `temporal` filters by the OWNING document (directed_vectors itself
    has no per-row temporal data — the doc carries it). When `temporal`
    is None the caller gets latest-live doc semantics by default
    (TemporalSpec()).
    """
    embedder = get_embedder()
    query_vec = await embedder.embed_query(query_text)
    literal = "[" + ",".join(f"{x:.7f}" for x in query_vec) + "]"

    spec = temporal or TemporalSpec()

    async with with_tenant(customer_id) as conn:
        params: list = [customer_id, literal, top_k]

        # build_predicate is shared with vector/bm25; its signature
        # requires both aliases. We don't have a chunk join here, so
        # we discard the chunk_sql fragment and apply only doc_sql.
        pred = build_predicate(
            spec, doc_alias="d", chunk_alias="d", next_param_index=len(params) + 1
        )
        params.extend(pred.params)

        # Two-stage query: inner DISTINCT ON collapses to one row per doc
        # (the best phrase for that doc), outer ORDER BY dist + LIMIT picks
        # the top_k closest documents globally. The naive single-stage
        # `DISTINCT ON (doc_id) ... ORDER BY doc_id, dist LIMIT k` returns
        # the first top_k DOC_IDS lexicographically, not the closest docs —
        # silently wrong once the table has more than top_k docs.
        rows = await conn.fetch(
            f"""
            SELECT * FROM (
                SELECT DISTINCT ON (dv.doc_id)
                       dv.vector_id,
                       dv.doc_id,
                       dv.source_text,
                       dv.embedding <=> $2::halfvec AS dist,
                       d.version AS doc_version,
                       d.source_system,
                       d.source_url,
                       d.title,
                       d.author_id,
                       d.created_at,
                       d.updated_at
                FROM directed_vectors dv
                JOIN documents d
                  ON dv.customer_id = d.customer_id
                 AND dv.doc_id      = d.doc_id
                WHERE dv.customer_id = $1
                  {pred.doc_sql}
                ORDER BY dv.doc_id, dv.embedding <=> $2::halfvec ASC
            ) per_doc_best
            ORDER BY dist ASC
            LIMIT $3
            """,
            *params,
        )

    return [
        DirectedHit(
            doc_id=r["doc_id"],
            doc_version=r["doc_version"],
            source_system=r["source_system"],
            source_url=r["source_url"],
            title=r["title"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            # Convert cosine distance -> similarity to match VectorHit's
            # convention (higher is better). Distance is in [0, 2]; clamp
            # at 0 to be safe against float jitter.
            score=max(0.0, 1.0 - float(r["dist"])),
            matched_text=r["source_text"],
            vector_id=str(r["vector_id"]),
            author_id=normalize_author_id(r["author_id"]),
        )
        for r in rows
    ]
