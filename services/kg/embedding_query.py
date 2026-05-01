"""Thin pgvector embedding-query helpers used by the classifier and the
candidates-queue layer-2 dedup. Replaces the originally-planned Pinecone
client wrapper — the repo uses pgvector, not Pinecone, so similarity
queries run as plain SQL via the existing asyncpg path.

Tenant isolation: callers must pass an asyncpg.Connection already inside
``with_tenant(customer_id)``. The helper does NOT enter ``with_tenant()``
itself because the classifier path typically batches multiple queries
inside one transaction and we don't want nested context-manager juggling.
``customer_id`` is also included in the WHERE clause as defense-in-depth
(RLS enforces it but the explicit predicate makes the query
self-documenting and survives an accidental connection-without-GUC).

DB error handling: surfaces an ``EmbeddingQueryError`` on asyncpg-side
failures so the classifier can degrade to rules-only matching
(spec §6 step 2 — what was originally a Pinecone failure fallback is now
a generic DB-error fallback wired into the classifier's connection retry
path).

Vector format: pgvector accepts a string literal of the form
``'[v1,v2,...]'::vector`` as a query parameter. We build the string with
``repr(float(...))`` so ints get coerced to floats and the round-trip
preserves precision (``f"{x}"`` truncates).

See spec §5.1, §6, §7.3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import asyncpg

log = logging.getLogger(__name__)


class EmbeddingQueryError(Exception):
    """Surfaces DB-side failures from embedding similarity queries.

    Classifier callers catch this and fall back to rules-only matching.
    """


@dataclass(frozen=True)
class EmbeddingMatch:
    """A single ``(class_id, score)`` pair from ``query_similar``.

    ``score`` is cosine similarity in ``[0, 1]``, computed as
    ``1 - (signature_embedding <=> q)`` so higher = more similar.
    """

    class_id: str
    score: float


def _format_vector(vec: list[float]) -> str:
    """Render a Python vector as the pgvector string literal ``'[v1,v2,...]'``.

    pgvector parses this format directly when cast via ``$N::vector``.
    ``repr(float(...))`` coerces ints to floats and avoids the precision
    loss of ``f"{x}"`` (which formats with default ``str`` precision).
    """
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"


async def query_similar(
    conn: asyncpg.Connection,
    *,
    customer_id: str,
    vector: list[float],
    top_k: int = 10,
) -> list[EmbeddingMatch]:
    """Return the ``top_k`` most-similar ``kg_classes`` by signature embedding.

    The connection MUST already be inside ``with_tenant(customer_id)`` so
    the RLS policy scopes the query to this tenant.

    score = ``1 - (signature_embedding <=> $1::vector)`` — higher is more
    similar; bounded to ``[0, 1]`` for normalized vectors.

    Args:
        conn: asyncpg connection inside ``with_tenant(customer_id)``.
        customer_id: Tenant key (also enforced by RLS; included as a
            self-documenting predicate).
        vector: 1536-dim embedding (the kg_classes column type is
            ``vector(1536)`` — we don't enforce dim here so callers can
            unit-test with shorter vectors against the same code path).
        top_k: Maximum number of matches to return. Must be > 0.

    Raises:
        ValueError: ``top_k <= 0`` or ``vector`` is empty.
        EmbeddingQueryError: An asyncpg- or socket-level error fired
            during fetch. The classifier catches this and degrades to
            rules-only matching.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not vector:
        raise ValueError("vector must be non-empty")
    literal = _format_vector(vector)
    try:
        rows = await conn.fetch(
            """
            SELECT
                class_id,
                1 - (signature_embedding <=> $1::vector) AS score
            FROM kg_classes
            WHERE customer_id = $2
              AND signature_embedding IS NOT NULL
            ORDER BY signature_embedding <=> $1::vector
            LIMIT $3
            """,
            literal,
            customer_id,
            top_k,
        )
    except (asyncpg.PostgresError, OSError) as e:
        log.warning(
            "kg.embedding_query_failed",
            extra={"customer_id": customer_id, "error": str(e)},
        )
        raise EmbeddingQueryError(str(e)) from e
    return [
        EmbeddingMatch(class_id=r["class_id"], score=float(r["score"])) for r in rows
    ]


async def query_similar_candidates(
    conn: asyncpg.Connection,
    *,
    customer_id: str,
    payload_hash: str,
    vector: list[float],
    threshold: float = 0.85,
    statuses: tuple[str, ...] = ("pending",),
) -> list[tuple[str, float]]:
    """Layer-2 dedup query against ``kg_candidates``.

    Among candidates with the SAME ``payload_hash`` and a ``status`` in
    ``statuses`` (default: pending only), return those whose
    ``notes_embedding`` cosine similarity to ``vector`` is ``>= threshold``.

    Used by the debugging agent's candidate-write path: the payload hash
    is the layer-1 candidate identifier; this query is the layer-2
    confirmation (spec §7.3) — on hash collision we compare notes
    embeddings and only increment ``repeat_count`` when cosine clears
    the threshold.

    Args:
        conn: asyncpg connection inside ``with_tenant(customer_id)``.
        customer_id: Tenant key.
        payload_hash: The layer-1 collision key.
        vector: ``notes_embedding`` for the new candidate.
        threshold: Cosine similarity floor in ``[0, 1]``. Default 0.85
            matches spec §7.3.
        statuses: Allowed candidate statuses. Default ``('pending',)``
            because resolved candidates (accepted/rejected/merged) should
            not be re-deduped against — they re-open a fresh candidate.

    Returns:
        List of ``(candidate_id, score)`` tuples ordered by similarity
        (most similar first). ``candidate_id`` is the UUID rendered as
        text so callers can pass it through JSON without ``uuid.UUID``
        serialization concerns.

    Raises:
        ValueError: ``threshold`` outside ``[0, 1]`` or ``vector`` empty.
        EmbeddingQueryError: An asyncpg- or socket-level error fired
            during fetch.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1]")
    if not vector:
        raise ValueError("vector must be non-empty")
    literal = _format_vector(vector)
    try:
        rows = await conn.fetch(
            """
            SELECT
                candidate_id::text AS candidate_id,
                1 - (notes_embedding <=> $1::vector) AS score
            FROM kg_candidates
            WHERE customer_id = $2
              AND payload_hash = $3
              AND status = ANY($4::text[])
              AND notes_embedding IS NOT NULL
              AND 1 - (notes_embedding <=> $1::vector) >= $5
            ORDER BY notes_embedding <=> $1::vector
            """,
            literal,
            customer_id,
            payload_hash,
            list(statuses),
            threshold,
        )
    except (asyncpg.PostgresError, OSError) as e:
        log.warning(
            "kg.candidate_dedup_query_failed",
            extra={
                "customer_id": customer_id,
                "payload_hash": payload_hash,
                "error": str(e),
            },
        )
        raise EmbeddingQueryError(str(e)) from e
    return [(r["candidate_id"], float(r["score"])) for r in rows]
