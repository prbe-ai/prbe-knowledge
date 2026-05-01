"""Top-K pgvector similarity search for the classifier path (spec §6 step 2).

Thin async wrapper around ``services.kg.embedding_query.query_similar``.
The classifier orchestrator (Phase 1.5+) catches ``EmbeddingQueryError``
and falls back to rules-only matching against the full class set —
spec §6 step 2's degraded-mode behavior ("if pgvector is down, slow, or
returns empty, classifier proceeds with rules-only against the full
class set"). This module's only job is to be that thin wrapper plus a
named log event the orchestrator can grep for in production.

The wrapper deliberately does *not* implement the degraded-mode policy
itself: deciding when to fall back, when to retry, or when to emit a
``no_match`` belongs in the orchestrator that ties rules + embedding +
tiebreaker together. Keeping this module policy-free keeps the test
surface tiny and the orchestrator's mock boundary stable.

The connection MUST already be inside ``with_tenant(customer_id)`` —
the underlying ``query_similar`` enforces this implicitly via RLS, and
a connection without the GUC set returns zero rows. That looks like
"no matches" to the orchestrator (legitimate empty result) rather than
"GUC not set" (caller bug), so callers MUST own the GUC contract.

Refs: docs/superpowers/specs/2026-04-29-debugging-knowledge-graph-design.md §6.
"""

from __future__ import annotations

import asyncpg

from services.kg.embedding_query import (
    EmbeddingMatch,
    EmbeddingQueryError,
    query_similar,
)
from shared.logging import get_logger

log = get_logger(__name__)


async def rank_by_embedding(
    *,
    conn: asyncpg.Connection,
    customer_id: str,
    vector: list[float],
    top_k: int = 10,
) -> list[EmbeddingMatch]:
    """Return the top-``top_k`` most similar classes from pgvector.

    Args:
        conn: asyncpg connection that has *already* entered
            ``with_tenant(customer_id)`` so RLS scopes the query to this
            tenant. Without that GUC, the query returns zero rows and
            the orchestrator misreads it as a legitimate empty result.
        customer_id: Tenant key (also passed through to ``query_similar``
            as a self-documenting WHERE predicate; RLS enforces it
            independently).
        vector: 1536-dim embedding produced from the incident signature.
        top_k: Maximum number of matches to return. Defaults to 10 to
            match the orchestrator's tiebreaker fan-out.

    Returns:
        List of ``EmbeddingMatch`` ordered by descending cosine
        similarity. Empty list if no classes have a non-null
        ``signature_embedding`` for this tenant.

    Raises:
        EmbeddingQueryError: pgvector / asyncpg failure. Caller (the
            classifier orchestrator) catches this to enter degraded
            mode. A structured ``kg.classifier.embedding_unavailable``
            warning is emitted before the re-raise so operators have a
            single grep-target for "we entered degraded mode".
    """
    try:
        return await query_similar(
            conn,
            customer_id=customer_id,
            vector=vector,
            top_k=top_k,
        )
    except EmbeddingQueryError as e:
        log.warning(
            "kg.classifier.embedding_unavailable",
            customer_id=customer_id,
            error=str(e),
        )
        raise
