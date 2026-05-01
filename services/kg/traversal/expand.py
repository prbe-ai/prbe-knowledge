"""1-hop expansion over the matched class's neighbors (spec §6 step 5).

After the classifier selects a class and ``edge_walk`` queues its
context sources, this step gathers nearby classes the agent might also
want to consult. The neighbor set is the union of the matched class's:

- ``related.analogous_to``
- ``related.overlaps_with``
- ``related.often_confused_with``

Spec §6 step 5 enumerates exactly these three relation types.
``related.regressed_by`` (a debugging-specific edge layered on top in
spec §4.2) points at *change events*, not peer classes, so it is
deliberately **excluded** from the 1-hop expansion union.

The neighbors are then ranked by cosine similarity to the matched
class's ``signature_embedding`` — the **same pgvector similarity
function the classifier uses in step 2**, so neighbor scores are
directly comparable to classifier match scores.

Implementation: one round-trip. A CTE pulls the matched class's vector
inside the same query that ranks neighbors, so we don't pay two network
hops. The connection MUST already be inside ``with_tenant(customer_id)``
— RLS scopes both the CTE and the outer SELECT to this tenant.

Edge cases:

- **Empty union** (no related ids after dedup): return ``[]`` without
  hitting the DB.
- **Matched class has NULL ``signature_embedding``**: the CTE returns
  zero rows -> outer query returns zero rows -> ``[]``. We do **not**
  fall back to lexical matching when the embedding is missing — caller
  can re-run after the next embed pass refreshes the column.
- **None of the related ids exist** in ``kg_classes`` (e.g. they were
  deleted but ``kg_check`` hasn't run since): outer query returns zero
  rows -> ``[]``. Silently dropping dangling refs here is correct;
  ``kg_check`` is the canonical link validator.
- **Related ids exist but have NULL embedding**: filtered out by the
  ``c.signature_embedding IS NOT NULL`` predicate. Same rationale.
- **DB error**: caught, logged at WARNING with a structured event
  (``kg.traversal.expand_failed``), and re-raised as ``ExpandError`` so
  callers have a typed surface for degraded-mode policy.

Refs: docs/superpowers/specs/2026-04-29-debugging-knowledge-graph-design.md §6.
"""

from __future__ import annotations

import asyncpg

from services.kg.embedding_query import EmbeddingMatch
from services.kg.schema import Frontmatter
from shared.logging import get_logger

log = get_logger(__name__)


class ExpandError(Exception):
    """Surfaces DB-side failures from 1-hop expansion queries.

    Mirrors ``EmbeddingQueryError`` so the traversal orchestrator can
    catch a typed exception and decide degraded-mode policy without
    parsing strings.
    """


_EXPAND_SQL = """
WITH base AS (
    SELECT signature_embedding
      FROM kg_classes
     WHERE customer_id = $1
       AND class_id    = $2
       AND signature_embedding IS NOT NULL
)
SELECT
    c.class_id,
    1 - (c.signature_embedding <=> base.signature_embedding) AS score
  FROM kg_classes c, base
 WHERE c.customer_id            = $1
   AND c.class_id               = ANY($3::text[])
   AND c.signature_embedding    IS NOT NULL
 ORDER BY c.signature_embedding <=> base.signature_embedding
 LIMIT $4
"""


async def expand_one_hop(
    *,
    conn: asyncpg.Connection,
    customer_id: str,
    frontmatter: Frontmatter,
    top_k: int = 10,
) -> list[EmbeddingMatch]:
    """1-hop expansion ranked by cosine on ``signature_embedding``.

    The connection MUST already be inside ``with_tenant(customer_id)`` so
    RLS scopes both the CTE that fetches the matched class's vector and
    the outer SELECT that ranks neighbors.

    Args:
        conn: asyncpg connection inside ``with_tenant(customer_id)``.
        customer_id: Tenant key (also a self-documenting WHERE predicate;
            RLS enforces it independently).
        frontmatter: The matched class's frontmatter. Its ``id`` and
            ``related`` fields drive the neighbor union.
        top_k: Maximum number of matches to return. Defaults to 10.

    Returns:
        Up to ``top_k`` ``EmbeddingMatch`` rows ordered most-similar
        first. Empty list when the union is empty, the matched class has
        no stored embedding, or none of the related ids resolve to rows
        with a non-null embedding. Caller is responsible for trimming
        the result to its remaining token budget.

    Raises:
        ExpandError: An asyncpg- or socket-level failure during the
            ranked fetch. A structured ``kg.traversal.expand_failed``
            warning is emitted before the re-raise so operators have a
            single grep-target.
    """
    # Union of the three spec-named relation types. ``regressed_by`` is
    # NOT in the spec's enumerated 1-hop list (§6 step 5), so it stays
    # out — that edge points at change events, not peer classes.
    union: set[str] = set()
    union.update(frontmatter.related.analogous_to)
    union.update(frontmatter.related.overlaps_with)
    union.update(frontmatter.related.often_confused_with)
    union.discard(frontmatter.id)  # drop self-references after dedup

    if not union:
        return []

    related_ids = sorted(union)  # deterministic param for the SQL bind
    try:
        rows = await conn.fetch(
            _EXPAND_SQL,
            customer_id,
            frontmatter.id,
            related_ids,
            top_k,
        )
    except (asyncpg.PostgresError, OSError) as e:
        log.warning(
            "kg.traversal.expand_failed",
            customer_id=customer_id,
            class_id=frontmatter.id,
            error=str(e),
        )
        raise ExpandError(str(e)) from e

    return [
        EmbeddingMatch(class_id=r["class_id"], score=float(r["score"]))
        for r in rows
    ]
