"""Bundle builder for inferred-edge extraction.

A Bundle is a token-budgeted snapshot of content related to an anchor
document. The extractor sends one bundle per LLM call. Bundle contents
are gathered in cheapest-first order and trimmed to fit the token budget.

CRITICAL tenant-isolation rules:
- Every SQL query has an explicit WHERE customer_id = $1.
- with_tenant(customer_id) is used for all DB access so the RLS GUC
  (app.current_customer_id) is set.
- After assembly, every BundleDoc.customer_id is asserted against
  bundle.customer_id. A mismatch means a cross-tenant leak -- we drop
  the offending doc, log loudly, and return an empty bundle (do NOT
  raise; that would block the queue worker).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import asyncpg

from shared.db import with_tenant
from shared.logging import get_logger

log = get_logger(__name__)

# Default token budget per bundle (approx. cl100k tokens).
DEFAULT_TOKEN_BUDGET = 60_000

# Chars-per-token estimate for cheap pre-LLM budget accounting.
# cl100k averages ~4 chars per token on English prose; 3.5 is conservative.
_CHARS_PER_TOKEN = 3.5

# Maximum 1-hop neighbors fetched per anchor.
_MAX_1HOP = 20

# Maximum vector-similar cross-source chunks fetched.
_MAX_VECTOR_SIMILAR = 5

# Maximum same-window chunks fetched.
_MAX_TIME_WINDOW = 10

# Time window in hours for "same-window" gather step.
_TIME_WINDOW_HOURS = 24


def _estimate_tokens(text: str) -> int:
    """Cheap char-count token estimate (no tiktoken dependency)."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


@dataclass(slots=True)
class BundleDoc:
    """One document's worth of content included in a bundle."""

    doc_id: str
    customer_id: str
    source_system: str
    title: str | None
    # Concatenated chunk content included for this doc.
    content: str
    token_count: int


@dataclass
class Bundle:
    """Token-budgeted snapshot sent to the extractor."""

    customer_id: str
    anchor_doc_id: str
    docs: list[BundleDoc] = field(default_factory=list)
    total_tokens: int = 0

    def _append(self, doc: BundleDoc) -> None:
        """Append doc unconditionally and update the running token total.

        Callers MUST check token_budget before calling — there is no internal
        guard. The earlier name `_add` returned `bool`, implying a conditional
        add; that was misleading since it always returned True.
        """
        self.docs.append(doc)
        self.total_tokens += doc.token_count

    @property
    def doc_ids(self) -> set[str]:
        return {d.doc_id for d in self.docs}


async def build_bundle(
    customer_id: str,
    anchor_doc_id: str,
    conn: asyncpg.Connection,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> Bundle:
    """Build a token-budgeted content bundle for `anchor_doc_id`.

    Gather order (cheapest-first):
      1. Anchor doc + its chunks
      2. 1-hop graph neighbors of the anchor (already-linked entities)
      3. Top-N vector-similar chunks from *different* source_systems
      4. Same time-window (+-24 h) docs sharing anchor keywords

    Every query uses explicit WHERE customer_id = $1 AND the connection is
    opened via with_tenant() so RLS is enforced at both layers.

    Returns an empty Bundle (docs=[]) if:
      - The anchor doc does not exist for this customer.
      - Any cross-tenant leak is detected (logged as an error).
    """
    bundle = Bundle(customer_id=customer_id, anchor_doc_id=anchor_doc_id)

    # ------------------------------------------------------------------ #
    # Step 1: anchor doc + its chunks                                      #
    # ------------------------------------------------------------------ #
    anchor_row = await conn.fetchrow(
        """
        SELECT d.doc_id, d.customer_id, d.source_system, d.title
        FROM documents d
        WHERE d.customer_id = $1
          AND d.doc_id = $2
          AND d.valid_to IS NULL
        LIMIT 1
        """,
        customer_id,
        anchor_doc_id,
    )
    if anchor_row is None:
        log.warning(
            "inferred_edges.bundle.anchor_missing",
            customer=customer_id,
            anchor_doc_id=anchor_doc_id,
        )
        return bundle

    anchor_source = anchor_row["source_system"]
    anchor_content = await _fetch_doc_content(conn, customer_id, anchor_doc_id, token_budget)
    if anchor_content:
        anchor_doc = BundleDoc(
            doc_id=anchor_doc_id,
            customer_id=anchor_row["customer_id"],
            source_system=anchor_source,
            title=anchor_row["title"],
            content=anchor_content,
            token_count=_estimate_tokens(anchor_content),
        )
        bundle._append(anchor_doc)

    # ------------------------------------------------------------------ #
    # Step 2: 1-hop graph neighbors                                        #
    # ------------------------------------------------------------------ #
    remaining = token_budget - bundle.total_tokens
    if remaining > 0:
        neighbor_doc_ids = await _fetch_1hop_neighbor_docs(
            conn, customer_id, anchor_doc_id, exclude=bundle.doc_ids
        )
        for doc_id in neighbor_doc_ids:
            if bundle.total_tokens >= token_budget:
                break
            doc_row = await conn.fetchrow(
                """
                SELECT d.doc_id, d.customer_id, d.source_system, d.title
                FROM documents d
                WHERE d.customer_id = $1
                  AND d.doc_id = $2
                  AND d.valid_to IS NULL
                """,
                customer_id,
                doc_id,
            )
            if doc_row is None:
                continue
            content = await _fetch_doc_content(
                conn, customer_id, doc_id, token_budget - bundle.total_tokens
            )
            if content:
                bundle._append(
                    BundleDoc(
                        doc_id=doc_id,
                        customer_id=doc_row["customer_id"],
                        source_system=doc_row["source_system"],
                        title=doc_row["title"],
                        content=content,
                        token_count=_estimate_tokens(content),
                    )
                )

    # ------------------------------------------------------------------ #
    # Step 3: top-N vector-similar cross-source chunks                    #
    # ------------------------------------------------------------------ #
    if bundle.total_tokens < token_budget:
        cross_doc_ids = await _fetch_vector_similar_cross_source(
            conn, customer_id, anchor_doc_id, anchor_source, exclude=bundle.doc_ids
        )
        for doc_id in cross_doc_ids:
            if bundle.total_tokens >= token_budget:
                break
            doc_row = await conn.fetchrow(
                """
                SELECT d.doc_id, d.customer_id, d.source_system, d.title
                FROM documents d
                WHERE d.customer_id = $1
                  AND d.doc_id = $2
                  AND d.valid_to IS NULL
                """,
                customer_id,
                doc_id,
            )
            if doc_row is None:
                continue
            content = await _fetch_doc_content(
                conn, customer_id, doc_id, token_budget - bundle.total_tokens
            )
            if content:
                bundle._append(
                    BundleDoc(
                        doc_id=doc_id,
                        customer_id=doc_row["customer_id"],
                        source_system=doc_row["source_system"],
                        title=doc_row["title"],
                        content=content,
                        token_count=_estimate_tokens(content),
                    )
                )

    # ------------------------------------------------------------------ #
    # Step 4: same time-window docs with overlapping keywords             #
    # ------------------------------------------------------------------ #
    if bundle.total_tokens < token_budget:
        window_doc_ids = await _fetch_time_window_docs(
            conn, customer_id, anchor_doc_id, exclude=bundle.doc_ids
        )
        for doc_id in window_doc_ids:
            if bundle.total_tokens >= token_budget:
                break
            doc_row = await conn.fetchrow(
                """
                SELECT d.doc_id, d.customer_id, d.source_system, d.title
                FROM documents d
                WHERE d.customer_id = $1
                  AND d.doc_id = $2
                  AND d.valid_to IS NULL
                """,
                customer_id,
                doc_id,
            )
            if doc_row is None:
                continue
            content = await _fetch_doc_content(
                conn, customer_id, doc_id, token_budget - bundle.total_tokens
            )
            if content:
                bundle._append(
                    BundleDoc(
                        doc_id=doc_id,
                        customer_id=doc_row["customer_id"],
                        source_system=doc_row["source_system"],
                        title=doc_row["title"],
                        content=content,
                        token_count=_estimate_tokens(content),
                    )
                )

    # ------------------------------------------------------------------ #
    # CRITICAL: cross-tenant leak validator                               #
    # Every doc in the bundle MUST have customer_id == bundle.customer_id.#
    # ------------------------------------------------------------------ #
    clean_docs: list[BundleDoc] = []
    leaked = False
    for doc in bundle.docs:
        if doc.customer_id != customer_id:
            leaked = True
            log.error(
                "inferred_edges.bundle.cross_tenant_leak_detected",
                bundle_customer=customer_id,
                doc_customer=doc.customer_id,
                doc_id=doc.doc_id,
                anchor_doc_id=anchor_doc_id,
            )
        else:
            clean_docs.append(doc)

    if leaked:
        # Drop everything and return an empty bundle. We log loudly above.
        # Do NOT raise -- that would DLQ the queue row.
        bundle.docs = []
        bundle.total_tokens = 0
        return bundle

    bundle.docs = clean_docs
    return bundle


# ---- private helpers --------------------------------------------------------


async def _fetch_doc_content(
    conn: asyncpg.Connection,
    customer_id: str,
    doc_id: str,
    remaining_budget: int,
) -> str:
    """Concatenate live chunks for doc_id, trimmed to remaining_budget tokens."""
    rows = await conn.fetch(
        """
        SELECT c.content, c.token_count
        FROM chunks c
        WHERE c.customer_id = $1
          AND c.doc_id = $2
          AND c.valid_to IS NULL
        ORDER BY c.chunk_index ASC
        """,
        customer_id,
        doc_id,
    )
    if not rows:
        return ""

    parts: list[str] = []
    used = 0
    budget_chars = int(remaining_budget * _CHARS_PER_TOKEN)
    total_chars = 0

    for row in rows:
        content = row["content"]
        total_chars += len(content)
        if total_chars > budget_chars:
            # Trim the last piece to exactly fit
            overflow = total_chars - budget_chars
            trimmed = content[: len(content) - overflow]
            if trimmed:
                parts.append(trimmed)
            break
        parts.append(content)
        used += row["token_count"]

    return "\n".join(parts)


async def _fetch_1hop_neighbor_docs(
    conn: asyncpg.Connection,
    customer_id: str,
    anchor_doc_id: str,
    exclude: set[str],
) -> list[str]:
    """Doc IDs reachable in 1 graph hop from any node that anchors anchor_doc_id."""
    rows = await conn.fetch(
        """
        WITH anchor_nodes AS (
            -- nodes whose doc_id == anchor_doc_id in their properties
            -- (graph_nodes store doc_id in properties->>'doc_id' for Document nodes)
            SELECT gn.node_id
            FROM graph_nodes gn
            WHERE gn.customer_id = $1
              AND gn.label = 'Document'
              AND gn.canonical_id = $2
        ),
        neighbor_nodes AS (
            SELECT DISTINCT CASE
                WHEN ge.from_node_id = an.node_id THEN ge.to_node_id
                ELSE ge.from_node_id
            END AS node_id
            FROM graph_edges ge
            JOIN anchor_nodes an
              ON ge.from_node_id = an.node_id OR ge.to_node_id = an.node_id
            WHERE ge.customer_id = $1
        ),
        neighbor_docs AS (
            SELECT DISTINCT gn.canonical_id AS doc_id
            FROM graph_nodes gn
            JOIN neighbor_nodes nn ON gn.node_id = nn.node_id
            WHERE gn.customer_id = $1
              AND gn.label = 'Document'
        )
        SELECT nd.doc_id
        FROM neighbor_docs nd
        WHERE nd.doc_id <> $2
        LIMIT $3
        """,
        customer_id,
        anchor_doc_id,
        _MAX_1HOP,
    )
    result = [r["doc_id"] for r in rows if r["doc_id"] not in exclude]
    return result


async def _fetch_vector_similar_cross_source(
    conn: asyncpg.Connection,
    customer_id: str,
    anchor_doc_id: str,
    anchor_source: str,
    exclude: set[str],
) -> list[str]:
    """Top-N doc IDs from different source_systems, ordered by embedding similarity.

    Uses pgvector cosine similarity against the anchor's first chunk embedding.
    Falls back to an empty list if no embedding is available or pgvector is
    not installed (the query will raise; we swallow it and return []).

    Wraps the query in a SAVEPOINT so that on failure (pgvector missing,
    zero-vector cosine NaN, halfvec cast error, etc.) we ROLLBACK to the
    savepoint instead of poisoning the outer transaction. Without this,
    a swallowed Python exception still leaves Postgres with an aborted
    transaction, and every subsequent query dies with
    InFailedSQLTransactionError.
    """
    await conn.execute("SAVEPOINT vector_similar")
    try:
        rows = await conn.fetch(
            """
            WITH anchor_embedding AS (
                SELECT c.embedding
                FROM chunks c
                WHERE c.customer_id = $1
                  AND c.doc_id = $2
                  AND c.valid_to IS NULL
                  AND c.embedding IS NOT NULL
                ORDER BY c.chunk_index ASC
                LIMIT 1
            ),
            similar AS (
                SELECT DISTINCT c.doc_id,
                       1 - (c.embedding <=> ae.embedding) AS similarity
                FROM chunks c
                CROSS JOIN anchor_embedding ae
                WHERE c.customer_id = $1
                  AND c.valid_to IS NULL
                  AND c.doc_id <> $2
                ORDER BY similarity DESC
                LIMIT 200
            )
            SELECT s.doc_id
            FROM similar s
            JOIN documents d ON d.doc_id = s.doc_id
                             AND d.customer_id = $1
                             AND d.valid_to IS NULL
                             AND d.source_system <> $3
            LIMIT $4
            """,
            customer_id,
            anchor_doc_id,
            anchor_source,
            _MAX_VECTOR_SIMILAR,
        )
        await conn.execute("RELEASE SAVEPOINT vector_similar")
        return [r["doc_id"] for r in rows if r["doc_id"] not in exclude]
    except Exception as exc:  # pgvector unavailable, zero-vec, or no embedding
        await conn.execute("ROLLBACK TO SAVEPOINT vector_similar")
        await conn.execute("RELEASE SAVEPOINT vector_similar")
        log.debug(
            "inferred_edges.bundle.vector_similar_failed",
            customer=customer_id,
            anchor=anchor_doc_id,
            error=str(exc),
        )
        return []


async def _fetch_time_window_docs(
    conn: asyncpg.Connection,
    customer_id: str,
    anchor_doc_id: str,
    exclude: set[str],
) -> list[str]:
    """Doc IDs within +-24 h of anchor_doc_id.updated_at, same customer."""
    rows = await conn.fetch(
        """
        WITH anchor_ts AS (
            SELECT d.updated_at
            FROM documents d
            WHERE d.customer_id = $1
              AND d.doc_id = $2
              AND d.valid_to IS NULL
            LIMIT 1
        )
        SELECT d.doc_id
        FROM documents d
        CROSS JOIN anchor_ts a
        WHERE d.customer_id = $1
          AND d.doc_id <> $2
          AND d.valid_to IS NULL
          AND d.updated_at BETWEEN a.updated_at - make_interval(hours => $3)
                               AND a.updated_at + make_interval(hours => $3)
        ORDER BY d.updated_at DESC
        LIMIT $4
        """,
        customer_id,
        anchor_doc_id,
        _TIME_WINDOW_HOURS,
        _MAX_TIME_WINDOW,
    )
    return [r["doc_id"] for r in rows if r["doc_id"] not in exclude]


async def build_bundle_with_tenant(
    customer_id: str,
    anchor_doc_id: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> Bundle:
    """Convenience wrapper that acquires its own tenant-scoped connection.

    Use this when you don't already have an open connection. The worker
    calls this from the drain loop.
    """
    async with with_tenant(customer_id) as conn:
        return await build_bundle(customer_id, anchor_doc_id, conn, token_budget)
