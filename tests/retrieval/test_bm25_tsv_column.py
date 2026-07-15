"""Regression test for the BM25 -> content_tsv migration (0062).

Two invariants:

1. The GENERATED expression on `chunks.content_tsv` must equal
   `to_tsvector('english', content)` for every row -- if a future migration
   ever changes the expression (e.g. drops the english config, swaps to
   simple, or wraps content in coalesce()), `ts_rank_cd` over the column
   will diverge from the historical expression-based path. Asserted by
   reading the column back and comparing element-for-element.

2. `bm25_search` returns the right result set for a deterministic corpus:
   chunks containing query terms surface, chunks without any matching
   lexeme do not. Order is checked as a containment + top-of-list spot
   check, NOT a strict-pairwise score comparison -- ts_rank_cd's exact
   adjacent-score values can shift with Postgres point releases or
   stemmer/stopword changes, and pinning them produces flaky failures
   on real-world drift that's not actually a bug.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engine.retrieval.retrievers.bm25 import bm25_search
from engine.shared.db import raw_conn
from engine.shared.models import TemporalSpec

# Use wall-clock so seeded docs stay "recent" no matter when CI runs --
# avoids a future temporal-default change quietly hiding the test corpus.
_NOW = datetime.now(UTC)


# Deterministic 5-chunk corpus.
#   c1: both terms, repeated and dense  -> top hit
#   c2: both terms, spread thin
#   c3: one term repeated, dense
#   c4: one term, short
#   c5: neither term -- @@ predicate must filter it out
_CORPUS: list[tuple[str, str]] = [
    ("c1", "auth refactor auth refactor token rotation auth refactor"),
    ("c2", "the auth subsystem needs a refactor before launch"),
    ("c3", "auth auth auth tokens rotated daily"),
    ("c4", "we should plan an auth review"),
    ("c5", "completely unrelated content about deployment pipelines"),
]


async def _seed_corpus(customer_id: str) -> None:
    """Insert one customer + one document + 5 chunks. The chunks all attach
    to the same doc so the BM25 join lands on a single row each."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at, acl
            ) VALUES (
                'doc-bm25-tsv', 1, $1,
                'manual_upload', 'doc-bm25-tsv', 'https://prbe.ai/doc-bm25-tsv',
                'raw_source', 'manual_upload.note', 'text/plain',
                'h-doc-bm25-tsv', 'BM25 tsv test doc', 100, 0,
                $2, $2, $2, $2, '{}'::jsonb
            )
            """,
            customer_id, _NOW,
        )
        for chunk_id, body in _CORPUS:
            await conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, customer_id,
                    chunk_index, content, content_hash, token_count, kind,
                    embedding, first_seen_version, last_seen_version
                ) VALUES (
                    $1, 'doc-bm25-tsv', $2, 0, $3, $4, 5, 'content',
                    array_fill(0::real, ARRAY[3072])::halfvec,
                    1, 1
                )
                """,
                chunk_id, customer_id, body, f"chash-{chunk_id}",
            )


@pytest.mark.asyncio
async def test_content_tsv_column_equals_english_to_tsvector(live_db) -> None:
    """The actual drift detector: every row's stored `content_tsv` must
    equal `to_tsvector('english', content)` byte-for-byte. If a future
    migration ever swaps the GENERATED expression (different config, added
    coalesce, normalization wrapper, etc.), this fails loud. This is what
    the docstring of the file promised -- ranking-snapshot tests cannot
    detect that drift since the column IS the expression by construction
    today, but they CAN flake on legitimate Postgres point-release changes
    to ts_rank_cd."""
    cust = "cust-bm25-tsv-equiv"
    await _seed_corpus(cust)

    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT chunk_id,
                   content_tsv = to_tsvector('english', content) AS matches
            FROM chunks
            WHERE customer_id = $1
            """,
            cust,
        )

    assert rows, "seeded corpus must produce rows"
    for r in rows:
        assert r["matches"] is True, (
            f"chunk {r['chunk_id']}: content_tsv diverged from "
            f"to_tsvector('english', content)"
        )


@pytest.mark.asyncio
async def test_bm25_search_returns_only_chunks_with_matching_lexemes(
    live_db,
) -> None:
    """Containment + spot check: chunks containing query terms surface,
    chunks without any matching lexeme do not, and the densest match is
    rank 1. We deliberately avoid pinning exact adjacent scores -- those
    can shift between Postgres point releases without indicating a bug,
    and a strict-`>` snapshot would flake on real-world drift."""
    cust = "cust-bm25-tsv-filter"
    await _seed_corpus(cust)

    hits = await bm25_search(
        cust,
        "auth refactor",
        top_k=10,
        temporal=TemporalSpec(),
    )

    returned_ids = [h.chunk_id for h in hits]

    # c5 has neither term -> @@ predicate must filter it out.
    assert "c5" not in returned_ids

    # All other chunks contain at least one query term and must surface.
    assert set(returned_ids) == {"c1", "c2", "c3", "c4"}

    # c1 is the densest match (both terms, repeated 3x each) -- it should
    # always be rank 1 regardless of ts_rank_cd's internal tuning. This is
    # a property assertion ("most-relevant doc wins"), not a score snapshot.
    assert hits[0].chunk_id == "c1"
    assert hits[0].score > 0.0

    # All returned hits are content chunks from the seeded doc with
    # positive scores.
    for hit in hits:
        assert hit.doc_id == "doc-bm25-tsv"
        assert hit.kind == "content"
        assert hit.score > 0.0


@pytest.mark.asyncio
async def test_bm25_search_excludes_chunks_with_no_matching_lexeme(
    live_db,
) -> None:
    """The @@ predicate must filter at the index, not just deprioritize.
    A query whose every token is absent from a chunk's content_tsv must
    leave that chunk out of the result set entirely -- otherwise ranking
    quality collapses on large corpora (every chunk gets a zero score and
    LIMIT becomes arbitrary)."""
    cust = "cust-bm25-tsv-filter-only"
    await _seed_corpus(cust)

    hits = await bm25_search(
        cust,
        "deployment",  # only c5 contains this token
        top_k=10,
        temporal=TemporalSpec(),
    )

    assert [h.chunk_id for h in hits] == ["c5"]
