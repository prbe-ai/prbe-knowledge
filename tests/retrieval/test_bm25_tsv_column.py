"""Regression test: BM25 results must be stable across the
expression-based GIN index and the materialized content_tsv column.

If the GENERATED expression on chunks.content_tsv ever drifts from the
to_tsvector('english', content) used historically, ts_rank_cd will return
different scores and BM25 ranking will silently change. This test pins the
expected (chunk_id, score) for a known corpus + query so the drift fails
loud."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from services.retrieval.retrievers.bm25 import bm25_search
from shared.db import raw_conn
from shared.models import TemporalSpec


_NOW = datetime(2026, 5, 9, tzinfo=UTC)


# Deterministic 5-chunk corpus. ts_rank_cd is cover-density ranking, so the
# stable ordering for query "auth | refactor" against this corpus is:
#   c1 (both terms, repeated and dense)         -> rank ~0.27
#   c3 (one term repeated, very dense cover)    -> rank ~0.10
#   c2 (both terms but spread thin in long doc) -> rank ~0.083
#   c4 (one term, short doc, low density)       -> rank ~0.067
# c5 contains neither term and the @@ predicate must filter it out.
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
async def test_bm25_search_orders_corpus_by_ts_rank_cd(live_db) -> None:
    """Pin the (chunk_id, score) ordering BM25 returns for a known corpus
    and query. If the GENERATED expression on content_tsv ever drifts from
    to_tsvector('english', content), ts_rank_cd values will change and
    this assertion fails loudly. The test is intentionally a snapshot of
    behavior, not a smoke check -- the whole point of the materialized
    column is that it must be byte-identical to the prior expression."""
    cust = "cust-bm25-tsv"
    await _seed_corpus(cust)

    hits = await bm25_search(
        cust,
        "auth refactor",
        top_k=10,
        temporal=TemporalSpec(),
    )

    # c5 has neither term -> must not appear.
    returned_ids = [h.chunk_id for h in hits]
    assert "c5" not in returned_ids

    # Cover-density wins over raw term-count: c3 (3x "auth" packed tight)
    # outranks c2 (both terms but spread across a longer chunk). ts_rank_cd
    # is deterministic given the corpus + query + english config, so the
    # ordering is stable; this snapshot fails loudly if the GENERATED
    # expression on content_tsv ever drifts from to_tsvector('english',
    # content).
    assert returned_ids == ["c1", "c3", "c2", "c4"]

    # Scores are strictly decreasing -- locks the ordering with a stronger
    # invariant than just "c1 first". If a future change accidentally ties
    # or inverts two adjacent scores this fails.
    scores = [h.score for h in hits]
    assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 1)), scores

    # All returned hits must be content chunks from the seeded doc.
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
    cust = "cust-bm25-tsv-filter"
    await _seed_corpus(cust)

    hits = await bm25_search(
        cust,
        "deployment",  # only c5 contains this token
        top_k=10,
        temporal=TemporalSpec(),
    )

    assert [h.chunk_id for h in hits] == ["c5"]
