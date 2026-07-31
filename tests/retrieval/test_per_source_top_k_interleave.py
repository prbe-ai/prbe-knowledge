"""per_source_top_k must survive the final LIMIT.

The window function already gave each source_system its own slot budget, and
then `ORDER BY score DESC LIMIT top_k` handed every slot back to whichever
source scores highest in the absolute -- undoing the guarantee one line after
computing it.

That is not a ranking nicety, it silently deletes a corpus. Measured on the
research cluster before the fix, on a keyed+keyless request with
per_source_top_k=20 and top_k=30:

    {github: 4, claude_code: 20, code_graph: 6}     <- custom_ingest ABSENT
    top_k=200 showed the same query returning custom_ingest from rank 61

Cosine and ts_rank scores are NOT comparable across sources: a terse structured
projection ("run abc123, status complete") never out-scores a chatty transcript
paragraph on a natural-language query, whatever its relevance. Rank within a
source is the only fair cross-source currency, so the final ORDER BY leads with
`_ps_rn` -- every source's rank-1 before any source's rank-2.

These tests seed sources with deliberately LOPSIDED score potential: `loud`
repeats the query terms so it dominates on raw score, `quiet` mentions them
once. Without interleaving, `quiet` is cut entirely.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engine.retrieval.retrievers.bm25 import bm25_search
from engine.shared.db import raw_conn
from engine.shared.models import TemporalSpec

_NOW = datetime.now(UTC)
_TERMS = "kubernetes helm namespace"

# (source_system, doc suffix, chunk bodies). `loud` wins every head-to-head
# score comparison; `quiet` is relevant but terse.
_SOURCES: list[tuple[str, str, list[str]]] = [
    ("claude_code", "loud", [f"{_TERMS} {_TERMS} {_TERMS} discussion turn {i}" for i in range(40)]),
    ("custom_ingest", "quiet", [f"run record mentioning {_TERMS} once" for _ in range(40)]),
]


def _doc(customer_id: str, suffix: str) -> str:
    return f"{customer_id}-{suffix}"


async def _seed(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )
        for source_system, suffix, chunks in _SOURCES:
            doc_id = _doc(customer_id, suffix)
            await conn.execute(
                """
                INSERT INTO documents (
                    doc_id, version, customer_id,
                    source_system, source_id, source_url,
                    doc_class, doc_type, content_type,
                    content_hash, title, body_size_bytes, body_token_count,
                    created_at, updated_at, valid_from, ingested_at, acl
                ) VALUES (
                    $1, 1, $2, $3, $1, 'https://prbe.ai/' || $1,
                    'raw_source', 'manual_upload.note', 'text/plain',
                    'h-' || $1, $4, 100, 0, $5, $5, $5, $5, '{}'::jsonb
                )
                """,
                doc_id, customer_id, source_system, f"{suffix} doc", _NOW,
            )
            for idx, body in enumerate(chunks):
                await conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, customer_id,
                        chunk_index, content, content_hash, token_count, kind,
                        embedding, first_seen_version, last_seen_version
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, 5, 'content',
                        array_fill(0::real, ARRAY[3072])::halfvec, 1, 1
                    )
                    """,
                    f"{doc_id}-c{idx}", doc_id, customer_id, idx, body,
                    f"chash-{doc_id}-{idx}",
                )


@pytest.mark.asyncio
async def test_per_source_top_k_survives_the_final_limit(live_db) -> None:
    """The regression: a quiet source must not be cut by a global score sort.

    top_k is deliberately smaller than the two sources' combined per-source
    budgets, so the LIMIT has to choose -- which is exactly the condition that
    used to delete a corpus.
    """
    cust = "cust-ps-interleave"
    await _seed(cust)

    hits = await bm25_search(
        cust, _TERMS, top_k=20, temporal=TemporalSpec(mode="all"), per_source_top_k=20
    )
    by_source: dict[str, int] = {}
    for h in hits:
        by_source[h.source_system] = by_source.get(h.source_system, 0) + 1

    assert "claude_code" in by_source, by_source
    assert "custom_ingest" in by_source, (
        f"the quiet source was cut entirely: {by_source} -- the global ORDER BY "
        "score undid the per-source partition"
    )


@pytest.mark.asyncio
async def test_interleave_is_roughly_balanced(live_db) -> None:
    """Neither source may take more than one slot beyond an even split.

    Asserted as a bound rather than an exact split: ties in ts_rank_cd break on
    chunk_id, so the exact interleave point is stable but not worth pinning.
    """
    cust = "cust-ps-balance"
    await _seed(cust)

    hits = await bm25_search(
        cust, _TERMS, top_k=20, temporal=TemporalSpec(mode="all"), per_source_top_k=20
    )
    by_source: dict[str, int] = {}
    for h in hits:
        by_source[h.source_system] = by_source.get(h.source_system, 0) + 1

    assert len(by_source) == 2, by_source
    assert max(by_source.values()) - min(by_source.values()) <= 1, by_source


@pytest.mark.asyncio
async def test_without_per_source_top_k_pure_score_order_is_unchanged(live_db) -> None:
    """The interleave applies ONLY to the per-source path.

    A caller that does not ask for per-source slotting still gets a straight
    score ranking, so this change cannot alter existing single-corpus traffic.
    """
    cust = "cust-ps-nochange"
    await _seed(cust)

    hits = await bm25_search(cust, _TERMS, top_k=10, temporal=TemporalSpec(mode="all"))
    assert hits, "expected the loud source to dominate an unslotted query"
    # The loud source repeats the terms, so it must lead on raw score.
    assert hits[0].source_system == "claude_code"
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), "unslotted path must stay score-ordered"
