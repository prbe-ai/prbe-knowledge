"""`sort_by="recency"` keeps an ANN candidate pool (E2).

Before this, the recency branch took NO ANN limit: `_build_inner_query` has
no distance predicate, so the inner query was the whole filtered scope and
the outer `ORDER BY updated_at DESC` returned the newest `top_k` chunks in
the tenant whatever the query said. "latest X" was "latest anything".

Now the inner query is the index's best `top_k * VECTOR_RECENCY_POOL_MULTIPLIER`
by distance and only THAT pool is sorted by time. With `per_source_top_k`
the recency path takes the same pool + per-source top-up strategy as
relevance (a single global recency pool would let one loud source starve a
quiet one) and only the per-source ranking switches to `updated_at`.
Pinned at the SQL level with the recording connection from conftest.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pytest
from structlog.testing import capture_logs

from engine.retrieval.retrievers import vector as vector_mod
from engine.shared.constants import VECTOR_RECENCY_POOL_MULTIPLIER
from tests.retrieval.conftest import RecordingConn

_INNER_ANN_LIMIT = re.compile(
    r"ORDER BY\s+c\.embedding_v2 <=> \$2::halfvec\s*\n\s*LIMIT\s+\$(\d+)"
)


def _pool_param_index(sql: str) -> int:
    m = _INNER_ANN_LIMIT.search(sql)
    assert m, f"inner query has no distance ORDER BY + LIMIT:\n{sql}"
    return int(m.group(1))


async def test_recency_takes_an_ann_pool_then_sorts_it_by_time(
    recorded: RecordingConn,
) -> None:
    await vector_mod.vector_search(
        customer_id="c1", query_text="q", top_k=10, sort_by="recency"
    )
    sql = recorded.sql or ""
    # Inner: the HNSW-servable distance order, bounded by the pool param.
    idx = _pool_param_index(sql)
    assert recorded.params[idx - 1] == 10 * VECTOR_RECENCY_POOL_MULTIPLIER
    # Outer: recency over that bounded pool, capped at $3 == top_k.
    assert "ORDER BY updated_at DESC, chunk_id" in sql
    assert sql.rstrip().endswith("LIMIT $3")
    assert recorded.params[2] == 10


async def test_relevance_pool_is_exactly_top_k(recorded: RecordingConn) -> None:
    """The multiplier applies to recency only; relevance is unchanged."""
    await vector_mod.vector_search(customer_id="c1", query_text="q", top_k=10)
    idx = _pool_param_index(recorded.sql or "")
    assert recorded.params[idx - 1] == 10
    assert "ORDER BY score DESC, chunk_id" in (recorded.sql or "")


async def test_recency_now_enables_the_iterative_scan_like_relevance(
    recorded: RecordingConn,
) -> None:
    """The iterative-scan mitigation was gated on `sort_by != "recency"`
    because recency had no ANN path. It has one now, and under-return
    from post-ANN filters applies to it the same way."""
    await vector_mod.vector_search(
        customer_id="c1", query_text="q", top_k=10, sort_by="recency"
    )
    assert any("iterative_scan" in s for s in recorded.statements), (
        recorded.statements
    )


async def test_recency_per_source_uses_the_pool_and_topup_strategy(
    recorded: RecordingConn,
) -> None:
    """recency + per_source_top_k no longer windows one global pool (which
    let a loud source's 180 nearest chunks starve a quiet one); it takes
    the same distance-ordered pool + per-source top-ups as relevance and
    ranks each source's slots by time in Python."""
    await vector_mod.vector_search(
        customer_id="c1", query_text="q", top_k=10, sort_by="recency",
        per_source_top_k=3,
    )
    pool_fetches = [(s, p) for s, p in recorded.fetched if "<=>" in s]
    assert pool_fetches, recorded.fetched
    sql, params = pool_fetches[0]
    assert "ROW_NUMBER()" not in sql
    assert "updated_at DESC" not in sql  # ranking moved to Python
    assert sql.rstrip().endswith("LIMIT $3")
    assert params[2] == max(10, vector_mod.PER_SOURCE_ANN_POOL)


def test_per_source_ranking_by_recency_hands_out_slots_newest_first() -> None:
    """The Python merge for rank_by="recency": within a source, newest first;
    across sources, every source's rank-1 before any rank-2."""
    now = datetime.now(UTC)

    def row(cid: str, src: str, age_days: int, score: float) -> dict:
        return {
            "chunk_id": cid, "source_system": src, "score": score,
            "updated_at": now - timedelta(days=age_days),
        }

    rows = [
        row("a-old", "A", 30, 0.99), row("a-new", "A", 1, 0.50),
        row("b-mid", "B", 10, 0.90), row("b-new", "B", 0, 0.10),
    ]
    out = vector_mod._rank_per_source(rows, per_source_top_k=1, top_k=10, rank_by="recency")
    assert [r["chunk_id"] for r in out] == ["b-new", "a-new"]
    out = vector_mod._rank_per_source(rows, per_source_top_k=1, top_k=10, rank_by="relevance")
    assert [r["chunk_id"] for r in out] == ["a-old", "b-mid"]


async def test_recency_pool_short_is_counted_not_guessed(recorded: RecordingConn) -> None:
    """The recording connection returns no rows, so a recency query is
    always short of top_k: exactly one `vector.recency_pool_short` event,
    with the numbers a dashboard needs."""
    with capture_logs() as logs:
        await vector_mod.vector_search(
            customer_id="c1", query_text="q", top_k=10, sort_by="recency"
        )
    short = [e for e in logs if e.get("event") == "vector.recency_pool_short"]
    assert len(short) == 1, logs
    assert short[0]["requested"] == 10
    assert short[0]["pool_size"] == 10 * VECTOR_RECENCY_POOL_MULTIPLIER
    assert short[0]["returned"] == 0


def test_multiplier_widens_the_pool() -> None:
    """A multiplier of 1 would make recency identical to relevance's pool
    and re-create the original miss for older on-topic chunks."""
    assert VECTOR_RECENCY_POOL_MULTIPLIER >= 2


@pytest.mark.parametrize("sort_by", ["relevance", "recency"])
async def test_single_query_paths_take_the_ann_semaphore(
    recorded: RecordingConn, monkeypatch: pytest.MonkeyPatch, sort_by: str
) -> None:
    """Every ANN statement shares one admission gate; the recency path used
    to run outside it."""
    entered: list[str] = []

    class _Sem:
        async def __aenter__(self) -> None:
            entered.append(sort_by)

        async def __aexit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(vector_mod, "_ANN_STATEMENT_SEMAPHORE", _Sem())
    await vector_mod.vector_search(customer_id="c1", query_text="q", top_k=5, sort_by=sort_by)
    assert entered == [sort_by]
