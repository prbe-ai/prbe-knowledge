"""`sort_by="recency"` keeps an ANN candidate pool (E2).

Before this, the recency branch took NO ANN limit: `_build_inner_query` has
no distance predicate, so the inner query was the whole filtered scope and
the outer `ORDER BY updated_at DESC` returned the newest `top_k` chunks in
the tenant whatever the query said. "latest X" was "latest anything".

Now the inner query is the index's best `top_k * VECTOR_RECENCY_POOL_MULTIPLIER`
by distance and only THAT pool is sorted by time. Pinned at the SQL level,
where the defect lived, with the same recording connection the ANN-shape
tests use.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import Any

import pytest

from engine.retrieval.retrievers import vector as vector_mod
from engine.shared.constants import VECTOR_RECENCY_POOL_MULTIPLIER


class _RecordingConn:
    """Captures the SQL `vector_search` builds without touching a database."""

    def __init__(self) -> None:
        self.sql: str | None = None
        self.params: tuple[Any, ...] = ()
        self.statements: list[str] = []

    async def execute(self, sql: str, *args: Any) -> None:
        self.statements.append(sql)

    async def fetch(self, sql: str, *params: Any) -> list[Any]:
        self.sql = sql
        self.params = params
        return []


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> _RecordingConn:
    conn = _RecordingConn()

    @asynccontextmanager
    async def _fake_with_tenant(customer_id: str):  # type: ignore[no-untyped-def]
        yield conn

    monkeypatch.setattr(vector_mod, "with_tenant", _fake_with_tenant)

    class _FakeEmbedder:
        async def embed_query(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(vector_mod, "get_embedder_v2", lambda: _FakeEmbedder())
    return conn


_INNER_ANN_LIMIT = re.compile(
    r"ORDER BY\s+c\.embedding_v2 <=> \$2::halfvec\s*\n\s*LIMIT\s+\$(\d+)"
)


def _pool_param_index(sql: str) -> int:
    m = _INNER_ANN_LIMIT.search(sql)
    assert m, f"inner query has no distance ORDER BY + LIMIT:\n{sql}"
    return int(m.group(1))


async def test_recency_takes_an_ann_pool_then_sorts_it_by_time(
    recorded: _RecordingConn,
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


async def test_relevance_pool_is_exactly_top_k(recorded: _RecordingConn) -> None:
    """The multiplier applies to recency only; relevance is unchanged."""
    await vector_mod.vector_search(customer_id="c1", query_text="q", top_k=10)
    idx = _pool_param_index(recorded.sql or "")
    assert recorded.params[idx - 1] == 10
    assert "ORDER BY score DESC, chunk_id" in (recorded.sql or "")


async def test_recency_now_enables_the_iterative_scan_like_relevance(
    recorded: _RecordingConn,
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


async def test_recency_per_source_windows_the_pool_not_the_table(
    recorded: _RecordingConn,
) -> None:
    """The per-source recency shape is unchanged (ROW_NUMBER window), but
    the thing it windows over is now the ANN pool."""
    await vector_mod.vector_search(
        customer_id="c1", query_text="q", top_k=10, sort_by="recency",
        per_source_top_k=3,
    )
    sql = recorded.sql or ""
    assert "ROW_NUMBER()" in sql
    idx = _pool_param_index(sql)
    assert recorded.params[idx - 1] == 10 * VECTOR_RECENCY_POOL_MULTIPLIER


def test_multiplier_widens_the_pool() -> None:
    """A multiplier of 1 would make recency identical to relevance's pool
    and re-create the original miss for older on-topic chunks."""
    assert VECTOR_RECENCY_POOL_MULTIPLIER >= 2
