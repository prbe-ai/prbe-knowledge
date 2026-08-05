"""The ANN ORDER BY must stay index-usable.

This is the guard that was missing when the `chunk_id` tiebreaker sat in the
ANN ORDER BY. Every correctness test passed the entire time, because the
results were RIGHT -- they just took 3,355 ms instead of 12.7 ms:

    ORDER BY embedding <=> $2, chunk_id  ->  Parallel Seq Scan + Sort  3,355 ms
    ORDER BY embedding <=> $2            ->  Index Scan (hnsw)            12.7 ms

Measured on the managed data plane against 203,454 rows. It is documented
pgvector behaviour (pgvector#760), not a planner quirk: an ANN index can only
answer "nearest first", so ANY second sort key forces exact distances for
every candidate row.

WHY SHAPE AND NOT `EXPLAIN`
---------------------------
The obvious test is to run EXPLAIN and assert "no Seq Scan". That test would
be actively misleading here, for two reasons:

  1. Postgres correctly chooses a sequential scan on a small table. The seeded
     test corpus is a few dozen rows, so the "bad" plan is the RIGHT plan
     there, and the assertion would only pass via `enable_seqscan = off` --
     which tests the planner override, not our SQL.
  2. A seq scan is legitimate at high selectivity even in production. A blanket
     plan-node prohibition encodes a rule that is false in general.

So this pins the PROPERTY that determines index usability -- the ANN ordering
expression is a single term -- which is deterministic, stats-independent, and
is the exact thing that regressed. The complementary latency check belongs in
a load environment with production-shaped data, not here.

Mutation-verified: putting `, c.chunk_id` back into `ann_order_sql` turns
`test_ann_order_by_is_a_single_expression` red, and removing the outer
tiebreak turns `test_determinism_is_preserved_outside_the_ann_pool` red.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import Any

import pytest

from engine.retrieval.retrievers import vector as vector_mod


class _RecordingConn:
    """Captures the SQL `vector_search` builds without touching a database."""

    def __init__(self) -> None:
        self.sql: str | None = None
        self.params: tuple[Any, ...] = ()
        self.statements: list[str] = []

    async def execute(self, sql: str, *args: Any) -> None:
        # SAVEPOINT / SET LOCAL hnsw.iterative_scan / RELEASE
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


def _ann_order_clause(sql: str) -> str:
    """The ORDER BY that the ANN pool is sorted by."""
    matches = re.findall(r"ORDER BY\s+(.+)", sql)
    ann = [m for m in matches if "<=>" in m]
    assert ann, f"no distance ORDER BY found in:\n{sql}"
    return ann[0].strip()


async def test_ann_order_by_is_a_single_expression(recorded: _RecordingConn) -> None:
    """The regression. A comma here means a second sort key, which means a
    seq scan, which means 3,355 ms."""
    await vector_mod.vector_search(customer_id="c1", query_text="q", top_k=10)

    clause = _ann_order_clause(recorded.sql or "")
    assert "<=>" in clause
    assert "," not in clause, (
        f"ANN ORDER BY carries a second sort key and cannot use the HNSW "
        f"index: {clause!r}"
    )
    assert "chunk_id" not in clause


async def test_determinism_is_preserved_outside_the_ann_pool(
    recorded: _RecordingConn,
) -> None:
    """Removing the tiebreaker must not make ordering nondeterministic.

    It moves OUT of the ANN ordering and into an outer sort over the bounded
    pool. Both halves matter: dropping the tiebreak entirely would trade a
    perf bug for a reproducibility bug.
    """
    await vector_mod.vector_search(customer_id="c1", query_text="q", top_k=10)
    sql = recorded.sql or ""

    assert "chunk_id" in sql
    ann_clause = _ann_order_clause(sql)
    # The tiebreak exists, and it is NOT the ANN clause.
    tiebreaks = [
        m for m in re.findall(r"ORDER BY\s+(.+)", sql) if "chunk_id" in m and "<=>" not in m
    ]
    assert tiebreaks, f"no deterministic tiebreak outside the ANN pool:\n{sql}"
    assert "chunk_id" not in ann_clause


async def test_ann_pool_is_bounded(recorded: _RecordingConn) -> None:
    """The ANN subquery must carry its OWN LIMIT, directly after its ORDER BY.

    Without it the outer sort receives every matching row and the index scan
    buys nothing -- the bounded pool is what makes the outer tiebreak cheap.

    The assertion is deliberately anchored to the ANN ORDER BY rather than
    "a LIMIT appears somewhere after the distance expression". The looser
    version silently passed mutation testing: the OUTER query's `LIMIT $3`
    sits later in the same string, so deleting the pool's LIMIT left the test
    green. Anchoring is the difference between a guard and a decoration.
    """
    await vector_mod.vector_search(customer_id="c1", query_text="q", top_k=10)
    sql = recorded.sql or ""
    assert re.search(
        r"ORDER BY\s+c\.embedding_v2 <=> \$\d+::halfvec\s*\n\s*LIMIT\s+\$\d+",
        sql,
    ), f"ANN ORDER BY is not immediately bounded by its own LIMIT:\n{sql}"


async def test_iterative_scan_enabled_on_the_ann_path(
    recorded: _RecordingConn,
) -> None:
    """Filtered ANN under-returns without it.

    pgvector applies filters AFTER the index scan, so with the default
    ef_search a filter matching ~10% of rows yields ~4 rows instead of top_k.
    This used to be gated on `source_keys` alone, which missed the fact that
    the visibility predicate is unconditional -- i.e. every ANN query is
    filtered.
    """
    await vector_mod.vector_search(customer_id="c1", query_text="q", top_k=10)
    joined = " ".join(recorded.statements)
    assert "hnsw.iterative_scan" in joined
    assert "SET LOCAL" in joined, (
        "must be SET LOCAL, not SET: pgbouncer runs in transaction pooling "
        "mode and a session-level SET would leak onto another tenant's query"
    )


async def test_recency_path_keeps_its_combined_ordering(
    recorded: _RecordingConn,
) -> None:
    """recency cannot use the ANN index by construction, so it is untouched."""
    await vector_mod.vector_search(
        customer_id="c1", query_text="q", top_k=10, sort_by="recency"
    )
    sql = recorded.sql or ""
    assert "updated_at DESC" in sql
