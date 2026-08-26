"""The per-source path must stay ON the ANN index.

THE REGRESSION THIS PINS (2026-08-26)
-------------------------------------
Unified search sends `per_source_top_k` on EVERY request, and the retriever
answered that flag by skipping the ANN LIMIT and windowing the full matching
set -- a Parallel Seq Scan over 626k joined rows + Sort, 37-52 seconds per
query, ~97% of the retrieval stage. Search timed out for every caller while a
comment claimed production used the fast path. Every correctness test passed
throughout, because the results were RIGHT -- the same blind spot
`test_ann_order_by_shape.py` documents for the tiebreak regression, one
branch over.

So these tests pin the SHAPE of what executes, in the same spirit: on the
per-source relevance path, every candidate-producing statement must carry
`ORDER BY <distance> LIMIT` (the only form HNSW can serve) and no statement
may window the un-limited inner query. Plus the merge semantics that replaced
the SQL window, which now live in Python where a unit test can actually reach
them.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from engine.retrieval.retrievers import vector as vector_mod

_TS = datetime(2026, 8, 26, tzinfo=UTC)


def _row(chunk_id: str, source: str, score: float) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": f"doc:{chunk_id}",
        "doc_version": 1,
        "source_system": source,
        "source_url": "https://x",
        "title": None,
        "author_id": None,
        "content": "...",
        "kind": "content",
        "created_at": _TS,
        "updated_at": _TS,
        "score": score,
    }


class _Dispatcher:
    """Answers each SQL by its shape, recording everything that executes.

    The per-source path opens several connections concurrently; a single
    shared dispatcher sees them all, which is exactly what the assertions
    need -- the property under test is "no statement anywhere full-scans".
    """

    def __init__(self) -> None:
        self.pool_rows: list[dict[str, Any]] = []
        self.source_rows: list[str] = []
        self.topup_rows: dict[str, list[dict[str, Any]]] = {}
        self.fetched: list[tuple[str, tuple[Any, ...]]] = []
        self.executed: list[str] = []

    async def execute(self, sql: str, *args: Any) -> None:
        self.executed.append(sql)

    async def fetch(self, sql: str, *params: Any) -> list[Any]:
        self.fetched.append((sql, params))
        if "WITH RECURSIVE" in sql:
            return [{"source_system": s} for s in self.source_rows]
        if "AND d.source_system = $" in sql:
            # top-up: the scalar source equality this path appends (last
            # parameter). Distinct from the caller's hard filter, which
            # spells `= ANY($N::text[])`.
            return self.topup_rows.get(params[-1], [])
        return self.pool_rows


@pytest.fixture
def db(monkeypatch: pytest.MonkeyPatch) -> _Dispatcher:
    dispatcher = _Dispatcher()

    @asynccontextmanager
    async def _fake_with_tenant(customer_id: str):  # type: ignore[no-untyped-def]
        yield dispatcher

    monkeypatch.setattr(vector_mod, "with_tenant", _fake_with_tenant)

    class _Embedder:
        async def embed_query(self, text: str) -> list[float]:
            return [0.25, 0.5]

    monkeypatch.setattr(vector_mod, "get_embedder_v2", lambda: _Embedder())
    return dispatcher


async def _search(db: _Dispatcher, **kwargs: Any) -> list[Any]:
    return await vector_mod.vector_search(
        "cust-1", "q", top_k=30, per_source_top_k=2, **kwargs
    )


# ============================================================
# Shape: every candidate query is index-servable
# ============================================================

async def test_every_candidate_query_is_ann_limited(db: _Dispatcher) -> None:
    """No statement on this path may scan without an ANN ORDER BY + LIMIT.

    This is the assertion that was missing while production full-scanned:
    the WINDOWED shape (`ROW_NUMBER() OVER` around an un-limited inner query)
    must not appear, and every SELECT that touches `chunks` must end in
    `ORDER BY <distance>` + `LIMIT` so HNSW can serve it."""
    db.pool_rows = [_row("c1", "github", 0.9)]
    db.source_rows = ["github"]
    await _search(db)

    candidate_sqls = [s for s, _ in db.fetched if "FROM chunks c" in s]
    assert candidate_sqls, "expected at least the pool query"
    for sql in candidate_sqls:
        assert "ROW_NUMBER()" not in sql, "the windowed full-scan shape is the regression"
        tail = sql[sql.rindex("WHERE") :]
        assert re.search(r"ORDER BY\s+c\.embedding_v2\s+<=>\s+\$2::halfvec\s+LIMIT \$\d+", sql), (
            "candidate query must be ANN-ordered and LIMITed:\n" + tail
        )


async def test_recency_per_source_keeps_the_windowed_shape(db: _Dispatcher) -> None:
    """recency cannot use the ANN index by construction, so its per-source
    branch legitimately keeps the window. Pinned so the fast-path fix cannot
    quietly widen into a path it does not serve."""
    db.pool_rows = [_row("c1", "github", 0.9)]
    await _search(db, sort_by="recency")
    assert any("ROW_NUMBER()" in s for s, _ in db.fetched)


async def test_iterative_scan_is_enabled_on_every_ann_connection(db: _Dispatcher) -> None:
    """Each concurrent ANN query runs on its own pooled connection, and each
    needs its own SET LOCAL -- the GUC does not travel between connections.
    Without it a quiet source's top-up under-returns exactly like the
    source_keys case the docstring documents."""
    db.pool_rows = [_row("c1", "github", 0.9), _row("c2", "github", 0.8)]
    db.source_rows = ["github", "custom_ingest"]
    db.topup_rows["custom_ingest"] = [_row("c9", "custom_ingest", 0.4)]
    await _search(db)

    ann_fetches = sum(1 for s, _ in db.fetched if "FROM chunks c" in s)
    iterscan_sets = sum(1 for s in db.executed if "hnsw.iterative_scan" in s)
    assert ann_fetches == 2  # pool + one top-up
    assert iterscan_sets == ann_fetches


# ============================================================
# Top-ups: only for sources the pool left short
# ============================================================

async def test_no_topups_when_the_pool_satisfies_every_quota(db: _Dispatcher) -> None:
    """A pool row count >= K per source means that source's true top-K is
    already in the pool (it is globally distance-ordered), so no second
    query. The common case costs exactly one ANN round-trip."""
    db.pool_rows = [
        _row("g1", "github", 0.9),
        _row("g2", "github", 0.8),
        _row("s1", "slack", 0.7),
        _row("s2", "slack", 0.6),
    ]
    db.source_rows = ["github", "slack"]
    await _search(db)

    topups = [s for s, _ in db.fetched if "AND d.source_system = $" in s]
    assert topups == []


async def test_topup_runs_only_for_the_short_source(db: _Dispatcher) -> None:
    """The rank-61 case: a quiet source missing from the pool gets its own
    ANN query; the loud one that filled its quota does not."""
    db.pool_rows = [
        _row("g1", "github", 0.9),
        _row("g2", "github", 0.8),
    ]
    db.source_rows = ["github", "custom_ingest"]
    db.topup_rows["custom_ingest"] = [_row("ci1", "custom_ingest", 0.3)]
    hits = await _search(db)

    topup_params = [p for s, p in db.fetched if "AND d.source_system = $" in s]
    assert [p[-1] for p in topup_params] == ["custom_ingest"]
    assert {h.source_system for h in hits} == {"github", "custom_ingest"}


async def test_caller_sources_list_is_the_quota_list(db: _Dispatcher) -> None:
    """When the caller passed `sources`, that list IS the quota set -- no
    discovery query. A source outside it must never get a top-up, because
    the hard filter already excludes its rows."""
    db.pool_rows = [_row("g1", "github", 0.9)]
    await _search(db, sources=["github", "slack"])

    assert not any("WITH RECURSIVE" in s for s, _ in db.fetched)
    topup_params = [p for s, p in db.fetched if "AND d.source_system = $" in s]
    assert sorted(p[-1] for p in topup_params) == ["github", "slack"]


# ============================================================
# Merge: the SQL window's semantics, now in Python
# ============================================================

async def test_interleave_ranks_before_scores(db: _Dispatcher) -> None:
    """Every source's rank-1 lands before any source's rank-2.

    Cosine scores are not comparable across sources -- ordering the merged
    set by score would hand every slot back to the chattiest corpus, undoing
    the entire guarantee."""
    db.pool_rows = [
        _row("g1", "github", 0.9),
        _row("g2", "github", 0.8),
        _row("s1", "slack", 0.2),
        _row("s2", "slack", 0.1),
    ]
    db.source_rows = ["github", "slack"]
    hits = await _search(db)

    assert [h.chunk_id for h in hits] == ["g1", "s1", "g2", "s2"]


async def test_per_source_quota_and_global_cap_hold(db: _Dispatcher) -> None:
    """At most K rows per source survive, and at most top_k rows total."""
    db.pool_rows = [_row(f"g{i}", "github", 0.9 - i / 100) for i in range(5)]
    db.source_rows = ["github"]
    hits = await _search(db)

    assert [h.chunk_id for h in hits] == ["g0", "g1"]  # K=2 of 5


async def test_pool_and_topup_overlap_dedupes(db: _Dispatcher) -> None:
    """A short source's pool rows reappear inside its top-up (the top-up is
    a superset by construction). The duplicate must collapse, or one chunk
    eats two of its source's K slots."""
    shared = _row("ci1", "custom_ingest", 0.5)
    db.pool_rows = [_row("g1", "github", 0.9), _row("g2", "github", 0.8), shared]
    db.source_rows = ["github", "custom_ingest"]
    db.topup_rows["custom_ingest"] = [dict(shared), _row("ci2", "custom_ingest", 0.4)]
    hits = await _search(db)

    ci = [h.chunk_id for h in hits if h.source_system == "custom_ingest"]
    assert ci == ["ci1", "ci2"]


async def test_ann_statements_respect_the_admission_bound(db: _Dispatcher, monkeypatch) -> None:
    """No more than the semaphore's worth of ANN statements in flight.

    Measured 2026-08-26: 4 concurrent sub-queries x (pool + top-ups) x 2
    replicas put ~20+ simultaneous ANN scans on a 2-vCPU Postgres --
    channel_total 214s vs channel_max 74s, pure thrash. Every statement was
    individually index-shaped and fast; the storm was the problem. Admission
    control is the fix, so this pins that the bound is actually applied to
    both statement kinds (pool AND top-up), not just declared."""
    import asyncio as aio

    in_flight = 0
    peak = 0
    real_fetch = _Dispatcher.fetch

    async def _counting_fetch(self: _Dispatcher, sql: str, *params: Any) -> list[Any]:
        nonlocal in_flight, peak
        if "FROM chunks c" in sql:
            in_flight += 1
            peak = max(peak, in_flight)
            await aio.sleep(0.01)  # let the gather actually overlap
            in_flight -= 1
        return await real_fetch(self, sql, *params)

    monkeypatch.setattr(_Dispatcher, "fetch", _counting_fetch)
    monkeypatch.setattr(vector_mod, "_ANN_STATEMENT_SEMAPHORE", vector_mod.asyncio.Semaphore(2))

    db.pool_rows = []
    db.source_rows = ["a", "b", "c", "d", "e", "f"]  # every source short -> 6 top-ups
    await _search(db)

    assert peak <= 2, f"admission bound violated: {peak} ANN statements in flight"
