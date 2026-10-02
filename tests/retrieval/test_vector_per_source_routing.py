"""Which statement a short source's top-up runs, and the bound on the HNSW one.

THE PROBLEM THIS PINS (research plane, 2026-10-02)
--------------------------------------------------
A per-source top-up filters on the JOINed documents row, so its HNSW walk has
to keep going until enough of that source's rows turn up. For a source that is
rare near the query the walk runs to `hnsw.max_scan_tuples`: new-workspace's
codex top-up -- a source that tenant does not have -- walked 18,935 tuples in
22.4 s to return nothing, while the exact plan answers in 25 ms; probe's
custom_ingest top-up took 8.9 s and 3 of its 20 rows were in the source's true
top 20. The planner cannot see it: `documents` is not partitioned, so it
priced that empty source at 2,076 docs.

So a short source is routed on its tenant's live-chunk count:
  - <= PER_SOURCE_EXACT_MAX_CHUNKS: the EXACT statement, ordered by an
    expression HNSW cannot serve (`score DESC, chunk_id`);
  - bigger: the HNSW top-up, with its walk bounded by
    PER_SOURCE_TOPUP_MAX_SCAN_TUPLES;
  - no count yet: the HNSW top-up, which is what ran before.
The count runs in the background and is cached; a search never waits for it.

These tests use the shape-answering fake from test_vector_per_source_ann.py;
the live half (binding, true top-K, the bound as pgvector reads it, RLS as a
non-superuser) is test_vector_per_source_routing_live.py.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from engine.retrieval.retrievers import vector as vector_mod
from engine.shared.models import TemporalMode, TemporalSpec
from tests.retrieval.test_vector_per_source_ann import (
    _Dispatcher,
    _row,
    _search,
    drain_size_counts,
    install_dispatcher,
)


@pytest.fixture
def db(monkeypatch: pytest.MonkeyPatch) -> _Dispatcher:
    return install_dispatcher(monkeypatch)


SMALL = vector_mod.PER_SOURCE_EXACT_MAX_CHUNKS
BIG = vector_mod.PER_SOURCE_EXACT_MAX_CHUNKS + 1

_ANN_ORDER = re.compile(r"ORDER BY\s+c\.embedding_v2\s+<=>\s+\$2::halfvec\s+LIMIT \$3\s*$")
_EXACT_ORDER = re.compile(r"ORDER BY score DESC, c\.chunk_id\s+LIMIT \$3\s*$")


def _topups(db: _Dispatcher) -> list[tuple[str, tuple[Any, ...]]]:
    return [(s, p) for s, p in db.fetched if "AND d.source_system = $" in s]


def _size_queries(db: _Dispatcher) -> list[tuple[str, tuple[Any, ...]]]:
    return [(s, p) for s, p in db.fetched if "AS live_chunks" in s]


def _two_short_sources(db: _Dispatcher) -> None:
    """github fills its quota (K=2) from the pool; pi and custom_ingest do not."""
    db.pool_rows = [_row("g1", "github", 0.9), _row("g2", "github", 0.8)]
    db.source_rows = ["github", "pi", "custom_ingest"]
    db.topup_rows["pi"] = [_row("p1", "pi", 0.3)]
    db.topup_rows["custom_ingest"] = [_row("c1", "custom_ingest", 0.2)]


async def _counted_search(db: _Dispatcher, **kwargs: Any) -> list[Any]:
    """A search after the tenant's counts are in: the first one schedules
    them (and itself takes the HNSW top-up), this one is routed. Only the
    routed search's statements stay in `db.fetched`."""
    await _search(db, **kwargs)
    db.fetched.clear()
    return await _search(db, **kwargs)


# ============================================================
# Routing
# ============================================================

async def test_small_source_takes_the_exact_statement(db: _Dispatcher) -> None:
    """At or under the threshold the top-up must not be an ANN query at all:
    ordered by an expression the HNSW index cannot serve, LIMITed to the
    source's quota, restricted to that one source."""
    _two_short_sources(db)
    db.sizes = {"pi": 361, "custom_ingest": SMALL}
    hits = await _counted_search(db)

    by_source = {p[-1]: s for s, p in _topups(db)}
    assert set(by_source) == {"pi", "custom_ingest"}
    for source, sql in by_source.items():
        assert _EXACT_ORDER.search(sql), f"{source} did not take the exact statement:\n{sql}"
        assert not _ANN_ORDER.search(sql)
    assert {h.source_system for h in hits} == {"github", "pi", "custom_ingest"}


async def test_big_source_keeps_the_ann_topup(db: _Dispatcher) -> None:
    """One over the threshold keeps today's index-served statement."""
    _two_short_sources(db)
    db.sizes = {"pi": 361, "custom_ingest": BIG}
    await _counted_search(db)

    by_source = {p[-1]: s for s, p in _topups(db)}
    assert _EXACT_ORDER.search(by_source["pi"])
    assert _ANN_ORDER.search(by_source["custom_ingest"]), by_source["custom_ingest"]


async def test_absent_source_is_exact_not_a_walk(db: _Dispatcher) -> None:
    """A requested source the tenant has none of is the worst HNSW case (the
    walk runs to the cap and finds nothing) and the cheapest exact one."""
    db.pool_rows = [_row("g1", "github", 0.9), _row("g2", "github", 0.8)]
    db.sizes = {"codex": 0}
    await _counted_search(db, sources=["github", "codex"])

    (sql, params), = _topups(db)
    assert params[-1] == "codex"
    assert _EXACT_ORDER.search(sql)


async def test_exact_statement_binds_the_quota_and_the_source(db: _Dispatcher) -> None:
    """$3 is the per-source quota and the last parameter is the source -- the
    same slots the ANN top-up uses, so the shared inner query binds both."""
    _two_short_sources(db)
    db.sizes = {"pi": 10, "custom_ingest": 10}
    await _counted_search(db)
    assert len(_topups(db)) == 2
    for sql, params in _topups(db):
        assert _EXACT_ORDER.search(sql)
        assert params[2] == 2  # per_source_top_k
        assert f"AND d.source_system = ${len(params)}" in sql


async def test_first_search_never_waits_for_the_count(db: _Dispatcher, monkeypatch) -> None:
    """The count is not cheap at this threshold (5.0 s cold for probe's three
    usual short sources), so it is never on the request path: the search
    that schedules it answers with the HNSW top-up while it runs."""
    _two_short_sources(db)
    db.sizes = {"pi": 1, "custom_ingest": 1}
    release = asyncio.Event()
    real_fetch = _Dispatcher.fetch

    async def _blocked_count(self: _Dispatcher, sql: str, *params: Any) -> list[Any]:
        if "AS live_chunks" in sql:
            await release.wait()
        return await real_fetch(self, sql, *params)

    monkeypatch.setattr(_Dispatcher, "fetch", _blocked_count)
    hits = await asyncio.wait_for(
        vector_mod.vector_search("cust-1", "q", top_k=30, per_source_top_k=2), timeout=2
    )
    assert {h.source_system for h in hits} == {"github", "pi", "custom_ingest"}
    assert all(_ANN_ORDER.search(s) for s, _ in _topups(db))
    assert vector_mod._SOURCE_SIZE_TASKS, "no count was scheduled"

    release.set()
    await drain_size_counts()
    db.fetched.clear()
    await _search(db)
    assert all(_EXACT_ORDER.search(s) for s, _ in _topups(db))


async def test_failed_count_keeps_the_ann_topup(db: _Dispatcher, monkeypatch) -> None:
    """The count is an optimisation. If it errors, the sources keep the
    statement that ran before this routing existed, and searches answer."""
    _two_short_sources(db)
    real_fetch = _Dispatcher.fetch

    async def _fetch(self: _Dispatcher, sql: str, *params: Any) -> list[Any]:
        if "AS live_chunks" in sql:
            raise asyncpg.exceptions.QueryCanceledError("canceling statement due to statement timeout")
        return await real_fetch(self, sql, *params)

    monkeypatch.setattr(_Dispatcher, "fetch", _fetch)
    hits = await _counted_search(db)

    assert all(_ANN_ORDER.search(s) for s, _ in _topups(db))
    assert {h.source_system for h in hits} == {"github", "pi", "custom_ingest"}
    assert not vector_mod._SOURCE_SIZE_INFLIGHT, "a failed count must not block the next one"


@pytest.mark.parametrize("mode", [TemporalMode.AS_OF, TemporalMode.ALL])
async def test_modes_the_count_cannot_bound_keep_the_ann_topup(db: _Dispatcher, mode) -> None:
    """AS_OF / ALL also match CLOSED chunks, which the live count does not
    see, so it is no bound on the exact statement's work there. No count is
    taken and the top-up is the one that ran before."""
    _two_short_sources(db)
    db.sizes = {"pi": 1, "custom_ingest": 1}
    spec = (
        TemporalSpec(mode=mode, as_of=datetime(2026, 9, 1, tzinfo=UTC))
        if mode == TemporalMode.AS_OF
        else TemporalSpec(mode=mode)
    )
    await _search(db, temporal=spec)
    await _search(db, temporal=spec)

    assert _size_queries(db) == []
    assert all(_ANN_ORDER.search(s) for s, _ in _topups(db))


async def test_changed_between_is_routed(db: _Dispatcher) -> None:
    """CHANGED_BETWEEN reads live chunks of live documents, so the live
    count bounds it and the routing applies."""
    _two_short_sources(db)
    db.sizes = {"pi": 1, "custom_ingest": BIG}
    spec = TemporalSpec(
        mode=TemporalMode.CHANGED_BETWEEN,
        since=datetime(2026, 9, 1, tzinfo=UTC),
        until=datetime(2026, 10, 1, tzinfo=UTC),
    )
    await _counted_search(db, temporal=spec)
    by_source = {p[-1]: s for s, p in _topups(db)}
    assert _EXACT_ORDER.search(by_source["pi"])
    assert _ANN_ORDER.search(by_source["custom_ingest"])


# ============================================================
# The bound on the HNSW walk
# ============================================================

def _record_executes(db: _Dispatcher) -> list[tuple[str, tuple[Any, ...]]]:
    calls: list[tuple[str, tuple[Any, ...]]] = []
    real_execute = db.execute

    async def _execute(sql: str, *args: Any) -> None:
        calls.append((sql, args))
        await real_execute(sql, *args)

    db.execute = _execute  # type: ignore[method-assign]
    return calls


async def test_ann_topup_bounds_its_walk(db: _Dispatcher) -> None:
    """Every HNSW top-up connection sets hnsw.max_scan_tuples to the bound,
    locally (set_config's is_local=true, so it dies with the transaction).
    Without it the walk runs to pgvector's 20,000 -- the 42.9 s statement."""
    _two_short_sources(db)
    db.sizes = {"pi": BIG, "custom_ingest": BIG}
    await _search(db)  # schedules the counts
    calls = _record_executes(db)
    db.fetched.clear()
    await _search(db)

    bounds = [(s, a) for s, a in calls if a and a[0] == "hnsw.max_scan_tuples"]
    assert len(_topups(db)) == 2
    assert [a for _, a in bounds] == [
        ("hnsw.max_scan_tuples", str(vector_mod.PER_SOURCE_TOPUP_MAX_SCAN_TUPLES))
    ] * 2
    assert all("set_config($1, $2, true)" in s for s, _ in bounds)


async def test_uncounted_ann_topup_is_bounded_too(db: _Dispatcher) -> None:
    """Before the count lands, every short source takes the HNSW top-up --
    and that walk is bounded as well."""
    _two_short_sources(db)
    calls = _record_executes(db)
    await vector_mod.vector_search("cust-1", "q", top_k=30, per_source_top_k=2)
    await drain_size_counts()
    assert sum(1 for _, a in calls if a and a[0] == "hnsw.max_scan_tuples") == 2


async def test_pool_and_exact_statements_are_not_bounded(db: _Dispatcher) -> None:
    """The bound is for the top-up walk only: the pool keeps pgvector's
    default (its filter is the whole quota list, so it does not walk far),
    and the exact statement has no walk to bound."""
    _two_short_sources(db)
    db.sizes = {"pi": 1, "custom_ingest": 1}
    await _search(db)
    calls = _record_executes(db)
    await _search(db)
    assert not any("max_scan_tuples" in str(a) for _, a in calls)


async def test_exact_statements_have_their_own_gate(db: _Dispatcher, monkeypatch) -> None:
    """Six small short sources: no more than the exact gate's worth of exact
    statements run at once, each without parallel workers."""
    in_flight = 0
    peak = 0
    real_fetch = _Dispatcher.fetch

    async def _counting(self: _Dispatcher, sql: str, *params: Any) -> list[Any]:
        nonlocal in_flight, peak
        if _EXACT_ORDER.search(sql):
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
        return await real_fetch(self, sql, *params)

    monkeypatch.setattr(_Dispatcher, "fetch", _counting)
    monkeypatch.setattr(vector_mod, "_EXACT_STATEMENT_SEMAPHORE", asyncio.Semaphore(2))
    db.pool_rows = []
    db.source_rows = ["a", "b", "c", "d", "e", "f"]
    db.sizes = dict.fromkeys(db.source_rows, 1)
    await _search(db)
    calls = _record_executes(db)
    db.fetched.clear()
    await _search(db)

    assert len(_topups(db)) == 6 and all(_EXACT_ORDER.search(s) for s, _ in _topups(db))
    assert peak == 2, f"exact gate not applied: {peak} in flight"
    parallel = [a for s, a in calls if "max_parallel_workers_per_gather" in s]
    assert parallel == [(str(vector_mod.PER_SOURCE_EXACT_PARALLEL_WORKERS),)] * 6


def test_exact_gate_is_below_the_ann_gate() -> None:
    assert 0 < vector_mod.PER_SOURCE_EXACT_MAX_CONCURRENT < vector_mod._ANN_STATEMENT_SEMAPHORE._value


def test_the_bound_is_below_pgvectors_default() -> None:
    """A bound at or above 20,000 bounds nothing."""
    assert 0 < vector_mod.PER_SOURCE_TOPUP_MAX_SCAN_TUPLES < 20_000


# ============================================================
# The count: background, one statement, cached, one in flight
# ============================================================

async def test_count_is_one_statement_for_the_short_sources_only(db: _Dispatcher) -> None:
    """github filled its quota from the pool, so it gets no top-up and no
    count; the two short sources are counted together, capped at the
    threshold + 1 so the count's own cost is bounded by the threshold."""
    _two_short_sources(db)
    db.sizes = {"github": BIG, "pi": 1, "custom_ingest": 1}
    await _search(db)

    (sql, params), = _size_queries(db)
    assert sorted(params[1]) == ["custom_ingest", "pi"]
    assert params[2] == vector_mod.PER_SOURCE_EXACT_MAX_CHUNKS + 1
    assert "LIMIT $3" in sql


async def test_no_count_when_nothing_is_short(db: _Dispatcher) -> None:
    db.pool_rows = [_row("g1", "github", 0.9), _row("g2", "github", 0.8)]
    db.source_rows = ["github"]
    await _search(db)
    assert _size_queries(db) == []


def _clock(monkeypatch) -> list[float]:
    clock = [1000.0]
    monkeypatch.setattr(vector_mod.time, "monotonic", lambda: clock[0])
    return clock


async def test_count_is_cached(db: _Dispatcher, monkeypatch) -> None:
    _clock(monkeypatch)
    _two_short_sources(db)
    db.sizes = {"pi": 1, "custom_ingest": 1}
    await _search(db)
    await _search(db)
    await _search(db)
    assert len(_size_queries(db)) == 1, "a fresh count was re-run"


async def test_recount_starts_before_expiry_and_routing_holds(db: _Dispatcher, monkeypatch) -> None:
    """Past SOURCE_SIZE_REFRESH_AFTER_SECONDS (but inside the TTL) the count
    is still used AND a recount starts, so a source in steady use does not
    drop to the HNSW top-up every TTL."""
    clock = _clock(monkeypatch)
    _two_short_sources(db)
    db.sizes = {"pi": 1, "custom_ingest": 1}
    await _search(db)
    clock[0] += vector_mod.SOURCE_SIZE_REFRESH_AFTER_SECONDS + 1
    db.fetched.clear()
    await _search(db)
    assert all(_EXACT_ORDER.search(s) for s, _ in _topups(db))
    assert len(_size_queries(db)) == 1, "no recount ahead of expiry"


async def test_expired_count_never_routes_exact(db: _Dispatcher, monkeypatch) -> None:
    """A count past the TTL is not used: until a new one lands the source
    takes the HNSW top-up, so a source that grew past the threshold cannot
    stay on the exact scan on the strength of an old count."""
    clock = _clock(monkeypatch)
    _two_short_sources(db)
    db.sizes = {"pi": 1, "custom_ingest": 1}
    await _search(db)
    clock[0] += vector_mod.SOURCE_SIZE_TTL_SECONDS + 1
    db.fetched.clear()
    await vector_mod.vector_search("cust-1", "q", top_k=30, per_source_top_k=2)
    assert all(_ANN_ORDER.search(s) for s, _ in _topups(db)), "an expired count routed exact"
    await drain_size_counts()
    assert len(_size_queries(db)) == 1, "an expired count was not recounted"


async def test_source_that_grows_past_the_threshold_leaves_the_exact_scan(
    db: _Dispatcher, monkeypatch
) -> None:
    clock = _clock(monkeypatch)
    _two_short_sources(db)
    db.sizes = {"pi": 1, "custom_ingest": SMALL}
    await _counted_search(db)
    assert _EXACT_ORDER.search({p[-1]: s for s, p in _topups(db)}["custom_ingest"])

    db.sizes["custom_ingest"] = BIG  # it grew
    clock[0] += vector_mod.SOURCE_SIZE_REFRESH_AFTER_SECONDS + 1
    await _search(db)  # still inside the TTL: routes on the old count, recounts
    db.fetched.clear()
    await _search(db)
    by_source = {p[-1]: s for s, p in _topups(db)}
    assert _ANN_ORDER.search(by_source["custom_ingest"]), "the grown source stayed exact"
    assert _EXACT_ORDER.search(by_source["pi"])


async def test_failed_recount_falls_back_and_waits_before_retrying(
    db: _Dispatcher, monkeypatch
) -> None:
    """A failed recount does not keep the old count alive past its TTL, and
    the next searches do not hammer the database with retries."""
    clock = _clock(monkeypatch)
    _two_short_sources(db)
    db.sizes = {"pi": 1, "custom_ingest": 1}
    await _search(db)

    real_fetch = _Dispatcher.fetch

    async def _failing_count(self: _Dispatcher, sql: str, *params: Any) -> list[Any]:
        if "AS live_chunks" in sql:
            self.fetched.append((sql, params))
            raise asyncpg.exceptions.QueryCanceledError("canceling statement due to statement timeout")
        return await real_fetch(self, sql, *params)

    monkeypatch.setattr(_Dispatcher, "fetch", _failing_count)
    clock[0] += vector_mod.SOURCE_SIZE_TTL_SECONDS + 1
    db.fetched.clear()
    for _ in range(3):
        await _search(db)
    assert all(_ANN_ORDER.search(s) for s, _ in _topups(db)), "routed exact on a count that failed to refresh"
    assert len(_size_queries(db)) == 1, "a failed count was retried with no delay"

    clock[0] += vector_mod.SOURCE_SIZE_RETRY_SECONDS + 1
    monkeypatch.setattr(_Dispatcher, "fetch", real_fetch)
    db.fetched.clear()
    await _search(db)
    assert len(_size_queries(db)) == 1, "not retried after the delay"
    db.fetched.clear()
    await _search(db)
    assert all(_EXACT_ORDER.search(s) for s, _ in _topups(db))


async def test_counts_across_a_tenant_burst_are_capped_not_queued(
    db: _Dispatcher, monkeypatch
) -> None:
    """Ten tenants' first searches at once: at most
    SOURCE_SIZE_MAX_CONCURRENT_COUNTS counts run (each holds a pool
    connection); the rest are skipped, not queued, and no search waits."""
    in_flight = 0
    peak = 0
    release = asyncio.Event()
    real_fetch = _Dispatcher.fetch

    async def _slow_count(self: _Dispatcher, sql: str, *params: Any) -> list[Any]:
        nonlocal in_flight, peak
        if "AS live_chunks" in sql:
            in_flight += 1
            peak = max(peak, in_flight)
            await release.wait()
            in_flight -= 1
        return await real_fetch(self, sql, *params)

    monkeypatch.setattr(_Dispatcher, "fetch", _slow_count)
    _two_short_sources(db)
    db.sizes = {"pi": 1, "custom_ingest": 1}
    tenants = [f"cust-{i}" for i in range(10)]
    await asyncio.wait_for(
        asyncio.gather(*(
            vector_mod.vector_search(t, "q", top_k=30, per_source_top_k=2) for t in tenants
        )),
        timeout=2,
    )
    assert len(vector_mod._SOURCE_SIZE_TASKS) == vector_mod.SOURCE_SIZE_MAX_CONCURRENT_COUNTS
    await asyncio.sleep(0)
    assert peak == vector_mod.SOURCE_SIZE_MAX_CONCURRENT_COUNTS
    release.set()
    await drain_size_counts()
    assert len(_size_queries(db)) == vector_mod.SOURCE_SIZE_MAX_CONCURRENT_COUNTS
    # the skipped tenants are counted by their next searches
    for t in tenants:
        await vector_mod.vector_search(t, "q", top_k=30, per_source_top_k=2)
        await drain_size_counts()
    assert sorted({p[0] for _, p in _size_queries(db)}) == sorted(tenants)


async def test_concurrent_searches_share_one_count(db: _Dispatcher, monkeypatch) -> None:
    """The pre-fan-out runs up to four sub-queries of one search at once, all
    missing the cache together; one count answers them all."""
    real_fetch = _Dispatcher.fetch

    async def _slow_count(self: _Dispatcher, sql: str, *params: Any) -> list[Any]:
        if "AS live_chunks" in sql:
            await asyncio.sleep(0.01)  # a real count suspends; let the others arrive
        return await real_fetch(self, sql, *params)

    monkeypatch.setattr(_Dispatcher, "fetch", _slow_count)
    _two_short_sources(db)
    db.sizes = {"pi": 1, "custom_ingest": 1}
    await asyncio.gather(*(_search(db) for _ in range(4)))
    assert len(_size_queries(db)) == 1


async def test_cache_is_per_tenant(db: _Dispatcher) -> None:
    _two_short_sources(db)
    db.sizes = {"pi": 1, "custom_ingest": 1}
    await vector_mod.vector_search("cust-1", "q", top_k=30, per_source_top_k=2)
    await vector_mod.vector_search("cust-2", "q", top_k=30, per_source_top_k=2)
    await drain_size_counts()
    assert sorted(p[0] for _, p in _size_queries(db)) == ["cust-1", "cust-2"]


def test_routes_exact_boundary() -> None:
    assert vector_mod._routes_exact(0)
    assert vector_mod._routes_exact(SMALL)
    assert not vector_mod._routes_exact(BIG)
    assert not vector_mod._routes_exact(None)
