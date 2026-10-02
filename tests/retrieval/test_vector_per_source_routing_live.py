"""The per-source top-up routing against a real Postgres + pgvector.

The unit half (test_vector_per_source_routing.py) pins which statement runs;
this file proves the statements do what the routing assumes:

  - the exact statement BINDS, returns the TRUE top-K for its source, and no
    plan for it can use the HNSW index (structural, not a cost accident);
  - the live-chunk count counts what the exact statement will read (live,
    embedded chunks of live document versions), stops at its cap, and reads 0
    for a source the tenant does not have;
  - the walk bound is what pgvector actually reads, ends with the
    transaction, and really stops the walk;
  - all of it holds as a NON-superuser under FORCE RLS, which is how
    production runs (`app`). The suite's own role is a superuser and bypasses
    RLS, so a test that only ran as it would prove nothing about isolation.

Needs the standard live database (schema.sql applied); runs on the CI image
`pgvector/pgvector:pg16` (no pg_search needed).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest
import pytest_asyncio

from engine.retrieval.retrievers import vector as vector_mod
from engine.shared import db as db_module
from engine.shared.models import TemporalSpec
from engine.shared.partitions import ensure_tenant_partition

pytestmark = pytest.mark.integration

TENANT = "route-live"
#: last_seen_version of a live chunk (chunks_live_sentinel_chk).
LIVE = 2_147_483_647
OTHER = "route-live-other"
NONSUPER = "prbe_vector_routing_app"
DIM = 3072
K = 5
SOURCES = ["claude_code", "pi", "custom_ingest", "codex"]  # codex: asked for, absent


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _lit(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


RNG = np.random.default_rng(31)
CENTER_BIG = _unit(RNG.standard_normal(DIM))
CENTER_PI = _unit(RNG.standard_normal(DIM))


async def _seed_tenant(conn, tenant: str, *, big: int, pi: int, ci: int) -> None:
    """claude_code clusters around one centre, pi around another, so a query
    near the first leaves pi out of a small pool -- pi becomes SHORT, which is
    the case the routing exists for."""
    await conn.execute(
        "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1,$1,$1)"
        " ON CONFLICT DO NOTHING",
        tenant,
    )
    await ensure_tenant_partition(conn, tenant)
    rows: list[tuple[str, str, str, int, Any, str]] = []
    plan = [("claude_code", big, CENTER_BIG, None), ("pi", pi, CENTER_PI, None),
            ("custom_ingest", ci, CENTER_PI, "artifacts")]
    for source, n, centre, key in plan:
        vecs = _unit(centre + 0.9 * _unit(RNG.standard_normal((n, DIM))))
        for i in range(n):
            doc_id = f"{tenant}:{source}:{i // 3}"
            if i % 3 == 0:
                meta = {"source_key": key if i % 6 == 0 else "elsewhere"} if key else {}
                await conn.execute(
                    """
                    INSERT INTO documents (customer_id, doc_id, version, source_system, source_id,
                                           source_url, doc_type, content_hash, created_at,
                                           updated_at, valid_from, acl, title, metadata)
                    VALUES ($1, $2, 1, $3, $2, 'https://x', $3 || '.doc', 'h', NOW(), NOW(),
                            NOW(), '{}'::jsonb, 't', $4::jsonb)
                    """,
                    tenant, doc_id, source, json.dumps(meta),
                )
            rows.append((tenant, doc_id, f"{doc_id}:c{i}", i, _lit(vecs[i]), source))
    await conn.executemany(
        """
        INSERT INTO chunks (customer_id, doc_id, chunk_id, chunk_index, content, content_hash,
                            token_count, first_seen_version, last_seen_version, embedding_v2,
                            kind, visibility)
        VALUES ($1, $2, $3, $4, 'c', 'h' || $4::int, 1, 1, $6, $5::text::halfvec, 'content', 'approved')
        """,
        [(*r[:5], LIVE) for r in rows],
    )


async def _seed_dead_pi_rows(conn) -> None:
    """Rows the count must NOT see: a closed chunk, a live chunk with no
    embedding yet, and a live document version with no chunk at all (pi:dead
    -- its only chunk belonged to version 1 and was closed with it)."""
    v = _lit(CENTER_PI)
    await conn.execute(
        """
        INSERT INTO chunks (customer_id, doc_id, chunk_id, chunk_index, content, content_hash,
                            token_count, first_seen_version, last_seen_version, embedding_v2,
                            valid_to, kind, visibility)
        VALUES ($1, $1 || ':pi:0', 'closed', 90, 'c', 'h90', 1, 1, 1, $2::text::halfvec, NOW(),
                'content', 'approved'),
               ($1, $1 || ':pi:0', 'unembedded', 91, 'c', 'h91', 1, 1, $3, NULL, NULL,
                'content', 'approved')
        """,
        TENANT, v, LIVE,
    )
    for version, valid_to in ((1, "NOW()"), (2, "NULL")):
        await conn.execute(
            f"""
            INSERT INTO documents (customer_id, doc_id, version, source_system, source_id,
                                   source_url, doc_type, content_hash, created_at, updated_at,
                                   valid_from, valid_to, acl, title)
            VALUES ($1, $1 || ':pi:dead', {version}, 'pi', 'dead', 'https://x', 'pi.doc', 'h',
                    NOW(), NOW(), NOW(), {valid_to}, '{{}}'::jsonb, 't')
            """,
            TENANT,
        )
    await conn.execute(
        """
        INSERT INTO chunks (customer_id, doc_id, chunk_id, chunk_index, content, content_hash,
                            token_count, first_seen_version, last_seen_version, embedding_v2,
                            valid_to, kind, visibility)
        VALUES ($1, $1 || ':pi:dead', 'old-version', 0, 'c', 'hd', 1, 1, 1, $2::text::halfvec,
                NOW(), 'content', 'approved')
        """,
        TENANT, v,
    )


@pytest_asyncio.fixture
async def seeded(live_db, monkeypatch):
    async with db_module.raw_conn() as conn:
        await _seed_tenant(conn, TENANT, big=240, pi=30, ci=30)
        await _seed_dead_pi_rows(conn)
        # Another tenant whose pi rows sit exactly where the query looks:
        # if isolation failed they would win every pi slot.
        await _seed_tenant(conn, OTHER, big=30, pi=30, ci=0)
        await conn.execute("ANALYZE documents")
        await conn.execute("ANALYZE chunks")
    # A small pool so a 300-row tenant leaves pi short, as a 400-row pool
    # leaves it short in a 450k-chunk one.
    monkeypatch.setattr(vector_mod, "PER_SOURCE_ANN_POOL", 10)
    await _fresh_counts(monkeypatch)
    yield
    await _drain()


@pytest_asyncio.fixture
async def as_nonsuper(seeded, monkeypatch):
    """Every vector-channel transaction runs as a NOSUPERUSER NOBYPASSRLS role,
    under the tenant GUC -- production's shape. Yields nothing; the patch is
    the point."""
    async with db_module.raw_conn() as conn:
        await conn.execute(
            f"""
            DO $$ BEGIN
              IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{NONSUPER}') THEN
                CREATE ROLE {NONSUPER} NOLOGIN NOSUPERUSER NOBYPASSRLS;
              END IF;
            END $$;
            """
        )
        await conn.execute(f"GRANT USAGE ON SCHEMA public TO {NONSUPER}")
        await conn.execute(f"GRANT SELECT ON documents, chunks, customers TO {NONSUPER}")
        assert not await conn.fetchval(
            "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = $1", NONSUPER
        )
    real = db_module.with_tenant

    @asynccontextmanager
    async def _wt(customer_id: str):
        async with real(customer_id) as conn:
            await conn.execute(f"SET LOCAL ROLE {NONSUPER}")
            yield conn

    monkeypatch.setattr(vector_mod, "with_tenant", _wt)
    yield


def _embedder(vec: np.ndarray):
    class _E:
        async def embed_query(self, text: str) -> list[float]:
            return [float(x) for x in vec]

    return lambda: _E()


async def _search(monkeypatch, vec: np.ndarray, **kw):
    monkeypatch.setattr(vector_mod, "get_embedder_v2", _embedder(vec))
    return await vector_mod.vector_search(
        TENANT, "q", top_k=len(SOURCES) * K, per_source_top_k=K, sources=SOURCES,
        temporal=TemporalSpec(), **kw
    )


async def _true_top_k(source: str, vec: np.ndarray, *, keys: list[str] | None = None) -> list[str]:
    """Brute force in numpy over the vectors as STORED (halfvec-rounded), live
    rows of live versions of TENANT only."""
    async with db_module.raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT c.chunk_id, c.embedding_v2::real[] AS v, d.metadata->>'source_key' AS k
            FROM chunks c JOIN documents d
              ON d.customer_id = c.customer_id AND d.doc_id = c.doc_id
             AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
            WHERE c.customer_id = $1 AND d.source_system = $2
              AND c.valid_to IS NULL AND d.valid_to IS NULL AND c.embedding_v2 IS NOT NULL
            """,
            TENANT, source,
        )
    if keys is not None:
        rows = [r for r in rows if r["k"] is None or r["k"] in keys]
    m = np.array([r["v"] for r in rows], dtype=np.float64)
    q = vec.astype(np.float64)
    sim = (m @ q) / (np.linalg.norm(m, axis=1) * np.linalg.norm(q))
    order = sorted(range(len(rows)), key=lambda i: (-sim[i], rows[i]["chunk_id"]))
    return [rows[i]["chunk_id"] for i in order[:K]]


def _queries() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    return [_unit(CENTER_BIG + 0.5 * _unit(rng.standard_normal(DIM))) for _ in range(4)]


# ============================================================
# The exact route returns the true top-K
# ============================================================

async def test_exact_route_returns_the_true_top_k(seeded, monkeypatch) -> None:
    recorded = _record_statements(monkeypatch)
    for q in _queries():
        hits = await _search(monkeypatch, q)
        got = [h.chunk_id for h in hits if h.source_system == "pi"]
        assert got == await _true_top_k("pi", q)
    kinds = {k for k, _, _ in recorded}
    assert "exact" in kinds, f"pi never took the exact route: {kinds}"


async def test_exact_route_honours_the_request_filters(seeded, monkeypatch) -> None:
    """The exact statement is the shared filtered SELECT with a different
    ORDER BY, so a source_keys filter still applies to it: half the
    custom_ingest docs carry another key and must not appear. Keyless docs
    stay admitted (production's request shape), which keeps the pool full of
    claude_code and custom_ingest short."""
    keys = ["artifacts"]
    recorded = _record_statements(monkeypatch)
    for q in _queries()[:2]:
        hits = await _search(monkeypatch, q, source_keys=keys, source_keys_include_keyless=True)
        got = [h.chunk_id for h in hits if h.source_system == "custom_ingest"]
        assert got == await _true_top_k("custom_ingest", q, keys=keys)
    assert ("exact", "custom_ingest") in {(k, s) for k, s, _ in recorded}


async def test_as_a_nonsuperuser_under_rls(as_nonsuper, monkeypatch) -> None:
    """Same truth as a NOSUPERUSER NOBYPASSRLS role. OTHER's pi rows sit at
    the same centre; a single one surfacing means isolation failed."""
    for q in _queries():
        hits = await _search(monkeypatch, q)
        assert all(h.chunk_id.startswith(TENANT + ":") for h in hits)
        got = [h.chunk_id for h in hits if h.source_system == "pi"]
        assert got == await _true_top_k("pi", q)


async def test_absent_source_returns_nothing_cheaply(seeded, monkeypatch) -> None:
    recorded = _record_statements(monkeypatch)
    hits = await _search(monkeypatch, _queries()[0])
    assert not [h for h in hits if h.source_system == "codex"]
    assert ("exact", "codex") in {(k, s) for k, s, _ in recorded}


# ============================================================
# No plan for the exact statement can use HNSW
# ============================================================

async def test_exact_statement_cannot_plan_the_index(seeded, monkeypatch) -> None:
    """With sorting disabled the planner takes ANY ordered path it has --
    the ANN top-up turns into an HNSW scan (the control, proving the check
    can see one), and the exact statement still cannot, because its ORDER BY
    is not one the index serves. Structural, not a cost accident on a small
    table."""
    recorded = _record_statements(monkeypatch)
    monkeypatch.setattr(vector_mod, "PER_SOURCE_EXACT_MAX_CHUNKS", 20)  # pi=30 -> ANN, codex=0 -> exact
    await _search(monkeypatch, _queries()[0])

    by_kind = {k: (sql, params) for k, s, (sql, params) in recorded}
    assert {"exact", "ann"} <= set(by_kind)
    for kind, expect_hnsw in (("ann", True), ("exact", False)):
        sql, params = by_kind[kind]
        plan = await _explain(sql, params, analyze=False, extra="SET LOCAL enable_sort = off")
        assert _uses_hnsw(plan) is expect_hnsw, f"{kind}: {json.dumps(plan)[:400]}"


# ============================================================
# The live-chunk count
# ============================================================

async def test_count_is_what_the_exact_statement_reads(seeded) -> None:
    """30 live embedded pi chunks; the closed one, the unembedded one and the
    one tied to a superseded version are out. codex: absent -> 0. OTHER's
    rows: invisible."""
    sizes = await vector_mod._count_source_sizes(TENANT, ["pi", "codex", "claude_code"])
    assert sizes == {"pi": 30, "codex": 0, "claude_code": 240}


async def test_count_stops_at_the_cap(seeded, monkeypatch) -> None:
    """Capped at threshold + 1, both through the documents gate (80 docs >=
    cap 51 -> no chunk join) and through the chunk join (pi: 11 docs < 51,
    30 chunks -> counted; with cap 21 -> stops at 21)."""
    monkeypatch.setattr(vector_mod, "PER_SOURCE_EXACT_MAX_CHUNKS", 50)
    assert await vector_mod._count_source_sizes(TENANT, ["claude_code"]) == {"claude_code": 51}
    monkeypatch.setattr(vector_mod, "PER_SOURCE_EXACT_MAX_CHUNKS", 20)
    assert await vector_mod._count_source_sizes(TENANT, ["pi"]) == {"pi": 21}


async def test_documents_without_embedded_chunks_do_not_count(seeded, monkeypatch) -> None:
    """A source with more live DOCUMENTS than the threshold but only a few
    live, embedded CHUNKS is small: the exact scan reads chunks, so chunks
    are what the count counts. 40 live docs (threshold 20): 3 with an
    embedded chunk, 5 whose only chunk has no embedding yet, 32 with none."""
    async with db_module.raw_conn() as conn:
        for i in range(40):
            doc_id = f"{TENANT}:linear:{i}"
            await conn.execute(
                """
                INSERT INTO documents (customer_id, doc_id, version, source_system, source_id,
                                       source_url, doc_type, content_hash, created_at,
                                       updated_at, valid_from, acl, title)
                VALUES ($1, $2, 1, 'linear', $2, 'https://x', 'linear.issue', 'h', NOW(),
                        NOW(), NOW(), '{}'::jsonb, 't')
                """,
                TENANT, doc_id,
            )
            if i < 8:
                await conn.execute(
                    """
                    INSERT INTO chunks (customer_id, doc_id, chunk_id, chunk_index, content,
                                        content_hash, token_count, first_seen_version,
                                        last_seen_version, embedding_v2, kind, visibility)
                    VALUES ($1, $2, $2 || ':c0', 0, 'c', 'h', 1, 1, $3,
                            CASE WHEN $4 THEN $5::text::halfvec END, 'content', 'approved')
                    """,
                    TENANT, doc_id, LIVE, i < 3, _lit(CENTER_PI),
                )
    monkeypatch.setattr(vector_mod, "PER_SOURCE_EXACT_MAX_CHUNKS", 20)
    sizes = await vector_mod._count_source_sizes(TENANT, ["linear"])
    assert sizes == {"linear": 3}
    assert vector_mod._routes_exact(sizes["linear"])


async def test_count_burst_leaves_the_pool_to_searches(live_db, settings, monkeypatch) -> None:
    """Eight tenants' first searches at once on a THREE-connection pool, with
    every count made to take 1.5 s in the database. Only
    SOURCE_SIZE_MAX_CONCURRENT_COUNTS counts may hold a connection; the rest
    are skipped, so every search finishes long before the count does. With
    one count per tenant in flight the three connections would be held by
    counts and the searches would queue behind them."""
    import asyncio
    import time

    await db_module.close_pool()
    await db_module.init_pool(settings.model_copy(update={"db_pool_max_size": 3, "db_pool_min_size": 1}))
    await _fresh_counts(monkeypatch, count=False)
    monkeypatch.setattr(
        vector_mod,
        "_SOURCE_SIZE_SQL",
        f"SELECT x.* FROM ({vector_mod._SOURCE_SIZE_SQL}) x CROSS JOIN pg_sleep(1.5)",
    )
    in_flight = 0
    peak = 0
    real_count = vector_mod._count_source_sizes

    async def _tracked(customer_id: str, sources: list[str]) -> dict[str, int]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            return await real_count(customer_id, sources)
        finally:
            in_flight -= 1

    monkeypatch.setattr(vector_mod, "_count_source_sizes", _tracked)
    monkeypatch.setattr(vector_mod, "get_embedder_v2", _embedder(_queries()[0]))

    started = time.perf_counter()
    await asyncio.gather(*(
        vector_mod.vector_search(
            f"burst-{i}", "q", top_k=20, per_source_top_k=K, sources=SOURCES,
            temporal=TemporalSpec(),
        )
        for i in range(8)
    ))
    searches_s = time.perf_counter() - started
    assert vector_mod._SOURCE_SIZE_TASKS, "no count was started"
    await _drain()

    assert searches_s < 1.2, f"searches waited on counts: {searches_s:.2f} s"
    assert peak == vector_mod.SOURCE_SIZE_MAX_CONCURRENT_COUNTS


async def test_count_as_a_nonsuperuser(as_nonsuper) -> None:
    sizes = await vector_mod._count_source_sizes(TENANT, ["pi", "codex"])
    assert sizes == {"pi": 30, "codex": 0}


# ============================================================
# The walk bound
# ============================================================

async def test_bound_is_what_pgvector_reads_and_ends_with_the_transaction(seeded) -> None:
    async with db_module.raw_conn() as conn:
        default = await _show_after_load(conn)
        async with conn.transaction():
            await vector_mod._bound_topup_walk(conn)
            assert await _show_after_load(conn) == str(vector_mod.PER_SOURCE_TOPUP_MAX_SCAN_TUPLES)
        assert await _show_after_load(conn) == default


async def test_bound_stops_the_walk(live_db, monkeypatch) -> None:
    """The HNSW top-up for a source the tenant does not have is the walk
    that never fills its LIMIT: uncapped it visits every live chunk of the
    tenant, capped it stops near the bound.

    pgvector's bound is approximate -- once reached, the scan still returns
    the candidates it had already discarded -- so the graph has to be big
    next to the initial search for the difference to show: 1,500 chunks,
    ef_search 10. Forced onto the index (enable_sort = off) because on 1,500
    rows the planner would rightly sort."""
    async with db_module.raw_conn() as conn:
        await _seed_tenant(conn, TENANT, big=1500, pi=0, ci=0)
        await conn.execute("ANALYZE chunks")
    monkeypatch.setattr(vector_mod, "PER_SOURCE_ANN_POOL", 10)
    await _fresh_counts(monkeypatch)
    monkeypatch.setattr(vector_mod, "PER_SOURCE_EXACT_MAX_CHUNKS", -1)  # every top-up -> ANN
    monkeypatch.setattr(vector_mod, "PER_SOURCE_TOPUP_MAX_SCAN_TUPLES", 100)
    recorded = _record_statements(monkeypatch)
    await _search(monkeypatch, _queries()[0])
    sql, params = next(sp for k, s, sp in recorded if k == "ann" and s == "pi")

    force = "SET LOCAL enable_sort = off; SET LOCAL hnsw.ef_search = 10"
    uncapped = _hnsw_rows(await _explain(sql, params, extra=force, bound=False))
    capped = _hnsw_rows(await _explain(sql, params, extra=force, bound=True))
    assert uncapped >= 1450, uncapped
    assert capped <= uncapped // 3, (capped, uncapped)


# ============================================================
# helpers
# ============================================================

async def _fresh_counts(monkeypatch, *, count: bool = True) -> None:
    """An empty per-process count state (and fresh gates, which bind to the
    test's event loop), then TENANT's counts taken the way the background
    refresh takes them -- so the searches under test are routed, as every
    search after a pod's first one is."""
    import asyncio

    monkeypatch.setattr(vector_mod, "_SOURCE_SIZE_CACHE", {})
    monkeypatch.setattr(vector_mod, "_SOURCE_SIZE_INFLIGHT", set())
    monkeypatch.setattr(vector_mod, "_SOURCE_SIZE_TASKS", set())
    monkeypatch.setattr(vector_mod, "_SOURCE_SIZE_RETRY_AT", {})
    monkeypatch.setattr(
        vector_mod, "_EXACT_STATEMENT_SEMAPHORE",
        asyncio.Semaphore(vector_mod.PER_SOURCE_EXACT_MAX_CONCURRENT),
    )
    if count:
        await vector_mod._refresh_source_sizes(TENANT, SOURCES)


async def _drain() -> None:
    import asyncio

    while vector_mod._SOURCE_SIZE_TASKS:
        await asyncio.gather(*list(vector_mod._SOURCE_SIZE_TASKS))


def _record_statements(monkeypatch) -> list[tuple[str, str, tuple[str, tuple]]]:
    """Wrap the vector module's with_tenant so every top-up statement is
    recorded as (kind, source, (sql, params)) while still executing for real."""
    recorded: list[tuple[str, str, tuple[str, tuple]]] = []
    real = vector_mod.with_tenant

    class _Rec:
        def __init__(self, conn) -> None:
            self._c = conn

        def __getattr__(self, name: str):
            return getattr(self._c, name)

        async def fetch(self, sql: str, *params: Any):
            if "AND d.source_system = $" in sql:
                kind = "exact" if "ORDER BY score DESC, c.chunk_id" in sql else "ann"
                recorded.append((kind, params[-1], (sql, params)))
            return await self._c.fetch(sql, *params)

    @asynccontextmanager
    async def _wt(customer_id: str):
        async with real(customer_id) as conn:
            yield _Rec(conn)

    monkeypatch.setattr(vector_mod, "with_tenant", _wt)
    return recorded


async def _explain(sql: str, params: tuple, *, analyze: bool = True, extra: str = "",
                   bound: bool = False) -> dict:
    async with db_module.with_tenant(TENANT) as conn:
        await conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
        if bound:
            await vector_mod._bound_topup_walk(conn)
        if extra:
            await conn.execute(extra)
        opts = "ANALYZE, FORMAT JSON" if analyze else "FORMAT JSON"
        out = await conn.fetchval(f"EXPLAIN ({opts}) {sql}", *params)
    return json.loads(out)[0]["Plan"]


def _walk(plan: dict):
    yield plan
    for child in plan.get("Plans", []) or []:
        yield from _walk(child)


def _uses_hnsw(plan: dict) -> bool:
    return any("embedding_v2" in (n.get("Index Name") or "") for n in _walk(plan))


def _hnsw_rows(plan: dict) -> int:
    nodes = [n for n in _walk(plan) if "embedding_v2" in (n.get("Index Name") or "")]
    assert nodes, "the forced plan did not use the HNSW index"
    return int(nodes[0]["Actual Rows"])


async def _show_after_load(conn) -> str:
    # pgvector adopts a namespaced placeholder only once it is loaded in the
    # session; touching an operator forces that, so SHOW reads pgvector's value.
    await conn.fetchval("SELECT '[1,0]'::vector <=> '[0,1]'::vector")
    return await conn.fetchval("SHOW hnsw.max_scan_tuples")
