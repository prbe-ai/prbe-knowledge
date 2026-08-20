"""Focused contract tests for internal ingestion statistics routes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import orjson
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport

from engine.shared import db as db_module
from engine.shared.config import Settings, get_settings
from engine.shared.db import raw_conn
from kb import stats_routes

CUSTOMER = "stats-customer"
OTHER_CUSTOMER = "stats-other-customer"
INTERNAL_KEY = "test-internal-key"


class _FakeConnection:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query: str, *args: object) -> list[dict[str, Any]]:
        self.fetch_calls.append((query, args))
        return self.rows


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", INTERNAL_KEY)
    get_settings.cache_clear()  # type: ignore[attr-defined]
    yield
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture
def fake_db(monkeypatch: pytest.MonkeyPatch) -> _FakeConnection:
    conn = _FakeConnection()

    @asynccontextmanager
    async def fake_with_tenant(customer_id: str) -> AsyncIterator[_FakeConnection]:
        assert customer_id == CUSTOMER
        yield conn

    monkeypatch.setattr(stats_routes, "with_tenant", fake_with_tenant)
    return conn


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app = FastAPI()
    app.include_router(stats_routes.router)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as value:
        yield value


@pytest_asyncio.fixture
async def stats_live_db(settings: Settings) -> AsyncIterator[None]:
    """DB lifecycle scoped to the two stats customers.

    The shared live_db fixture still truncates incident_investigations, which
    migration 0092 intentionally dropped. Mirror its stale-loop reset and
    guaranteed cleanup without depending on that stale global table list.
    """
    customer_ids = [CUSTOMER, OTHER_CUSTOMER]
    db_module.reset_pool()
    await db_module.init_pool(settings)
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "DELETE FROM customers WHERE customer_id = ANY($1::text[])",
            customer_ids,
        )
    try:
        yield None
    finally:
        async with db_module.raw_conn() as conn:
            await conn.execute(
                "DELETE FROM customers WHERE customer_id = ANY($1::text[])",
                customer_ids,
            )
        await db_module.close_pool()


def _headers(*, customer: bool = True) -> dict[str, str]:
    headers = {"X-Internal-Knowledge-Key": INTERNAL_KEY}
    if customer:
        headers["X-Prbe-Customer"] = CUSTOMER
    return headers


async def _seed_document(
    conn: Any,
    *,
    customer_id: str,
    doc_id: str,
    source: str = "claude_code",
    metadata: dict[str, str] | None = None,
    parent_doc_id: str | None = None,
    ingested_at: datetime,
    valid_to: datetime | None = None,
    deleted_at: datetime | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO documents (
            doc_id, version, customer_id,
            source_system, source_id, source_url,
            doc_class, doc_type, content_type,
            content_hash, body_size_bytes, body_token_count,
            created_at, updated_at, valid_from, valid_to, deleted_at, ingested_at,
            parent_doc_id, acl, metadata
        ) VALUES (
            $1, 1, $2,
            $3, $1, '',
            'raw_source', 'claude_code.session', 'text/plain',
            $4, 0, 0,
            $5, $5, $5, $6, $7, $5,
            $8, '{}'::jsonb, $9::jsonb
        )
        """,
        doc_id,
        customer_id,
        source,
        f"hash:{customer_id}:{doc_id}",
        ingested_at,
        valid_to,
        deleted_at,
        parent_doc_id,
        orjson.dumps(metadata or {}).decode(),
    )


async def _seed_chunk(
    conn: Any,
    *,
    customer_id: str,
    doc_id: str,
    suffix: str,
    valid_to: datetime | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, doc_id, customer_id,
            chunk_index, content, content_hash, token_count,
            first_seen_version, last_seen_version, valid_to
        ) VALUES (
            $1, $2, $3,
            0, $4, $5, 1,
            1, 1, $6
        )
        """,
        f"{customer_id}:{doc_id}:{suffix}",
        doc_id,
        customer_id,
        f"content:{suffix}",
        f"chunk-hash:{customer_id}:{doc_id}:{suffix}",
        valid_to,
    )


@pytest.mark.asyncio
async def test_device_stats_requires_internal_key(
    client: httpx.AsyncClient,
    fake_db: _FakeConnection,
) -> None:
    response = await client.get(
        "/api/stats/ingestion/claude_code/devices",
        headers={"X-Prbe-Customer": CUSTOMER},
    )

    assert response.status_code == 401
    assert fake_db.fetch_calls == []


@pytest.mark.asyncio
async def test_device_stats_requires_customer_header(
    client: httpx.AsyncClient,
    fake_db: _FakeConnection,
) -> None:
    response = await client.get(
        "/api/stats/ingestion/claude_code/devices",
        headers=_headers(customer=False),
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "missing X-Prbe-Customer"}
    assert fake_db.fetch_calls == []


@pytest.mark.asyncio
async def test_device_stats_rejects_non_device_source(
    client: httpx.AsyncClient,
    fake_db: _FakeConnection,
) -> None:
    response = await client.get(
        "/api/stats/ingestion/github/devices",
        headers=_headers(),
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "source does not support per-device stats: github"
    }
    assert fake_db.fetch_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["claude_code", "codex"])
async def test_device_stats_returns_grouped_live_counts(
    client: httpx.AsyncClient,
    fake_db: _FakeConnection,
    source: str,
) -> None:
    last_ingested_at = datetime(2026, 7, 21, 14, 30, tzinfo=UTC)
    fake_db.rows = [
        {
            "device_id": "device-new",
            "docs": 7,
            "chunks": 19,
            "last_ingested_at": last_ingested_at,
        },
        {
            "device_id": "device-empty",
            "docs": 1,
            "chunks": 0,
            "last_ingested_at": None,
        },
    ]

    response = await client.get(
        f"/api/stats/ingestion/{source}/devices",
        headers=_headers(),
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "customer_id": CUSTOMER,
        "source": source,
        "devices": [
            {
                "device_id": "device-new",
                "docs": 7,
                "chunks": 19,
                "last_ingested_at": "2026-07-21T14:30:00+00:00",
            },
            {
                "device_id": "device-empty",
                "docs": 1,
                "chunks": 0,
                "last_ingested_at": None,
            },
        ],
    }

    assert len(fake_db.fetch_calls) == 1
    query, args = fake_db.fetch_calls[0]
    normalized_query = " ".join(query.split())
    assert args == (CUSTOMER, source)
    assert "WITH live_docs AS" in normalized_query
    assert "SELECT DISTINCT ON (d.customer_id, d.doc_id)" in normalized_query
    assert "COUNT(DISTINCT d.doc_id) AS docs" in normalized_query
    assert "COUNT(c.chunk_id) AS chunks" in normalized_query
    assert "LEFT JOIN live_docs parent" in normalized_query
    assert "NULLIF(BTRIM(d.parent_doc_id), '')" in normalized_query
    assert "NULLIF(BTRIM(d.metadata->>'parent_doc_id'), '')" in normalized_query
    assert "NULLIF(BTRIM(parent.metadata->>'device_id'), '')" in normalized_query
    assert "LEFT JOIN chunks c" in normalized_query
    assert "c.valid_to IS NULL" in normalized_query
    assert "d.source_system = $2" in normalized_query
    assert "d.valid_to IS NULL" in normalized_query
    assert "d.deleted_at IS NULL" in normalized_query
    assert "WHERE d.device_id IS NOT NULL" in normalized_query
    assert "ORDER BY last_ingested_at DESC, device_id" in normalized_query
    assert normalized_query.endswith("LIMIT 10")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_device_stats_attributes_live_children_to_their_live_parent(
    client: httpx.AsyncClient,
    stats_live_db: None,
) -> None:
    base = datetime(2026, 7, 21, 12, tzinfo=UTC)
    parent_doc_id = "claude_code:stats-customer:session-a"

    async with raw_conn() as conn:
        await conn.executemany(
            """
            INSERT INTO customers(customer_id, display_name, api_key_hash)
            VALUES ($1, $1, $1)
            """,
            [(CUSTOMER,), (OTHER_CUSTOMER,)],
        )

        await _seed_document(
            conn,
            customer_id=CUSTOMER,
            doc_id=parent_doc_id,
            metadata={"device_id": "device-a"},
            ingested_at=base,
        )
        await _seed_document(
            conn,
            customer_id=CUSTOMER,
            doc_id="claude_code:stats-customer:session-a:qa:0",
            parent_doc_id=parent_doc_id,
            ingested_at=base + timedelta(minutes=1),
        )
        await _seed_document(
            conn,
            customer_id=CUSTOMER,
            doc_id="claude_code:stats-customer:session-a:decision:0",
            metadata={"parent_doc_id": parent_doc_id},
            ingested_at=base + timedelta(minutes=2),
        )
        await _seed_document(
            conn,
            customer_id=CUSTOMER,
            doc_id="claude_code:stats-customer:session-a:file-ref:0",
            metadata={"device_id": "device-a"},
            parent_doc_id=parent_doc_id,
            ingested_at=base + timedelta(minutes=3),
        )
        await _seed_document(
            conn,
            customer_id=CUSTOMER,
            doc_id="claude_code:stats-customer:session-b",
            metadata={"device_id": "device-b"},
            ingested_at=base + timedelta(minutes=4),
        )

        # These live children must not inherit across bitemporal, source, or
        # tenant boundaries.
        stale_parent = "claude_code:stats-customer:stale-parent"
        await _seed_document(
            conn,
            customer_id=CUSTOMER,
            doc_id=stale_parent,
            metadata={"device_id": "stale-device"},
            ingested_at=base,
            valid_to=base + timedelta(minutes=1),
        )
        await _seed_document(
            conn,
            customer_id=CUSTOMER,
            doc_id=f"{stale_parent}:qa:0",
            parent_doc_id=stale_parent,
            ingested_at=base + timedelta(minutes=5),
        )

        deleted_parent = "claude_code:stats-customer:deleted-parent"
        await _seed_document(
            conn,
            customer_id=CUSTOMER,
            doc_id=deleted_parent,
            metadata={"device_id": "deleted-device"},
            ingested_at=base,
            deleted_at=base + timedelta(minutes=1),
        )
        await _seed_document(
            conn,
            customer_id=CUSTOMER,
            doc_id=f"{deleted_parent}:qa:0",
            parent_doc_id=deleted_parent,
            ingested_at=base + timedelta(minutes=6),
        )

        foreign_parent = "claude_code:foreign:session"
        await _seed_document(
            conn,
            customer_id=OTHER_CUSTOMER,
            doc_id=foreign_parent,
            metadata={"device_id": "foreign-device"},
            ingested_at=base,
        )
        await _seed_document(
            conn,
            customer_id=CUSTOMER,
            doc_id="claude_code:stats-customer:foreign-child",
            parent_doc_id=foreign_parent,
            ingested_at=base + timedelta(minutes=7),
        )

        codex_parent = "codex:stats-customer:session"
        await _seed_document(
            conn,
            customer_id=CUSTOMER,
            doc_id=codex_parent,
            source="codex",
            metadata={"device_id": "codex-device"},
            ingested_at=base,
        )
        await _seed_document(
            conn,
            customer_id=CUSTOMER,
            doc_id="claude_code:stats-customer:codex-child",
            parent_doc_id=codex_parent,
            ingested_at=base + timedelta(minutes=8),
        )

        for suffix in ("parent-1", "parent-2"):
            await _seed_chunk(
                conn,
                customer_id=CUSTOMER,
                doc_id=parent_doc_id,
                suffix=suffix,
            )
        await _seed_chunk(
            conn,
            customer_id=CUSTOMER,
            doc_id=parent_doc_id,
            suffix="parent-stale",
            valid_to=base + timedelta(minutes=1),
        )
        for suffix in ("column-child-1", "column-child-2"):
            await _seed_chunk(
                conn,
                customer_id=CUSTOMER,
                doc_id="claude_code:stats-customer:session-a:qa:0",
                suffix=suffix,
            )
        for doc_id, suffix in (
            ("claude_code:stats-customer:session-a:decision:0", "metadata-child"),
            ("claude_code:stats-customer:session-a:file-ref:0", "stamped-child"),
            ("claude_code:stats-customer:session-b", "direct-device"),
            (f"{stale_parent}:qa:0", "stale-parent-child"),
            (f"{deleted_parent}:qa:0", "deleted-parent-child"),
            ("claude_code:stats-customer:foreign-child", "foreign-child"),
            ("claude_code:stats-customer:codex-child", "codex-child"),
        ):
            await _seed_chunk(
                conn,
                customer_id=CUSTOMER,
                doc_id=doc_id,
                suffix=suffix,
            )

    response = await client.get(
        "/api/stats/ingestion/claude_code/devices",
        headers=_headers(),
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "customer_id": CUSTOMER,
        "source": "claude_code",
        "devices": [
            {
                "device_id": "device-b",
                "docs": 1,
                "chunks": 1,
                "last_ingested_at": "2026-07-21T12:04:00+00:00",
            },
            {
                "device_id": "device-a",
                "docs": 4,
                "chunks": 6,
                "last_ingested_at": "2026-07-21T12:03:00+00:00",
            },
        ],
    }


# ---------------------------------------------------------------------------
# GET /api/stats/ingestion — the aggregate route.
#
# It had no tests. It also grew a cache and a concurrent fan-out, and both are
# the kind of thing that looks right and is wrong once: a cache keyed too
# loosely leaks one tenant's counts to another, and a fan-out that quietly
# serialises costs the same as before while reading as if it were fixed.
# ---------------------------------------------------------------------------


#: Fixed so the isoformat() assertion below is not a tautology against now().
_INGESTED_AT = datetime(2026, 8, 19, 12, 30, tzinfo=UTC)


class _RoutedConnection:
    """Returns rows chosen by which of the four aggregates is being run.

    `_FakeConnection` above answers every query with the same list, which was
    fine when the route ran its four queries down one connection and the test
    only cared about the device route. The aggregate route needs docs, chunks,
    queue and backfill to be distinguishable or the assembly logic is untested.
    """

    def __init__(self, rows_by_table: dict[str, list[dict[str, Any]]]) -> None:
        self._rows_by_table = rows_by_table
        self.fetch_calls: list[str] = []
        self.concurrent_peak = 0
        self._in_flight = 0

    @staticmethod
    def _table_of(query: str) -> str:
        for table in ("ingestion_queue", "backfill_state", "chunks", "documents"):
            if table in query:
                return table
        raise AssertionError(f"unrecognised stats query: {query}")

    async def fetch(self, query: str, *args: object) -> list[dict[str, Any]]:
        table = self._table_of(query)
        self.fetch_calls.append(table)
        self._in_flight += 1
        self.concurrent_peak = max(self.concurrent_peak, self._in_flight)
        try:
            # Yield to the loop so a genuinely concurrent caller has the chance
            # to overlap. A sequential caller simply never does, which is the
            # difference `concurrent_peak` is measuring.
            await asyncio.sleep(0)
            return self._rows_by_table.get(table, [])
        finally:
            self._in_flight -= 1


_DOC_ROWS = [
    {"source_system": "github", "docs": 12, "last_ingested_at": _INGESTED_AT},
    {"source_system": "claude_code", "docs": 3, "last_ingested_at": None},
]
_CHUNK_ROWS = [
    {"source_system": "github", "chunks": 240},
    {"source_system": "claude_code", "chunks": 9},
]
_QUEUE_ROWS = [
    {"source_system": "github", "status": "pending", "n": 4},
    {"source_system": "github", "status": "dlq", "n": 1},
]
_BACKFILL_ROWS = [
    {
        "source_system": "github",
        "status": "complete",
        "events_enqueued": 12,
        "last_error": None,
        "started_at": _INGESTED_AT,
        "last_progress_at": None,
        "completed_at": _INGESTED_AT,
    }
]


@pytest.fixture(autouse=True)
def _clear_stats_cache() -> Iterator[None]:
    """The cache is module state; leaking it across tests makes order matter."""
    stats_routes.reset_stats_cache()
    yield
    stats_routes.reset_stats_cache()


@pytest.fixture
def routed_db(monkeypatch: pytest.MonkeyPatch) -> _RoutedConnection:
    conn = _RoutedConnection(
        {
            "documents": _DOC_ROWS,
            "chunks": _CHUNK_ROWS,
            "ingestion_queue": _QUEUE_ROWS,
            "backfill_state": _BACKFILL_ROWS,
        }
    )
    seen_customers: list[str] = []

    @asynccontextmanager
    async def fake_with_tenant(customer_id: str) -> AsyncIterator[_RoutedConnection]:
        seen_customers.append(customer_id)
        yield conn

    monkeypatch.setattr(stats_routes, "with_tenant", fake_with_tenant)
    conn.seen_customers = seen_customers  # type: ignore[attr-defined]
    return conn


@pytest.mark.asyncio
async def test_ingestion_stats_requires_internal_key(
    client: httpx.AsyncClient,
    routed_db: _RoutedConnection,
) -> None:
    response = await client.get(
        "/api/stats/ingestion", headers={"X-Prbe-Customer": CUSTOMER}
    )

    assert response.status_code == 401
    assert routed_db.fetch_calls == []


@pytest.mark.asyncio
async def test_ingestion_stats_requires_customer_header(
    client: httpx.AsyncClient,
    routed_db: _RoutedConnection,
) -> None:
    response = await client.get("/api/stats/ingestion", headers=_headers(customer=False))

    assert response.status_code == 400
    assert routed_db.fetch_calls == []


@pytest.mark.asyncio
async def test_ingestion_stats_assembles_per_source_rows_and_totals(
    client: httpx.AsyncClient,
    routed_db: _RoutedConnection,
) -> None:
    response = await client.get("/api/stats/ingestion", headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["totals"] == {"docs": 15, "chunks": 249}
    # Ordered by descending doc count, so the biggest source leads the table.
    assert [row["source"] for row in body["sources"]] == ["github", "claude_code"]
    github = body["sources"][0]
    assert github["chunks"] == 240
    assert github["pending"] == 4
    assert github["dlq"] == 1
    assert github["processing"] == 0
    assert github["last_ingested_at"] == _INGESTED_AT.isoformat()
    assert body["backfills"][0]["status"] == "complete"


@pytest.mark.asyncio
async def test_ingestion_stats_runs_its_four_queries_concurrently(
    client: httpx.AsyncClient,
    routed_db: _RoutedConnection,
) -> None:
    response = await client.get("/api/stats/ingestion", headers=_headers())

    assert response.status_code == 200
    assert sorted(routed_db.fetch_calls) == [
        "backfill_state",
        "chunks",
        "documents",
        "ingestion_queue",
    ]
    # The point of the change: all four in flight at once. Sequential execution
    # gives a peak of 1 and would still pass every other assertion here.
    assert routed_db.concurrent_peak == 4


@pytest.mark.asyncio
async def test_ingestion_stats_serves_the_second_call_from_cache(
    client: httpx.AsyncClient,
    routed_db: _RoutedConnection,
) -> None:
    first = await client.get("/api/stats/ingestion", headers=_headers())
    calls_after_first = len(routed_db.fetch_calls)
    second = await client.get("/api/stats/ingestion", headers=_headers())

    assert first.json() == second.json()
    assert calls_after_first == 4
    assert len(routed_db.fetch_calls) == 4, "second call re-queried the database"


@pytest.mark.asyncio
async def test_ingestion_stats_refresh_bypasses_the_cache(
    client: httpx.AsyncClient,
    routed_db: _RoutedConnection,
) -> None:
    await client.get("/api/stats/ingestion", headers=_headers())
    response = await client.get(
        "/api/stats/ingestion", params={"refresh": "true"}, headers=_headers()
    )

    assert response.status_code == 200
    assert len(routed_db.fetch_calls) == 8


@pytest.mark.asyncio
async def test_ingestion_stats_cache_does_not_cross_tenants(
    client: httpx.AsyncClient,
    routed_db: _RoutedConnection,
) -> None:
    await client.get("/api/stats/ingestion", headers=_headers())
    response = await client.get(
        "/api/stats/ingestion",
        headers={
            "X-Internal-Knowledge-Key": INTERNAL_KEY,
            "X-Prbe-Customer": OTHER_CUSTOMER,
        },
    )

    assert response.status_code == 200
    assert response.json()["customer_id"] == OTHER_CUSTOMER
    # A cache keyed on anything but the tenant would have served CUSTOMER's
    # numbers here, with the right customer_id stamped on top of them.
    assert len(routed_db.fetch_calls) == 8
    assert routed_db.seen_customers.count(OTHER_CUSTOMER) == 4  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_ingestion_stats_requeries_once_the_ttl_lapses(
    client: httpx.AsyncClient,
    routed_db: _RoutedConnection,
) -> None:
    """Age the cached entry rather than patch the clock.

    `stats_routes.time` IS the stdlib time module, so monkeypatching
    `monotonic` on it also rewires asyncio's event loop, which calls it on
    every iteration -- a fake clock built for two reads gets drained by the
    loop and the test dies in a fixture finalizer instead of failing usefully.
    Reaching into the cache entry is the narrower lie.
    """
    await client.get("/api/stats/ingestion", headers=_headers())
    cached = stats_routes._stats_cache[CUSTOMER]
    stats_routes._stats_cache[CUSTOMER] = stats_routes._CachedStats(
        payload=cached.payload,
        fetched_at=cached.fetched_at - stats_routes._STATS_CACHE_TTL_S - 1.0,
    )

    response = await client.get("/api/stats/ingestion", headers=_headers())

    assert response.status_code == 200
    assert len(routed_db.fetch_calls) == 8


@pytest.mark.asyncio
async def test_a_slow_tenant_does_not_block_another_tenants_stats(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One global lock here would be a cross-tenant outage, not a slow page.

    The lock is held across the whole DB round trip, and Starlette does not
    cancel a handler when the client disconnects — so research-os hitting its
    10s relay timeout would not release it. A single wedged tenant would hold
    every other tenant's /knowledge header until the statement timeout.
    """
    blocked = asyncio.Event()
    released = asyncio.Event()

    @asynccontextmanager
    async def fake_with_tenant(customer_id: str) -> AsyncIterator[Any]:
        if customer_id == CUSTOMER:
            blocked.set()
            await released.wait()
        yield _RoutedConnection(
            {
                "documents": _DOC_ROWS,
                "chunks": _CHUNK_ROWS,
                "ingestion_queue": _QUEUE_ROWS,
                "backfill_state": _BACKFILL_ROWS,
            }
        )

    monkeypatch.setattr(stats_routes, "with_tenant", fake_with_tenant)

    slow = asyncio.create_task(client.get("/api/stats/ingestion", headers=_headers()))
    await asyncio.wait_for(blocked.wait(), timeout=5)

    # CUSTOMER is wedged mid-query holding its own lock. OTHER_CUSTOMER must
    # still be served.
    fast = await asyncio.wait_for(
        client.get(
            "/api/stats/ingestion",
            headers={
                "X-Internal-Knowledge-Key": INTERNAL_KEY,
                "X-Prbe-Customer": OTHER_CUSTOMER,
            },
        ),
        timeout=5,
    )
    assert fast.status_code == 200
    assert fast.json()["customer_id"] == OTHER_CUSTOMER

    released.set()
    assert (await asyncio.wait_for(slow, timeout=5)).status_code == 200


@pytest.mark.asyncio
async def test_cache_and_lock_tables_do_not_grow_without_bound(
    client: httpx.AsyncClient,
    routed_db: _RoutedConnection,
) -> None:
    """Both per-tenant dicts are pruned on write, or the managed plane leaks."""
    await client.get("/api/stats/ingestion", headers=_headers())
    cached = stats_routes._stats_cache[CUSTOMER]
    stats_routes._stats_cache[CUSTOMER] = stats_routes._CachedStats(
        payload=cached.payload,
        fetched_at=cached.fetched_at - stats_routes._STATS_CACHE_TTL_S - 1.0,
    )

    await client.get(
        "/api/stats/ingestion",
        headers={
            "X-Internal-Knowledge-Key": INTERNAL_KEY,
            "X-Prbe-Customer": OTHER_CUSTOMER,
        },
    )

    assert CUSTOMER not in stats_routes._stats_cache
    assert CUSTOMER not in stats_routes._stats_cache_locks
    assert OTHER_CUSTOMER in stats_routes._stats_cache
