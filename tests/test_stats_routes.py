"""Focused contract tests for internal ingestion statistics routes."""

from __future__ import annotations

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
