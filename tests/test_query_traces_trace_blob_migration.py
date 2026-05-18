"""Migration 0079 assertions: query_traces.trace_blob_key column.

Verifies the column exists, is nullable text, and that
write_query_trace round-trips a non-NULL blob_key end-to-end.
The live_db fixture runs `alembic upgrade head` against a containerized
Postgres before the suite, so we just inspect the result.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from services.retrieval.usage import (
    EVENT_TYPE_QUERY,
    QueryTrace,
    write_query_trace,
)
from shared.db import raw_conn, with_tenant


@pytest.mark.asyncio
async def test_trace_blob_key_column_exists(live_db) -> None:
    """Migration 0079 added trace_blob_key (text, nullable) to query_traces."""
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'query_traces'
              AND column_name = 'trace_blob_key'
            """
        )
    assert len(rows) == 1
    col = rows[0]
    assert col["data_type"] == "text"
    assert col["is_nullable"] == "YES"


@pytest.mark.asyncio
async def test_write_query_trace_round_trips_blob_key(
    live_db, _seed_customer
) -> None:
    """A QueryTrace with trace_blob_key set lands on the row and reads back."""
    customer_id = _seed_customer
    request_id = str(uuid.uuid4())
    blob_key = "search-traces/2026-05-17/abc-123.json.gz"

    await write_query_trace(
        QueryTrace(
            customer_id=customer_id,
            request_id=request_id,
            event_type=EVENT_TYPE_QUERY,
            request_payload={"query": "test"},
            response_payload={"results": []},
            occurred_at=datetime.now(UTC),
            trace_blob_key=blob_key,
        )
    )

    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            "SELECT trace_blob_key FROM query_traces WHERE request_id = $1",
            request_id,
        )
    assert row is not None
    assert row["trace_blob_key"] == blob_key


@pytest.mark.asyncio
async def test_write_query_trace_with_null_blob_key_succeeds(
    live_db, _seed_customer
) -> None:
    """Sampled-out rows omit trace_blob_key — the INSERT still succeeds."""
    customer_id = _seed_customer
    request_id = str(uuid.uuid4())

    await write_query_trace(
        QueryTrace(
            customer_id=customer_id,
            request_id=request_id,
            event_type=EVENT_TYPE_QUERY,
            request_payload={"query": "test"},
            response_payload={"results": []},
            occurred_at=datetime.now(UTC),
            # trace_blob_key omitted (default None)
        )
    )

    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            "SELECT trace_blob_key FROM query_traces WHERE request_id = $1",
            request_id,
        )
    assert row is not None
    assert row["trace_blob_key"] is None


@pytest.fixture
async def _seed_customer(live_db) -> str:
    """Insert a customer row so query_traces FK can resolve."""
    customer_id = f"cust-{uuid.uuid4().hex[:8]}"
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, $2, $3) ON CONFLICT (customer_id) DO NOTHING",
            customer_id,
            f"{customer_id} display",
            f"hash-{customer_id}",
        )
    return customer_id
