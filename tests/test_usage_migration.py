"""Migration assertions for usage_events.

Verifies the table, indexes, FK, and RLS policy exist after
`alembic upgrade head` (which the live_db fixture's containerized Postgres
already ran). Doesn't reapply the migration — it just inspects the result
of the standard local migrate.
"""

from __future__ import annotations

import pytest

from engine.shared.db import raw_conn


@pytest.mark.asyncio
async def test_usage_events_table_exists(live_db) -> None:
    """The columns + types + nullability match the migration spec."""
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'usage_events'
            ORDER BY ordinal_position
            """
        )

    cols = {r["column_name"]: (r["data_type"], r["is_nullable"]) for r in rows}
    assert cols["event_id"] == ("uuid", "NO")
    assert cols["customer_id"] == ("text", "NO")
    assert cols["occurred_at"][0] == "timestamp with time zone"
    assert cols["caller_kind"] == ("text", "NO")
    assert cols["caller_subject"] == ("text", "YES")
    assert cols["event_type"] == ("text", "NO")
    assert cols["request_id"] == ("uuid", "YES")
    assert cols["endpoint"] == ("text", "NO")
    assert cols["summary"] == ("text", "YES")
    assert cols["status"] == ("text", "NO")
    assert cols["error_class"] == ("text", "YES")
    assert cols["latency_ms"] == ("integer", "YES")
    assert cols["result_count"] == ("integer", "YES")
    assert cols["metadata"] == ("jsonb", "NO")


@pytest.mark.asyncio
async def test_usage_events_indexes_exist(live_db) -> None:
    """Three indexes back the feed, type-filtered feed, and FTS search."""
    async with raw_conn() as conn:
        names = {
            r["indexname"]
            for r in await conn.fetch(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'usage_events'"
            )
        }
    assert "idx_usage_events_customer_time" in names
    assert "idx_usage_events_customer_type_time" in names
    assert "idx_usage_events_search" in names


@pytest.mark.asyncio
async def test_usage_events_rls_policy_exists(live_db) -> None:
    """RLS is enabled and the tenant-isolation policy is present."""
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT relrowsecurity, relforcerowsecurity
            FROM pg_class
            WHERE relname = 'usage_events'
            """
        )
        assert row is not None
        assert row["relrowsecurity"] is True
        assert row["relforcerowsecurity"] is True

        policies = await conn.fetch(
            "SELECT polname FROM pg_policy WHERE polrelid = 'usage_events'::regclass"
        )
        assert any(p["polname"] == "usage_events_tenant_isolation" for p in policies)


@pytest.mark.asyncio
async def test_usage_events_customer_fk_cascades(live_db) -> None:
    """Deleting a customer drops their usage_events rows (no orphan rows)."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ('cust-fk-cascade', 'fk', 'hash-fk')
            """
        )
        # Direct INSERT bypassing RLS is fine in raw_conn (no GUC set).
        # But the policy still applies to writes when ENABLE+FORCE — so
        # we set the GUC inline for the insert.
        await conn.execute("SELECT set_config('app.current_customer_id', 'cust-fk-cascade', false)")
        await conn.execute(
            """
            INSERT INTO usage_events (customer_id, caller_kind, event_type, endpoint, status)
            VALUES ('cust-fk-cascade', 'mcp', 'knowledge.retrieve', '/retrieve', 'ok')
            """
        )
        await conn.execute("SELECT set_config('app.current_customer_id', '', false)")

        await conn.execute("DELETE FROM customers WHERE customer_id = 'cust-fk-cascade'")
        # Need GUC to read the row back (or its absence). After the cascade,
        # the row is gone regardless of RLS visibility.
        await conn.execute("SELECT set_config('app.current_customer_id', 'cust-fk-cascade', false)")
        n = await conn.fetchval(
            "SELECT COUNT(*) FROM usage_events WHERE customer_id = 'cust-fk-cascade'"
        )
        assert n == 0
