"""Smoke test that migration 0081 applies and the table is shaped as expected."""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.asyncio


async def test_table_exists_with_expected_columns() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL not set")
    import asyncpg
    conn = await asyncpg.connect(dsn=dsn)
    try:
        rows = await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'customer_incident_mcp_servers' "
            "ORDER BY column_name"
        )
        columns = {r["column_name"] for r in rows}
        expected = {
            "customer_id", "mcp_kind", "mcp_url", "secret_ciphertext",
            "enabled", "created_at", "updated_at",
        }
        assert expected.issubset(columns), f"missing: {expected - columns}"
    finally:
        await conn.close()


async def test_rls_force_enabled() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL not set")
    import asyncpg
    conn = await asyncpg.connect(dsn=dsn)
    try:
        row = await conn.fetchrow(
            "SELECT relrowsecurity AS rls, relforcerowsecurity AS force_rls "
            "FROM pg_class WHERE relname = 'customer_incident_mcp_servers'",
        )
        assert row["rls"] is True
        assert row["force_rls"] is True
    finally:
        await conn.close()


async def test_policy_has_both_using_and_with_check() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL not set")
    import asyncpg
    conn = await asyncpg.connect(dsn=dsn)
    try:
        row = await conn.fetchrow(
            "SELECT qual IS NOT NULL AS has_using, "
            "with_check IS NOT NULL AS has_with_check "
            "FROM pg_policies WHERE tablename = 'customer_incident_mcp_servers' "
            "AND policyname = 'tenant_isolation'"
        )
        assert row is not None
        assert row["has_using"] is True
        assert row["has_with_check"] is True
    finally:
        await conn.close()
