"""Smoke test that the 0080 migration ran and the table is shaped as expected.

This test is an integration smoke against the locally-applied schema.
It will skip if no DATABASE_URL is configured (CI uses Alembic
upgrade in its own job; this is the local-developer signal).
"""
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
            "WHERE table_name = 'incident_investigations' ORDER BY column_name"
        )
        columns = {r["column_name"] for r in rows}
        expected = {
            "customer_id", "incident_doc_id", "current_report_doc_id",
            "state", "versions", "reviewer_id", "reviewed_at",
            "created_at", "updated_at",
        }
        assert expected.issubset(columns), f"missing columns: {expected - columns}"
    finally:
        await conn.close()


async def test_rls_policy_present() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL not set")
    import asyncpg
    conn = await asyncpg.connect(dsn=dsn)
    try:
        row = await conn.fetchrow(
            "SELECT polname FROM pg_policy "
            "WHERE polrelid = 'incident_investigations'::regclass"
        )
        assert row is not None and row["polname"] == "tenant_isolation"
    finally:
        await conn.close()
