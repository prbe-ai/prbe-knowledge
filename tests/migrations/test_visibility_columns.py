"""Smoke test that migration 0082 ran and visibility columns + partial
indexes are shaped as expected on ``documents`` and ``chunks``.

This is an integration smoke against the locally-applied schema.
Skips if no DATABASE_URL is configured (CI applies alembic upgrade in
its own job; this is the local-developer signal). Pattern mirrors
``tests/test_incident_investigations_migration.py``.
"""
from __future__ import annotations

import os

import pytest


def _dsn() -> str | None:
    """Return a vanilla asyncpg DSN if one is configured, else None.

    Local conftest sets DATABASE_URL to a ``postgresql://`` URL, which
    asyncpg accepts directly. We strip ``+psycopg`` driver suffixes if
    they sneak in from a parent env.
    """
    raw = os.environ.get("DATABASE_URL")
    if not raw:
        return None
    # asyncpg doesn't understand the SQLAlchemy-style "+driver" suffix.
    return raw.replace("postgresql+psycopg://", "postgresql://", 1)


async def test_visibility_columns_present_on_documents_and_chunks() -> None:
    dsn = _dsn()
    if not dsn:
        pytest.skip("DATABASE_URL not set")
    import asyncpg
    conn = await asyncpg.connect(dsn=dsn)
    try:
        for table in ("documents", "chunks"):
            row = await conn.fetchrow(
                "SELECT data_type, is_nullable, column_default "
                "FROM information_schema.columns "
                "WHERE table_name = $1 AND column_name = 'visibility'",
                table,
            )
            assert row is not None, f"{table}.visibility column missing"
            assert row["data_type"] == "text", (
                f"{table}.visibility expected text, got {row['data_type']}"
            )
            assert row["is_nullable"] == "NO", (
                f"{table}.visibility should be NOT NULL"
            )
            # The default is captured as a quoted literal by Postgres.
            assert row["column_default"] is not None
            assert "approved" in row["column_default"], (
                f"{table}.visibility default expected to include 'approved', "
                f"got {row['column_default']!r}"
            )
    finally:
        await conn.close()


async def test_visibility_check_constraints_present() -> None:
    dsn = _dsn()
    if not dsn:
        pytest.skip("DATABASE_URL not set")
    import asyncpg
    conn = await asyncpg.connect(dsn=dsn)
    try:
        for table, constraint in (
            ("documents", "documents_visibility_chk"),
            ("chunks", "chunks_visibility_chk"),
        ):
            row = await conn.fetchrow(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = $1::regclass AND conname = $2",
                table, constraint,
            )
            assert row is not None, (
                f"{constraint} on {table} missing"
            )
    finally:
        await conn.close()


async def test_visibility_partial_indexes_present() -> None:
    dsn = _dsn()
    if not dsn:
        pytest.skip("DATABASE_URL not set")
    import asyncpg
    conn = await asyncpg.connect(dsn=dsn)
    try:
        for table, index in (
            ("chunks", "chunks_visibility_approved_idx"),
            ("documents", "documents_visibility_approved_idx"),
        ):
            row = await conn.fetchrow(
                "SELECT indexdef FROM pg_indexes "
                "WHERE schemaname = 'public' "
                "AND tablename = $1 AND indexname = $2",
                table, index,
            )
            assert row is not None, f"{index} on {table} missing"
            # Partial-index predicate must filter on visibility = 'approved'.
            indexdef = row["indexdef"]
            assert "visibility" in indexdef and "approved" in indexdef, (
                f"{index} on {table} is not the expected partial index: "
                f"{indexdef!r}"
            )
            assert "WHERE" in indexdef.upper(), (
                f"{index} on {table} should be a partial index: {indexdef!r}"
            )
    finally:
        await conn.close()


async def test_existing_rows_default_to_approved() -> None:
    """The migration adds the column with a server default; existing
    rows MUST come out as 'approved' (no draft rows pre-exist)."""
    dsn = _dsn()
    if not dsn:
        pytest.skip("DATABASE_URL not set")
    import asyncpg
    conn = await asyncpg.connect(dsn=dsn)
    try:
        for table in ("documents", "chunks"):
            row = await conn.fetchrow(
                f"SELECT COUNT(*) AS non_approved "
                f"FROM {table} WHERE visibility <> 'approved'"
            )
            assert row is not None
            assert row["non_approved"] == 0, (
                f"{table} has rows with visibility != 'approved' "
                f"({row['non_approved']}) after 0082 — backfill regressed?"
            )
    finally:
        await conn.close()
