"""Exercise the supported fresh bootstrap, including its real canonical schema."""

import asyncio
import os
from urllib.parse import urlsplit

import asyncpg
import pytest


def test_fresh_bootstrap_includes_tenant_fenced_session_receipts(monkeypatch):
    dsn = os.environ.get("PRBE_SCHEMA_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("requires dedicated local backfill_schema_test database")
    parsed = urlsplit(dsn)
    assert parsed.hostname in {"127.0.0.1", "localhost"}
    assert parsed.path == "/backfill_schema_test"
    monkeypatch.setenv("DATABASE_URL", dsn)
    monkeypatch.setenv("DATABASE_URL_SYNC", dsn.replace("postgresql://", "postgresql+psycopg://", 1))

    async def reset():
        connection = await asyncpg.connect(dsn)
        try:
            await connection.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        finally:
            await connection.close()

    asyncio.run(reset())
    from scripts.migrate import main
    main()

    async def verify():
        connection = await asyncpg.connect(dsn)
        try:
            rows = await connection.fetch(
                "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
                "WHERE relnamespace='public'::regnamespace AND relname=ANY($1::text[])",
                ["session_streams", "session_batch_receipts"],
            )
            assert len(rows) == 2
            assert all(row["relrowsecurity"] and row["relforcerowsecurity"] for row in rows)
            assert await connection.fetchval("SELECT count(*) FROM pg_policies WHERE "
                "schemaname='public' AND tablename=ANY($1::text[]) AND policyname='tenant_isolation'",
                ["session_streams", "session_batch_receipts"]) == 2
            return await connection.fetchval("SELECT version_num FROM alembic_version")
        finally:
            await connection.close()

    before = asyncio.run(verify())
    main()  # supported existing-database no-op must preserve both fences
    assert asyncio.run(verify()) == before
