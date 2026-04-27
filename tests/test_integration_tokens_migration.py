"""Verify migration 0010 creates the device columns and partial unique indexes
with the right behavior: existing connectors keep one-row-per-(customer,source);
device-scoped sources can have many rows per (customer,source) keyed by device_id.

Test runs against a freshly migrated local Postgres.
"""
from __future__ import annotations

import asyncpg
import pytest

from shared.config import get_settings


@pytest.mark.asyncio
async def test_singleton_unique_index_blocks_duplicate_non_device_rows() -> None:
    dsn = get_settings().database_url
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute("DELETE FROM integration_tokens WHERE customer_id = 'mig-test-cust'")
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) VALUES ('mig-test-cust', 'mig', 'mig-test-hash') ON CONFLICT DO NOTHING"
        )
        await conn.execute(
            """
            INSERT INTO integration_tokens (customer_id, source_system, access_token_encrypted, status)
            VALUES ('mig-test-cust', 'slack', 'enc-1', 'active')
            """
        )
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(
                """
                INSERT INTO integration_tokens (customer_id, source_system, access_token_encrypted, status)
                VALUES ('mig-test-cust', 'slack', 'enc-2', 'active')
                """
            )
    finally:
        await conn.execute("DELETE FROM integration_tokens WHERE customer_id = 'mig-test-cust'")
        await conn.close()


@pytest.mark.asyncio
async def test_device_unique_index_allows_many_rows_with_distinct_device_id() -> None:
    dsn = get_settings().database_url
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute("DELETE FROM integration_tokens WHERE customer_id = 'mig-test-cust'")
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) VALUES ('mig-test-cust', 'mig', 'mig-test-hash') ON CONFLICT DO NOTHING"
        )
        await conn.execute(
            """
            INSERT INTO integration_tokens
                (customer_id, source_system, access_token_encrypted, status,
                 device_id, device_metadata)
            VALUES
                ('mig-test-cust', 'claude_code', 'enc-1', 'active', 'dev-1', '{}'::jsonb),
                ('mig-test-cust', 'claude_code', 'enc-2', 'active', 'dev-2', '{}'::jsonb),
                ('mig-test-cust', 'claude_code', 'enc-3', 'active', 'dev-3', '{}'::jsonb)
            """
        )
        rows = await conn.fetch(
            """
            SELECT device_id FROM integration_tokens
            WHERE customer_id = 'mig-test-cust' AND source_system = 'claude_code'
            ORDER BY device_id
            """
        )
        assert [r["device_id"] for r in rows] == ["dev-1", "dev-2", "dev-3"]

        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(
                """
                INSERT INTO integration_tokens
                    (customer_id, source_system, access_token_encrypted, status,
                     device_id, device_metadata)
                VALUES ('mig-test-cust', 'claude_code', 'enc-dup', 'active', 'dev-1', '{}'::jsonb)
                """
            )
    finally:
        await conn.execute("DELETE FROM integration_tokens WHERE customer_id = 'mig-test-cust'")
        await conn.close()
