"""Migration 0148: session_streams accepts protocol 3 beside protocol 2.

The 0128 check was an inline column constraint, so the migration finds it by
definition, not by name. The second test replays the upgrade against a table
built with the 0128 DDL: if the lookup ever misses, the old `= 2` check
survives next to the new one and a protocol-3 stream is still refused.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import asyncpg
import pytest

from engine.shared.db import raw_conn

_MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "db/migrations/versions/20261006_0148_session_streams_protocol_3.py"
)


def _migration():
    spec = importlib.util.spec_from_file_location("m0148", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _insert(conn, table: str, version: int) -> None:
    await conn.execute(
        f"INSERT INTO {table}(customer_id, source_system, session_id, stream_id,"
        " protocol_version, prefix_sha256) VALUES ('mig0148', 'claude_code', $1, 's', $2, 'p')",
        f"s-{version}",
        version,
    )


@pytest.mark.asyncio
async def test_the_schema_accepts_protocol_2_and_3_and_nothing_else(live_db) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash)"
            " VALUES ('mig0148', 'mig', 'mig0148-hash') ON CONFLICT DO NOTHING"
        )
        try:
            for version in (2, 3):
                await _insert(conn, "session_streams", version)
            with pytest.raises(asyncpg.CheckViolationError):
                await _insert(conn, "session_streams", 4)
        finally:
            await conn.execute("DELETE FROM session_streams WHERE customer_id = 'mig0148'")
            await conn.execute("DELETE FROM customers WHERE customer_id = 'mig0148'")


@pytest.mark.asyncio
async def test_the_upgrade_replaces_the_0128_check_found_by_definition(live_db) -> None:
    async with raw_conn() as conn:
        await conn.execute("DROP SCHEMA IF EXISTS mig0148 CASCADE")
        await conn.execute("CREATE SCHEMA mig0148")
        try:
            # 0128's column, verbatim: an inline, default-named check.
            await conn.execute(
                "CREATE TABLE mig0148.session_streams (customer_id TEXT, source_system TEXT,"
                " session_id TEXT, stream_id TEXT,"
                " protocol_version INTEGER NOT NULL CHECK (protocol_version = 2),"
                " prefix_sha256 TEXT)"
            )
            async with conn.transaction():
                await conn.execute("SET LOCAL search_path = mig0148")
                await conn.execute(_migration().UPGRADE_SQL)
            checks = await conn.fetch(
                "SELECT conname, pg_get_constraintdef(oid) AS def FROM pg_constraint"
                " WHERE conrelid = 'mig0148.session_streams'::regclass AND contype = 'c'"
            )
            assert [r["conname"] for r in checks] == ["session_streams_protocol_version_check"]
            await _insert(conn, "mig0148.session_streams", 3)
            with pytest.raises(asyncpg.CheckViolationError):
                await _insert(conn, "mig0148.session_streams", 4)
        finally:
            await conn.execute("DROP SCHEMA IF EXISTS mig0148 CASCADE")


def test_the_downgrade_refuses() -> None:
    with pytest.raises(RuntimeError):
        _migration().downgrade()
