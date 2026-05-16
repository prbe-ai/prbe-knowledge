"""Verify the 0071 migration adds the expected columns to query_traces.

Uses the same live_db + raw_conn pattern as other migration tests in this
repo. The live_db fixture starts a containerised Postgres with all
migrations already applied (alembic upgrade head), so this test inspects
the result rather than re-running the migration.

Note: `alembic_runner.run_migration_at` / `async_pool` / `alembic_config`
fixtures do not exist in this codebase; the plan template was adapted to
match the actual test infrastructure pattern.
"""

from __future__ import annotations

import pytest

from shared.db import raw_conn


@pytest.mark.asyncio
@pytest.mark.integration
async def test_migration_adds_columns(live_db) -> None:
    """All seven 0071 columns exist with the correct types and nullability."""
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = 'query_traces'
              AND column_name IN (
                'grounding_bundle', 'router_raw', 'intents_count',
                'intent_dispatch', 'cache_tokens', 'router_model',
                'failure_recovered'
              )
            ORDER BY column_name
            """
        )

    cols = {r["column_name"]: dict(r) for r in rows}

    assert set(cols.keys()) == {
        "cache_tokens", "failure_recovered", "grounding_bundle",
        "intent_dispatch", "intents_count", "router_model", "router_raw",
    }
    assert cols["failure_recovered"]["is_nullable"] == "NO"
    assert cols["failure_recovered"]["column_default"] == "false"
    assert cols["intents_count"]["data_type"] == "integer"
    assert cols["grounding_bundle"]["data_type"] == "jsonb"
    assert cols["router_raw"]["data_type"] == "jsonb"
    assert cols["intent_dispatch"]["data_type"] == "jsonb"
    assert cols["cache_tokens"]["data_type"] == "jsonb"
    assert cols["router_model"]["data_type"] == "text"
    assert cols["grounding_bundle"]["is_nullable"] == "YES"
    assert cols["router_raw"]["is_nullable"] == "YES"
    assert cols["intents_count"]["is_nullable"] == "YES"
    assert cols["intent_dispatch"]["is_nullable"] == "YES"
    assert cols["cache_tokens"]["is_nullable"] == "YES"
    assert cols["router_model"]["is_nullable"] == "YES"
