"""Migration assertions for the wiki v4 schema (migration 0041).

Tests the post-upgrade shape: triage_targets gone, source_ts NOT NULL,
dlq_reason / dlq_at present, status CHECK accepts dlq + synthesis_skipped
(but rejects verifier_rejected), composite index ix_wsq_drain_cursor
exists.

Doesn't reapply the migration — it just inspects the result of the
standard local migrate (the `live_db` fixture's containerized Postgres
already ran upgrade head).

A round-trip test would require running `alembic downgrade -1` mid-test,
which would corrupt sibling tests' DB state. The integration test here
covers the upgrade shape; downgrade correctness is exercised by hand
against a scratch DB before deploy (the standard alembic discipline).
"""

from __future__ import annotations

import pytest

from engine.shared.db import raw_conn


@pytest.mark.asyncio
async def test_wsq_columns_match_v4_shape(live_db) -> None:
    """source_ts NOT NULL; dlq_reason/dlq_at nullable; triage_targets gone."""
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'wiki_synthesis_queue'
            """
        )
    cols = {r["column_name"]: (r["data_type"], r["is_nullable"]) for r in rows}

    # Removed by 0041.
    assert "triage_targets" not in cols

    # New columns.
    assert cols["source_ts"][0] == "timestamp with time zone"
    assert cols["source_ts"][1] == "NO"
    assert cols["dlq_reason"] == ("text", "YES")
    assert cols["dlq_at"][0] == "timestamp with time zone"
    assert cols["dlq_at"][1] == "YES"


@pytest.mark.asyncio
async def test_wsq_status_check_accepts_v4_states(live_db) -> None:
    """The CHECK constraint accepts dlq + synthesis_skipped, rejects
    the v3 verifier_rejected name."""
    async with raw_conn() as conn:
        # Seed a customer for the FK.
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'mig-v4-cust', 'h') ON CONFLICT DO NOTHING",
            "mig-v4-cust",
        )
        # Insert with status='dlq' must succeed.
        await conn.execute(
            """
            INSERT INTO wiki_synthesis_queue (
                customer_id, doc_id, doc_version, source_system, doc_type,
                status, source_ts, dlq_reason, dlq_at
            )
            VALUES ($1, 'd:1', 1, 'github', 'github.commit',
                    'dlq', NOW(), 'agent.test', NOW())
            """,
            "mig-v4-cust",
        )
        # Insert with status='synthesis_skipped' must succeed.
        await conn.execute(
            """
            INSERT INTO wiki_synthesis_queue (
                customer_id, doc_id, doc_version, source_system, doc_type,
                status, source_ts
            )
            VALUES ($1, 'd:2', 1, 'github', 'github.commit',
                    'synthesis_skipped', NOW())
            """,
            "mig-v4-cust",
        )
        # Insert with the legacy 'verifier_rejected' must fail the CHECK.
        # asyncpg raises CheckViolationError but matching the bare class would
        # require importing asyncpg.exceptions; expectations against the
        # message keeps the test loose-coupled.
        from asyncpg.exceptions import CheckViolationError

        with pytest.raises(CheckViolationError):
            await conn.execute(
                """
                INSERT INTO wiki_synthesis_queue (
                    customer_id, doc_id, doc_version, source_system, doc_type,
                    status, source_ts
                )
                VALUES ($1, 'd:3', 1, 'github', 'github.commit',
                        'verifier_rejected', NOW())
                """,
                "mig-v4-cust",
            )
        # Cleanup.
        await conn.execute(
            "DELETE FROM wiki_synthesis_queue WHERE customer_id = $1",
            "mig-v4-cust",
        )
        await conn.execute(
            "DELETE FROM customers WHERE customer_id = $1", "mig-v4-cust"
        )


@pytest.mark.asyncio
async def test_wsq_drain_cursor_index_exists(live_db) -> None:
    """The composite index ix_wsq_drain_cursor backs next_events()."""
    async with raw_conn() as conn:
        names = {
            r["indexname"]
            for r in await conn.fetch(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'wiki_synthesis_queue'"
            )
        }
    assert "ix_wsq_drain_cursor" in names
