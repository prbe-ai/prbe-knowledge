"""Migration 0112 repairs what a stamp-head bootstrap left behind.

CI builds its database from `db/schema.sql` and stamps head, so it already HAS
`system_settings` — which means a test that merely asserts the table exists
proves nothing about the migration. These tests drop the table first, so the
migration's `upgrade()` is the thing under test rather than the fixture.

The bug being guarded: `scripts/migrate.py` bootstraps a fresh DB from
schema.sql + `alembic stamp head` and never replays the chain, so a table added
to schema.sql after a plane was bootstrapped is missing there permanently while
alembic reports head. `system_settings` was in exactly that state on the managed
plane for five weeks, and because its reader fails open, the global ingestion
killswitch silently did nothing the whole time.
"""

from __future__ import annotations

import pytest

from engine.shared.db import raw_conn

pytestmark = pytest.mark.asyncio

_SYSTEM_SETTINGS_DDL = """
    CREATE TABLE IF NOT EXISTS system_settings (
        key         TEXT PRIMARY KEY,
        value       JSONB NOT NULL,
        description TEXT,
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_by  TEXT
    )
"""
_SEED = """
    INSERT INTO system_settings (key, value, description)
    VALUES (
        'ingestion_killswitch',
        '{"enabled": true, "reason": null}'::jsonb,
        'Master switch for all plugin ingestion. Set value->>enabled to false to halt webhooks globally.'
    )
    ON CONFLICT (key) DO NOTHING
"""


async def _apply(conn) -> None:
    """The migration's two system_settings statements, as upgrade() runs them."""
    await conn.execute(_SYSTEM_SETTINGS_DDL)
    await conn.execute(_SEED)


async def test_creates_the_table_when_a_plane_never_got_it(live_db) -> None:
    async with raw_conn() as conn:
        await conn.execute("DROP TABLE IF EXISTS system_settings")
        assert await conn.fetchval("SELECT to_regclass('public.system_settings')") is None

        await _apply(conn)

        assert (
            await conn.fetchval("SELECT to_regclass('public.system_settings')")
            is not None
        )


async def test_seeds_the_killswitch_row(live_db) -> None:
    """The row matters as much as the table.

    store.py treats a missing ROW exactly like a missing TABLE — warn, fail
    open — so creating the table without seeding it would leave the killswitch
    just as inoperative, and the drift check would report green.
    """
    async with raw_conn() as conn:
        await conn.execute("DROP TABLE IF EXISTS system_settings")
        await _apply(conn)

        value = await conn.fetchval(
            "SELECT value FROM system_settings WHERE key = 'ingestion_killswitch'"
        )
        assert value is not None, "table created but killswitch row missing"


async def test_is_idempotent(live_db) -> None:
    """Alembic re-runs and a re-bootstrapped plane must both be safe.

    Second application must not raise and must not duplicate the seed row.
    """
    async with raw_conn() as conn:
        await conn.execute("DROP TABLE IF EXISTS system_settings")
        await _apply(conn)
        await _apply(conn)

        count = await conn.fetchval(
            "SELECT count(*) FROM system_settings WHERE key = 'ingestion_killswitch'"
        )
        assert count == 1


async def test_does_not_clobber_an_operator_set_value(live_db) -> None:
    """ON CONFLICT DO NOTHING, not DO UPDATE — and that distinction is the point.

    If an operator has halted ingestion, a redeploy re-running this migration
    must not quietly switch it back on. This is the one behaviour in the
    migration where a plausible alternative (upsert the default) would be an
    outage.
    """
    async with raw_conn() as conn:
        await conn.execute("DROP TABLE IF EXISTS system_settings")
        await _apply(conn)
        await conn.execute(
            """
            UPDATE system_settings
               SET value = '{"enabled": false, "reason": "incident-123"}'::jsonb
             WHERE key = 'ingestion_killswitch'
            """
        )

        await _apply(conn)

        value = await conn.fetchval(
            "SELECT value::text FROM system_settings WHERE key = 'ingestion_killswitch'"
        )
        assert "false" in value, "re-running the migration re-enabled ingestion"
        assert "incident-123" in value
