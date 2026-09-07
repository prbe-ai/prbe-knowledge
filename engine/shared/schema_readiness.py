"""Keep new ingestion roles unready until the additive GitHub schema is present.

Managed Helm runs migrations in a post-upgrade hook, without ``--wait``. New
pods may start first, so neither HTTP readiness nor any composed worker may
start using the 0127 tables before that transaction commits. Existing pods
continue serving their legacy paths while replacement pods wait here.
"""

from __future__ import annotations

import asyncio

from engine.shared.db import raw_conn
from engine.shared.logging import get_logger

log = get_logger(__name__)

SCHEMA_WAIT_SECONDS = 120.0
SCHEMA_POLL_SECONDS = 1.0
_SCHEMA_READY_SQL = """
    WITH required(table_name, columns) AS (VALUES
        ('github_installations', ARRAY['history_lease_id', 'history_heartbeat_at',
                                      'history_last_claimed_at']),
        ('github_backfill_jobs', ARRAY[]::text[]),
        ('github_document_bindings', ARRAY['repository', 'live_present', 'history_present']),
        ('github_worker_capabilities', ARRAY[]::text[]),
        ('github_source_gates', ARRAY[]::text[]),
        ('github_backfill_retry_receipts', ARRAY[]::text[]),
        ('ingestion_queue', ARRAY['github_installation_id', 'github_generation',
                                 'github_job_id', 'github_payload', 'github_lease_id'])
    )
    SELECT bool_and(to_regclass(table_name) IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM unnest(columns) AS required_column(name)
         WHERE NOT EXISTS (
            SELECT 1 FROM pg_attribute
             WHERE attrelid = to_regclass(table_name)
               AND attname = required_column.name AND attnum > 0 AND NOT attisdropped
         )
    )) FROM required
"""


class GitHubSchemaNotReady(RuntimeError):
    """The role must not publish readiness or start processing work yet."""


async def github_control_schema_ready() -> bool:
    """Check the same relations workers resolve through their search path.

    Managed databases can keep these tables in ag_catalog rather than public.
    Resolve each visible relation independently, including the queue altered by
    the migration; checking only current_schema() misses mixed-schema installs.
    """
    async with raw_conn() as conn:
        return bool(await conn.fetchval(_SCHEMA_READY_SQL))


async def wait_for_github_control_schema(
    *,
    timeout_seconds: float = SCHEMA_WAIT_SECONDS,
    poll_seconds: float = SCHEMA_POLL_SECONDS,
) -> None:
    """Wait for the 0127 transaction, retaining no DB connection while sleeping.

    Test the additive schema floor rather than one exact Alembic head so later
    migrations remain compatible. Fresh bootstrap and upgrade both create these
    tables and queue columns before this binary may handle legacy or v2 work.
    The whole wait, including database acquisition/query time, is bounded.
    """
    waiting = False
    try:
        async with asyncio.timeout(timeout_seconds):
            while True:
                if await github_control_schema_ready():
                    return
                if not waiting:
                    log.info("startup.waiting_for_github_schema", migration="0127")
                    waiting = True
                await asyncio.sleep(poll_seconds)
    except TimeoutError:
        raise GitHubSchemaNotReady(
            "GitHub control schema 0127 is not ready; complete database migrations "
            "before starting this role"
        ) from None
