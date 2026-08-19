"""Shared test fixtures.

Tests that need a live Postgres + MinIO expect:

    docker compose up -d        # includes a one-shot `migrate` service
    # (or run the migration manually: python scripts/migrate.py)

The `live_db` fixture truncates Phase 0 tables between runs so tests start clean.

Important: we override env vars at module import so they beat any `.env` file
pydantic-settings would otherwise read (the user's real .env may point at Neon
prod, which tests must never touch).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

# Force local/test env vars BEFORE anything imports Settings. Env vars override
# any `.env` file on the filesystem in pydantic-settings' lookup order.
_TEST_ENV = {
    "ENVIRONMENT": "local",
    # PRBE_TEST_DATABASE_URL lets concurrent checkouts on one machine run
    # against their own database instead of racing each other's migrations
    # in the shared compose Postgres (the container name is fixed, so two
    # worktrees otherwise share one schema). ENFORCED local-only below —
    # never a way to point destructive test fixtures at a real deployment.
    "DATABASE_URL": os.environ.get(
        "PRBE_TEST_DATABASE_URL",
        "postgresql://prbe:prbe@localhost:5432/prbe_knowledge",
    ),
    "R2_ENDPOINT_URL": "http://localhost:9000",
    "R2_ACCESS_KEY_ID": "minioadmin",
    "R2_SECRET_ACCESS_KEY": "minioadmin",
    "R2_BUCKET_PREFIX": "prbe-test",
    "OPENAI_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "TOKEN_ENCRYPTION_KEY": "VQzt8cN0Q8dUJYwQZUWaGKg_uvDyF-58DyHJ6m5f8ww=",
}
# The override's whole point is OTHER local databases, so the guard pins
# the host, not the DSN: a CI misconfiguration or poisoned environment
# must not aim the TRUNCATE-everything fixtures at a real deployment.
_TEST_DB_HOST = (_TEST_ENV["DATABASE_URL"].split("@", 1)[-1]).split("/", 1)[0].split(":", 1)[0]
if _TEST_DB_HOST not in ("localhost", "127.0.0.1", "::1"):
    raise RuntimeError(
        "PRBE_TEST_DATABASE_URL must point at localhost — refusing to run "
        f"destructive test fixtures against host {_TEST_DB_HOST!r}"
    )

for _k, _v in _TEST_ENV.items():
    os.environ[_k] = _v

# Mirror service boot: importing the handlers package fires the
# @register_connector decorators, which also populate shared.source_registry
# (per-source doc_type prefix / priority / decay profiles). Retrieval tests
# (fusion decay, doc-type resolver) read those profiles.
import kb.handlers  # noqa: E402,F401
from engine.shared import db as db_module  # noqa: E402
from engine.shared.config import Settings, get_settings  # noqa: E402


@pytest.fixture(scope="session")
def anyio_backend() -> str:  # pragma: no cover — pytest-asyncio auto-mode
    return "asyncio"


@pytest.fixture(scope="session")
def settings() -> Settings:
    get_settings.cache_clear()  # type: ignore[attr-defined]
    return Settings()


TRUNCATE_SQL = """
    TRUNCATE TABLE
        graph_edges,
        graph_nodes,
        audit_log,
        usage_events,
        query_traces,
        ingestion_events,
        failed_chunks,
        integration_tokens,
        backfill_state,
        manual_uploads,
        ingestion_queue,
        acl_snapshots,
        chunks,
        directed_vectors,
        documents,
        customer_source_mapping,
        code_repo_state,
        customers
    RESTART IDENTITY CASCADE;
"""
# `incident_investigations` was listed here until migration 0092
# (`drop_incident_pivot_tables`) dropped it along with
# `customer_incident_mcp_servers`, `wiki_review_queue` and
# `customer_postmortem_templates` -- the only one of the four this list
# ever named. TRUNCATE takes the whole list or none, so a correctly
# migrated database failed EVERY live-DB test at fixture setup with
# `UndefinedTableError`, before a single assertion ran.


@pytest_asyncio.fixture
async def live_db(settings: Settings) -> AsyncIterator[None]:
    """Initialize a fresh pool on the current event loop, truncate, yield, close."""
    # Drop any pool left over from a prior test — its loop is already closed,
    # so a graceful close() would crash. Reset the reference, then init fresh.
    db_module.reset_pool()
    await db_module.init_pool(settings)
    async with db_module.raw_conn() as conn:
        await conn.execute(TRUNCATE_SQL)
    try:
        yield None
    finally:
        async with db_module.raw_conn() as conn:
            await conn.execute(TRUNCATE_SQL)
        await db_module.close_pool()
