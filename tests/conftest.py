"""Shared test fixtures.

Tests that need a live Postgres + MinIO expect:

    docker compose up -d
    scripts/neon-migrate.sh local

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
    "DATABASE_URL": "postgresql://prbe:prbe@localhost:5432/prbe_knowledge",
    "R2_ENDPOINT_URL": "http://localhost:9000",
    "R2_ACCESS_KEY_ID": "minioadmin",
    "R2_SECRET_ACCESS_KEY": "minioadmin",
    "R2_BUCKET_PREFIX": "prbe-test",
    "OPENAI_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "TOKEN_ENCRYPTION_KEY": "VQzt8cN0Q8dUJYwQZUWaGKg_uvDyF-58DyHJ6m5f8ww=",
}
for _k, _v in _TEST_ENV.items():
    os.environ[_k] = _v

from shared.config import Settings, get_settings  # noqa: E402
from shared.db import close_pool, init_pool, raw_conn  # noqa: E402


@pytest.fixture(scope="session")
def settings() -> Settings:
    get_settings.cache_clear()  # type: ignore[attr-defined]
    return Settings()


@pytest_asyncio.fixture
async def live_db(settings: Settings) -> AsyncIterator[None]:
    """Spins the async pool up, truncates tenant-scoped tables, yields, tears down."""
    await init_pool(settings)
    async with raw_conn() as conn:
        await conn.execute(
            """
            TRUNCATE
                chunks, documents, ingestion_queue, ingestion_events,
                failed_chunks, acl_snapshots, graph_edges, graph_nodes,
                integration_tokens, backfill_state, query_cache, audit_log,
                customers
            RESTART IDENTITY CASCADE
            """
        )
    yield
    await close_pool()
