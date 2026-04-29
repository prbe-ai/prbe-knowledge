"""Async Postgres pool + RLS binding.

Every query runs inside `with_tenant()` which sets the GUC `app.current_customer_id`
at transaction start. That GUC powers the RLS policies on graph_nodes / graph_edges.

Non-tenant operations (bootstrap, cron reclaim) use `raw_conn()`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg

from shared.config import Settings, get_settings
from shared.exceptions import DatabaseUnavailable, TenantIsolationError
from shared.logging import get_logger

log = get_logger(__name__)

_pool: asyncpg.Pool | None = None


async def init_pool(settings: Settings | None = None) -> asyncpg.Pool:
    """Initialize the module-level pool. Call once at process start."""
    global _pool
    if _pool is not None:
        return _pool

    settings = settings or get_settings()
    attempts = max(1, settings.db_init_retry_attempts)
    base = settings.db_init_retry_base_seconds
    last_exc: BaseException | None = None
    # Retry tolerates Neon cold-wake on first connect after scale-to-zero.
    for attempt in range(1, attempts + 1):
        try:
            _pool = await asyncpg.create_pool(
                dsn=settings.database_url,
                min_size=settings.db_pool_min_size,
                max_size=settings.db_pool_max_size,
                command_timeout=settings.db_statement_timeout_ms / 1000,
                statement_cache_size=0,  # pgbouncer-compatible
            )
            return _pool
        except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            backoff = base * (2 ** (attempt - 1))
            log.warning(
                "db.init_pool.retry",
                attempt=attempt,
                attempts=attempts,
                exc_type=type(exc).__name__,
                backoff_seconds=backoff,
            )
            await asyncio.sleep(backoff)
    raise DatabaseUnavailable(str(last_exc)) from last_exc


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def reset_pool() -> None:
    """Drop the module reference without gracefully closing.

    Used only by tests that reuse module state across event loops — a
    graceful close would touch the pool's creation loop, which may
    already be closed and raises `RuntimeError: Event loop is closed`.
    """
    global _pool
    _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise DatabaseUnavailable("pool not initialized — call init_pool() first")
    return _pool


@asynccontextmanager
async def raw_conn() -> AsyncIterator[asyncpg.Connection]:
    """Connection without any tenant GUC set. For bootstrap, cron, infra ops only."""
    async with get_pool().acquire() as conn:
        yield conn


@asynccontextmanager
async def with_tenant(customer_id: str) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a connection, start a transaction, bind the tenant GUC.

    RLS policies on graph_nodes / graph_edges require this to be set.
    Anything that does `SELECT` on graph tables outside this context will
    return zero rows silently — that's by design.
    """
    if not customer_id:
        raise TenantIsolationError("with_tenant() requires a non-empty customer_id")

    async with get_pool().acquire() as conn, conn.transaction():
        # set_config with is_local=true scopes the GUC to this tx only.
        await conn.execute(
            "SELECT set_config('app.current_customer_id', $1, true)",
            customer_id,
        )
        yield conn


async def health_check() -> bool:
    try:
        async with raw_conn() as conn:
            val = await conn.fetchval("SELECT 1")
            return val == 1
    except Exception:
        return False
