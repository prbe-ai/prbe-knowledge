"""Async Postgres pool + RLS binding.

Every query runs inside `with_tenant()` which sets the GUC `app.current_customer_id`
at transaction start. That GUC powers the RLS policies on graph_nodes / graph_edges.

Non-tenant operations (bootstrap, cron reclaim) use `raw_conn()`.

search_path: prbe-knowledge's tables live in `ag_catalog` (the Apache-AGE
extension's schema; see migration 0066 for the historical context). The
pool's ``on_connect`` hook pins ``search_path = ag_catalog, public, "$user"``
on every fresh connection — defense-in-depth so callers don't need a
``SET search_path`` of their own and so the data plane works under any
role whose default search_path hasn't been pre-set
(``probe`` is set in prbe-backend's ``0005_probe_role_search_path``;
``probe_app`` should be too, but the on_connect hook means a missed
``ALTER ROLE`` won't silently route every query to ``public`` and 503).
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

# Order matches prbe-backend's `0005_probe_role_search_path`: ag_catalog first
# so the AGE-extension-owned tables resolve without a schema-qualified name;
# public second for any third-party extensions (pg_trgm, vector, pg_search)
# that landed there; "$user" last as a courtesy for per-role private schemas.
_CONNECTION_SEARCH_PATH = 'ag_catalog, public, "$user"'


async def apply_connection_setup(conn: asyncpg.Connection) -> None:
    """Per-connection bootstrap. Runs once on connect, NOT per transaction.

    - Pins search_path to ``ag_catalog, public, "$user"`` so app code can
      reference `graph_nodes` (etc.) without schema qualification regardless
      of the connecting role's per-role default. Critical under the
      shared-managed cluster where the data plane connects as ``probe_app``
      and AGE's install-time search_path hijack put every table in
      ``ag_catalog`` (see migration 0066).

    Does NOT set ``app.current_customer_id`` — that's per-tenant context
    and is set per-transaction by `with_tenant(customer_id)`.

    Public so non-pool consumers (e.g. NotifyListener / nightly_trigger
    one-shots) that bypass the pool via ``asyncpg.connect()`` can apply
    the same hook the pool's ``init=`` runs.
    """
    # SET (not SET LOCAL) so the value persists for the connection's
    # lifetime, not just the implicit txn that issued it.
    await conn.execute(f"SET search_path = {_CONNECTION_SEARCH_PATH}")


# Back-compat alias: existing internal callers still reference the
# leading-underscore name. Both point at the same coroutine; the
# public name is preferred for new call sites.
_setup_connection = apply_connection_setup


async def init_pool(settings: Settings | None = None) -> asyncpg.Pool:
    """Initialize the module-level pool. Call once at process start."""
    global _pool
    if _pool is not None:
        return _pool

    settings = settings or get_settings()
    attempts = max(1, settings.db_init_retry_attempts)
    base = settings.db_init_retry_base_seconds
    backoff_cap = settings.db_init_retry_backoff_cap_seconds
    last_exc: BaseException | None = None
    # Short bounded retry: covers transient boot blips (NetworkPolicy
    # settling, DNS, pool limits, a credential race). The ceiling lives in
    # shared.constants (DB_INIT_RETRY_*) so it stays one explicit knob.
    for attempt in range(1, attempts + 1):
        try:
            _pool = await asyncpg.create_pool(
                dsn=settings.database_url,
                min_size=settings.db_pool_min_size,
                max_size=settings.db_pool_max_size,
                command_timeout=settings.db_statement_timeout_ms / 1000,
                timeout=settings.db_connect_timeout_seconds,
                statement_cache_size=0,  # pgbouncer-compatible
                init=_setup_connection,
            )
            return _pool
        except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            backoff = min(base * (2 ** (attempt - 1)), backoff_cap)
            log.warning(
                "db.init_pool.retry",
                attempt=attempt,
                attempts=attempts,
                exc_type=type(exc).__name__,
                backoff_seconds=backoff,
            )
            await asyncio.sleep(backoff)
    raise DatabaseUnavailable(
        f"could not connect to Postgres after {attempts} attempt(s): "
        f"{type(last_exc).__name__}: {last_exc}"
    ) from last_exc


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
