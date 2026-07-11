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

Role discipline (RLS non-superuser cutover, bug #46): in production the
``DATABASE_URL`` should point at the non-privileged ``probe_app`` role,
NOT the ``probe`` superuser. Superuser (and BYPASSRLS) connections
silently bypass FORCE RLS, which would defeat the whole tenant-isolation
story. ``init_pool`` probes the connected role at startup: in any
non-local env a privileged role WARN-logs ``db.superuser_in_managed_env``
and — when ``settings.require_non_superuser_db`` is enabled (opt-in;
the default is warn-only, because the live managed cluster still has
privileged infra connections in flight) — refuses to start
(TenantIsolationError). Flip ``REQUIRE_NON_SUPERUSER_DB=true`` to enforce
fail-closed; see ``docs/database-url-cutover.md`` for the operator switch.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg

from shared.config import Settings, get_settings, validate_boot_secrets
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
    # Fail fast on shipped placeholder secrets in any non-local env —
    # init_pool is the one choke point every service entrypoint passes
    # through (same rationale as the superuser probe below).
    validate_boot_secrets(settings)
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
            try:
                await _log_connected_role(_pool, settings)
            except TenantIsolationError:
                # Fail-closed role guard tripped: don't leave a live pool
                # behind, and don't retry — a privileged DSN is a
                # misconfiguration, not a transient blip.
                guard_failed_pool, _pool = _pool, None
                await guard_failed_pool.close()
                raise
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


async def _log_connected_role(pool: asyncpg.Pool, settings: Settings) -> None:
    """Emit one log line on boot identifying the role the pool connects as.

    Defense-in-depth for the bug #46 cutover (DATABASE_URL switch from
    ``probe`` superuser to non-privileged ``probe_app``). Superuser
    connections silently bypass FORCE RLS, so a misconfigured production
    DSN would produce zero RLS enforcement with no other visible signal.

    Behavior:
    - Always logs ``db.role`` at INFO with role + is_superuser + bypass_rls.
    - WARN-logs ``db.superuser_in_managed_env`` when running on a
      non-local environment as a superuser OR a BYPASSRLS role — the
      operator-visible signal that the cutover hasn't happened yet (or
      has regressed).
    - When ``settings.require_non_superuser_db`` is enabled (opt-in;
      warn-only by default), that same privileged-role detection raises
      TenantIsolationError so the process refuses to boot with RLS
      silently disabled. Enable with REQUIRE_NON_SUPERUSER_DB=true once
      every pooled service is confirmed on a non-privileged role.

    The probe itself stays best-effort: any probe failure is swallowed
    (the boot path mustn't block on a probe). Local dev runs as
    superuser by design, so neither the warning nor the guard fires
    there.
    """
    try:
        async with pool.acquire() as conn:
            role = await conn.fetchval("SELECT current_user")
            is_superuser = await conn.fetchval(
                "SELECT current_setting('is_superuser', true)::bool"
            )
            # BYPASSRLS is a separate role attribute: a non-superuser role
            # can still bypass FORCE RLS with it (e.g. probe_admin).
            # session_user == the authenticated role (the pool's on_connect
            # never SET ROLEs), and pg_roles is world-readable.
            bypass_rls = await conn.fetchval(
                "SELECT rolbypassrls FROM pg_roles WHERE rolname = session_user"
            )
    except Exception as exc:  # pragma: no cover — best-effort probe
        log.debug(
            "db.role_probe_failed",
            error=str(exc),
            error_class=type(exc).__name__,
        )
        # Fail closed when enforcement is on: if we could not PROVE the role
        # is unprivileged, refuse to boot rather than silently trusting a
        # DSN that might bypass RLS. Warn-only mode keeps the best-effort
        # swallow (probe errors must not block a boot that isn't enforcing).
        if settings.require_non_superuser_db and settings.environment != "local":
            raise TenantIsolationError(
                "refusing to start: could not verify the DATABASE_URL role is "
                "non-privileged (role probe failed) while REQUIRE_NON_SUPERUSER_DB "
                "is enabled. Fix the probe error, or set "
                "REQUIRE_NON_SUPERUSER_DB=false to boot warn-only.",
                environment=settings.environment,
                error=str(exc),
            ) from exc
        return

    # Skip the INFO line entirely on `local` -- it pollutes stdout in
    # subprocess-based CLI tests (`tests/synth/test_seed_db.py` asserts empty
    # stdout). Local dev doesn't need the cutover signal anyway.
    privileged = bool(is_superuser) or bool(bypass_rls)
    if settings.environment != "local":
        log.info(
            "db.role",
            role=role,
            is_superuser=bool(is_superuser),
            bypass_rls=bool(bypass_rls),
        )
    if privileged and settings.environment != "local":
        log.warning(
            "db.superuser_in_managed_env",
            role=role,
            environment=settings.environment,
            is_superuser=bool(is_superuser),
            bypass_rls=bool(bypass_rls),
            enforced=settings.require_non_superuser_db,
            remediation=(
                "DATABASE_URL connects as a superuser or BYPASSRLS role, "
                "which bypasses FORCE RLS. Switch to the non-privileged "
                "probe_app role — see docs/database-url-cutover.md."
            ),
        )
        if settings.require_non_superuser_db:
            raise TenantIsolationError(
                "refusing to start: DATABASE_URL connects as a privileged "
                "role (superuser or BYPASSRLS), which silently disables "
                "FORCE RLS tenant isolation. Switch to the non-privileged "
                "probe_app role, or set REQUIRE_NON_SUPERUSER_DB=false to "
                "boot warn-only.",
                role=str(role),
                environment=settings.environment,
                is_superuser=bool(is_superuser),
                bypass_rls=bool(bypass_rls),
            )


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
        # Standalone community mode: fall back to the single configured tenant
        # so RLS stays ON (and is trivially satisfied) when no per-request
        # tenant is resolved. Unset in hosted => raises exactly as before.
        customer_id = get_settings().default_customer_id
        if not customer_id:
            raise TenantIsolationError(
                "with_tenant() requires a non-empty customer_id "
                "(or DEFAULT_CUSTOMER_ID in single-tenant mode)"
            )

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
