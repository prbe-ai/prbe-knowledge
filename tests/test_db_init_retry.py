"""Cold-start retry: `init_pool` must tolerate transient connect failures.

Background: the first `asyncpg.create_pool` after a pod boot can fail on a
transient blip (NetworkPolicy settling, DNS, pool limits, a credential
race). `init_pool` retries with a *short, bounded* exponential backoff
before raising `DatabaseUnavailable` — the ceiling is the
`DB_INIT_RETRY_*` constants in `shared.constants`, deliberately single-digit
seconds so a blip recovers fast and a real outage surfaces a readable error
rather than a silent ~minute-long hang.
"""

from __future__ import annotations

import pytest

from shared import db as db_module
from shared.config import Settings
from shared.constants import (
    DB_INIT_RETRY_ATTEMPTS,
    DB_INIT_RETRY_BACKOFF_CAP_SECONDS,
    DB_INIT_RETRY_BASE_SECONDS,
)
from shared.exceptions import DatabaseUnavailable, TenantIsolationError


class _FakePool:
    async def close(self) -> None:
        return None


def _settings(attempts: int = 6, connect_timeout: float = 0.5) -> Settings:
    return Settings(
        database_url="postgresql://prbe:prbe@localhost:5432/prbe_knowledge",
        db_init_retry_attempts=attempts,
        db_init_retry_base_seconds=0.01,
        db_connect_timeout_seconds=connect_timeout,
    )


@pytest.mark.asyncio
async def test_init_pool_retries_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    db_module.reset_pool()
    calls = {"n": 0}

    async def fake_create_pool(*args: object, **kwargs: object) -> _FakePool:
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("connection refused")
        return _FakePool()

    monkeypatch.setattr(db_module.asyncpg, "create_pool", fake_create_pool)

    pool = await db_module.init_pool(_settings())

    assert calls["n"] == 3
    assert isinstance(pool, _FakePool)
    db_module.reset_pool()


@pytest.mark.asyncio
async def test_init_pool_raises_database_unavailable_after_all_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_module.reset_pool()
    calls = {"n": 0}

    async def fake_create_pool(*args: object, **kwargs: object) -> _FakePool:
        calls["n"] += 1
        raise OSError("connection refused")

    monkeypatch.setattr(db_module.asyncpg, "create_pool", fake_create_pool)

    with pytest.raises(DatabaseUnavailable) as excinfo:
        await db_module.init_pool(_settings(attempts=3))

    assert calls["n"] == 3
    # Exhaustion must surface a readable startup error, not a silent hang.
    msg = str(excinfo.value)
    assert "3 attempt" in msg
    assert "connection refused" in msg
    db_module.reset_pool()


def test_default_backoff_ceiling_is_tight() -> None:
    """The shipped defaults must keep total boot retry time single-digit
    seconds — a transient blip recovers fast; we don't sit in a ~minute-long
    crash-loop tail anymore."""
    assert DB_INIT_RETRY_ATTEMPTS <= 5
    assert DB_INIT_RETRY_BASE_SECONDS <= 1.0
    assert DB_INIT_RETRY_BACKOFF_CAP_SECONDS <= 10.0

    # Sum of the per-retry sleeps the loop would perform (attempts - 1
    # retries), each capped at DB_INIT_RETRY_BACKOFF_CAP_SECONDS.
    total_sleep = sum(
        min(DB_INIT_RETRY_BASE_SECONDS * (2 ** (i - 1)), DB_INIT_RETRY_BACKOFF_CAP_SECONDS)
        for i in range(1, DB_INIT_RETRY_ATTEMPTS)
    )
    assert total_sleep <= 15.0, f"boot retry sleep ceiling too long: {total_sleep}s"

    # And the Settings defaults track the constants.
    settings = Settings(database_url="postgresql://prbe:prbe@localhost:5432/prbe_knowledge")
    assert settings.db_init_retry_attempts == DB_INIT_RETRY_ATTEMPTS
    assert settings.db_init_retry_base_seconds == DB_INIT_RETRY_BASE_SECONDS
    assert settings.db_init_retry_backoff_cap_seconds == DB_INIT_RETRY_BACKOFF_CAP_SECONDS


@pytest.mark.asyncio
async def test_init_pool_backoff_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-attempt backoff never exceeds the cap, even if base is large."""
    db_module.reset_pool()
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def always_fail(*args: object, **kwargs: object) -> _FakePool:
        raise OSError("connection refused")

    monkeypatch.setattr(db_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(db_module.asyncpg, "create_pool", always_fail)

    settings = Settings(
        database_url="postgresql://prbe:prbe@localhost:5432/prbe_knowledge",
        db_init_retry_attempts=6,
        db_init_retry_base_seconds=30.0,  # would explode to 30,60,120,... uncapped
        db_init_retry_backoff_cap_seconds=2.0,
    )
    with pytest.raises(DatabaseUnavailable):
        await db_module.init_pool(settings)

    assert sleeps == [2.0, 2.0, 2.0, 2.0, 2.0]  # attempts - 1 retries, all capped
    db_module.reset_pool()


@pytest.mark.asyncio
async def test_init_pool_passes_connect_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    db_module.reset_pool()
    captured: dict[str, object] = {}

    async def fake_create_pool(*args: object, **kwargs: object) -> _FakePool:
        captured.update(kwargs)
        return _FakePool()

    monkeypatch.setattr(db_module.asyncpg, "create_pool", fake_create_pool)

    settings = _settings(connect_timeout=0.25)
    await db_module.init_pool(settings)

    assert captured["timeout"] == settings.db_connect_timeout_seconds
    db_module.reset_pool()


# ---------------------------------------------------------------------------
# Role-warning probe (bug #46 cutover guard)
# ---------------------------------------------------------------------------
#
# `init_pool` issues a one-shot warning at boot when DATABASE_URL connects
# as a superuser in any non-local environment. The probe is best-effort
# (boot must not block on its failure) but it must fire when it should
# and stay quiet otherwise — both branches are pinned here so an
# accidental refactor doesn't silently drop the cutover signal.


class _FakeConn:
    def __init__(self, role: str, is_superuser: bool, bypass_rls: bool = False) -> None:
        self._role = role
        self._is_superuser = is_superuser
        self._bypass_rls = bypass_rls

    async def fetchval(self, query: str, *args: object) -> object:
        if "rolbypassrls" in query:
            return self._bypass_rls
        if "current_user" in query:
            return self._role
        if "is_superuser" in query:
            return self._is_superuser
        return None


class _FakeAcquireCtx:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *exc: object) -> None:
        return None


class _RoleProbePool(_FakePool):
    """FakePool that answers the role probe with a fixed role + flags."""

    def __init__(self, role: str, is_superuser: bool, bypass_rls: bool = False) -> None:
        self._conn = _FakeConn(role, is_superuser, bypass_rls)

    def acquire(self) -> _FakeAcquireCtx:
        return _FakeAcquireCtx(self._conn)


@pytest.mark.asyncio
async def test_init_pool_warns_when_superuser_in_non_local_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import structlog.testing

    db_module.reset_pool()

    async def fake_create_pool(*_args: object, **_kwargs: object) -> _RoleProbePool:
        return _RoleProbePool(role="probe", is_superuser=True)

    monkeypatch.setattr(db_module.asyncpg, "create_pool", fake_create_pool)

    settings = Settings(
        database_url="postgresql://probe:probe@neon/prbe",
        environment="main",
        db_init_retry_attempts=1,
        db_init_retry_base_seconds=0.01,
        require_non_superuser_db=False,  # pin the warn-only escape hatch
    )
    with structlog.testing.capture_logs() as logs:
        await db_module.init_pool(settings)

    events = [entry.get("event") for entry in logs]
    assert "db.superuser_in_managed_env" in events, events
    db_module.reset_pool()


@pytest.mark.asyncio
async def test_init_pool_refuses_superuser_when_guard_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in fail-closed: with require_non_superuser_db=True, a superuser DSN
    in a non-local env refuses to boot and closes the pool. (The default is
    warn-only — see test_init_pool_warns_when_superuser_in_non_local_env.)"""
    db_module.reset_pool()

    async def fake_create_pool(*_args: object, **_kwargs: object) -> _RoleProbePool:
        return _RoleProbePool(role="probe", is_superuser=True)

    monkeypatch.setattr(db_module.asyncpg, "create_pool", fake_create_pool)

    settings = Settings(
        database_url="postgresql://probe:guard@neon/prbe",
        environment="main",
        require_non_superuser_db=True,
        db_init_retry_attempts=1,
        db_init_retry_base_seconds=0.01,
    )
    with pytest.raises(TenantIsolationError):
        await db_module.init_pool(settings)

    # The guard must not leave a live pool behind.
    assert db_module._pool is None
    db_module.reset_pool()


@pytest.mark.asyncio
async def test_init_pool_refuses_bypassrls_non_superuser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BYPASSRLS bypasses FORCE RLS without rolsuper — the guard must
    catch it too (e.g. a probe_admin-shaped role in DATABASE_URL)."""
    db_module.reset_pool()

    async def fake_create_pool(*_args: object, **_kwargs: object) -> _RoleProbePool:
        return _RoleProbePool(role="probe_admin", is_superuser=False, bypass_rls=True)

    monkeypatch.setattr(db_module.asyncpg, "create_pool", fake_create_pool)

    settings = Settings(
        database_url="postgresql://probe_admin:guard@neon/prbe",
        environment="main",
        require_non_superuser_db=True,
        db_init_retry_attempts=1,
        db_init_retry_base_seconds=0.01,
    )
    with pytest.raises(TenantIsolationError):
        await db_module.init_pool(settings)

    assert db_module._pool is None
    db_module.reset_pool()


@pytest.mark.asyncio
async def test_init_pool_quiet_when_probe_app_in_managed_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import structlog.testing

    db_module.reset_pool()

    async def fake_create_pool(*_args: object, **_kwargs: object) -> _RoleProbePool:
        return _RoleProbePool(role="probe_app", is_superuser=False)

    monkeypatch.setattr(db_module.asyncpg, "create_pool", fake_create_pool)

    settings = Settings(
        database_url="postgresql://probe_app:x@neon/prbe",
        environment="main",
        db_init_retry_attempts=1,
        db_init_retry_base_seconds=0.01,
    )
    with structlog.testing.capture_logs() as logs:
        await db_module.init_pool(settings)

    events = [entry.get("event") for entry in logs]
    assert "db.superuser_in_managed_env" not in events, events
    # The neutral INFO probe still fires so ops can see the role.
    assert "db.role" in events, events
    db_module.reset_pool()


@pytest.mark.asyncio
async def test_init_pool_quiet_when_superuser_but_local_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import structlog.testing

    db_module.reset_pool()

    async def fake_create_pool(*_args: object, **_kwargs: object) -> _RoleProbePool:
        return _RoleProbePool(role="prbe", is_superuser=True)

    monkeypatch.setattr(db_module.asyncpg, "create_pool", fake_create_pool)

    settings = Settings(
        database_url="postgresql://prbe:prbe@localhost/prbe_knowledge",
        environment="local",
        db_init_retry_attempts=1,
        db_init_retry_base_seconds=0.01,
    )
    with structlog.testing.capture_logs() as logs:
        await db_module.init_pool(settings)

    events = [entry.get("event") for entry in logs]
    assert "db.superuser_in_managed_env" not in events, events
    db_module.reset_pool()
