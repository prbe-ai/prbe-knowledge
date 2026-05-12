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
from shared.exceptions import DatabaseUnavailable


class _FakePool:
    async def close(self) -> None:  # pragma: no cover — never called in these tests
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
