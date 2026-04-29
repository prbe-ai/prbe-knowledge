"""Cold-start retry: `init_pool` must tolerate transient connect failures.

Background: the worker's first `asyncpg.create_pool` to Neon can fail when
the endpoint is scaled-to-zero and slow to wake. `init_pool` retries with
exponential backoff before raising `DatabaseUnavailable`.
"""

from __future__ import annotations

import pytest

from shared import db as db_module
from shared.config import Settings
from shared.exceptions import DatabaseUnavailable


class _FakePool:
    async def close(self) -> None:  # pragma: no cover — never called in these tests
        return None


def _settings(attempts: int = 6) -> Settings:
    return Settings(
        database_url="postgresql://prbe:prbe@localhost:5432/prbe_knowledge",
        db_init_retry_attempts=attempts,
        db_init_retry_base_seconds=0.01,
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

    with pytest.raises(DatabaseUnavailable):
        await db_module.init_pool(_settings(attempts=3))

    assert calls["n"] == 3
    db_module.reset_pool()
