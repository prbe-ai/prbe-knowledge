"""Tests for the global ingestion killswitch store.

DB calls are mocked so these run without docker compose. Two layers:
  - cache TTL semantics (single-flight, 30s window, force_refresh bypass)
  - fail-OPEN behavior on missing row, JSON decode error, DB exception
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with a cold cache."""
    from services.system_settings import store

    store.invalidate_cache()
    yield
    store.invalidate_cache()


def _make_pool(rows: list[dict | None] | Exception):
    """Build a mock that mimics shared.db.get_pool().acquire().fetchrow().

    `rows` can be a list (one entry per acquire() call) or an Exception
    that every acquire() raises before fetchrow runs.
    """

    class _Conn:
        def __init__(self, queue: list[dict | None]):
            self._queue = queue
            self.calls = 0

        async def fetchrow(self, *_args, **_kwargs):
            self.calls += 1
            if not self._queue:
                return None
            return self._queue.pop(0)

    class _Pool:
        def __init__(self):
            self.conn = _Conn(list(rows) if isinstance(rows, list) else [])
            self._raise = rows if isinstance(rows, Exception) else None
            self.acquire_calls = 0

        @asynccontextmanager
        async def _ctx(self):
            self.acquire_calls += 1
            if self._raise is not None:
                raise self._raise
            yield self.conn

        def acquire(self):
            return self._ctx()

    return _Pool()


@pytest.mark.asyncio
async def test_returns_enabled_when_row_says_enabled() -> None:
    from services.system_settings import store

    pool = _make_pool([{"value": json.dumps({"enabled": True, "reason": None})}])
    with mock.patch.object(store, "get_pool", return_value=pool):
        ks = await store.get_ingestion_killswitch()
    assert ks.enabled is True
    assert ks.reason is None


@pytest.mark.asyncio
async def test_returns_disabled_with_reason() -> None:
    from services.system_settings import store

    payload = {"enabled": False, "reason": "maintenance window 20:00-21:00 UTC"}
    pool = _make_pool([{"value": json.dumps(payload)}])
    with mock.patch.object(store, "get_pool", return_value=pool):
        ks = await store.get_ingestion_killswitch()
    assert ks.enabled is False
    assert ks.reason == "maintenance window 20:00-21:00 UTC"


@pytest.mark.asyncio
async def test_accepts_dict_value_not_just_str() -> None:
    """asyncpg with a JSONB codec returns dict; without one returns str.
    The store must handle both."""
    from services.system_settings import store

    pool = _make_pool([{"value": {"enabled": False, "reason": "x"}}])
    with mock.patch.object(store, "get_pool", return_value=pool):
        ks = await store.get_ingestion_killswitch()
    assert ks.enabled is False
    assert ks.reason == "x"


@pytest.mark.asyncio
async def test_missing_row_fails_open() -> None:
    """A missing row (e.g. mid-deploy before seed) defaults to enabled.
    Better to keep ingesting than halt every customer over a half-deployed
    state."""
    from services.system_settings import store

    pool = _make_pool([None])
    with mock.patch.object(store, "get_pool", return_value=pool):
        ks = await store.get_ingestion_killswitch()
    assert ks.enabled is True
    assert ks.reason is None


@pytest.mark.asyncio
async def test_db_exception_fails_open() -> None:
    """DB unreachable: log + return enabled rather than halting ingestion."""
    from services.system_settings import store

    pool = _make_pool(RuntimeError("connection refused"))
    with mock.patch.object(store, "get_pool", return_value=pool):
        ks = await store.get_ingestion_killswitch()
    assert ks.enabled is True


@pytest.mark.asyncio
async def test_garbage_json_fails_open() -> None:
    from services.system_settings import store

    pool = _make_pool([{"value": "not valid json {"}])
    with mock.patch.object(store, "get_pool", return_value=pool):
        ks = await store.get_ingestion_killswitch()
    assert ks.enabled is True


@pytest.mark.asyncio
async def test_unexpected_value_type_fails_open() -> None:
    from services.system_settings import store

    pool = _make_pool([{"value": 42}])
    with mock.patch.object(store, "get_pool", return_value=pool):
        ks = await store.get_ingestion_killswitch()
    assert ks.enabled is True


@pytest.mark.asyncio
async def test_cache_hit_within_ttl_skips_db() -> None:
    """Hot path: webhook handler calls this on every POST. The DB must
    only be hit once per 30s window."""
    from services.system_settings import store

    pool = _make_pool([{"value": json.dumps({"enabled": True, "reason": None})}])
    with mock.patch.object(store, "get_pool", return_value=pool):
        await store.get_ingestion_killswitch()
        await store.get_ingestion_killswitch()
        await store.get_ingestion_killswitch()
    # Only the FIRST call hits the pool.
    assert pool.acquire_calls == 1
    assert pool.conn.calls == 1


@pytest.mark.asyncio
async def test_force_refresh_bypasses_cache() -> None:
    """The /api/internal/ingestion-status endpoint uses force_refresh=True
    so admin polling never sees stale cache."""
    from services.system_settings import store

    pool = _make_pool(
        [
            {"value": json.dumps({"enabled": True, "reason": None})},
            {"value": json.dumps({"enabled": False, "reason": "flipped"})},
        ]
    )
    with mock.patch.object(store, "get_pool", return_value=pool):
        ks1 = await store.get_ingestion_killswitch()
        ks2 = await store.get_ingestion_killswitch(force_refresh=True)
    assert ks1.enabled is True
    assert ks2.enabled is False
    assert pool.acquire_calls == 2


@pytest.mark.asyncio
async def test_cache_expires_after_ttl() -> None:
    """After TTL elapses, the next call refetches."""
    from services.system_settings import store

    pool = _make_pool(
        [
            {"value": json.dumps({"enabled": True, "reason": None})},
            {"value": json.dumps({"enabled": False, "reason": "flipped"})},
        ]
    )
    with mock.patch.object(store, "get_pool", return_value=pool):
        ks1 = await store.get_ingestion_killswitch()
        # Fast-forward time past the TTL.
        with mock.patch.object(store, "time") as mock_time:
            mock_time.monotonic.return_value = ks1.fetched_at + store._CACHE_TTL_S + 1
            ks2 = await store.get_ingestion_killswitch()
    assert ks1.enabled is True
    assert ks2.enabled is False
    assert pool.acquire_calls == 2


@pytest.mark.asyncio
async def test_concurrent_callers_single_flight() -> None:
    """Thundering herd: 50 simultaneous webhook handlers all call this on
    a cold cache. The DB must only be hit once."""
    from services.system_settings import store

    pool = _make_pool([{"value": json.dumps({"enabled": True, "reason": None})}])
    with mock.patch.object(store, "get_pool", return_value=pool):
        results = await asyncio.gather(
            *(store.get_ingestion_killswitch() for _ in range(50))
        )
    assert all(ks.enabled is True for ks in results)
    # Single-flight guarantee: only one DB read despite 50 callers.
    assert pool.acquire_calls == 1
