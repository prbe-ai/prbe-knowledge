"""Polling-scheduler framework tests (PR B foundation).

Covers the framework only — no per-source poller is registered here.
Tests use a stub poller that registers as ``SourceSystem.SLACK`` for
the duration of the test (then unregisters) so the scheduler's
dispatch + cursor management is exercised end-to-end without any
real upstream API call.

Test surface:
  * registry: register / get / re-register / collision
  * cursor helpers: list_due_cursors / load_cursor / advance_cursor /
    stamp_error
  * scheduler.tick_once: happy path, error path, poller raise,
    no-poller-registered skip, document sink invocation
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

import services.ingestion.polling.base as polling_base
from services.ingestion.polling.base import (
    BasePoller,
    PollResult,
    get_poller,
    register_poller,
    registered_sources,
)
from services.ingestion.polling.cursors import (
    advance_cursor,
    list_due_cursors,
    load_cursor,
    stamp_error,
)
from services.ingestion.polling.scheduler import PollScheduler
from shared.constants import SourceSystem
from shared.db import raw_conn, with_tenant

_TENANT = "test-polling-tenant"


@pytest_asyncio.fixture(autouse=True)
async def _clean_cursors(live_db):
    """Wipe ingestion_cursors before + after each test. live_db already
    truncates the broad set of per-tenant tables but doesn't touch this
    one — it's new in migration 0072 and not in the conftest list."""
    async with raw_conn() as conn:
        await conn.execute("TRUNCATE TABLE ingestion_cursors")
    yield
    async with raw_conn() as conn:
        await conn.execute("TRUNCATE TABLE ingestion_cursors")


@pytest_asyncio.fixture
async def _seed_customer():
    """The cursor rows FK-cascade off customers.customer_id, so we need
    a row in `customers` to satisfy the constraint. The truncate fixture
    above wipes ingestion_cursors but not customers — insert idempotently."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash, status)
            VALUES ($1, 'Polling Test Co', md5('test-polling-key'), 'active')
            ON CONFLICT (customer_id) DO NOTHING
            """,
            _TENANT,
        )
    yield


# --- registry ------------------------------------------------------------


class _StubPoller(BasePoller):
    """Test stub. Returns whatever the test put into _next_result."""

    source = SourceSystem.SLACK
    _next_result: PollResult = PollResult(documents=[], next_cursor=None)
    _calls: list[tuple[str, str, str | None]] = []

    @classmethod
    def reset(cls) -> None:
        cls._next_result = PollResult(documents=[], next_cursor=None)
        cls._calls = []

    async def poll(
        self,
        *,
        customer_id: str,
        resource_id: str,
        cursor: str | None,
    ) -> PollResult:
        type(self)._calls.append((customer_id, resource_id, cursor))
        return type(self)._next_result


@pytest.fixture
def _register_stub_poller():
    """Register the stub for the duration of the test; unregister after.
    Side-stepping `register_poller`'s collision check by clearing the
    registry entry directly during cleanup."""
    _StubPoller.reset()
    register_poller(SourceSystem.SLACK, _StubPoller)
    yield _StubPoller
    polling_base._REGISTRY.pop(SourceSystem.SLACK, None)


def test_register_get_poller():
    register_poller(SourceSystem.GITHUB, _StubPoller)
    try:
        assert get_poller(SourceSystem.GITHUB) is _StubPoller
        assert SourceSystem.GITHUB in registered_sources()
    finally:
        polling_base._REGISTRY.pop(SourceSystem.GITHUB, None)


def test_register_poller_idempotent_same_class():
    register_poller(SourceSystem.GITHUB, _StubPoller)
    try:
        # Re-registering the SAME class is a no-op, not a raise.
        register_poller(SourceSystem.GITHUB, _StubPoller)
        assert get_poller(SourceSystem.GITHUB) is _StubPoller
    finally:
        polling_base._REGISTRY.pop(SourceSystem.GITHUB, None)


def test_register_poller_collision_raises():
    class _OtherPoller(BasePoller):
        source = SourceSystem.GITHUB

        async def poll(self, *, customer_id, resource_id, cursor):
            return PollResult(documents=[])

    register_poller(SourceSystem.GITHUB, _StubPoller)
    try:
        with pytest.raises(RuntimeError, match="already registered"):
            register_poller(SourceSystem.GITHUB, _OtherPoller)
    finally:
        polling_base._REGISTRY.pop(SourceSystem.GITHUB, None)


def test_get_poller_unknown_returns_none():
    assert get_poller(SourceSystem.MANUAL_UPLOAD) is None


# --- cursor helpers ------------------------------------------------------


@pytest.mark.asyncio
async def test_advance_cursor_inserts_first_then_updates(_seed_customer):
    # First call inserts.
    await advance_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="C123",
        new_cursor_value="ts:1000",
    )
    loaded = await load_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="C123",
    )
    assert loaded is not None
    assert loaded.cursor_value == "ts:1000"
    assert loaded.last_error is None

    # Second call updates the same row (composite PK).
    await advance_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="C123",
        new_cursor_value="ts:2000",
    )
    loaded = await load_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="C123",
    )
    assert loaded.cursor_value == "ts:2000"


@pytest.mark.asyncio
async def test_advance_cursor_none_keeps_existing_value(_seed_customer):
    """Pollers pass next_cursor=None when the upstream returned empty;
    the existing value must stick (COALESCE in the UPSERT)."""
    await advance_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="C123",
        new_cursor_value="ts:1000",
    )
    await advance_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="C123",
        new_cursor_value=None,
    )
    loaded = await load_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="C123",
    )
    assert loaded.cursor_value == "ts:1000"


@pytest.mark.asyncio
async def test_stamp_error_records_without_advancing_cursor(_seed_customer):
    await advance_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="C123",
        new_cursor_value="ts:1000",
    )
    await stamp_error(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="C123",
        error="rate_limited 429",
    )
    loaded = await load_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="C123",
    )
    assert loaded.cursor_value == "ts:1000"  # unchanged
    assert loaded.last_error == "rate_limited 429"
    assert loaded.last_error_at is not None


@pytest.mark.asyncio
async def test_list_due_cursors_filters_by_age(_seed_customer):
    # Insert a row, then backdate its polled_at.
    await advance_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="recent",
        new_cursor_value="x",
    )
    await advance_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="stale",
        new_cursor_value="x",
    )
    async with with_tenant(_TENANT) as conn:
        await conn.execute(
            """
            UPDATE ingestion_cursors
               SET polled_at = $1
             WHERE resource_id = 'stale'
            """,
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

    due = await list_due_cursors(min_age_seconds=300, limit=10)
    due_ids = {r.resource_id for r in due}
    assert "stale" in due_ids
    assert "recent" not in due_ids


# --- scheduler.tick_once -------------------------------------------------


@pytest.mark.asyncio
async def test_tick_one_runs_registered_poller_and_advances_cursor(
    _seed_customer, _register_stub_poller
):
    _StubPoller._next_result = PollResult(
        documents=[{"id": "doc1"}],
        next_cursor="ts:9999",
    )
    # Seed an old cursor row so it's "due".
    await advance_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="C123",
        new_cursor_value="ts:5000",
    )
    async with with_tenant(_TENANT) as conn:
        await conn.execute(
            "UPDATE ingestion_cursors SET polled_at = $1",
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

    documents_seen: list[tuple[str, list]] = []

    async def _sink(cust_id: str, docs: list) -> None:
        documents_seen.append((cust_id, docs))

    scheduler = PollScheduler(
        tick_interval_seconds=1, min_resource_age_seconds=60, sink=_sink
    )
    processed = await scheduler.tick_once()

    assert processed == 1
    assert _StubPoller._calls == [(_TENANT, "C123", "ts:5000")]
    assert documents_seen == [(_TENANT, [{"id": "doc1"}])]

    loaded = await load_cursor(
        customer_id=_TENANT, source=SourceSystem.SLACK, resource_id="C123"
    )
    assert loaded.cursor_value == "ts:9999"
    assert loaded.last_error is None


@pytest.mark.asyncio
async def test_tick_one_stamps_error_on_poll_error_field(
    _seed_customer, _register_stub_poller
):
    _StubPoller._next_result = PollResult(documents=[], error="429 rate limited")

    await advance_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="C123",
        new_cursor_value="ts:1000",
    )
    async with with_tenant(_TENANT) as conn:
        await conn.execute(
            "UPDATE ingestion_cursors SET polled_at = $1",
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

    scheduler = PollScheduler(min_resource_age_seconds=60)
    await scheduler.tick_once()

    loaded = await load_cursor(
        customer_id=_TENANT, source=SourceSystem.SLACK, resource_id="C123"
    )
    # Cursor unchanged; error stamped.
    assert loaded.cursor_value == "ts:1000"
    assert loaded.last_error == "429 rate limited"


@pytest.mark.asyncio
async def test_tick_one_stamps_error_on_poller_raised(
    _seed_customer, _register_stub_poller
):
    class _ExplodingPoller(BasePoller):
        source = SourceSystem.SLACK

        async def poll(self, *, customer_id, resource_id, cursor):
            raise RuntimeError("boom")

    # Swap the registered stub for an exploding one.
    polling_base._REGISTRY[SourceSystem.SLACK] = _ExplodingPoller

    await advance_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="C123",
        new_cursor_value="ts:1000",
    )
    async with with_tenant(_TENANT) as conn:
        await conn.execute(
            "UPDATE ingestion_cursors SET polled_at = $1",
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

    scheduler = PollScheduler(min_resource_age_seconds=60)
    await scheduler.tick_once()

    loaded = await load_cursor(
        customer_id=_TENANT, source=SourceSystem.SLACK, resource_id="C123"
    )
    assert loaded.cursor_value == "ts:1000"
    assert loaded.last_error is not None
    assert "RuntimeError" in loaded.last_error
    assert "boom" in loaded.last_error


@pytest.mark.asyncio
async def test_tick_one_skips_unregistered_source(_seed_customer):
    """A cursor row for a source with no registered poller is skipped
    (logged at debug). The row's polled_at is NOT bumped — we don't want
    to busy-loop, but we also don't want to drop the row."""
    # Source = NOTION; nothing registered for it.
    await advance_cursor(
        customer_id=_TENANT,
        source=SourceSystem.NOTION,
        resource_id="page-xyz",
        new_cursor_value="t:1",
    )
    async with with_tenant(_TENANT) as conn:
        await conn.execute(
            "UPDATE ingestion_cursors SET polled_at = $1",
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

    scheduler = PollScheduler(min_resource_age_seconds=60)
    processed = await scheduler.tick_once()
    # The row is processed (counted) but the poll is a no-op.
    assert processed == 1
    loaded = await load_cursor(
        customer_id=_TENANT,
        source=SourceSystem.NOTION,
        resource_id="page-xyz",
    )
    assert loaded.cursor_value == "t:1"
    assert loaded.last_error is None


@pytest.mark.asyncio
async def test_tick_one_no_due_rows_returns_zero(_seed_customer, _register_stub_poller):
    # Insert a fresh row; min_resource_age_seconds=300 means it's not
    # due yet.
    await advance_cursor(
        customer_id=_TENANT,
        source=SourceSystem.SLACK,
        resource_id="C123",
        new_cursor_value="ts:1000",
    )

    scheduler = PollScheduler(min_resource_age_seconds=300)
    processed = await scheduler.tick_once()
    assert processed == 0
    # Poller was not called.
    assert _StubPoller._calls == []
