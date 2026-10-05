"""Tests for ``services.kg.advisory_lock.tenant_xact_lock``.

Coverage:

- Unit (no DB needed): empty-customer-id error, no-active-transaction error.
  These hit the validation gates before any SQL runs and use a tiny fake
  connection. They MUST pass even when Postgres isn't running.
- Live-DB: happy path acquires the lock inside an asyncpg transaction,
  and the concurrent-acquire test confirms a second waiter blocks until
  the first holder commits. Skipped automatically when Postgres isn't
  reachable (see ``tests/kg/conftest.py``).
"""

from __future__ import annotations

import asyncio

import pytest

from services.kg.advisory_lock import tenant_xact_lock

# ---------------------------------------------------------------------------
# Unit tests — no DB required.
# ---------------------------------------------------------------------------


class _FakeConn:
    """Minimal asyncpg-shaped stub for the validation-gate unit tests.

    ``tenant_xact_lock`` only touches ``conn.is_in_transaction()`` and
    ``conn.execute(...)`` before yielding. The empty-customer-id and
    no-transaction paths short-circuit before ``execute`` runs, so this
    fake never needs to actually talk to Postgres.
    """

    def __init__(self, *, in_transaction: bool) -> None:
        self._in_transaction = in_transaction
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def is_in_transaction(self) -> bool:
        return self._in_transaction

    async def execute(self, sql: str, *args: object) -> str:
        self.executed.append((sql, args))
        return "SELECT 1"


@pytest.mark.asyncio
async def test_rejects_empty_customer_id() -> None:
    """An empty customer_id must raise before any SQL runs."""
    conn = _FakeConn(in_transaction=True)
    with pytest.raises(ValueError, match="customer_id"):
        async with tenant_xact_lock(conn, customer_id=""):  # type: ignore[arg-type]
            pass
    assert conn.executed == []


@pytest.mark.asyncio
async def test_rejects_without_active_transaction() -> None:
    """Calling outside a transaction must raise before any SQL runs."""
    conn = _FakeConn(in_transaction=False)
    with pytest.raises(RuntimeError, match="transaction"):
        async with tenant_xact_lock(conn, customer_id="cust-test-no-txn"):  # type: ignore[arg-type]
            pass
    assert conn.executed == []


# ---------------------------------------------------------------------------
# Live-DB tests — auto-skipped if Postgres isn't reachable.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquires_within_transaction(live_db_conn) -> None:  # type: ignore[no-untyped-def]
    """Happy path: acquire inside a transaction, no exception."""
    async with (
        live_db_conn.transaction(),
        tenant_xact_lock(live_db_conn, customer_id="cust-test-acquire"),
    ):
        # Lock held; nothing else to assert — getting here without
        # raising is the success condition.
        pass


@pytest.mark.asyncio
async def test_blocks_concurrent_acquire(pg_pool) -> None:  # type: ignore[no-untyped-def]
    """A second acquire on the same customer_id blocks until the first releases.

    Two pool connections take the same lock. The second must NOT make
    progress while the first is held; once the first commits and releases
    its xact, the second proceeds.
    """
    customer_id = "cust-test-concurrency"
    held = asyncio.Event()
    release_first = asyncio.Event()

    async def hold_lock() -> None:
        async with (
            pg_pool.acquire() as conn1, conn1.transaction(),
            tenant_xact_lock(conn1, customer_id=customer_id),
        ):
            held.set()
            await release_first.wait()

    holder = asyncio.create_task(hold_lock())
    try:
        await asyncio.wait_for(held.wait(), timeout=5.0)

        async def try_acquire_second() -> None:
            async with (
                pg_pool.acquire() as conn2, conn2.transaction(),
                tenant_xact_lock(conn2, customer_id=customer_id),
            ):
                pass

        second = asyncio.create_task(try_acquire_second())
        # Second should NOT complete while the holder still holds the lock.
        done, _pending = await asyncio.wait({second}, timeout=0.5)
        assert second not in done, "second acquire should be blocked"

        # Release the holder; the second waiter should now complete.
        release_first.set()
        await asyncio.wait_for(holder, timeout=5.0)
        await asyncio.wait_for(second, timeout=5.0)
    finally:
        # Belt-and-suspenders: ensure background task can't outlive the test
        # if an assertion above fired before holder was awaited.
        release_first.set()
        if not holder.done():
            await asyncio.wait_for(holder, timeout=5.0)
