"""Tests for the per-page advisory lock taken in WikiAgentRuntime
``_persist_update`` / ``_persist_create``.

Cross-machine bootstrap fan-out can race two crawlers on the same
(customer, page_slug). The lock serializes the read-then-write so
each writer integrates with the previous's committed content.

These tests need a live Postgres connection because pg_advisory_xact_lock
is what we're exercising.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from shared.config import Settings
from shared.db import raw_conn
from shared.locks import advisory_lock_key

CUSTOMER = "wiki-page-lock-cust"


@pytest_asyncio.fixture
async def seeded_customer(live_db: None, settings: Settings) -> AsyncIterator[None]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'wiki-page-lock', 'h') ON CONFLICT DO NOTHING",
            CUSTOMER,
        )
    yield None


@pytest.mark.asyncio
async def test_per_page_lock_serializes_concurrent_writes(
    seeded_customer: None,
) -> None:
    """Two coroutines acquire ``pg_advisory_xact_lock`` on the same page
    key sequentially: the second blocks until the first's transaction
    commits. Without the lock, both would acquire instantly.

    This pins the wire-level contract that ``_persist_update`` /
    ``_persist_create`` rely on. The full read-then-write path is
    exercised by the worker integration tests; this is the lock-only
    regression guard.
    """
    page_slug = "decision:rollback-policy"
    lock_key = advisory_lock_key("page", CUSTOMER, page_slug)

    first_acquired = asyncio.Event()
    second_acquired = asyncio.Event()
    release_first = asyncio.Event()

    async def first() -> None:
        async with raw_conn() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock($1)", lock_key)
            first_acquired.set()
            await release_first.wait()
            # On exit, txn commits and lock releases.

    async def second() -> None:
        async with raw_conn() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock($1)", lock_key)
            second_acquired.set()

    t1 = asyncio.create_task(first())
    await asyncio.wait_for(first_acquired.wait(), timeout=2.0)

    t2 = asyncio.create_task(second())
    # Give second a moment to attempt the lock; it should still be
    # blocked on first's hold.
    await asyncio.sleep(0.2)
    assert not second_acquired.is_set(), (
        "second writer acquired the lock while first still holds it; "
        "advisory lock contract is broken"
    )

    # Release first; second should now acquire promptly.
    release_first.set()
    await asyncio.wait_for(second_acquired.wait(), timeout=2.0)
    await asyncio.gather(t1, t2)


@pytest.mark.asyncio
async def test_different_page_keys_do_not_block_each_other(
    seeded_customer: None,
) -> None:
    """Lock is per-(customer, page_slug). Two writers targeting
    DIFFERENT pages must acquire concurrently."""
    key_a = advisory_lock_key("page", CUSTOMER, "service_card:auth")
    key_b = advisory_lock_key("page", CUSTOMER, "decision:rollback-policy")

    a_acquired = asyncio.Event()
    b_acquired = asyncio.Event()
    proceed = asyncio.Event()

    async def hold(key: int, ack: asyncio.Event) -> None:
        async with raw_conn() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock($1)", key)
            ack.set()
            await proceed.wait()

    t_a = asyncio.create_task(hold(key_a, a_acquired))
    t_b = asyncio.create_task(hold(key_b, b_acquired))
    await asyncio.wait_for(a_acquired.wait(), timeout=2.0)
    await asyncio.wait_for(b_acquired.wait(), timeout=2.0)
    proceed.set()
    await asyncio.gather(t_a, t_b)


@pytest.mark.asyncio
async def test_different_customers_do_not_block_each_other(
    seeded_customer: None,
) -> None:
    """Lock includes customer_id so two tenants writing the same slug
    never serialize."""
    key_one = advisory_lock_key("page", "cust-one", "service_card:auth")
    key_two = advisory_lock_key("page", "cust-two", "service_card:auth")
    assert key_one != key_two

    one_acquired = asyncio.Event()
    two_acquired = asyncio.Event()
    proceed = asyncio.Event()

    async def hold(key: int, ack: asyncio.Event) -> None:
        async with raw_conn() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock($1)", key)
            ack.set()
            await proceed.wait()

    t1 = asyncio.create_task(hold(key_one, one_acquired))
    t2 = asyncio.create_task(hold(key_two, two_acquired))
    await asyncio.wait_for(one_acquired.wait(), timeout=2.0)
    await asyncio.wait_for(two_acquired.wait(), timeout=2.0)
    proceed.set()
    await asyncio.gather(t1, t2)
