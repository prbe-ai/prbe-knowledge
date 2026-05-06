"""Tests for ``BootstrapListener`` — the bootstrap fly app's NOTIFY router.

The orchestrator itself is exercised in ``test_bootstrap_orchestrator.py``;
this file owns the cross-machine routing contract: every machine in the
wiki-bootstrap fly app receives every NOTIFY (Postgres broadcasts), so
the listener relies on a per-customer ``pg_try_advisory_xact_lock`` to
ensure exactly one machine drains each trigger.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import orjson
import pytest
import pytest_asyncio

from services.synthesis.bootstrap_app import (
    BootstrapListener,
    _bootstrap_dispatch_lock_key,
)
from services.synthesis.bootstrap_orchestrator import BootstrapOrchestrator
from shared.config import Settings, get_settings
from shared.db import raw_conn

CUSTOMER = "bootstrap-app-test-cust"


@pytest_asyncio.fixture
async def seeded_customer(live_db: None) -> AsyncIterator[None]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'bootstrap-app-test', 'h') ON CONFLICT DO NOTHING",
            CUSTOMER,
        )
    yield None


@pytest_asyncio.fixture
async def http_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


def _settings() -> Settings:
    get_settings.cache_clear()  # type: ignore[attr-defined]
    return Settings()


class _RecordingOrchestrator(BootstrapOrchestrator):
    """Stand-in that records which customer_ids it was asked to drain.

    Subclasses the real orchestrator so the listener's type expectations
    stay satisfied; overrides ``bootstrap`` to skip the actual fan-out.
    """

    def __init__(self, *, settings: Settings, http: httpx.AsyncClient) -> None:
        super().__init__(settings=settings, http=http)
        self.calls: list[str] = []

    async def bootstrap(self, *, customer_id: str, **_kwargs):  # type: ignore[override]
        self.calls.append(customer_id)


@pytest.mark.asyncio
async def test_dispatch_skips_when_lock_held(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """Two BootstrapListener._dispatch calls for the same customer don't
    both invoke the orchestrator.

    Pre-acquiring the dispatch lock from a separate connection simulates
    "the other fly machine got there first." The listener under test
    should observe that the lock is held and bail with
    ``bootstrap_listener.skip_concurrent`` — no orchestrator call.
    Once the held lock releases, a fresh dispatch on the same listener
    runs cleanly.
    """
    orch = _RecordingOrchestrator(settings=_settings(), http=http_client)
    listener = BootstrapListener(dsn="unused", orchestrator=orch)

    payload = orjson.dumps(
        {"customer_id": CUSTOMER, "sources": [], "wipe_first": False}
    ).decode()
    lock_key = _bootstrap_dispatch_lock_key(CUSTOMER)

    # Hold the lock from another connection — simulates a peer machine
    # already mid-drain for this customer.
    async with raw_conn() as held_conn, held_conn.transaction():
        acquired = await held_conn.fetchval(
            "SELECT pg_try_advisory_xact_lock($1)", lock_key
        )
        assert acquired is True

        await listener._dispatch(payload)
        # Listener saw the held lock and bailed; orchestrator never ran.
        assert orch.calls == []

    # Outside the held-conn txn, the lock has released. A fresh dispatch
    # must succeed — proves the lock isn't leaking from the skipped path.
    await listener._dispatch(payload)
    assert orch.calls == [CUSTOMER]


@pytest.mark.asyncio
async def test_dispatch_lock_keys_differ_per_customer(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """Different customers get different lock keys — a drain in flight
    for customer A must not block a drain for customer B."""
    other = "bootstrap-app-test-cust-other"
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'bootstrap-app-test-other', 'h') ON CONFLICT DO NOTHING",
            other,
        )

    orch = _RecordingOrchestrator(settings=_settings(), http=http_client)
    listener = BootstrapListener(dsn="unused", orchestrator=orch)

    cust_a_lock = _bootstrap_dispatch_lock_key(CUSTOMER)
    async with raw_conn() as held_conn, held_conn.transaction():
        await held_conn.execute("SELECT pg_advisory_xact_lock($1)", cust_a_lock)
        # While customer A is locked, customer B's dispatch still runs.
        payload_b = orjson.dumps(
            {"customer_id": other, "sources": [], "wipe_first": False}
        ).decode()
        await listener._dispatch(payload_b)
        assert orch.calls == [other]
