"""Nightly trigger step B0: the queue-seed reconcile.

The reconcile is the guarantee behind every best-effort seeding path
(Normalizer enqueue swallow, settings-toggle BackgroundTasks, SQL flag
flips), so what is pinned here is the guarantee itself: every enabled
active tenant gets seeded, disabled and inactive tenants are skipped,
and one tenant's failure does not starve the rest.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engine.shared.db import raw_conn
from kb.synthesis import nightly_trigger
from tests.wiki_fixtures import insert_customer, insert_document

_NOW = datetime(2026, 8, 1, tzinfo=UTC)


async def _customer(customer_id: str, *, enabled: bool, status: str = "active") -> None:
    await insert_customer(
        customer_id,
        preferences='{"wiki_generation_enabled": true}' if enabled else "{}",
        status=status,
    )


async def _doc(customer_id: str, doc_id: str) -> None:
    await insert_document(customer_id, doc_id, created_at=_NOW)


async def _pending_count(customer_id: str) -> int:
    async with raw_conn() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM wiki_synthesis_queue "
            "WHERE customer_id = $1 AND status = 'pending'",
            customer_id,
        )


@pytest.mark.asyncio
async def test_reconcile_seeds_enabled_and_skips_disabled(live_db: None) -> None:
    await _customer("recon-on", enabled=True)
    await _customer("recon-off", enabled=False)
    await _customer("recon-inactive", enabled=True, status="suspended")
    await _doc("recon-on", "doc:on")
    await _doc("recon-off", "doc:off")
    await _doc("recon-inactive", "doc:inactive")

    summary = await nightly_trigger.reconcile_missing_queue_rows()

    assert summary == {"customers": 1, "seeded": 1, "failures": 0}
    assert await _pending_count("recon-on") == 1
    assert await _pending_count("recon-off") == 0
    assert await _pending_count("recon-inactive") == 0


@pytest.mark.asyncio
async def test_reconcile_is_idempotent(live_db: None) -> None:
    await _customer("recon-idem", enabled=True)
    await _doc("recon-idem", "doc:a")

    first = await nightly_trigger.reconcile_missing_queue_rows()
    second = await nightly_trigger.reconcile_missing_queue_rows()

    assert first["seeded"] == 1
    assert second["seeded"] == 0
    assert await _pending_count("recon-idem") == 1


@pytest.mark.asyncio
async def test_one_broken_tenant_does_not_starve_the_rest(
    live_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _customer("recon-bad", enabled=True)
    await _customer("recon-good", enabled=True)
    await _doc("recon-bad", "doc:bad")
    await _doc("recon-good", "doc:good")

    real_seed = nightly_trigger.seed_missing_docs

    async def exploding_seed(conn, customer_id, *, limit=None):
        if customer_id == "recon-bad":
            raise RuntimeError("boom")
        return await real_seed(conn, customer_id, limit=limit)

    monkeypatch.setattr(nightly_trigger, "seed_missing_docs", exploding_seed)

    summary = await nightly_trigger.reconcile_missing_queue_rows()

    assert summary == {"customers": 2, "seeded": 1, "failures": 1}
    assert await _pending_count("recon-good") == 1
    assert await _pending_count("recon-bad") == 0


@pytest.mark.asyncio
async def test_reconcile_makes_progress_in_bounded_batches(
    live_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backlog bigger than one batch still seeds completely — each batch
    is its own bounded statement, so a huge tenant cannot fail
    all-or-nothing against the statement_timeout every night."""
    await _customer("recon-batch", enabled=True)
    for i in range(3):
        await _doc("recon-batch", f"doc:{i}")

    monkeypatch.setattr(nightly_trigger, "WIKI_RECONCILE_SEED_BATCH", 1)

    summary = await nightly_trigger.reconcile_missing_queue_rows()

    assert summary == {"customers": 1, "seeded": 3, "failures": 0}
    assert await _pending_count("recon-batch") == 3


@pytest.mark.asyncio
async def test_reconcile_setup_failure_does_not_block_the_notify(
    live_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main()'s contract: NOTHING in step B0 may cost tenants their
    nightly drain — including the pool failing to open. Before the guard
    covered init_pool, a transient DB blip at pool-open crashed main()
    ahead of the notify pass."""

    async def exploding_init_pool(settings):
        raise RuntimeError("db unreachable")

    notified: list[str] = []

    async def fake_trigger(dsn: str) -> int:
        notified.append(dsn)
        return 0

    monkeypatch.setattr(nightly_trigger, "init_pool", exploding_init_pool)
    monkeypatch.setattr(nightly_trigger, "trigger_nightly_synthesis", fake_trigger)

    await nightly_trigger.main()  # must not raise

    assert notified, "the notify pass must run even when reconcile setup fails"
