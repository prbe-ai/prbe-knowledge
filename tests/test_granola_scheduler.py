"""GranolaScheduler tests — the new poller process's only real job.

Covers _fetch_due_customers SQL: must include ACTIVE tokens with
COMPLETE/FAILED backfills past the staleness window, must exclude
PENDING/RUNNING (already enqueued) and recently-progressed rows.
"""

from __future__ import annotations

import pytest

from services.ingestion.poller import GranolaScheduler
from shared.constants import (
    BackfillStatus,
    IntegrationStatus,
    SourceSystem,
)
from shared.db import raw_conn


async def _seed_customer(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash, status)
            VALUES ($1, $2, $3, 'active')
            """,
            customer_id,
            customer_id,
            f"hash-{customer_id}",
        )


async def _seed_granola_integration(
    customer_id: str,
    *,
    token_status: str = IntegrationStatus.ACTIVE.value,
    bf_status: str = BackfillStatus.COMPLETE.value,
    last_progress_minutes_ago: float | None = 10.0,
) -> None:
    """Insert a Granola integration_tokens row + backfill_state row."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO integration_tokens
                (customer_id, source_system, access_token_encrypted,
                 scope, status)
            VALUES ($1, $2, $3, 'tier:enterprise', $4)
            """,
            customer_id,
            SourceSystem.GRANOLA.value,
            "encrypted-stub",
            token_status,
        )
        if last_progress_minutes_ago is None:
            await conn.execute(
                """
                INSERT INTO backfill_state
                    (customer_id, source_system, status, events_enqueued,
                     last_progress_at)
                VALUES ($1, $2, $3, 0, NULL)
                """,
                customer_id,
                SourceSystem.GRANOLA.value,
                bf_status,
            )
        else:
            await conn.execute(
                """
                INSERT INTO backfill_state
                    (customer_id, source_system, status, events_enqueued,
                     last_progress_at)
                VALUES ($1, $2, $3, 0,
                        NOW() - make_interval(secs => $4))
                """,
                customer_id,
                SourceSystem.GRANOLA.value,
                bf_status,
                last_progress_minutes_ago * 60,
            )


@pytest.mark.asyncio
async def test_scheduler_picks_up_completed_stale_active(live_db) -> None:
    """The basic happy path: a token that's done its initial backfill and
    hasn't progressed in 10 minutes is due for a re-poll."""
    await _seed_customer("cust-stale")
    await _seed_granola_integration(
        "cust-stale", last_progress_minutes_ago=10.0
    )

    scheduler = GranolaScheduler(interval_seconds=300)
    due = await scheduler._fetch_due_customers()

    assert due == ["cust-stale"]


@pytest.mark.asyncio
async def test_scheduler_skips_recently_progressed(live_db) -> None:
    """A customer whose last_progress_at is fresh (1 min ago) shouldn't be
    re-polled yet — the 5-min interval hasn't elapsed."""
    await _seed_customer("cust-fresh")
    await _seed_granola_integration(
        "cust-fresh", last_progress_minutes_ago=1.0
    )

    scheduler = GranolaScheduler(interval_seconds=300)
    due = await scheduler._fetch_due_customers()

    assert due == []


@pytest.mark.asyncio
async def test_scheduler_skips_pending_and_running(live_db) -> None:
    """Already-enqueued and in-flight backfills are skipped — re-enqueueing
    them would be a no-op anyway, but skipping in SQL avoids the round-trip."""
    await _seed_customer("cust-pending")
    await _seed_customer("cust-running")
    await _seed_granola_integration(
        "cust-pending",
        bf_status=BackfillStatus.PENDING.value,
        last_progress_minutes_ago=10.0,
    )
    await _seed_granola_integration(
        "cust-running",
        bf_status=BackfillStatus.RUNNING.value,
        last_progress_minutes_ago=10.0,
    )

    scheduler = GranolaScheduler(interval_seconds=300)
    due = await scheduler._fetch_due_customers()

    assert due == []


@pytest.mark.asyncio
async def test_scheduler_includes_failed_for_retry(live_db) -> None:
    """FAILED backfills should be re-enqueued so transient failures self-heal."""
    await _seed_customer("cust-failed")
    await _seed_granola_integration(
        "cust-failed",
        bf_status=BackfillStatus.FAILED.value,
        last_progress_minutes_ago=10.0,
    )

    scheduler = GranolaScheduler(interval_seconds=300)
    due = await scheduler._fetch_due_customers()

    assert due == ["cust-failed"]


@pytest.mark.asyncio
async def test_scheduler_skips_revoked_tokens(live_db) -> None:
    """A revoked integration shouldn't be polled — the token is dead."""
    await _seed_customer("cust-revoked")
    await _seed_granola_integration(
        "cust-revoked",
        token_status=IntegrationStatus.REVOKED.value,
        last_progress_minutes_ago=10.0,
    )

    scheduler = GranolaScheduler(interval_seconds=300)
    due = await scheduler._fetch_due_customers()

    assert due == []


@pytest.mark.asyncio
async def test_scheduler_includes_never_progressed(live_db) -> None:
    """A row with last_progress_at IS NULL (e.g., backfill that completed
    via a path that doesn't set the field) should be picked up."""
    await _seed_customer("cust-null")
    await _seed_granola_integration(
        "cust-null",
        bf_status=BackfillStatus.COMPLETE.value,
        last_progress_minutes_ago=None,
    )

    scheduler = GranolaScheduler(interval_seconds=300)
    due = await scheduler._fetch_due_customers()

    assert due == ["cust-null"]


@pytest.mark.asyncio
async def test_scheduler_orders_oldest_first(live_db) -> None:
    """Oldest stale customers should be processed first when there are many.
    Predictable ordering matters for fairness under tick contention."""
    await _seed_customer("cust-old")
    await _seed_customer("cust-newer")
    await _seed_granola_integration(
        "cust-old", last_progress_minutes_ago=30.0
    )
    await _seed_granola_integration(
        "cust-newer", last_progress_minutes_ago=10.0
    )

    scheduler = GranolaScheduler(interval_seconds=300)
    due = await scheduler._fetch_due_customers()

    assert due == ["cust-old", "cust-newer"]
