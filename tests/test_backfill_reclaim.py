"""ReclaimLoop tests for stale-heartbeat backfill_state rows.

A worker that dies mid-paginate (rolling deploy, OOM, network blip) leaves
its backfill_state row stranded with status='running' forever — claim only
picks 'pending'. ReclaimLoop._reclaim_once flips stale 'running' rows back
to 'pending', preserving last_cursor and events_enqueued so the next worker
resumes exactly where the dead one stopped.

Also covers the runner-side heartbeat loop: an unconditional ping decoupled
from progress, so a healthy-but-paused runner is not falsely reclaimed.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from services.ingestion.backfill_runner import _heartbeat_loop
from services.ingestion.worker import ReclaimLoop
from shared.constants import BackfillStatus, SourceSystem
from shared.db import raw_conn


async def _insert_customer(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'x', 'y') ON CONFLICT DO NOTHING",
            customer_id,
        )


async def _insert_backfill_state(
    *,
    customer_id: str,
    source: SourceSystem,
    status: str,
    heartbeat_offset_seconds: int | None,
    last_cursor: str | None = None,
    events_enqueued: int = 0,
) -> None:
    """Insert a backfill_state row with heartbeat_at = NOW() - <offset>s.

    `heartbeat_offset_seconds=None` leaves heartbeat_at NULL (e.g. for
    'pending' rows that have never been claimed).
    """
    async with raw_conn() as conn:
        if heartbeat_offset_seconds is None:
            await conn.execute(
                """
                INSERT INTO backfill_state
                    (customer_id, source_system, status, last_cursor,
                     events_enqueued, started_at, heartbeat_at)
                VALUES ($1, $2, $3, $4, $5, NULL, NULL)
                """,
                customer_id,
                source.value,
                status,
                last_cursor,
                events_enqueued,
            )
        else:
            await conn.execute(
                """
                INSERT INTO backfill_state
                    (customer_id, source_system, status, last_cursor,
                     events_enqueued, started_at, heartbeat_at)
                VALUES ($1, $2, $3, $4, $5,
                        NOW() - make_interval(secs => $6),
                        NOW() - make_interval(secs => $6))
                """,
                customer_id,
                source.value,
                status,
                last_cursor,
                events_enqueued,
                heartbeat_offset_seconds,
            )


@pytest.mark.asyncio
async def test_reclaim_flips_stale_running_to_pending(live_db) -> None:
    """Stale 'running' row is flipped to 'pending'; cursor + counter preserved."""
    await _insert_customer("cust-stale")
    await _insert_backfill_state(
        customer_id="cust-stale",
        source=SourceSystem.SLACK,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=600,  # 10 minutes
        last_cursor="abc",
        events_enqueued=1600,
    )

    loop = ReclaimLoop(backfill_threshold_seconds=300)
    queue_n, backfill_n = await loop._reclaim_once()
    assert queue_n == 0
    assert backfill_n == 1

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, heartbeat_at, started_at, last_cursor, "
            "events_enqueued, last_error "
            "FROM backfill_state WHERE customer_id='cust-stale'"
        )
    assert row["status"] == BackfillStatus.PENDING.value
    assert row["heartbeat_at"] is None
    assert row["started_at"] is None
    assert row["last_cursor"] == "abc"
    assert row["events_enqueued"] == 1600
    assert "reclaimed: heartbeat stale" in (row["last_error"] or "")


@pytest.mark.asyncio
async def test_reclaim_leaves_fresh_running_alone(live_db) -> None:
    """A 'running' row with a fresh heartbeat (< threshold) is untouched."""
    await _insert_customer("cust-fresh")
    await _insert_backfill_state(
        customer_id="cust-fresh",
        source=SourceSystem.SLACK,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=30,  # 30s — well under threshold
        last_cursor="cur-fresh",
        events_enqueued=42,
    )

    loop = ReclaimLoop(backfill_threshold_seconds=300)
    _queue_n, backfill_n = await loop._reclaim_once()
    assert backfill_n == 0

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, heartbeat_at, last_cursor, events_enqueued, last_error "
            "FROM backfill_state WHERE customer_id='cust-fresh'"
        )
    assert row["status"] == BackfillStatus.RUNNING.value
    assert row["heartbeat_at"] is not None
    assert row["last_cursor"] == "cur-fresh"
    assert row["events_enqueued"] == 42
    assert row["last_error"] is None


@pytest.mark.asyncio
async def test_reclaim_leaves_pending_alone(live_db) -> None:
    """A 'pending' row (never claimed) is untouched even with NULL heartbeat."""
    await _insert_customer("cust-pending")
    await _insert_backfill_state(
        customer_id="cust-pending",
        source=SourceSystem.SLACK,
        status=BackfillStatus.PENDING.value,
        heartbeat_offset_seconds=None,
        last_cursor=None,
        events_enqueued=0,
    )

    loop = ReclaimLoop(backfill_threshold_seconds=300)
    _queue_n, backfill_n = await loop._reclaim_once()
    assert backfill_n == 0

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, last_error FROM backfill_state "
            "WHERE customer_id='cust-pending'"
        )
    assert row["status"] == BackfillStatus.PENDING.value
    assert row["last_error"] is None


@pytest.mark.asyncio
async def test_reclaim_leaves_complete_alone(live_db) -> None:
    """A 'complete' row is untouched even if heartbeat_at is ancient."""
    await _insert_customer("cust-done")
    await _insert_backfill_state(
        customer_id="cust-done",
        source=SourceSystem.SLACK,
        status=BackfillStatus.COMPLETE.value,
        heartbeat_offset_seconds=99999,  # ancient — but status='complete'
        last_cursor="final",
        events_enqueued=5000,
    )

    loop = ReclaimLoop(backfill_threshold_seconds=300)
    _queue_n, backfill_n = await loop._reclaim_once()
    assert backfill_n == 0

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, last_cursor, events_enqueued, last_error "
            "FROM backfill_state WHERE customer_id='cust-done'"
        )
    assert row["status"] == BackfillStatus.COMPLETE.value
    assert row["last_cursor"] == "final"
    assert row["events_enqueued"] == 5000
    assert row["last_error"] is None


# ---- _heartbeat_loop: liveness decoupled from progress -------------------


async def _heartbeat_at(customer_id: str):
    async with raw_conn() as conn:
        return await conn.fetchval(
            "SELECT heartbeat_at FROM backfill_state WHERE customer_id = $1",
            customer_id,
        )


@pytest.mark.asyncio
async def test_heartbeat_loop_advances_without_progress(live_db) -> None:
    """Heartbeat ticks even when no events are being enqueued."""
    await _insert_customer("cust-hb")
    await _insert_backfill_state(
        customer_id="cust-hb",
        source=SourceSystem.SLACK,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=600,  # stale on purpose
        last_cursor="cur",
        events_enqueued=100,
    )
    before = await _heartbeat_at("cust-hb")

    task = asyncio.create_task(
        _heartbeat_loop("cust-hb", SourceSystem.SLACK, interval_seconds=0.05)
    )
    try:
        await asyncio.sleep(0.2)  # ~3-4 ticks
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    after = await _heartbeat_at("cust-hb")
    assert after > before


@pytest.mark.asyncio
async def test_heartbeat_loop_skips_non_running_rows(live_db) -> None:
    """A 'pending' row never has its heartbeat written by the liveness loop."""
    await _insert_customer("cust-pending-hb")
    await _insert_backfill_state(
        customer_id="cust-pending-hb",
        source=SourceSystem.SLACK,
        status=BackfillStatus.PENDING.value,
        heartbeat_offset_seconds=None,  # NULL
    )

    task = asyncio.create_task(
        _heartbeat_loop(
            "cust-pending-hb", SourceSystem.SLACK, interval_seconds=0.05
        )
    )
    try:
        await asyncio.sleep(0.2)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert await _heartbeat_at("cust-pending-hb") is None


@pytest.mark.asyncio
async def test_heartbeat_loop_cancels_cleanly(live_db) -> None:
    """Cancellation propagates as CancelledError, no stray exceptions."""
    await _insert_customer("cust-cancel")
    await _insert_backfill_state(
        customer_id="cust-cancel",
        source=SourceSystem.SLACK,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=10,
    )

    task = asyncio.create_task(
        _heartbeat_loop("cust-cancel", SourceSystem.SLACK, interval_seconds=10)
    )
    await asyncio.sleep(0.05)  # let the task start its sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
