"""ReclaimLoop tests for stale-heartbeat backfill_state rows.

A worker that dies mid-paginate (rolling deploy, OOM, network blip) leaves
its backfill_state row stranded with status='running' forever — claim only
picks 'pending'. ReclaimLoop._reclaim_once flips stale 'running' rows back
to 'pending', preserving last_cursor and events_enqueued so the next worker
resumes exactly where the dead one stopped.

Also covers the runner-side heartbeat loop: an unconditional ping decoupled
from progress, so a healthy-but-paused runner is not falsely reclaimed.

And the claim-ownership token: every in-loop UPDATE filters on started_at
(set fresh on each claim) so a runner whose row was reclaimed mid-flight
detects preemption and bails without clobbering the new owner's state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from services.ingestion.backfill_runner import (
    BackfillReclaimedError,
    _heartbeat_loop,
    _load_resume_state,
    _mark_done,
    _mark_failed,
    _release_for_resume,
    _update_progress,
    enqueue_slack_channel_backfill,
)
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
) -> datetime | None:
    """Insert a backfill_state row with heartbeat_at = NOW() - <offset>s.

    `heartbeat_offset_seconds=None` leaves heartbeat_at NULL (e.g. for
    'pending' rows that have never been claimed). Returns the started_at
    that was set, which tests use as the claim ownership token.
    """
    async with raw_conn() as conn:
        if heartbeat_offset_seconds is None:
            row = await conn.fetchrow(
                """
                INSERT INTO backfill_state
                    (customer_id, source_system, status, last_cursor,
                     events_enqueued, started_at, heartbeat_at)
                VALUES ($1, $2, $3, $4, $5, NULL, NULL)
                RETURNING started_at
                """,
                customer_id,
                source.value,
                status,
                last_cursor,
                events_enqueued,
            )
        else:
            row = await conn.fetchrow(
                """
                INSERT INTO backfill_state
                    (customer_id, source_system, status, last_cursor,
                     events_enqueued, started_at, heartbeat_at)
                VALUES ($1, $2, $3, $4, $5,
                        NOW() - make_interval(secs => $6),
                        NOW() - make_interval(secs => $6))
                RETURNING started_at
                """,
                customer_id,
                source.value,
                status,
                last_cursor,
                events_enqueued,
                heartbeat_offset_seconds,
            )
    return row["started_at"] if row else None


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
    started_at = await _insert_backfill_state(
        customer_id="cust-hb",
        source=SourceSystem.SLACK,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=600,  # stale on purpose
        last_cursor="cur",
        events_enqueued=100,
    )
    before = await _heartbeat_at("cust-hb")

    task = asyncio.create_task(
        _heartbeat_loop(
            "cust-hb",
            SourceSystem.SLACK,
            interval_seconds=0.05,
            claim_token=started_at,
        )
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
    # Any token is fine — status='pending' filter excludes the row regardless.
    fake_token = datetime.now(UTC)

    task = asyncio.create_task(
        _heartbeat_loop(
            "cust-pending-hb",
            SourceSystem.SLACK,
            interval_seconds=0.05,
            claim_token=fake_token,
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
async def test_heartbeat_loop_skips_when_claim_token_mismatch(live_db) -> None:
    """A different claim's token doesn't write to a running row."""
    await _insert_customer("cust-hb-other")
    real_started_at = await _insert_backfill_state(
        customer_id="cust-hb-other",
        source=SourceSystem.SLACK,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=600,
    )
    before = await _heartbeat_at("cust-hb-other")
    other_token = real_started_at - timedelta(seconds=1)  # different token

    task = asyncio.create_task(
        _heartbeat_loop(
            "cust-hb-other",
            SourceSystem.SLACK,
            interval_seconds=0.05,
            claim_token=other_token,
        )
    )
    try:
        await asyncio.sleep(0.2)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    after = await _heartbeat_at("cust-hb-other")
    assert after == before  # heartbeat untouched — wrong token


@pytest.mark.asyncio
async def test_heartbeat_loop_cancels_cleanly(live_db) -> None:
    """Cancellation propagates as CancelledError, no stray exceptions."""
    await _insert_customer("cust-cancel")
    started_at = await _insert_backfill_state(
        customer_id="cust-cancel",
        source=SourceSystem.SLACK,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=10,
    )

    task = asyncio.create_task(
        _heartbeat_loop(
            "cust-cancel",
            SourceSystem.SLACK,
            interval_seconds=10,
            claim_token=started_at,
        )
    )
    await asyncio.sleep(0.05)  # let the task start its sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ---- ownership token: _update_progress, _mark_done, _mark_failed ---------


@pytest.mark.asyncio
async def test_load_resume_state_returns_cursor_and_count(live_db) -> None:
    await _insert_customer("cust-resume")
    started_at = await _insert_backfill_state(
        customer_id="cust-resume",
        source=SourceSystem.SLACK,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=30,
        last_cursor="resume-cursor",
        events_enqueued=1600,
    )

    state = await _load_resume_state("cust-resume", SourceSystem.SLACK)
    assert state is not None
    assert state.cursor == "resume-cursor"
    assert state.events_enqueued == 1600
    assert state.started_at == started_at


@pytest.mark.asyncio
async def test_update_progress_succeeds_with_matching_token(live_db) -> None:
    await _insert_customer("cust-up-ok")
    started_at = await _insert_backfill_state(
        customer_id="cust-up-ok",
        source=SourceSystem.SLACK,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=30,
        last_cursor="old",
        events_enqueued=100,
    )

    await _update_progress(
        "cust-up-ok", SourceSystem.SLACK, "new", 125, claim_token=started_at
    )

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT last_cursor, events_enqueued FROM backfill_state "
            "WHERE customer_id='cust-up-ok'"
        )
    assert row["last_cursor"] == "new"
    assert row["events_enqueued"] == 125


@pytest.mark.asyncio
async def test_enqueue_slack_channel_backfill_creates_channel_cursor(live_db) -> None:
    await _insert_customer("cust-slack-channel")

    result = await enqueue_slack_channel_backfill("cust-slack-channel", "CNEW")

    assert result.queued is True
    assert result.reason == "inserted"
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, last_cursor, events_enqueued FROM backfill_state "
            "WHERE customer_id='cust-slack-channel'"
        )
    cursor = json.loads(row["last_cursor"])
    assert row["status"] == BackfillStatus.PENDING.value
    assert row["events_enqueued"] == 0
    assert cursor["active"] == {"CNEW": None}
    assert cursor["mode"] == "channel_join"


@pytest.mark.asyncio
async def test_running_slack_backfill_defers_new_channel_until_done(live_db) -> None:
    await _insert_customer("cust-slack-deferred")
    started_at = await _insert_backfill_state(
        customer_id="cust-slack-deferred",
        source=SourceSystem.SLACK,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=30,
        last_cursor=json.dumps({"active": {"COLD": None}, "done": []}),
        events_enqueued=10,
    )

    result = await enqueue_slack_channel_backfill("cust-slack-deferred", "CNEW")
    assert result.queued is True
    assert result.reason == "deferred_until_running_backfill_finishes"

    await _update_progress(
        "cust-slack-deferred",
        SourceSystem.SLACK,
        json.dumps({"active": {}, "done": ["COLD"]}),
        25,
        claim_token=started_at,
    )
    await _mark_done(
        "cust-slack-deferred",
        SourceSystem.SLACK,
        25,
        json.dumps({"active": {}, "done": ["COLD"]}),
        claim_token=started_at,
    )

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, started_at, heartbeat_at, completed_at, "
            "last_cursor, events_enqueued FROM backfill_state "
            "WHERE customer_id='cust-slack-deferred'"
        )
    cursor = json.loads(row["last_cursor"])
    assert row["status"] == BackfillStatus.PENDING.value
    assert row["started_at"] is None
    assert row["heartbeat_at"] is None
    assert row["completed_at"] is None
    assert row["events_enqueued"] == 0
    assert cursor["active"] == {"CNEW": None}


@pytest.mark.asyncio
async def test_update_progress_raises_on_token_mismatch(live_db) -> None:
    """Reaper-then-reclaim case: started_at advanced, our writes must bail."""
    await _insert_customer("cust-up-bad")
    real_token = await _insert_backfill_state(
        customer_id="cust-up-bad",
        source=SourceSystem.SLACK,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=30,
        last_cursor="cur",
        events_enqueued=42,
    )
    stale_token = real_token - timedelta(seconds=10)

    with pytest.raises(BackfillReclaimedError):
        await _update_progress(
            "cust-up-bad",
            SourceSystem.SLACK,
            "should-not-write",
            999,
            claim_token=stale_token,
        )

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT last_cursor, events_enqueued FROM backfill_state "
            "WHERE customer_id='cust-up-bad'"
        )
    assert row["last_cursor"] == "cur"  # unchanged
    assert row["events_enqueued"] == 42


@pytest.mark.asyncio
async def test_update_progress_raises_when_status_not_running(live_db) -> None:
    """Reaper flipped status='pending' — runner's progress writes must bail."""
    await _insert_customer("cust-up-pending")
    started_at = await _insert_backfill_state(
        customer_id="cust-up-pending",
        source=SourceSystem.SLACK,
        status=BackfillStatus.PENDING.value,  # already reclaimed
        heartbeat_offset_seconds=None,
        last_cursor="cur",
        events_enqueued=42,
    )

    with pytest.raises(BackfillReclaimedError):
        await _update_progress(
            "cust-up-pending",
            SourceSystem.SLACK,
            "x",
            50,
            claim_token=started_at or datetime.now(UTC),
        )


@pytest.mark.asyncio
async def test_mark_done_skips_on_token_mismatch(live_db) -> None:
    await _insert_customer("cust-md-bad")
    real_token = await _insert_backfill_state(
        customer_id="cust-md-bad",
        source=SourceSystem.SLACK,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=30,
    )
    stale_token = real_token - timedelta(seconds=10)

    # No exception — silent no-op for terminal calls.
    await _mark_done(
        "cust-md-bad", SourceSystem.SLACK, 99, "x", claim_token=stale_token
    )

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, completed_at FROM backfill_state "
            "WHERE customer_id='cust-md-bad'"
        )
    assert row["status"] == BackfillStatus.RUNNING.value  # unchanged
    assert row["completed_at"] is None


@pytest.mark.asyncio
async def test_mark_failed_skips_on_token_mismatch(live_db) -> None:
    await _insert_customer("cust-mf-bad")
    real_token = await _insert_backfill_state(
        customer_id="cust-mf-bad",
        source=SourceSystem.SLACK,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=30,
    )
    stale_token = real_token - timedelta(seconds=10)

    await _mark_failed(
        "cust-mf-bad",
        SourceSystem.SLACK,
        "I do not own this row",
        claim_token=stale_token,
    )

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, last_error FROM backfill_state "
            "WHERE customer_id='cust-mf-bad'"
        )
    assert row["status"] == BackfillStatus.RUNNING.value
    assert row["last_error"] is None


@pytest.mark.asyncio
async def test_update_progress_preserves_cumulative_count_across_resume(
    live_db,
) -> None:
    """F2: a resumed run picks up where the prior run left off, then increments."""
    await _insert_customer("cust-cumulative")
    started_at = await _insert_backfill_state(
        customer_id="cust-cumulative",
        source=SourceSystem.SLACK,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=30,
        last_cursor="prior",
        events_enqueued=1600,  # accumulated from prior run
    )

    state = await _load_resume_state("cust-cumulative", SourceSystem.SLACK)
    assert state is not None
    # Caller initializes its local counter from the prior cumulative count.
    enqueued = state.events_enqueued
    enqueued += 25  # one PROGRESS_EVERY_N_EVENTS tick worth
    await _update_progress(
        "cust-cumulative",
        SourceSystem.SLACK,
        "after-25-more",
        enqueued,
        claim_token=started_at,
    )

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT events_enqueued, last_cursor FROM backfill_state "
            "WHERE customer_id='cust-cumulative'"
        )
    assert row["events_enqueued"] == 1625  # cumulative, not 25
    assert row["last_cursor"] == "after-25-more"


@pytest.mark.asyncio
async def test_release_for_resume_flips_running_to_pending(live_db) -> None:
    """SIGTERM path: release the claim immediately so a peer can resume without
    waiting for the 5-min stale-heartbeat reclaim cron. last_cursor and
    events_enqueued are preserved."""
    await _insert_customer("cust-release")
    started_at = await _insert_backfill_state(
        customer_id="cust-release",
        source=SourceSystem.LINEAR,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=10,  # fresh — we're killing it ourselves
        last_cursor="cursor-mid-flight",
        events_enqueued=420,
    )

    released = await _release_for_resume(
        "cust-release", SourceSystem.LINEAR, claim_token=started_at
    )
    assert released is True

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, started_at, heartbeat_at, last_cursor, "
            "events_enqueued FROM backfill_state "
            "WHERE customer_id='cust-release'"
        )
    assert row["status"] == BackfillStatus.PENDING.value
    assert row["started_at"] is None
    assert row["heartbeat_at"] is None
    assert row["last_cursor"] == "cursor-mid-flight"
    assert row["events_enqueued"] == 420


@pytest.mark.asyncio
async def test_release_for_resume_no_op_on_token_mismatch(live_db) -> None:
    """If a competing worker already re-claimed the row (started_at advanced),
    releasing with the stale token must NOT clobber the new owner's state."""
    await _insert_customer("cust-release-mismatch")
    await _insert_backfill_state(
        customer_id="cust-release-mismatch",
        source=SourceSystem.LINEAR,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=10,
        last_cursor="new-owner-cursor",
        events_enqueued=10,
    )

    stale_token = datetime.now(UTC) - timedelta(hours=1)
    released = await _release_for_resume(
        "cust-release-mismatch", SourceSystem.LINEAR, claim_token=stale_token
    )
    assert released is False

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, last_cursor FROM backfill_state "
            "WHERE customer_id='cust-release-mismatch'"
        )
    assert row["status"] == BackfillStatus.RUNNING.value
    assert row["last_cursor"] == "new-owner-cursor"


@pytest.mark.asyncio
async def test_release_for_resume_no_op_when_status_not_running(live_db) -> None:
    """If the row is already pending/done/failed, release is a no-op."""
    await _insert_customer("cust-release-done")
    started_at = await _insert_backfill_state(
        customer_id="cust-release-done",
        source=SourceSystem.LINEAR,
        status=BackfillStatus.COMPLETE.value,
        heartbeat_offset_seconds=10,
        last_cursor="finished",
        events_enqueued=999,
    )
    assert started_at is not None

    released = await _release_for_resume(
        "cust-release-done", SourceSystem.LINEAR, claim_token=started_at
    )
    assert released is False

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM backfill_state "
            "WHERE customer_id='cust-release-done'"
        )
    assert row["status"] == BackfillStatus.COMPLETE.value


@pytest.mark.asyncio
async def test_release_survives_cascading_cancel_of_outer_task(live_db) -> None:
    """SIGTERM during deploy: the outer task awaiting the release gets cancelled
    a second time before the asyncpg roundtrip completes. With asyncio.shield,
    the UPDATE still lands and the row flips to 'pending'. Without shield, the
    row stayed stuck in 'running' until the 5-min reclaim cron swept it."""
    await _insert_customer("cust-cascade-cancel")
    started_at = await _insert_backfill_state(
        customer_id="cust-cascade-cancel",
        source=SourceSystem.LINEAR,
        status=BackfillStatus.RUNNING.value,
        heartbeat_offset_seconds=10,
        last_cursor="cursor-mid-flight",
        events_enqueued=420,
    )

    async def cancel_handler() -> None:
        # Mirrors the except asyncio.CancelledError: block in run_backfill.
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(
                _release_for_resume(
                    "cust-cascade-cancel", SourceSystem.LINEAR, started_at
                )
            )

    task = asyncio.create_task(cancel_handler())
    # Yield once so the task enters the await on shield, then cancel — this
    # reproduces the production timing where asyncio.gather tears the runner
    # task down before the asyncpg UPDATE roundtrip completes.
    await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # The shielded UPDATE may still be running on the loop after task is done;
    # poll briefly for the row flip rather than assuming sync completion.
    deadline = asyncio.get_event_loop().time() + 2.0
    while asyncio.get_event_loop().time() < deadline:
        async with raw_conn() as conn:
            row = await conn.fetchrow(
                "SELECT status FROM backfill_state "
                "WHERE customer_id='cust-cascade-cancel'"
            )
        if row["status"] == BackfillStatus.PENDING.value:
            break
        await asyncio.sleep(0.05)

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, last_cursor, events_enqueued FROM backfill_state "
            "WHERE customer_id='cust-cascade-cancel'"
        )
    assert row["status"] == BackfillStatus.PENDING.value
    assert row["last_cursor"] == "cursor-mid-flight"
    assert row["events_enqueued"] == 420
