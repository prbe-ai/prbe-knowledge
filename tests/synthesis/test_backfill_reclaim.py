"""Tests for the bootstrap-run reclaim path.

Covers:
- Stale `kind='bootstrap'` row in `running` → flipped to `pending`
  (NOT `failed`), error contains the reclaim marker. Worker reclaims
  on next claim tick.
- Fresh `running` row (started recently) → untouched.
- Rows in `complete`/`failed`/`partial`/`cancelled` terminal states →
  never swept, even if `started_at` is older than the threshold.
- `kind='wake'` (daily-replay) row in `running` → untouched. Reclaim
  only targets bootstrap kind because daily replays use the queue's
  own heartbeat-based reclaim.
- Reclaim returns the row count (idempotency check on second pass).
- BackfillReclaimLoop swallows exceptions in the inner tick so a
  transient DB blip doesn't kill the whole loop.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from engine.shared.config import Settings
from engine.shared.db import raw_conn
from kb.synthesis import backfill_reclaim
from kb.synthesis.backfill_reclaim import (
    BackfillReclaimLoop,
    reclaim_stale_backfill_runs,
)

CUSTOMER = "wiki-bootstrap-reclaim-cust"


@pytest_asyncio.fixture
async def reset_db(live_db: None, settings: Settings) -> AsyncIterator[None]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash, preferences) "
            "VALUES ($1, 'wiki-bootstrap-reclaim', 'h', $2::jsonb) "
            "ON CONFLICT (customer_id) DO UPDATE SET preferences = EXCLUDED.preferences",
            CUSTOMER,
            '{"wiki_generation_enabled": true}',
        )
    yield None


async def _seed_run_row(
    *,
    customer_id: str,
    kind: str,
    status: str,
    source: str | None,
    started_age_hours: float,
    error: str | None = None,
) -> int:
    """Direct INSERT into wiki_synthesis_runs with the requested state.

    Bypasses the orchestrator path so each test can pin kind, status,
    started_at, and error independently.
    """
    started_at = datetime.now(UTC) - timedelta(hours=started_age_hours)
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO wiki_synthesis_runs (
                customer_id, kind, stage, source,
                started_at, status, error
            )
            VALUES ($1, $2, 'synthesis', $3, $4, $5, $6)
            RETURNING run_id
            """,
            customer_id,
            kind,
            source,
            started_at,
            status,
            error,
        )
    return int(row["run_id"])


@pytest.mark.asyncio
async def test_reclaim_flips_stale_running_to_pending(reset_db: None) -> None:
    """A row that's been 'running' beyond the threshold goes back to
    'pending' (NOT 'failed'). Workers re-claim it on the next tick.
    `finished_at` is intentionally NOT populated — the row isn't
    finished, it's being requeued."""
    run_id = await _seed_run_row(
        customer_id=CUSTOMER,
        kind="bootstrap",
        status="running",
        source="github",
        started_age_hours=7.0,  # 1h past 6h threshold
    )
    reclaimed = await reclaim_stale_backfill_runs()
    assert reclaimed == 1

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, finished_at, error FROM wiki_synthesis_runs WHERE run_id = $1",
            run_id,
        )
    assert row["status"] == "pending"
    assert row["finished_at"] is None
    assert row["error"] is not None
    assert "reclaimed:" in row["error"]


@pytest.mark.asyncio
async def test_reclaim_preserves_existing_error_text(reset_db: None) -> None:
    """When the row already has a non-empty error, the reclaim marker
    is appended with a `' | '` separator instead of clobbering it."""
    run_id = await _seed_run_row(
        customer_id=CUSTOMER,
        kind="bootstrap",
        status="running",
        source="github",
        started_age_hours=7.0,
        error="rate limit hit on commit page 42",
    )
    reclaimed = await reclaim_stale_backfill_runs()
    assert reclaimed == 1

    async with raw_conn() as conn:
        err = await conn.fetchval(
            "SELECT error FROM wiki_synthesis_runs WHERE run_id = $1",
            run_id,
        )
    assert err.startswith("rate limit hit on commit page 42")
    assert " | reclaimed:" in err


@pytest.mark.asyncio
async def test_reclaim_leaves_fresh_running_alone(reset_db: None) -> None:
    """A bootstrap row that started 30 minutes ago is well within the
    6h threshold — reclaim must leave it as-is."""
    run_id = await _seed_run_row(
        customer_id=CUSTOMER,
        kind="bootstrap",
        status="running",
        source="github",
        started_age_hours=0.5,  # 30 min
    )
    reclaimed = await reclaim_stale_backfill_runs()
    assert reclaimed == 0

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, finished_at, error FROM wiki_synthesis_runs WHERE run_id = $1",
            run_id,
        )
    assert row["status"] == "running"
    assert row["finished_at"] is None
    assert row["error"] is None


@pytest.mark.asyncio
async def test_reclaim_leaves_terminal_states_alone(reset_db: None) -> None:
    """Rows in any terminal state (complete/failed/partial/cancelled)
    are untouched even when started_at is older than the threshold —
    the WHERE filter is `status='running'`. Pending rows are also
    untouched (they're queued, not stuck)."""
    complete_id = await _seed_run_row(
        customer_id=CUSTOMER,
        kind="bootstrap",
        status="complete",
        source="github",
        started_age_hours=8.0,
    )
    failed_id = await _seed_run_row(
        customer_id=CUSTOMER,
        kind="bootstrap",
        status="failed",
        source="linear",
        started_age_hours=8.0,
    )
    partial_id = await _seed_run_row(
        customer_id=CUSTOMER,
        kind="bootstrap",
        status="partial",
        source="slack",
        started_age_hours=8.0,
    )
    cancelled_id = await _seed_run_row(
        customer_id=CUSTOMER,
        kind="bootstrap",
        status="cancelled",
        source="notion",
        started_age_hours=8.0,
    )
    pending_id = await _seed_run_row(
        customer_id=CUSTOMER,
        kind="bootstrap",
        status="pending",
        source="granola",
        started_age_hours=8.0,
    )

    reclaimed = await reclaim_stale_backfill_runs()
    assert reclaimed == 0

    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT run_id, status FROM wiki_synthesis_runs WHERE run_id = ANY($1::bigint[])",
            [complete_id, failed_id, partial_id, cancelled_id, pending_id],
        )
    by_id = {r["run_id"]: r["status"] for r in rows}
    assert by_id[complete_id] == "complete"
    assert by_id[failed_id] == "failed"
    assert by_id[partial_id] == "partial"
    assert by_id[cancelled_id] == "cancelled"
    assert by_id[pending_id] == "pending"


@pytest.mark.asyncio
async def test_reclaim_only_targets_bootstrap_kind(reset_db: None) -> None:
    """A `kind='wake'` (daily replay) row in `running` for 7h is
    explicitly NOT touched — daily replays drive their own queue
    rows that have a separate heartbeat-based reclaim path
    (services/synthesis/reclaim.py)."""
    wake_id = await _seed_run_row(
        customer_id=CUSTOMER,
        kind="wake",
        status="running",
        source=None,
        started_age_hours=7.0,
    )
    scheduled_id = await _seed_run_row(
        customer_id=CUSTOMER,
        kind="scheduled",
        status="running",
        source=None,
        started_age_hours=7.0,
    )
    bootstrap_id = await _seed_run_row(
        customer_id=CUSTOMER,
        kind="bootstrap",
        status="running",
        source="github",
        started_age_hours=7.0,
    )

    reclaimed = await reclaim_stale_backfill_runs()
    assert reclaimed == 1

    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT run_id, status FROM wiki_synthesis_runs WHERE run_id = ANY($1::bigint[])",
            [wake_id, scheduled_id, bootstrap_id],
        )
    by_id = {r["run_id"]: r["status"] for r in rows}
    assert by_id[wake_id] == "running"
    assert by_id[scheduled_id] == "running"
    # New semantics: bootstrap row goes back to pending, not failed.
    assert by_id[bootstrap_id] == "pending"


@pytest.mark.asyncio
async def test_reclaim_does_not_touch_v4_runs(reset_db: None) -> None:
    """Defense-in-depth: even if some other kind shows up in a 'running'
    state for >threshold (e.g. an onboarding row from a v4 daily-replay
    flavor), bootstrap reclaim must not touch it. Same WHERE filter as
    above; explicit second test pinning the contract on a separate kind
    so a future widening of the filter trips here."""
    onboarding_id = await _seed_run_row(
        customer_id=CUSTOMER,
        kind="onboarding",
        status="running",
        source=None,
        started_age_hours=12.0,
    )
    reclaimed = await reclaim_stale_backfill_runs()
    assert reclaimed == 0
    async with raw_conn() as conn:
        status = await conn.fetchval(
            "SELECT status FROM wiki_synthesis_runs WHERE run_id = $1",
            onboarding_id,
        )
    assert status == "running"


@pytest.mark.asyncio
async def test_reclaim_returns_count_and_is_idempotent(reset_db: None) -> None:
    """Three stale rows → first pass returns 3, second pass returns 0
    (idempotent: WHERE filter excludes already-flipped rows)."""
    for source in ("github", "linear", "slack"):
        await _seed_run_row(
            customer_id=CUSTOMER,
            kind="bootstrap",
            status="running",
            source=source,
            started_age_hours=7.0,
        )

    first = await reclaim_stale_backfill_runs()
    assert first == 3

    second = await reclaim_stale_backfill_runs()
    assert second == 0


@pytest.mark.asyncio
async def test_reclaim_loop_swallows_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop's inner tick must catch and log exceptions so a
    transient DB blip doesn't kill the whole reclaim task. We patch
    `reclaim_stale_backfill_runs` to raise on the first call and
    succeed on the second, then assert both calls were attempted
    and no exception propagated out of the loop.

    Pure unit test — doesn't touch the DB. `live_db` is intentionally
    not used. The loop's `interval_seconds=0` makes `wait_for` time
    out immediately so iterations happen within the test's awaits.
    """
    import asyncio as _asyncio

    call_count = 0

    async def flaky_reclaim(**_: object) -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated DB blip")
        return 0

    monkeypatch.setattr(
        backfill_reclaim,
        "reclaim_stale_backfill_runs",
        flaky_reclaim,
    )

    # interval_seconds=0 → wait_for(timeout=0) raises TimeoutError on
    # every iteration without sleeping, so the loop spins as fast as
    # asyncio's event loop will let it.
    loop = BackfillReclaimLoop(threshold_hours=6, interval_seconds=0)
    task = _asyncio.create_task(loop.run())

    # Yield until the loop has called flaky_reclaim at least twice
    # (proving the first exception didn't kill the task) — bounded
    # so a regression doesn't hang the test forever.
    for _ in range(200):
        await _asyncio.sleep(0)
        if call_count >= 2:
            break

    loop.shutdown()
    task.cancel()
    with contextlib.suppress(_asyncio.CancelledError):
        await task

    assert task.done()
    # Crucially: the task must not have died with an exception. If
    # the loop had let RuntimeError escape, .exception() would
    # return the RuntimeError instead of the CancelledError we
    # expect from our own cancel().
    with contextlib.suppress(_asyncio.CancelledError):
        # .exception() raises CancelledError on a cancelled task;
        # suppressing that is the assertion that nothing else escaped.
        exc = task.exception()
        assert exc is None, f"loop leaked exception: {exc!r}"

    assert call_count >= 2, (
        f"loop should have retried after the first failure; got {call_count} calls"
    )
