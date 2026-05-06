"""Tests for ``BackfillWorker`` — the bootstrap fly app's queue-claim worker.

The worker LISTENs on ``WIKI_BACKFILL_CHANNEL`` (wake hint) +
``WIKI_BACKFILL_CANCEL_CHANNEL`` (force-cancel), claims pending
``wiki_synthesis_runs`` rows via ``FOR UPDATE SKIP LOCKED``, runs each
(customer, source) crawl under a per-pair advisory lock + per-machine
semaphore, and writes the terminal status.

These tests pin the routing contract: claim atomicity across machines,
semaphore caps, advisory-lock skip on concurrent runs, NOTIFY-driven
hard cancel, and the zero-page-halt -> failed mapping. Most run against
a live Postgres; the few that don't are explicitly marked as such.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import asyncpg
import httpx
import orjson
import pytest
import pytest_asyncio

from services.synthesis.backfill_app import (
    BackfillWorker,
    _backfill_run_lock_key,
)
from services.synthesis.crawlers.base import (
    BackfillAgent,
    BackfillAgentResult,
)
from shared.config import Settings, get_settings
from shared.constants import WIKI_BACKFILL_CANCEL_CHANNEL
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
        # Clean any stale wiki_synthesis_runs rows from a prior test run
        # so claim ordering is deterministic.
        await conn.execute(
            "DELETE FROM wiki_synthesis_runs WHERE customer_id = $1",
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


# ---------------------------------------------------------------------------
# Mock crawler — small BackfillAgent stub the worker can drive.
# ---------------------------------------------------------------------------


class _MockCrawler(BackfillAgent):
    """Plain BackfillAgent that returns a configurable result."""

    sleep_seconds: float = 0.0
    pages_created_value: int = 0
    pages_updated_value: int = 0
    halt_reason_value: str | None = None
    raise_on_run: BaseException | None = None

    def system_prompt(self) -> str:
        return "mock"

    def source_api_tools(self) -> list[dict[str, Any]]:
        return []

    async def dispatch_source_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def run(self) -> BackfillAgentResult:
        started = datetime.now(UTC)
        if self.sleep_seconds:
            await asyncio.sleep(self.sleep_seconds)
        if self.raise_on_run is not None:
            raise self.raise_on_run
        return BackfillAgentResult(
            source=self.source,
            customer_id=self.customer_id,
            run_id=self.run_id,
            pages_created=self.pages_created_value,
            pages_updated=self.pages_updated_value,
            halt_reason=self.halt_reason_value,
            started_at=started,
            finished_at=datetime.now(UTC),
        )


def _make_factory(
    *,
    source: str,
    sleep_seconds: float = 0.0,
    pages_created: int = 0,
    pages_updated: int = 0,
    halt_reason: str | None = None,
    raise_on_run: BaseException | None = None,
):
    def factory(**kwargs: Any) -> BackfillAgent:
        cls = type(
            f"_Mock_{source}",
            (_MockCrawler,),
            {
                "source": source,
                "sleep_seconds": sleep_seconds,
                "pages_created_value": pages_created,
                "pages_updated_value": pages_updated,
                "halt_reason_value": halt_reason,
                "raise_on_run": raise_on_run,
            },
        )
        return cls(**kwargs)

    return factory


async def _seed_pending(*, source: str) -> int:
    async with raw_conn() as conn:
        return int(
            await conn.fetchval(
                """
                INSERT INTO wiki_synthesis_runs
                    (customer_id, kind, stage, source, status)
                VALUES ($1, 'bootstrap', 'synthesis', $2, 'pending')
                RETURNING run_id
                """,
                CUSTOMER,
                source,
            )
        )


def _build_worker(
    *,
    parallelism: int = 6,
    crawler_factories: dict[str, Any] | None = None,
    http: httpx.AsyncClient,
) -> BackfillWorker:
    settings = _settings()
    return BackfillWorker(
        dsn=settings.database_url,
        http=http,
        settings=settings,
        parallelism=parallelism,
        crawler_factories=crawler_factories,
    )


# ---------------------------------------------------------------------------
# Claim ordering / FOR UPDATE SKIP LOCKED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_claim_skip_locked_no_overlap(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """Two BackfillWorker instances racing on 5 pending rows: each claim
    returns a distinct row, total claimed = 5, no double-claim."""
    seeded_ids = [await _seed_pending(source=f"src{i}") for i in range(5)]

    worker_a = _build_worker(http=http_client)
    worker_b = _build_worker(http=http_client)

    claims_a: list[Any] = []
    claims_b: list[Any] = []

    async def drain(worker: BackfillWorker, sink: list[Any]) -> None:
        while True:
            claim = await worker._claim_one()
            if claim is None:
                return
            sink.append(claim.run_id)

    await asyncio.gather(drain(worker_a, claims_a), drain(worker_b, claims_b))
    all_claimed = claims_a + claims_b
    assert sorted(all_claimed) == sorted(seeded_ids)
    # Each row claimed exactly once (no overlap).
    assert len(set(all_claimed)) == len(all_claimed)


# ---------------------------------------------------------------------------
# Per-(customer, source) advisory lock skips concurrent run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_run_advisory_lock_skips_concurrent(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """Pre-acquire the per-(customer, source) lock from another conn;
    BackfillWorker._run_one bails without invoking the agent."""
    invoked: list[int] = []

    def factory(**kwargs: Any) -> BackfillAgent:
        cls = type(
            "_Spy",
            (_MockCrawler,),
            {
                "source": "spy",
                "sleep_seconds": 0.0,
                "pages_created_value": 0,
                "pages_updated_value": 0,
                "halt_reason_value": None,
                "raise_on_run": None,
            },
        )
        agent = cls(**kwargs)
        original_run = agent.run

        async def tracked_run() -> BackfillAgentResult:
            invoked.append(1)
            return await original_run()

        agent.run = tracked_run  # type: ignore[method-assign]
        return agent

    run_id = await _seed_pending(source="spy")
    worker = _build_worker(
        http=http_client,
        crawler_factories={"spy": factory},
    )

    # Hold the per-run lock from a separate conn.
    lock_key = _backfill_run_lock_key(CUSTOMER, "spy")
    async with raw_conn() as held_conn, held_conn.transaction():
        acquired = await held_conn.fetchval("SELECT pg_try_advisory_xact_lock($1)", lock_key)
        assert acquired is True

        # Manually claim the row first (mirrors what run() does after
        # waking on a NOTIFY) so _run_one sees it; the per-run lock
        # check happens inside _run_one BEFORE the agent runs.
        claim = await worker._claim_one()
        assert claim is not None
        assert claim.run_id == run_id

        await worker._run_one(claim)

    assert invoked == [], "agent should not have been invoked under contended lock"
    # Row is left at 'running' (worker doesn't flip back; reclaim
    # handles that if the original holder dies).
    async with raw_conn() as conn:
        status = await conn.fetchval(
            "SELECT status FROM wiki_synthesis_runs WHERE run_id = $1",
            run_id,
        )
    assert status == "running"


# ---------------------------------------------------------------------------
# Zero-page halt mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_page_halt_marks_failed(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """Crawler returns halt_reason + zero pages -> row marked 'failed'
    with error='halt:<reason>'. Mirrors PR #126's contract; now lives
    inside the worker instead of the orchestrator."""
    run_id = await _seed_pending(source="auth_missing")
    worker = _build_worker(
        http=http_client,
        crawler_factories={
            "auth_missing": _make_factory(source="auth_missing", halt_reason="auth.missing")
        },
    )
    claim = await worker._claim_one()
    assert claim is not None
    assert claim.run_id == run_id
    await worker._run_one(claim)

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, error FROM wiki_synthesis_runs WHERE run_id = $1",
            run_id,
        )
    assert row["status"] == "failed"
    assert row["error"] == "halt:auth.missing"


@pytest.mark.asyncio
async def test_partial_status_when_halted_with_pages(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """halt_reason + pages produced -> 'partial' (stalled but productive)."""
    run_id = await _seed_pending(source="stalled")
    worker = _build_worker(
        http=http_client,
        crawler_factories={
            "stalled": _make_factory(
                source="stalled",
                halt_reason="stall",
                pages_created=2,
                pages_updated=1,
            )
        },
    )
    claim = await worker._claim_one()
    assert claim is not None
    await worker._run_one(claim)

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, pages_created, pages_updated, error "
            "FROM wiki_synthesis_runs WHERE run_id = $1",
            run_id,
        )
    assert row["status"] == "partial"
    assert row["pages_created"] == 2
    assert row["pages_updated"] == 1
    assert row["error"] is None


@pytest.mark.asyncio
async def test_clean_run_marks_complete(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """Crawler returns no halt, no error -> 'complete' with counters."""
    run_id = await _seed_pending(source="winner")
    worker = _build_worker(
        http=http_client,
        crawler_factories={
            "winner": _make_factory(source="winner", pages_created=3, pages_updated=2)
        },
    )
    claim = await worker._claim_one()
    assert claim is not None
    await worker._run_one(claim)

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, pages_created, pages_updated FROM wiki_synthesis_runs "
            "WHERE run_id = $1",
            run_id,
        )
    assert row["status"] == "complete"
    assert row["pages_created"] == 3
    assert row["pages_updated"] == 2


# ---------------------------------------------------------------------------
# Cancellation paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_one_propagates_cancelled_error_and_marks_row(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """Cancelling the _run_one task mid-crawl: the task ends in cancelled
    state, the row is marked 'cancelled', and CancelledError propagates
    to the awaiter (not swallowed by the broad except Exception)."""
    run_id = await _seed_pending(source="long_runner")

    started = asyncio.Event()

    class _LongCrawler(_MockCrawler):
        source = "long_runner"

        async def run(self) -> BackfillAgentResult:
            started.set()
            await asyncio.sleep(60)  # cancel arrives long before this returns
            return await super().run()

    worker = _build_worker(
        http=http_client,
        crawler_factories={
            "long_runner": lambda **kwargs: _LongCrawler(**kwargs),
        },
    )
    claim = await worker._claim_one()
    assert claim is not None

    task = asyncio.create_task(worker._run_one(claim))
    await asyncio.wait_for(started.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async with raw_conn() as conn:
        status = await conn.fetchval(
            "SELECT status FROM wiki_synthesis_runs WHERE run_id = $1",
            run_id,
        )
    # Marked cancelled by _run_one's CancelledError handler.
    assert status == "cancelled"


@pytest.mark.asyncio
async def test_handle_cancel_payload_cancels_matching_in_flight_task(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """A cancel payload referencing run_id N cancels the in-flight task
    keyed at N. Doesn't touch unrelated tasks."""
    run_id_a = await _seed_pending(source="a")
    await _seed_pending(source="b")

    started_a = asyncio.Event()
    started_b = asyncio.Event()

    class _A(_MockCrawler):
        source = "a"

        async def run(self) -> BackfillAgentResult:
            started_a.set()
            await asyncio.sleep(60)
            return await super().run()

    class _B(_MockCrawler):
        source = "b"

        async def run(self) -> BackfillAgentResult:
            started_b.set()
            await asyncio.sleep(60)
            return await super().run()

    worker = _build_worker(
        http=http_client,
        crawler_factories={
            "a": lambda **kwargs: _A(**kwargs),
            "b": lambda **kwargs: _B(**kwargs),
        },
    )
    # Manually replicate what run() does: claim + spawn task into
    # in_flight map. Don't acquire the semaphore (we're not exercising
    # the cap here), the cancel path is independent of it.
    claim_a = await worker._claim_one()
    claim_b = await worker._claim_one()
    assert claim_a is not None and claim_b is not None
    task_a = asyncio.create_task(worker._run_one(claim_a))
    task_b = asyncio.create_task(worker._run_one(claim_b))
    worker._in_flight[claim_a.run_id] = task_a
    worker._in_flight[claim_b.run_id] = task_b

    try:
        await asyncio.wait_for(started_a.wait(), timeout=2.0)
        await asyncio.wait_for(started_b.wait(), timeout=2.0)

        payload = orjson.dumps({"customer_id": CUSTOMER, "run_ids": [run_id_a]}).decode("utf-8")
        await worker._handle_cancel_payload(payload)

        # task_a was cancelled; task_b is still running.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task_a, timeout=2.0)
        assert not task_b.done()
    finally:
        # Clean up the still-running task_b.
        task_b.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task_b


@pytest.mark.asyncio
async def test_handle_cancel_payload_unknown_run_id_is_noop(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """A cancel referencing an unknown run_id silently no-ops; doesn't
    crash the cancel processor."""
    worker = _build_worker(http=http_client)
    payload = orjson.dumps({"customer_id": CUSTOMER, "run_ids": [99_999_999]}).decode("utf-8")
    await worker._handle_cancel_payload(payload)  # must not raise


@pytest.mark.asyncio
async def test_handle_cancel_payload_unparseable_is_noop(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    worker = _build_worker(http=http_client)
    await worker._handle_cancel_payload("{not json")  # must not raise


# ---------------------------------------------------------------------------
# Cancel NOTIFY arriving via the LISTENing connection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_cancel_notify_cancels_task(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """Send a real NOTIFY on WIKI_BACKFILL_CANCEL_CHANNEL with a run_id
    matching an in-flight task; the worker's cancel listener picks it
    up, the processor cancels the task, and the row is marked
    'cancelled' within ~2s."""
    settings = _settings()
    run_id = await _seed_pending(source="long")

    started = asyncio.Event()

    class _LongCrawler(_MockCrawler):
        source = "long"

        async def run(self) -> BackfillAgentResult:
            started.set()
            await asyncio.sleep(60)
            return await super().run()

    worker = _build_worker(
        http=http_client,
        crawler_factories={"long": lambda **kwargs: _LongCrawler(**kwargs)},
    )
    # Spin the worker as a background task so its cancel listener +
    # processor are alive.
    worker_task = asyncio.create_task(worker.run())
    try:
        # Wait for the crawler to actually start.
        await asyncio.wait_for(started.wait(), timeout=5.0)

        # Fire the cancel NOTIFY.
        notify_conn = await asyncpg.connect(settings.database_url)
        try:
            payload = orjson.dumps({"customer_id": CUSTOMER, "run_ids": [run_id]}).decode("utf-8")
            await notify_conn.execute(
                "SELECT pg_notify($1, $2)",
                WIKI_BACKFILL_CANCEL_CHANNEL,
                payload,
            )
        finally:
            await notify_conn.close()

        # Wait up to 5s for the row to land in 'cancelled'.
        async with raw_conn() as conn:
            for _ in range(50):
                status = await conn.fetchval(
                    "SELECT status FROM wiki_synthesis_runs WHERE run_id = $1",
                    run_id,
                )
                if status == "cancelled":
                    break
                await asyncio.sleep(0.1)
        assert status == "cancelled"
    finally:
        worker.shutdown()
        try:
            await asyncio.wait_for(worker_task, timeout=10.0)
        except (TimeoutError, Exception):
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await worker_task


# ---------------------------------------------------------------------------
# Lock key shape regression — derivations don't collide
# ---------------------------------------------------------------------------


def test_lock_keys_distinct_per_salt_and_parts() -> None:
    """A regression that the three salts the codebase uses
    (backfill-trigger, backfill-run, page) yield different lock keys
    for the same customer/source so trigger lock + worker lock + page
    lock can't accidentally serialize on the same value."""
    from shared.locks import advisory_lock_key

    a = advisory_lock_key("backfill-trigger", "c1")
    b = advisory_lock_key("backfill-run", "c1", "github")
    c = advisory_lock_key("page", "c1", "team:engineering")
    assert a != b != c
    assert a != c


# ---------------------------------------------------------------------------
# Semaphore cap — pure unit (no DB) check on the worker's bookkeeping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_semaphore_cap_respected_under_concurrent_spawn(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """Pre-load 6 pending rows; instantiate worker with parallelism=2.
    Spin the worker; observe ``len(_in_flight) <= 2`` at every
    sample, even while pending rows remain."""
    for i in range(6):
        await _seed_pending(source=f"capped{i}")

    started_events: list[asyncio.Event] = [asyncio.Event() for _ in range(6)]
    release_event = asyncio.Event()

    class _Holder(_MockCrawler):
        source = "capped"  # overwritten below

        async def run(self) -> BackfillAgentResult:
            # Find the matching event by source's trailing index.
            try:
                idx = int(self.source.removeprefix("capped"))
            except ValueError:
                idx = 0
            if 0 <= idx < len(started_events):
                started_events[idx].set()
            await release_event.wait()
            return BackfillAgentResult(
                source=self.source,
                customer_id=self.customer_id,
                run_id=self.run_id,
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
            )

    factories: dict[str, Any] = {}
    for i in range(6):
        s = f"capped{i}"
        factories[s] = lambda src=s, **kwargs: type(  # type: ignore[misc]
            f"_H_{src}", (_Holder,), {"source": src}
        )(**kwargs)

    worker = _build_worker(
        http=http_client,
        parallelism=2,
        crawler_factories=factories,
    )
    worker_task = asyncio.create_task(worker.run())
    try:
        # Wait for the cap to fill up.
        for _ in range(50):
            if len(worker._in_flight) >= 2:
                break
            await asyncio.sleep(0.05)
        # Sample over a short window — at no point should >2 be in flight.
        max_observed = 0
        for _ in range(10):
            max_observed = max(max_observed, len(worker._in_flight))
            await asyncio.sleep(0.05)
        assert max_observed <= 2, f"semaphore cap breached: max={max_observed}"
        assert max_observed == 2, "expected the cap to be saturated"
    finally:
        release_event.set()
        worker.shutdown()
        try:
            await asyncio.wait_for(worker_task, timeout=10.0)
        except (TimeoutError, Exception):
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await worker_task


# ---------------------------------------------------------------------------
# Shutdown drains in-flight tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_shutdown_drains_in_flight(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """Start the worker with N tasks running; signal shutdown; assert
    all tasks complete or cancel before run() returns."""
    for i in range(3):
        await _seed_pending(source=f"draining{i}")

    proceed = asyncio.Event()

    class _PausingCrawler(_MockCrawler):
        source = "drain"

        async def run(self) -> BackfillAgentResult:
            await proceed.wait()
            return BackfillAgentResult(
                source=self.source,
                customer_id=self.customer_id,
                run_id=self.run_id,
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
            )

    factories: dict[str, Any] = {}
    for i in range(3):
        s = f"draining{i}"
        factories[s] = lambda src=s, **kwargs: type(  # type: ignore[misc]
            f"_P_{src}", (_PausingCrawler,), {"source": src}
        )(**kwargs)

    worker = _build_worker(
        http=http_client,
        parallelism=3,
        crawler_factories=factories,
    )
    worker_task = asyncio.create_task(worker.run())

    # Wait until tasks are in flight.
    for _ in range(50):
        if len(worker._in_flight) >= 3:
            break
        await asyncio.sleep(0.05)

    # Release the crawlers + signal shutdown roughly in parallel; the
    # worker should drain them before returning.
    proceed.set()
    worker.shutdown()
    await asyncio.wait_for(worker_task, timeout=15.0)
    assert worker_task.done()
    # All in-flight tasks should be settled.
    assert all(t.done() for t in worker._in_flight.values())
