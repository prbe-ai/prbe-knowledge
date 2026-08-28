"""The drain must outlive a dependency blip, and must not lie about being alive.

Regression tests for the 2026-08-27 ingestion stall. A node replacement took
the Postgres cluster away at 19:16 UTC; it came back 53 minutes later. The
worker did not.

Three separate defects turned a ~50 minute dependency outage into 26 hours of
frozen indexing for six tenants:

  1. The claim was the one unguarded DB call in the drain loop, so a single
     `ConnectionRefusedError` propagated through `Worker.run`'s gather and out
     of `run_worker_forever`, ending the process's main coroutine.
  2. Unwinding did not end the PROCESS. The exception reached `asyncio.run()`'s
     teardown, where `_cancel_all_tasks()` blocked forever on a task that
     ignored cancellation. `py-spy` found the container still sitting in that
     teardown 26h later, with the `ConnectionRefusedError` trapped unprinted in
     the `__exit__` frame.
  3. `/health` only ever asked whether Postgres answered. uvicorn's listening
     socket outlived the drain, so the liveness probe got 200 for 26 hours and
     Kubernetes never restarted the corpse.

Defect 3 is why nobody noticed, so it gets the most coverage here: a health
check that cannot distinguish a working worker from a dead one is worse than
no health check, because it actively suppresses the restart that would have
fixed this within two minutes.
"""

from __future__ import annotations

import asyncio

import pytest

import engine.ingest.worker as worker_mod
from engine.ingest.worker import Worker, _build_health_app, note_progress


def _bare_worker() -> Worker:
    """A Worker carrying only the fields the claim loop touches.

    Sidesteps `Worker.__init__`, which builds a Normalizer and a connector
    context — neither of which participates in the loop's error handling.
    """
    w = Worker.__new__(Worker)
    w._shutdown = asyncio.Event()
    w._claim_coalesce_max = 1
    return w


# ---------------------------------------------------------------------------
# 1. a dependency blip must not end the drain
# ---------------------------------------------------------------------------


async def test_claim_loop_survives_a_connection_refused_and_keeps_draining(
    monkeypatch,
) -> None:
    """The exact production failure: Postgres refuses, the drain must continue.

    Before the guard this call stack ended the process. `ConnectionRefusedError`
    is an `OSError`, raised from `_claim_one`'s pool acquire.
    """
    monkeypatch.setattr(worker_mod, "QUEUE_ERROR_BACKOFF_SECONDS", 0.01)
    w = _bare_worker()
    calls: list[int] = []

    async def flaky_tick(poll_interval: float) -> None:
        calls.append(1)
        if len(calls) <= 2:
            raise ConnectionRefusedError(111, "Connect call failed")
        # Postgres is back; the drain claims again and we stop the test.
        w._shutdown.set()

    monkeypatch.setattr(w, "_claim_tick", flaky_tick)

    await asyncio.wait_for(w._claim_loop(poll_interval=0.01), timeout=5)

    assert len(calls) == 3, (
        "the loop must retry after a dependency error, not exit on the first one"
    )


async def test_claim_loop_backs_off_between_failures(monkeypatch) -> None:
    """A failing dependency is retried on the backoff, not hammered."""
    monkeypatch.setattr(worker_mod, "QUEUE_ERROR_BACKOFF_SECONDS", 0.05)
    w = _bare_worker()
    calls: list[int] = []

    async def always_fails(poll_interval: float) -> None:
        calls.append(1)
        raise ConnectionRefusedError(111, "Connect call failed")

    monkeypatch.setattr(w, "_claim_tick", always_fails)

    task = asyncio.create_task(w._claim_loop(poll_interval=0.01))
    await asyncio.sleep(0.18)
    w._shutdown.set()
    await asyncio.wait_for(task, timeout=5)

    # ~0.18s of 0.05s backoffs. Without a backoff this would be thousands.
    assert 2 <= len(calls) <= 6, f"expected a backed-off retry cadence, got {len(calls)}"


async def test_claim_loop_still_propagates_cancellation(monkeypatch) -> None:
    """The broad `except Exception` must NOT swallow a real shutdown.

    `CancelledError` is a BaseException, so it passes through — this asserts
    that, because a drain that ignores cancellation is the very thing that
    hung teardown for 26 hours in defect 2.
    """
    monkeypatch.setattr(worker_mod, "QUEUE_ERROR_BACKOFF_SECONDS", 0.01)
    w = _bare_worker()

    async def cancelled_tick(poll_interval: float) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(w, "_claim_tick", cancelled_tick)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(w._claim_loop(poll_interval=0.01), timeout=5)


# ---------------------------------------------------------------------------
# 2. /health must assert the drain is running, not just that the DB answers
# ---------------------------------------------------------------------------


def _client(monkeypatch, *, db_ok: bool = True, threshold: float | None = 180.0):
    """A TestClient over the health app with the DB probe stubbed."""
    from fastapi.testclient import TestClient

    import engine.shared.db as db_mod

    async def fake_health_check() -> bool:
        return db_ok

    # `_build_health_app` imports health_check at call time, so patch the source.
    monkeypatch.setattr(db_mod, "health_check", fake_health_check)
    return TestClient(_build_health_app(drain_stall_threshold_seconds=threshold))


def test_health_is_ok_while_the_drain_is_ticking(monkeypatch) -> None:
    client = _client(monkeypatch)
    note_progress()

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["drain"] is True


def test_health_is_degraded_when_the_drain_has_stopped(monkeypatch) -> None:
    """THE test. A reachable database is not evidence that work is happening.

    Simulated by a threshold the beacon is already older than — the same state
    the pod was in for 26h while answering 200.
    """
    client = _client(monkeypatch, threshold=-1.0)

    resp = client.get("/health")

    assert resp.status_code == 503, (
        "a worker whose drain has stopped must FAIL liveness so it gets restarted"
    )
    body = resp.json()
    assert body["drain"] is False
    assert body["db"] is True, "the DB was healthy; the drain is what died"
    assert body["status"] == "degraded"


def test_health_still_fails_when_the_database_is_down(monkeypatch) -> None:
    """The original DB check is preserved, not replaced."""
    client = _client(monkeypatch, db_ok=False)
    note_progress()

    resp = client.get("/health")

    assert resp.status_code == 503
    assert resp.json()["db"] is False


def test_health_without_a_threshold_keeps_db_only_behaviour(monkeypatch) -> None:
    """Callers that run no drain (a plain DB probe) must not go permanently 503."""
    client = _client(monkeypatch, threshold=None)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert "drain_idle_seconds" not in resp.json()


async def test_a_long_row_keeps_the_beacon_fresh_via_its_heartbeat(monkeypatch) -> None:
    """A row taking many minutes must never be mistaken for a dead worker.

    The heartbeat is the drain's sign of life while `_process` is busy, so it
    feeds the beacon. Without this a slow-but-healthy row would fail liveness
    and get itself killed halfway through.
    """
    w = _bare_worker()
    monkeypatch.setattr(worker_mod, "QUEUE_HEARTBEAT_INTERVAL_SECONDS", 0.01)

    executed: list[int] = []

    class _Conn:
        async def execute(self, *args, **kwargs) -> None:
            executed.append(1)

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *exc) -> bool:
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    monkeypatch.setattr(worker_mod, "get_pool", lambda: _Pool())

    # Push the beacon into the past, then let one heartbeat land.
    worker_mod._last_progress_monotonic -= 10_000
    stale_before = worker_mod.seconds_since_progress()

    task = asyncio.create_task(w._heartbeat(queue_id=1))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert executed, "the heartbeat must have written at least once"
    assert worker_mod.seconds_since_progress() < stale_before, (
        "a heartbeat write is a sign of life and must refresh the beacon"
    )


# ---------------------------------------------------------------------------
# 3. a failing heartbeat must not escape _process and skip the row's bookkeeping
# ---------------------------------------------------------------------------


async def test_a_dead_heartbeat_does_not_escape_process(monkeypatch) -> None:
    """The exact escape path that ended the worker on 2026-08-27.

    `cancel()` on an already-dead task is a no-op, so `await` re-raises what it
    stored — and from inside a `finally` that exception bypasses every `except`
    clause in `_process`. The row's error handling never runs, and the drain
    dies carrying an exception nobody recorded.

    A heartbeat is bookkeeping. Its failure must not decide the fate of the row.
    """
    from engine.shared.constants import SourceSystem

    w = _bare_worker()

    async def dead_heartbeat(queue_id: int) -> None:
        raise ConnectionRefusedError(111, "Connect call failed")

    marked: list[int] = []

    class _Normalizer:
        async def process_queue_row(self, **kwargs):
            # Give the heartbeat task a chance to be scheduled and die.
            await asyncio.sleep(0)
            return "outcome"

    async def fake_mark_done(queue_id, *args, **kwargs) -> None:
        marked.append(queue_id)

    # Direct assignment: `_bare_worker` skips __init__, so these attributes do
    # not exist yet and monkeypatch.setattr would refuse them.
    w._heartbeat = dead_heartbeat
    w._normalizer = _Normalizer()
    w._mark_done = fake_mark_done

    row = {
        "queue_id": 34987,
        "customer_id": "probe",
        "source_system": SourceSystem.CLAUDE_CODE.value,
        "source_event_id": "ce8c4274",
        "payload_s3_key": "k",
        "payload_s3_keys": ["k"],
        "version": 3,
        "attempts": 3,
    }

    # Must NOT raise. Before the fix this propagated ConnectionRefusedError.
    await asyncio.wait_for(w._process(row), timeout=5)

    assert marked == [34987], (
        "the row must still be committed; a failed heartbeat must not skip it"
    )
