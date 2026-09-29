"""The redactd client lets scans overlap, except large ones, and a crash
restarts the daemon once however many scans saw it.

Until 2026-09-29 `RedactdSupervisor.scan` held one lock across every round
trip, so a 5.5 MB session's ~15 s scan made each small document in the process
wait for it: 31.5 s measured for a document whose own scan costs 4 ms. These
tests drive `scan` through a stand-in for `_request` -- the one-round-trip
transport, same signature and response shape -- so they need no daemon;
tests/test_redactd.py runs the same property against the real one.
"""

from __future__ import annotations

import threading
import time

import pytest

from engine.ingest import redactd
from engine.ingest.redactd import SERIAL_SCAN_BYTES, RedactdSupervisor

_LARGE = "x" * SERIAL_SCAN_BYTES
_SMALL = "a small document"


def _clean(payload: dict) -> dict:
    return {"ok": True, "findings": [[] for _ in payload["texts"]]}


def _in_thread(fn) -> tuple[threading.Thread, dict]:
    out: dict = {}

    def run() -> None:
        try:
            out["result"] = fn()
        except BaseException as exc:  # surfaced by the test's assertions
            out["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, out


def test_a_small_scan_does_not_wait_behind_a_large_one(monkeypatch) -> None:
    supervisor = RedactdSupervisor()
    large_in_flight = threading.Event()
    release_large = threading.Event()

    def request(payload: dict) -> dict:
        if len(payload["texts"][0]) >= SERIAL_SCAN_BYTES:
            large_in_flight.set()
            assert release_large.wait(10)
        return _clean(payload)

    monkeypatch.setattr(supervisor, "_request", request)
    large, large_out = _in_thread(lambda: supervisor.scan([_LARGE]))
    assert large_in_flight.wait(5)
    try:
        small, small_out = _in_thread(lambda: supervisor.scan([_SMALL]))
        small.join(2)
        assert not small.is_alive(), "the small scan is queued behind the large one"
        assert small_out == {"result": [[]]}
    finally:
        release_large.set()
        large.join(5)
    assert large_out == {"result": [[]]}


def test_large_scans_still_take_turns(monkeypatch) -> None:
    """redactd's 30 s deadline is per request; two multi-MB scans sharing the
    pod's CPUs could each run past it, so they stay one at a time."""
    supervisor = RedactdSupervisor()
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def request(payload: dict) -> dict:
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return _clean(payload)

    monkeypatch.setattr(supervisor, "_request", request)
    runs = [_in_thread(lambda: supervisor.scan([_LARGE])) for _ in range(4)]
    for thread, out in runs:
        thread.join(5)
        assert out == {"result": [[]]}
    assert state["peak"] == 1


def test_small_scans_overlap(monkeypatch) -> None:
    supervisor = RedactdSupervisor()
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}
    all_in = threading.Barrier(3, timeout=5)

    def request(payload: dict) -> dict:
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        all_in.wait()  # only returns once three scans are in flight together
        with lock:
            state["now"] -= 1
        return _clean(payload)

    monkeypatch.setattr(supervisor, "_request", request)
    runs = [_in_thread(lambda: supervisor.scan([_SMALL])) for _ in range(3)]
    for thread, out in runs:
        thread.join(5)
        assert out == {"result": [[]]}
    assert state["peak"] == 3


def test_scans_failing_on_one_dead_daemon_restart_it_once(monkeypatch) -> None:
    supervisor = RedactdSupervisor()
    restarts: list[int] = []
    all_failed = threading.Barrier(4, timeout=5)

    def request(payload: dict) -> dict:
        if supervisor._generation == 0:
            all_failed.wait()  # every scan sees the same dead daemon
            raise ConnectionRefusedError("daemon is gone")
        return _clean(payload)

    def restart() -> None:
        restarts.append(supervisor._generation)
        supervisor._generation += 1

    monkeypatch.setattr(supervisor, "_request", request)
    monkeypatch.setattr(supervisor, "_restart_locked", restart)
    runs = [_in_thread(lambda: supervisor.scan([_SMALL])) for _ in range(4)]
    for thread, out in runs:
        thread.join(5)
        assert out == {"result": [[]]}
    assert restarts == [0]


def test_a_scan_that_fails_after_the_restart_still_fails_closed(monkeypatch) -> None:
    supervisor = RedactdSupervisor()

    def request(payload: dict) -> dict:
        raise ConnectionRefusedError("daemon is gone")

    monkeypatch.setattr(supervisor, "_request", request)
    monkeypatch.setattr(supervisor, "_restart_locked", lambda: None)
    with pytest.raises(redactd.ScanUnavailable, match="unreachable"):
        supervisor.scan([_SMALL])


class _AliveDaemon:
    """Stands in for the daemon's Popen: running, never exits."""

    def poll(self) -> None:
        return None


def test_a_timeout_on_a_live_daemon_does_not_restart_it_under_other_scans(monkeypatch) -> None:
    # A timed-out scan used to restart redactd, dropping every other scan's
    # connection mid-response (review of #602, 2026-09-29).
    supervisor = RedactdSupervisor()
    supervisor._proc = _AliveDaemon()
    other_in_flight = threading.Event()
    release_other = threading.Event()
    restarts: list[int] = []

    def request(payload: dict) -> dict:
        if payload["texts"][0] == "slow":
            assert other_in_flight.wait(5)
            raise TimeoutError("timed out")
        other_in_flight.set()
        assert release_other.wait(10)
        return _clean(payload)

    monkeypatch.setattr(supervisor, "_request", request)
    monkeypatch.setattr(supervisor, "_restart_locked", lambda: restarts.append(1))
    other, other_out = _in_thread(lambda: supervisor.scan([_SMALL]))
    assert other_in_flight.wait(5)
    try:
        with pytest.raises(redactd.ScanUnavailable, match="timed out"):
            supervisor.scan(["slow"])
    finally:
        release_other.set()
    other.join(5)
    assert other_out == {"result": [[]]}
    assert restarts == []


def test_a_timeout_with_nothing_else_in_flight_restarts_a_wedged_daemon(monkeypatch) -> None:
    supervisor = RedactdSupervisor()
    supervisor._proc = _AliveDaemon()
    calls: list[str] = []

    def request(payload: dict) -> dict:
        calls.append("req")
        if len(calls) == 1:
            raise TimeoutError("timed out")
        return _clean(payload)

    def restart() -> None:
        calls.append("restart")
        supervisor._generation += 1

    monkeypatch.setattr(supervisor, "_request", request)
    monkeypatch.setattr(supervisor, "_restart_locked", restart)
    assert supervisor.scan([_SMALL]) == [[]]
    assert calls == ["req", "restart", "req"]
