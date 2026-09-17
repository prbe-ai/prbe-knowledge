"""The backlog-age signal: what it measures, and what it must never do.

The 33-minute median wait on 2026-09-16 was found by hand days later. Every
existing signal was green throughout -- the liveness probe answers on a
database ping, and the drain-stall beacon only notices when the loop stops
entirely, so a drain running at a third of its rate is exactly as alive as one
keeping up.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from engine.ingest import queue_age

_ROOT = Path(__file__).resolve().parent.parent


def test_the_sample_measures_arrival_not_last_touch() -> None:
    """`enqueued_at` is bumped by every transcript batch, so age measured that
    way reports ~zero on precisely the workload filling the queue."""
    assert "first_enqueued_at" in queue_age._SAMPLE_SQL
    assert "MIN(first_enqueued_at)" in queue_age._SAMPLE_SQL
    assert "MIN(enqueued_at)" not in queue_age._SAMPLE_SQL


def test_processing_rows_count_towards_the_backlog() -> None:
    """Counting only `pending` would hide a worker that claims rows promptly
    and then crawls -- the exact failure this exists to catch."""
    assert "status IN ('pending', 'processing')" in queue_age._SAMPLE_SQL


def test_health_reads_the_snapshot_and_never_queries() -> None:
    """A scan-shaped query behind a liveness probe turns "the queue is deep"
    into "restart the thing that drains it"."""
    src = (_ROOT / "engine" / "ingest" / "worker.py").read_text()
    health = src[src.index("def _build_health_app") :]
    assert "queue_age.latest()" in health
    assert "sample_once" not in health, "the probe must not touch the database for this"


def test_backlog_depth_is_not_part_of_liveness() -> None:
    """`ok` is db + drain only. A deep queue is a thing to tell a human about,
    not a reason for kubelet to kill the only worker draining it."""
    src = (_ROOT / "engine" / "ingest" / "worker.py").read_text()
    m = re.search(r"^\s*ok = (.+)$", src, re.M)
    assert m, "could not find the health verdict"
    assert m.group(1).strip() == "db_ok and drain_ok", m.group(1)


def test_the_starting_snapshot_is_readable_before_any_sample() -> None:
    """/health is live the moment the process is, which is before the first
    sample a minute later."""
    assert queue_age.latest().as_body() == {
        "queue_pending": 0,
        "queue_processing": 0,
        "queue_oldest_age_seconds": 0,
    }


@pytest.mark.asyncio
async def test_a_failing_sample_never_stops_the_reporter(monkeypatch, caplog) -> None:
    """Never let the measurement break the thing measured: a database blip
    during a sample must not take down the process that drains the queue."""
    calls = {"n": 0}
    reporter = queue_age.QueueAgeReporter()

    async def _boom():
        # Deterministic: the fake decides when the loop ends, so this asserts
        # "it survived a failure and came back" rather than "0.05s was enough
        # wall time", which is a flake waiting for a loaded CI box.
        calls["n"] += 1
        if calls["n"] >= 3:
            reporter.shutdown()
        raise RuntimeError("database went away")

    monkeypatch.setattr(queue_age, "sample_once", _boom)
    monkeypatch.setattr(queue_age, "SAMPLE_INTERVAL_SECONDS", 0)
    await asyncio.wait_for(reporter.run(), timeout=5)
    assert calls["n"] == 3, "the loop stopped at the first failure"


@pytest.mark.asyncio
async def test_posthog_being_down_does_not_fail_the_sample(monkeypatch) -> None:
    """`shared.ops_alert.capture` swallows its own failures by design; this
    asserts the reporter does not reintroduce the coupling around it."""
    seen = {}
    reporter = queue_age.QueueAgeReporter()

    async def _sample():
        queue_age._latest = queue_age.QueueAge(pending=7, processing=1, oldest_age_seconds=1234)
        return queue_age._latest

    def _capture(event, properties=None):
        seen["event"] = event
        seen["properties"] = properties
        reporter.shutdown()
        raise RuntimeError("posthog unreachable")

    monkeypatch.setattr(queue_age, "sample_once", _sample)
    monkeypatch.setattr(queue_age, "capture", _capture)
    monkeypatch.setattr(queue_age, "SAMPLE_INTERVAL_SECONDS", 0)
    await asyncio.wait_for(reporter.run(), timeout=5)
    assert seen["event"] == "ingestion_queue_age"
    assert seen["properties"]["oldest_age_seconds"] == 1234
    assert queue_age.latest().pending == 7


def test_the_migration_backfills_rather_than_stamping_now() -> None:
    """A NOT NULL DEFAULT NOW() column stamps every existing row with the
    migration's own clock, so the fleet reports a backlog age of zero at
    exactly the moment someone starts trusting the number."""
    src = (
        _ROOT / "db" / "migrations" / "versions"
        / "20260917_0137_queue_first_enqueued_at.py"
    ).read_text()
    assert "UPDATE ingestion_queue SET first_enqueued_at = enqueued_at" in src


def test_the_queue_priority_default_is_not_the_top_tier() -> None:
    """100 used to mean "a live webhook", the safe default. Under the tier
    table it means research content -- the TOP tier for any future insert that
    forgets to name one."""
    schema = (_ROOT / "db" / "schema.sql").read_text()
    m = re.search(r"priority\s+SMALLINT NOT NULL DEFAULT (\d+)", schema)
    assert m, "priority column not found in schema.sql"
    assert int(m.group(1)) == 75, m.group(1)
