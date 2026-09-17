"""One slot per session, and a CAS miss that does not cost five minutes.

Both properties are expressed in SQL text built at runtime, so nothing else in
the suite would notice them regressing. Measured on 2026-09-16 before the fix:
259 CAS misses in 22 hours, each one a whole session re-normalized and thrown
away, and one row at attempts=1510.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_WORKER = (_ROOT / "engine" / "ingest" / "worker.py").read_text()
_APP = (_ROOT / "kb" / "ingestion_app.py").read_text()


def _session_upsert() -> str:
    start = _APP.index("ON CONFLICT (customer_id, source_system, source_event_id) DO UPDATE")
    return _APP[start : _APP.index("RETURNING queue_id", start)]


def test_a_processing_row_keeps_its_status() -> None:
    """The bug: every batch set `status = 'pending'`, including while a slot
    had the row in `processing`. A second slot then claimed the same session
    and both re-normalized it from every batch."""
    upsert = _session_upsert()
    assert "WHEN ingestion_queue.status = 'processing'" in upsert
    assert not re.search(r"^\s*status = 'pending',\s*$", upsert, re.M), (
        "an unconditional 'pending' is what let a second slot claim a row "
        "another slot was already working"
    )


def test_the_version_still_bumps_for_a_processing_row() -> None:
    """Keeping the status is only safe because the version bump is what tells
    the running worker its payload grew: its CAS commit misses, and the miss
    is the signal to re-run against the extended array."""
    upsert = _session_upsert()
    assert "version = ingestion_queue.version + 1" in upsert
    assert "payload_s3_keys = ingestion_queue.payload_s3_keys" in upsert


def test_enqueued_at_still_moves_for_the_session_completer() -> None:
    """`session_completer` reads MAX(enqueued_at) to decide a session is idle.
    A fix that froze it for processing rows would make live sessions look
    finished."""
    assert "enqueued_at = NOW()" in _session_upsert()


def _cas_miss_blocks() -> list[str]:
    """Each `worker.cas_retry` log through to the `return` that follows it."""
    out = []
    for m in re.finditer(r'"worker\.cas_retry"', _WORKER):
        tail = _WORKER[m.start() : m.start() + 1400]
        stop = re.search(r"\n            return\b|\n                return\b", tail)
        out.append(tail[: stop.end()] if stop else tail)
    return out


def test_every_cas_miss_releases_the_row() -> None:
    """THREE paths, not two: _mark_done, _mark_skipped and _on_error. A row
    left `processing` waits out QUEUE_RECLAIM_THRESHOLD_SECONDS (300s) holding
    a slot against its tenant's in-flight cap -- five minutes of latency on
    exactly the sessions whose batches are arriving right now."""
    blocks = _cas_miss_blocks()
    assert len(blocks) == 3, f"expected 3 CAS-miss paths, found {len(blocks)}"
    for block in blocks:
        assert "UPDATE ingestion_queue" in block, block[:200]
        assert "QueueStatus.PENDING.value" in block, block[:200]


def test_the_release_cannot_resurrect_a_finished_row() -> None:
    """Guarded on status: another path may already have marked the row done or
    dead between the CAS miss and this UPDATE."""
    for block in _cas_miss_blocks():
        assert "AND status = $3" in block
        assert "QueueStatus.PROCESSING.value" in block


def test_the_reclaim_loop_is_still_the_backstop() -> None:
    """The release is an optimisation on the common path, not a replacement
    for reclaim: a worker that dies mid-row writes no release at all."""
    assert "QUEUE_RECLAIM_THRESHOLD_SECONDS" in _WORKER
    assert "class ReclaimLoop" in _WORKER
