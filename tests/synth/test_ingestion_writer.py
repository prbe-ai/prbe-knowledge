"""Tests for IngestionWriter local mode (Plan 2 Task 11).

Integrate-mode tests are added in Task 14.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.output.writer import IngestionWriter


def _slack_doc(source_event_id: str = "doc-1") -> SynthDoc:
    return SynthDoc(
        id=source_event_id,
        source=Source.SLACK,
        source_event_id=source_event_id,
        text="hello",
        occurred_at=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
        channel="#standup",
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-x",
        archetype="STANDUP_UPDATE",
        personas=("gh:alice",),
        services_mentioned=("payments",),
    )


def _notion_doc(source_event_id: str = "page-1") -> SynthDoc:
    return SynthDoc(
        id=source_event_id,
        source=Source.NOTION,
        source_event_id=source_event_id,
        text="On-call handoff page body",
        occurred_at=datetime(2026, 5, 4, 10, 0, tzinfo=UTC),
        channel=None,
        page_id=source_event_id,
        thread_parent_id=None,
        scenario_id="scn-y",
        archetype="ON_CALL_HANDOFF",
        personas=("gh:alice", "gh:bob"),
        services_mentioned=("payments",),
    )


@pytest.mark.asyncio
async def test_local_writes_slack_envelope_to_disk(tmp_path: Path) -> None:
    writer = IngestionWriter(out_dir=tmp_path)
    await writer.write(_slack_doc("doc-1"))
    await writer.close()
    path = tmp_path / "raw" / "slack" / "doc-1.json"
    assert path.exists()
    payload = orjson.loads(path.read_bytes())
    assert payload["type"] == "event_callback"


@pytest.mark.asyncio
async def test_local_writes_notion_envelope_to_disk(tmp_path: Path) -> None:
    writer = IngestionWriter(out_dir=tmp_path)
    await writer.write(_notion_doc("page-1"))
    await writer.close()
    path = tmp_path / "raw" / "notion" / "page-1.json"
    assert path.exists()
    payload = orjson.loads(path.read_bytes())
    assert payload["type"] == "page.updated"


@pytest.mark.asyncio
async def test_local_overwrite_on_repeat_write(tmp_path: Path) -> None:
    writer = IngestionWriter(out_dir=tmp_path)
    await writer.write(_slack_doc("doc-1"))
    await writer.write(_slack_doc("doc-1"))  # second write must not raise
    path = tmp_path / "raw" / "slack" / "doc-1.json"
    assert path.exists()


@pytest.mark.asyncio
async def test_local_unsupported_source_raises(tmp_path: Path) -> None:
    """Granola/Claude_Code sources (not yet implemented) should raise ValueError."""
    writer = IngestionWriter(out_dir=tmp_path)

    # Use a Source value that is genuinely unimplemented.  We reach into the
    # enum to find one that is NOT in the 5 dispatched sources; if all are
    # supported this test is vacuously skipped.
    supported = {Source.SLACK, Source.NOTION, Source.GITHUB, Source.LINEAR, Source.SENTRY}
    unsupported = [s for s in Source if s not in supported]
    if not unsupported:
        pytest.skip("All known sources are now implemented; no unsupported source to test.")

    doc = SynthDoc(
        id="x",
        source=unsupported[0],
        source_event_id="x",
        text="",
        occurred_at=datetime(2026, 5, 1, tzinfo=UTC),
        channel=None,
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-z",
        archetype="STANDUP_UPDATE",
        personas=(),
        services_mentioned=(),
    )
    with pytest.raises(ValueError, match="Unsupported source for envelope wrapping"):
        await writer.write(doc)


@pytest.mark.asyncio
async def test_local_writes_github_envelope_to_disk(tmp_path: Path) -> None:
    """Plan 3 dispatch: github source → IngestionWriter calls github_wrapper.wrap."""
    import orjson

    doc = SynthDoc(
        id="scn-test-github-0",
        source=Source.GITHUB,
        source_event_id="scn-test-github-0",
        text="Title: Sample PR\n\nBody text",
        occurred_at=datetime(2026, 5, 1, tzinfo=UTC),
        channel=None,
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-test",
        archetype="INCIDENT",
        personas=("gh:alice",),
        services_mentioned=("payments",),
        priority=10,
    )
    writer = IngestionWriter(out_dir=tmp_path)
    await writer.write(doc)
    await writer.close()

    out_file = tmp_path / "raw" / "github" / "scn-test-github-0.json"
    assert out_file.exists(), "expected raw/github/<id>.json"
    payload = orjson.loads(out_file.read_bytes())
    # github_wrapper produces pull_request.opened for non-BIG_REFACTOR archetypes
    assert "pull_request" in payload or "issue" in payload


# ---------------------------------------------------------------------------
# Task 14: integrate-mode tests
# ---------------------------------------------------------------------------


def _mock_bucket() -> AsyncMock:
    bucket = AsyncMock()
    bucket.bucket_for = MagicMock(return_value="prbe-synth-bucket")
    bucket.put = AsyncMock(return_value=None)
    return bucket


def _mock_db() -> AsyncMock:
    db = AsyncMock()
    db.executemany = AsyncMock(return_value=None)
    return db


@pytest.mark.asyncio
async def test_integrate_writes_local_and_bucket_and_queues_row(tmp_path: Path) -> None:
    bucket = _mock_bucket()
    db = _mock_db()
    writer = IngestionWriter(
        out_dir=tmp_path,
        mode="integrate",
        customer_id="cust-eval-test-01",
        bucket=bucket,
        db=db,
    )
    await writer.write(_slack_doc("doc-1"))
    await writer.close()

    # Local file written
    assert (tmp_path / "raw" / "slack" / "doc-1.json").exists()
    # R2 put called once with the customer-scoped key
    assert bucket.put.await_count == 1
    args = bucket.put.await_args.args
    assert args[1] == "raw/slack/cust-eval-test-01/synth/doc-1.json"
    # ingestion_queue insert flushed on close
    assert db.executemany.await_count == 1
    sql = db.executemany.await_args.args[0]
    assert "INSERT INTO ingestion_queue" in sql
    assert "ON CONFLICT" in sql


@pytest.mark.asyncio
async def test_integrate_batches_at_50_writes(tmp_path: Path) -> None:
    bucket = _mock_bucket()
    db = _mock_db()
    writer = IngestionWriter(
        out_dir=tmp_path,
        mode="integrate",
        customer_id="cust-eval-test-01",
        bucket=bucket,
        db=db,
    )
    for i in range(50):
        await writer.write(_slack_doc(f"doc-{i}"))
    # Flush should have triggered exactly once at the 50th write.
    assert db.executemany.await_count == 1
    await writer.close()
    # close() with empty batch is a no-op.
    assert db.executemany.await_count == 1


@pytest.mark.asyncio
async def test_integrate_close_flushes_residual_batch(tmp_path: Path) -> None:
    bucket = _mock_bucket()
    db = _mock_db()
    writer = IngestionWriter(
        out_dir=tmp_path,
        mode="integrate",
        customer_id="cust-eval-test-01",
        bucket=bucket,
        db=db,
    )
    for i in range(10):
        await writer.write(_slack_doc(f"doc-{i}"))
    assert db.executemany.await_count == 0  # under batch threshold
    await writer.close()
    assert db.executemany.await_count == 1


@pytest.mark.asyncio
async def test_integrate_requires_customer_id_bucket_db(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="integrate mode requires"):
        IngestionWriter(out_dir=tmp_path, mode="integrate")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("PRBE_TEST_DB_URL"),
    reason="PRBE_TEST_DB_URL env not set; skipping live integration test",
)
async def test_integrate_round_trip_against_test_db(tmp_path: Path) -> None:
    """Live integration smoke. Requires PRBE_TEST_DB_URL pointing at a
    disposable Postgres + a real ObjectStore. Skipped in standard CI.
    """
    pytest.skip("placeholder: implementer wires this against shared.db helpers")
