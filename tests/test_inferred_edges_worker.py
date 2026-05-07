"""Unit tests for the inferred-edges side-queue worker.

Tests drain-loop behavior (dequeue, ack on success, attempt++ on failure,
SKIP LOCKED concurrency) using mocked DB connections.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.ingestion.inferred_edges.worker import (
    _MAX_ATTEMPTS,
    InferredEdgesWorker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    queue_id: int = 1,
    customer_id: str = "cust-worker-test",
    anchor_doc_id: str = "doc-anchor",
    attempts: int = 0,
) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda self, k: {
        "id": queue_id,
        "customer_id": customer_id,
        "anchor_doc_id": anchor_doc_id,
        "extractor_id": "inferred_edges:v1",
        "attempts": attempts,
    }[k]
    return row


# ---------------------------------------------------------------------------
# Tests: _claim_one
# ---------------------------------------------------------------------------


class _AsyncContextManagerMock:
    """A reusable async context manager that yields a fixed value."""

    def __init__(self, return_value):
        self._value = return_value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *_):
        return None


def _make_mock_pool(fetchrow_return=None) -> MagicMock:
    """Create a synchronous MagicMock pool with async acquire/transaction context managers."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    mock_conn.execute = AsyncMock()
    # transaction() must return an async context manager directly (not a coroutine)
    mock_conn.transaction = MagicMock(return_value=_AsyncContextManagerMock(mock_conn))
    # acquire() must return an async context manager directly
    mock_pool.acquire = MagicMock(return_value=_AsyncContextManagerMock(mock_conn))
    mock_pool._mock_conn = mock_conn  # expose for assertions
    return mock_pool


@pytest.mark.anyio
async def test_claim_one_returns_none_on_empty_queue() -> None:
    """_claim_one returns None when no pending rows exist."""
    worker = InferredEdgesWorker(concurrency=1)
    mock_pool = _make_mock_pool(fetchrow_return=None)

    with patch("services.ingestion.inferred_edges.worker.get_pool", return_value=mock_pool):
        result = await worker._claim_one()

    assert result is None


@pytest.mark.anyio
async def test_claim_one_marks_processing() -> None:
    """_claim_one marks the claimed row with processing_started_at."""
    worker = InferredEdgesWorker(concurrency=1)
    row = _make_row()
    mock_pool = _make_mock_pool(fetchrow_return=row)

    with patch("services.ingestion.inferred_edges.worker.get_pool", return_value=mock_pool):
        result = await worker._claim_one()

    assert result is row
    # Verify the UPDATE was called to mark processing
    mock_conn = mock_pool._mock_conn
    mock_conn.execute.assert_called_once()
    call_sql = mock_conn.execute.call_args[0][0]
    assert "processing_started_at" in call_sql


# ---------------------------------------------------------------------------
# Tests: _process success path
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_process_success_marks_done() -> None:
    """On successful extraction, _process marks the row done_at."""
    worker = InferredEdgesWorker(concurrency=1)
    row = _make_row()

    # Mock bundle with content
    mock_bundle = MagicMock()
    mock_bundle.docs = [MagicMock()]
    mock_bundle.customer_id = "cust-worker-test"
    mock_bundle.anchor_doc_id = "doc-anchor"

    # Mock extraction result with no edges
    from services.ingestion.inferred_edges.extractor import ExtractionResult
    mock_extraction = ExtractionResult(bundle_failed=False, cost_usd=0.005)

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()

    with (
        patch("services.ingestion.inferred_edges.worker.with_tenant") as mock_tenant,
        patch("services.ingestion.inferred_edges.worker.build_bundle", return_value=mock_bundle),
        patch("services.ingestion.inferred_edges.worker.extract_edges", return_value=mock_extraction),
        patch("services.ingestion.inferred_edges.worker._mark_done") as mock_mark_done,
    ):
        mock_tenant.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_tenant.return_value.__aexit__ = AsyncMock(return_value=None)

        await worker._process(row)

    mock_mark_done.assert_called_once_with(1)


@pytest.mark.anyio
async def test_process_failure_marks_error() -> None:
    """On exception, _process marks error but does NOT mark done."""
    worker = InferredEdgesWorker(concurrency=1)
    row = _make_row()

    with (
        patch("services.ingestion.inferred_edges.worker.with_tenant") as mock_tenant,
        patch("services.ingestion.inferred_edges.worker.build_bundle", side_effect=RuntimeError("DB is down")),
        patch("services.ingestion.inferred_edges.worker._mark_done") as mock_mark_done,
        patch("services.ingestion.inferred_edges.worker._mark_error") as mock_mark_error,
    ):
        mock_tenant.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
        mock_tenant.return_value.__aexit__ = AsyncMock(return_value=None)

        await worker._process(row)

    mock_mark_done.assert_not_called()
    mock_mark_error.assert_called_once()
    error_arg = mock_mark_error.call_args[0][1]
    assert "DB is down" in error_arg


# ---------------------------------------------------------------------------
# Tests: bundle failure path
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_process_bundle_failed_marks_error_not_done() -> None:
    """When extraction.bundle_failed is True, row is marked error, not done."""
    worker = InferredEdgesWorker(concurrency=1)
    row = _make_row()

    mock_bundle = MagicMock()
    mock_bundle.docs = [MagicMock()]
    mock_bundle.customer_id = "cust-worker-test"
    mock_bundle.anchor_doc_id = "doc-anchor"

    from services.ingestion.inferred_edges.extractor import ExtractionResult
    mock_extraction = ExtractionResult(
        bundle_failed=True,
        bundle_fail_reason="unknown_endpoint_ratio=3/4",
    )

    mock_conn = AsyncMock()

    with (
        patch("services.ingestion.inferred_edges.worker.with_tenant") as mock_tenant,
        patch("services.ingestion.inferred_edges.worker.build_bundle", return_value=mock_bundle),
        patch("services.ingestion.inferred_edges.worker.extract_edges", return_value=mock_extraction),
        patch("services.ingestion.inferred_edges.worker._mark_done") as mock_mark_done,
        patch("services.ingestion.inferred_edges.worker._mark_error") as mock_mark_error,
    ):
        mock_tenant.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_tenant.return_value.__aexit__ = AsyncMock(return_value=None)

        await worker._process(row)

    mock_mark_done.assert_not_called()
    mock_mark_error.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: shutdown
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_worker_shutdown_stops_claim_loop() -> None:
    """After shutdown(), the claim loop terminates without claiming rows."""
    worker = InferredEdgesWorker(concurrency=1)

    call_count = 0

    async def _fake_claim_one():
        nonlocal call_count
        call_count += 1
        # Trigger shutdown on first claim attempt
        worker.shutdown()
        return None

    worker._claim_one = _fake_claim_one

    await asyncio.wait_for(worker.run(), timeout=5.0)
    # The loop ran at least once but terminated
    assert call_count >= 1


# ---------------------------------------------------------------------------
# Tests: max_attempts gate
# ---------------------------------------------------------------------------


def test_max_attempts_constant() -> None:
    """MAX_ATTEMPTS is 3 per the design doc."""
    assert _MAX_ATTEMPTS == 3
