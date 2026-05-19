"""Ingestion worker dispatches the post-approval resolution-check seam.

When the normalizer's outcome carries ``resolution_check_doc_ids``
(populated only for PD ``incident.resolved`` and incident.io
``incident_closed_v2`` events), the worker fires
``services.post_approval.dispatch.on_resolution_event`` for each id —
that's how the worker side of the (approved ∧ resolved) edge gets
signaled.

These tests stub the normalizer + queue layer so we exercise just the
worker → dispatch wiring, not the full Phase A/B persistence path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.normalizer import NormalizeOutcome
from services.ingestion.worker import Worker
from shared.config import Settings

pytestmark = pytest.mark.asyncio


def _build_worker() -> Worker:
    settings = Settings(environment="local")
    return Worker(
        ConnectorContext(settings=settings, http=httpx.AsyncClient()),
        max_attempts=1,
        concurrency=1,
    )


@dataclass(slots=True)
class _FakeRow:
    """Minimal asyncpg.Record stand-in for Worker._process inputs."""

    queue_id: int
    customer_id: str
    source_system: str
    source_event_id: str
    payload_s3_keys: list[str]
    payload_s3_key: str
    version: int
    attempts: int

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def _make_row(
    *,
    customer_id: str = "cust-worker-1",
    source_system: str = "pagerduty",
    source_event_id: str = "pd:incident:PD-INC-001:incident.resolved",
    queue_id: int = 1001,
) -> _FakeRow:
    return _FakeRow(
        queue_id=queue_id,
        customer_id=customer_id,
        source_system=source_system,
        source_event_id=source_event_id,
        payload_s3_keys=[f"raw/pagerduty/{customer_id}/test.json"],
        payload_s3_key=f"raw/pagerduty/{customer_id}/test.json",
        version=1,
        attempts=0,
    )


async def test_worker_dispatches_on_resolution_event_when_flag_set() -> None:
    """When the normalizer's NormalizeOutcome carries doc ids in
    resolution_check_doc_ids, the worker calls on_resolution_event for
    each id."""
    worker = _build_worker()
    row = _make_row()

    fake_outcome = NormalizeOutcome(
        doc_ids=["pd:incident:PD-INC-001"],
        chunk_count=1,
        failed_chunk_count=0,
        resolution_check_doc_ids=["pd:incident:PD-INC-001"],
    )

    with (
        patch.object(
            worker._normalizer, "process_queue_row",
            new_callable=AsyncMock, return_value=fake_outcome,
        ),
        patch.object(worker, "_mark_done", new_callable=AsyncMock),
        patch.object(worker, "_heartbeat", new_callable=AsyncMock),
        patch(
            "services.ingestion.worker."
            "post_approval_dispatch.on_resolution_event",
            new_callable=AsyncMock,
        ) as fake_dispatch,
    ):
        await worker._process(row)

    fake_dispatch.assert_awaited_once_with(
        customer_id="cust-worker-1",
        incident_doc_id="pd:incident:PD-INC-001",
    )


async def test_worker_does_not_dispatch_when_flag_unset() -> None:
    """For routine events (incident.triggered, incident.acknowledged,
    every non-PD/iio source), resolution_check_doc_ids is empty and the
    worker must NOT call the dispatch seam."""
    worker = _build_worker()
    row = _make_row(source_event_id="pd:incident:PD-INC-001:incident.triggered")

    fake_outcome = NormalizeOutcome(
        doc_ids=["pd:incident:PD-INC-001"],
        chunk_count=1,
        failed_chunk_count=0,
        resolution_check_doc_ids=[],  # routine event — no dispatch
    )

    with (
        patch.object(
            worker._normalizer, "process_queue_row",
            new_callable=AsyncMock, return_value=fake_outcome,
        ),
        patch.object(worker, "_mark_done", new_callable=AsyncMock),
        patch.object(worker, "_heartbeat", new_callable=AsyncMock),
        patch(
            "services.ingestion.worker."
            "post_approval_dispatch.on_resolution_event",
            new_callable=AsyncMock,
        ) as fake_dispatch,
    ):
        await worker._process(row)

    fake_dispatch.assert_not_awaited()


async def test_worker_dispatch_failure_does_not_poison_queue_row() -> None:
    """A transient orchestrator hiccup must NOT push the queue row off
    the happy path — ingestion has already persisted, re-processing
    would re-embed for nothing. The worker logs the failure and still
    calls _mark_done.
    """
    worker = _build_worker()
    row = _make_row()

    fake_outcome = NormalizeOutcome(
        doc_ids=["pd:incident:PD-INC-001"],
        chunk_count=1,
        failed_chunk_count=0,
        resolution_check_doc_ids=["pd:incident:PD-INC-001"],
    )

    with (
        patch.object(
            worker._normalizer, "process_queue_row",
            new_callable=AsyncMock, return_value=fake_outcome,
        ),
        patch.object(worker, "_mark_done", new_callable=AsyncMock) as fake_done,
        patch.object(worker, "_heartbeat", new_callable=AsyncMock),
        patch(
            "services.ingestion.worker."
            "post_approval_dispatch.on_resolution_event",
            new_callable=AsyncMock,
            side_effect=RuntimeError("orchestrator down"),
        ) as fake_dispatch,
    ):
        await worker._process(row)

    fake_dispatch.assert_awaited_once()
    fake_done.assert_awaited_once()  # happy path completion


async def test_worker_dispatches_each_resolution_doc_id() -> None:
    """If a single event resolves multiple logical incidents (e.g. a
    batched webhook), the worker must call on_resolution_event once per
    doc_id."""
    worker = _build_worker()
    row = _make_row()

    fake_outcome = NormalizeOutcome(
        doc_ids=["pd:incident:A", "pd:incident:B"],
        chunk_count=2,
        failed_chunk_count=0,
        resolution_check_doc_ids=["pd:incident:A", "pd:incident:B"],
    )

    with (
        patch.object(
            worker._normalizer, "process_queue_row",
            new_callable=AsyncMock, return_value=fake_outcome,
        ),
        patch.object(worker, "_mark_done", new_callable=AsyncMock),
        patch.object(worker, "_heartbeat", new_callable=AsyncMock),
        patch(
            "services.ingestion.worker."
            "post_approval_dispatch.on_resolution_event",
            new_callable=AsyncMock,
        ) as fake_dispatch,
    ):
        await worker._process(row)

    assert fake_dispatch.await_count == 2
    awaited_doc_ids = {
        call.kwargs["incident_doc_id"]
        for call in fake_dispatch.await_args_list
    }
    assert awaited_doc_ids == {"pd:incident:A", "pd:incident:B"}


async def test_normalize_outcome_resolution_check_doc_ids_default_empty() -> None:
    """Routine NormalizeOutcome construction (no resolution flag) leaves
    resolution_check_doc_ids empty — keeps every existing connector
    unchanged."""
    outcome = NormalizeOutcome(
        doc_ids=["x"], chunk_count=1, failed_chunk_count=0,
    )
    assert outcome.resolution_check_doc_ids == []
