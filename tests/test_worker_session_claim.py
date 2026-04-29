"""Same-session claim serialization in `Worker._claim_one`.

Regression for incident 2026-04-29: two batches of the same Claude Code
session were being processed concurrently across worker machines, both
upserting the same canonical graph_nodes row (DOCUMENT and PERSON for the
session) and contending on the row lock — the second worker timed out at
db_statement_timeout (30s) and retried, until DLQ.

The fix: `_claim_one`'s SQL has a NOT EXISTS clause that skips a pending
row whose session_key (`split_part(source_event_id, ':', 1)`) matches an
in-flight processing row for the same customer + source. Connectors
without a colon in source_event_id (slack, github, notion, linear,
granola, sentry) get session_key == full id, so each row is its own
session and they never serialize against each other.

These tests pin both halves of that contract.
"""

from __future__ import annotations

import httpx
import pytest

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.worker import Worker
from shared import db as db_module
from shared.config import Settings
from shared.constants import QueueStatus


async def _enqueue(
    *,
    customer_id: str,
    source_system: str,
    source_event_id: str,
    status: str = QueueStatus.PENDING.value,
    priority: int = 100,
) -> int:
    """Insert a synthetic ingestion_queue row and return its queue_id."""
    async with db_module.raw_conn() as conn:
        # Seed the customer if it doesn't exist — ingestion_queue's FK requires it.
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'test-hash')
            ON CONFLICT DO NOTHING
            """,
            customer_id,
        )
        queue_id = await conn.fetchval(
            """
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id,
                 payload_s3_key, status, priority,
                 started_at, heartbeat_at)
            VALUES ($1, $2, $3, $4, $5, $6,
                    CASE WHEN $5 = 'processing' THEN NOW() ELSE NULL END,
                    CASE WHEN $5 = 'processing' THEN NOW() ELSE NULL END)
            RETURNING queue_id
            """,
            customer_id,
            source_system,
            source_event_id,
            f"raw/{source_system}/{customer_id}/{source_event_id}.json",
            status,
            priority,
        )
        return int(queue_id)


def _make_worker() -> Worker:
    settings = Settings(environment="local")
    ctx = ConnectorContext(settings=settings, http=httpx.AsyncClient())
    return Worker(ctx, max_attempts=5, concurrency=1)


@pytest.mark.asyncio
async def test_claim_skips_pending_when_same_session_is_processing(live_db) -> None:
    """The whole point: batch 1 of a CC session can't claim while batch 0 runs."""
    worker = _make_worker()

    session_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"  # ULID-style; arbitrary
    # Batch 0 is already in-flight on some other worker.
    in_flight_qid = await _enqueue(
        customer_id="cust-cc-1",
        source_system="claude_code",
        source_event_id=f"{session_id}:0",
        status=QueueStatus.PROCESSING.value,
    )
    # Batch 1 is pending and would normally be claimable.
    blocked_qid = await _enqueue(
        customer_id="cust-cc-1",
        source_system="claude_code",
        source_event_id=f"{session_id}:1",
        status=QueueStatus.PENDING.value,
    )

    claimed = await worker._claim_one()
    assert claimed is None, (
        f"_claim_one returned a row (queue_id={claimed['queue_id'] if claimed else None}, "
        f"event_id={claimed['source_event_id'] if claimed else None}) but should "
        f"have skipped queue_id={blocked_qid} because session {session_id} already "
        f"has an in-flight row at queue_id={in_flight_qid}"
    )

    # Now mark the in-flight row done. The pending batch 1 should claim cleanly.
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "UPDATE ingestion_queue SET status='done', completed_at=NOW() WHERE queue_id=$1",
            in_flight_qid,
        )

    claimed = await worker._claim_one()
    assert claimed is not None, "batch 1 should claim once batch 0 is done"
    assert claimed["queue_id"] == blocked_qid


@pytest.mark.asyncio
async def test_claim_does_not_block_different_session_for_same_customer(live_db) -> None:
    """A different session's batch must claim freely even when one session is busy."""
    worker = _make_worker()

    busy_session = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    free_session = "01HW9MX7VQ8B0Q0F5Q8V4M3K2H"

    await _enqueue(
        customer_id="cust-cc-2",
        source_system="claude_code",
        source_event_id=f"{busy_session}:0",
        status=QueueStatus.PROCESSING.value,
    )
    free_qid = await _enqueue(
        customer_id="cust-cc-2",
        source_system="claude_code",
        source_event_id=f"{free_session}:0",
        status=QueueStatus.PENDING.value,
    )

    claimed = await worker._claim_one()
    assert claimed is not None and claimed["queue_id"] == free_qid


@pytest.mark.asyncio
async def test_claim_does_not_serialize_connectors_without_colon(live_db) -> None:
    """Connectors whose source_event_id has no colon get session_key = full id.

    Two distinct slack events must never serialize against each other.
    """
    worker = _make_worker()

    in_flight = await _enqueue(
        customer_id="cust-slack-1",
        source_system="slack",
        source_event_id="evt-A1B2C3",  # no colon → session_key = "evt-A1B2C3"
        status=QueueStatus.PROCESSING.value,
    )
    pending = await _enqueue(
        customer_id="cust-slack-1",
        source_system="slack",
        source_event_id="evt-D4E5F6",  # no colon → session_key = "evt-D4E5F6" (different)
        status=QueueStatus.PENDING.value,
    )

    claimed = await worker._claim_one()
    assert claimed is not None, (
        f"slack pending row (queue_id={pending}) was wrongly skipped — "
        f"it has a different session_key from the in-flight row (queue_id={in_flight})"
    )
    assert claimed["queue_id"] == pending


@pytest.mark.asyncio
async def test_claim_isolates_by_customer_and_source(live_db) -> None:
    """The NOT EXISTS clause matches on customer + source + session_key.

    Same session_id under different customer (or different source) must NOT
    serialize. Two customers running CC sessions that happen to share a UUID
    must process in parallel.
    """
    worker = _make_worker()

    shared_session = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    await _enqueue(
        customer_id="cust-A",
        source_system="claude_code",
        source_event_id=f"{shared_session}:0",
        status=QueueStatus.PROCESSING.value,
    )
    other_customer_qid = await _enqueue(
        customer_id="cust-B",
        source_system="claude_code",
        source_event_id=f"{shared_session}:0",
        status=QueueStatus.PENDING.value,
    )

    claimed = await worker._claim_one()
    assert claimed is not None and claimed["queue_id"] == other_customer_qid, (
        "different customer's queue row was wrongly skipped — same-session "
        "serialization must scope to (customer_id, source_system, session_key)"
    )
