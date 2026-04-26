from datetime import UTC, datetime

import pytest

from services.ingestion.handlers.base import make_default_context
from services.ingestion.handlers.claude_code import ClaudeCodeConnector
from shared.constants import SourceSystem
from shared.models import WebhookEvent
from shared.storage import get_store


def _make_event(customer_id: str, session_id: str, batch_seq: int) -> WebhookEvent:
    return WebhookEvent(
        customer_id=customer_id,
        source_system=SourceSystem.CLAUDE_CODE,
        source_event_id=f"{session_id}:{batch_seq}",
        received_at=datetime.now(UTC),
        payload_s3_key=f"raw/claude_code/{customer_id}/{session_id}/{batch_seq}.jsonl",
        raw_payload={
            "device_id": "dev-1",
            "session_id": session_id,
            "batch_seq": batch_seq,
            "cwd": "/tmp/p",
            "events": [],  # we re-read full from R2
        },
        headers={},
    )


@pytest.mark.asyncio
async def test_fetch_supplementary_merges_all_batches_for_session() -> None:
    customer = "fs-test-cust"
    session = "sess-1"
    store = get_store()
    bucket = store.bucket_for(customer)
    await store.ensure_bucket(bucket)

    # Three batches written by the daemon
    for batch_seq, line in enumerate(
        [b'{"line_no":0,"role":"user","content":"hi"}\n',
         b'{"line_no":1,"role":"assistant","content":"hello"}\n',
         b'{"line_no":2,"role":"user","content":"continue"}\n'],
    ):
        await store.put(
            bucket,
            f"raw/claude_code/{customer}/{session}/{batch_seq}.jsonl",
            line,
        )

    c = ClaudeCodeConnector(make_default_context())
    event = _make_event(customer, session, 2)
    hydrated = await c.fetch_supplementary(event, token=None)

    assert hydrated["session_id"] == session
    assert len(hydrated["events"]) == 3
    assert [e["line_no"] for e in hydrated["events"]] == [0, 1, 2]
    assert hydrated["session_complete"] is False

    await store.delete_bucket_recursive(bucket)


@pytest.mark.asyncio
async def test_fetch_supplementary_skips_finalize_marker() -> None:
    """The session-completer cron writes raw/.../{session}/finalize.marker
    alongside the daemon's batch files. fetch_supplementary must NOT include
    the marker's body as an event."""
    customer = "fs-finalize-cust"
    session = "sess-final"
    store = get_store()
    bucket = store.bucket_for(customer)
    await store.ensure_bucket(bucket)

    # One real batch
    await store.put(
        bucket,
        f"raw/claude_code/{customer}/{session}/0.jsonl",
        b'{"line_no":0,"role":"user","content":"hi"}\n',
    )
    # Cron-written finalize marker (the kind that previously polluted events)
    await store.put(
        bucket,
        f"raw/claude_code/{customer}/{session}/finalize.marker",
        b'{"finalize":true}',
    )

    c = ClaudeCodeConnector(make_default_context())
    event = WebhookEvent(
        customer_id=customer,
        source_system=SourceSystem.CLAUDE_CODE,
        source_event_id=f"{session}:finalize",
        received_at=datetime.now(UTC),
        payload_s3_key=f"raw/claude_code/{customer}/{session}/finalize.marker",
        raw_payload={"device_id": "dev-1", "session_id": session, "events": [], "cwd": None},
        headers={},
    )
    hydrated = await c.fetch_supplementary(event, token=None)

    # Only the real batch event survives — the finalize body is filtered out.
    assert len(hydrated["events"]) == 1
    assert hydrated["events"][0]["line_no"] == 0
    assert hydrated["session_complete"] is True  # :finalize suffix forces it

    await store.delete_bucket_recursive(bucket)


@pytest.mark.asyncio
async def test_fetch_supplementary_dedupes_overlapping_line_nos() -> None:
    """If the daemon reships a batch (rare but possible on transient errors),
    duplicate line_no values should be deduplicated."""
    customer = "fs-dedup-cust"
    session = "sess-dup"
    store = get_store()
    bucket = store.bucket_for(customer)
    await store.ensure_bucket(bucket)

    # Batch 0 has line_no 0,1; batch 1 has line_no 1,2 (line_no=1 overlaps).
    await store.put(
        bucket,
        f"raw/claude_code/{customer}/{session}/0.jsonl",
        b'{"line_no":0,"role":"user"}\n{"line_no":1,"role":"assistant"}\n',
    )
    await store.put(
        bucket,
        f"raw/claude_code/{customer}/{session}/1.jsonl",
        b'{"line_no":1,"role":"assistant"}\n{"line_no":2,"role":"user"}\n',
    )

    c = ClaudeCodeConnector(make_default_context())
    event = _make_event(customer, session, 1)
    hydrated = await c.fetch_supplementary(event, token=None)

    assert [e["line_no"] for e in hydrated["events"]] == [0, 1, 2]
    assert hydrated["session_complete"] is False

    await store.delete_bucket_recursive(bucket)
