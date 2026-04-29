"""fetch_supplementary post-migration 0026 reads `event.payload_s3_keys`
(coalesced array) and merges every batch's webhook-envelope contents.

Pre-coalescing it listed a per-session R2 prefix that live traffic
never wrote to, which silently lost all-but-the-latest batch's events
per session. These tests pin the new behavior.
"""
from datetime import UTC, datetime

import orjson
import pytest

from services.ingestion.handlers.base import make_default_context
from services.ingestion.handlers.claude_code import ClaudeCodeConnector
from shared.constants import SourceSystem
from shared.models import WebhookEvent
from shared.storage import get_store


def _envelope(*, session_id: str, batch_seq: int, events: list[dict]) -> bytes:
    """Match what services/ingestion/main.py:webhook writes to R2."""
    return orjson.dumps({
        "_headers": {},
        "payload": {
            "device_id": "dev-1",
            "session_id": session_id,
            "batch_seq": batch_seq,
            "cwd": "/tmp/p",
            "events": events,
        },
        "received_at": datetime.now(UTC).isoformat(),
        "trace_id": f"test-{session_id}-{batch_seq}",
    })


def _make_event(
    customer_id: str,
    session_id: str,
    payload_s3_keys: list[str],
    *,
    source_event_id: str | None = None,
) -> WebhookEvent:
    return WebhookEvent(
        customer_id=customer_id,
        source_system=SourceSystem.CLAUDE_CODE,
        source_event_id=source_event_id or session_id,
        received_at=datetime.now(UTC),
        payload_s3_key=payload_s3_keys[0] if payload_s3_keys else "",
        payload_s3_keys=payload_s3_keys,
        raw_payload={
            "device_id": "dev-1",
            "session_id": session_id,
            "events": [],
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

    keys: list[str] = []
    for batch_seq, ev in enumerate([
        {"line_no": 0, "role": "user", "content": "hi"},
        {"line_no": 1, "role": "assistant", "content": "hello"},
        {"line_no": 2, "role": "user", "content": "continue"},
    ]):
        key = f"raw/claude_code/{customer}/2026/04/29/{session}:{batch_seq}.json"
        keys.append(key)
        await store.put(bucket, key, _envelope(
            session_id=session, batch_seq=batch_seq, events=[ev],
        ))

    c = ClaudeCodeConnector(make_default_context())
    event = _make_event(customer, session, keys)
    hydrated = await c.fetch_supplementary(event, token=None)

    assert hydrated["session_id"] == session
    assert len(hydrated["events"]) == 3
    assert [e["line_no"] for e in hydrated["events"]] == [0, 1, 2]
    assert hydrated["session_complete"] is False

    await store.delete_bucket_recursive(bucket)


@pytest.mark.asyncio
async def test_fetch_supplementary_detects_finalize_marker() -> None:
    """The session-completer cron upserts finalize.marker into the live
    row's payload_s3_keys array. fetch_supplementary detects the marker
    by key suffix and forces session_complete=True. The marker's empty
    events array contributes nothing to the merge — only the real batch's
    events survive.
    """
    customer = "fs-finalize-cust"
    session = "sess-final"
    store = get_store()
    bucket = store.bucket_for(customer)
    await store.ensure_bucket(bucket)

    live_key = f"raw/claude_code/{customer}/2026/04/29/{session}:0.json"
    marker_key = f"raw/claude_code/{customer}/{session}/finalize.marker"
    await store.put(bucket, live_key, _envelope(
        session_id=session, batch_seq=0,
        events=[{"line_no": 0, "role": "user", "content": "hi"}],
    ))
    # The cron's marker is itself an envelope-shaped placeholder with
    # finalize:true, events:[]. fetch_supplementary detects the marker
    # via the key suffix, not via the body content.
    await store.put(bucket, marker_key, orjson.dumps({
        "device_id": "cron-finalize",
        "session_id": session,
        "batch_seq": -1,
        "cwd": None,
        "events": [],
        "finalize": True,
    }))

    c = ClaudeCodeConnector(make_default_context())
    event = _make_event(customer, session, [live_key, marker_key])
    hydrated = await c.fetch_supplementary(event, token=None)

    assert len(hydrated["events"]) == 1
    assert hydrated["events"][0]["line_no"] == 0
    assert hydrated["session_complete"] is True

    await store.delete_bucket_recursive(bucket)


@pytest.mark.asyncio
async def test_fetch_supplementary_dedupes_overlapping_line_nos() -> None:
    """Daemon retries can ship the same batch twice. Duplicate line_no
    values across the array dedupe at merge time."""
    customer = "fs-dedup-cust"
    session = "sess-dup"
    store = get_store()
    bucket = store.bucket_for(customer)
    await store.ensure_bucket(bucket)

    # Batch 0 has line_no 0,1; batch 1 has line_no 1,2 (line_no=1 overlaps).
    key0 = f"raw/claude_code/{customer}/2026/04/29/{session}:0.json"
    key1 = f"raw/claude_code/{customer}/2026/04/29/{session}:1.json"
    await store.put(bucket, key0, _envelope(
        session_id=session, batch_seq=0,
        events=[
            {"line_no": 0, "role": "user"},
            {"line_no": 1, "role": "assistant"},
        ],
    ))
    await store.put(bucket, key1, _envelope(
        session_id=session, batch_seq=1,
        events=[
            {"line_no": 1, "role": "assistant"},
            {"line_no": 2, "role": "user"},
        ],
    ))

    c = ClaudeCodeConnector(make_default_context())
    event = _make_event(customer, session, [key0, key1])
    hydrated = await c.fetch_supplementary(event, token=None)

    assert [e["line_no"] for e in hydrated["events"]] == [0, 1, 2]
    assert hydrated["session_complete"] is False

    await store.delete_bucket_recursive(bucket)
