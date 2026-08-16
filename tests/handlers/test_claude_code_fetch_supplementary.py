"""fetch_supplementary post-migration 0026 reads `event.payload_s3_keys`
(coalesced array) and merges every batch's webhook-envelope contents.

Pre-coalescing it listed a per-session R2 prefix that live traffic
never wrote to, which silently lost all-but-the-latest batch's events
per session. These tests pin the new behavior.
"""
from datetime import UTC, datetime

import orjson
import pytest

from engine.ingest.handlers.base import make_default_context
from engine.shared.constants import SourceSystem
from engine.shared.models import WebhookEvent
from kb.handlers.claude_code import ClaudeCodeConnector


class _StubStore:
    def __init__(self) -> None:
        self.blobs: dict[tuple[str, str], bytes] = {}

    async def bucket_for(self, customer_id: str) -> str:
        return f"test-bucket-{customer_id}"

    async def ensure_bucket(self, bucket: str) -> None:
        return None

    async def put(self, bucket: str, key: str, body: bytes) -> None:
        self.blobs[(bucket, key)] = body

    async def get(self, bucket: str, key: str) -> bytes:
        return self.blobs[(bucket, key)]

    async def delete_bucket_recursive(self, bucket: str) -> None:
        keys = [k for k in self.blobs if k[0] == bucket]
        for key in keys:
            self.blobs.pop(key, None)


@pytest.fixture
def stub_store(monkeypatch: pytest.MonkeyPatch) -> _StubStore:
    store = _StubStore()
    from kb.handlers import claude_code as cc_mod

    monkeypatch.setattr(cc_mod, "get_store", lambda: store)
    return store


def _envelope(
    *,
    session_id: str,
    batch_seq: int,
    events: list[dict],
    extra_payload: dict | None = None,
) -> bytes:
    """Match what services/ingestion/main.py:webhook writes to R2."""
    payload = {
        "device_id": "dev-1",
        "session_id": session_id,
        "batch_seq": batch_seq,
        "cwd": "/tmp/p",
        "events": events,
    }
    if extra_payload:
        payload.update(extra_payload)
    return orjson.dumps({
        "_headers": {},
        "payload": payload,
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
async def test_fetch_supplementary_merges_all_batches_for_session(
    stub_store: _StubStore,
) -> None:
    customer = "fs-test-cust"
    session = "sess-1"
    store = stub_store
    bucket = await store.bucket_for(customer)
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
async def test_fetch_supplementary_carries_identity_from_later_payloads(
    stub_store: _StubStore,
) -> None:
    """Coalesced rows parse event.raw_payload from the oldest payload.

    If a session began before the gateway added name/email/hostname, those
    labels only appear on later payloads. fetch_supplementary must surface
    them so normalize() does not rewrite the active session back to a plain
    title.
    """
    customer = "fs-identity-cust"
    session = "sess-identity"
    store = stub_store
    bucket = await store.bucket_for(customer)
    await store.ensure_bucket(bucket)

    key0 = f"raw/claude_code/{customer}/2026/04/29/{session}:0.json"
    key1 = f"raw/claude_code/{customer}/2026/04/29/{session}:1.json"
    await store.put(bucket, key0, _envelope(
        session_id=session,
        batch_seq=0,
        events=[{"line_no": 0, "role": "user", "content": "before deploy"}],
        extra_payload={"employee_id": "emp-1"},
    ))
    await store.put(bucket, key1, _envelope(
        session_id=session,
        batch_seq=1,
        events=[{"line_no": 1, "role": "assistant", "content": "after deploy"}],
        extra_payload={
            "employee_id": "emp-1",
            "employee_name": "Richard Wei",
            "employee_email": "richard@prbe.ai",
            "employee_hostname": "Richards-MacBook-Pro.local",
        },
    ))

    c = ClaudeCodeConnector(make_default_context())
    event = _make_event(customer, session, [key0, key1])
    hydrated = await c.fetch_supplementary(event, token=None)

    assert hydrated["employee_id"] == "emp-1"
    assert hydrated["employee_name"] == "Richard Wei"
    assert hydrated["employee_email"] == "richard@prbe.ai"
    assert hydrated["employee_hostname"] == "Richards-MacBook-Pro.local"
    assert [e["line_no"] for e in hydrated["events"]] == [0, 1]

    await store.delete_bucket_recursive(bucket)


@pytest.mark.asyncio
async def test_fetch_supplementary_detects_finalize_marker(
    stub_store: _StubStore,
) -> None:
    """The session-completer cron upserts finalize.marker into the live
    row's payload_s3_keys array. fetch_supplementary detects the marker
    by key suffix and forces session_complete=True. The marker's empty
    events array contributes nothing to the merge — only the real batch's
    events survive.
    """
    customer = "fs-finalize-cust"
    session = "sess-final"
    store = stub_store
    bucket = await store.bucket_for(customer)
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
async def test_fetch_supplementary_dedupes_overlapping_line_nos(
    stub_store: _StubStore,
) -> None:
    """Daemon retries can ship the same batch twice. Duplicate line_no
    values across the array dedupe at merge time."""
    customer = "fs-dedup-cust"
    session = "sess-dup"
    store = stub_store
    bucket = await store.bucket_for(customer)
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


@pytest.mark.asyncio
async def test_client_finalize_payload_completes_the_session(
    stub_store: _StubStore,
) -> None:
    """The tap's SessionEnd finalize must mark the session complete.

    It arrives as an ordinary coalesced payload carrying `finalize: true` and
    no events, keyed like any other batch — NOT as the cron's dedicated
    `.../finalize.marker` object. Before this was honored the gateway route
    authenticated, forwarded and stored the payload, returned 202, and nothing
    ever set complete — so a cleanly-ended session was never mined for units.
    """
    customer = "fs-finalize-cust"
    session = "sess-finalize"
    store = stub_store
    bucket = await store.bucket_for(customer)
    await store.ensure_bucket(bucket)

    batch_key = f"raw/claude_code/{customer}/2026/04/29/{session}:0.json"
    await store.put(bucket, batch_key, _envelope(
        session_id=session,
        batch_seq=0,
        events=[{"line_no": 0, "raw": {"type": "user", "content": "hi"}}],
    ))

    # The gateway rebuilds a finalize body from validated fields only:
    # finalize + session_id + device_id + server-stamped identity. No events,
    # no batch_seq — hence the bare-session_id R2 key.
    finalize_key = f"raw/claude_code/{customer}/2026/04/29/{session}.json"
    await store.put(bucket, finalize_key, orjson.dumps({
        "_headers": {},
        "payload": {
            "finalize": True,
            "session_id": session,
            "device_id": "dev-1",
            "employee_id": "emp-1",
        },
        "received_at": datetime.now(UTC).isoformat(),
        "trace_id": f"test-{session}-finalize",
    }))

    c = ClaudeCodeConnector(make_default_context())
    hydrated = await c.fetch_supplementary(
        _make_event(customer, session, [batch_key, finalize_key]), None
    )

    assert hydrated["session_complete"] is True
    # The finalize payload contributes no events — the real transcript must
    # survive it intact, or the extractor gets an empty session to mine.
    assert len(hydrated["events"]) == 1


@pytest.mark.asyncio
async def test_batches_without_any_finalize_stay_incomplete(
    stub_store: _StubStore,
) -> None:
    """The negative half: ordinary traffic must NOT look finished.

    Guards the branch above from degenerating into "always complete", which
    would re-run the extraction LLM on every batch of every live session.
    """
    customer = "fs-open-cust"
    session = "sess-open"
    store = stub_store
    bucket = await store.bucket_for(customer)
    await store.ensure_bucket(bucket)

    key = f"raw/claude_code/{customer}/2026/04/29/{session}:0.json"
    await store.put(bucket, key, _envelope(
        session_id=session,
        batch_seq=0,
        events=[{"line_no": 0, "raw": {"type": "user", "content": "still going"}}],
    ))

    c = ClaudeCodeConnector(make_default_context())
    hydrated = await c.fetch_supplementary(_make_event(customer, session, [key]), None)

    assert hydrated["session_complete"] is False
