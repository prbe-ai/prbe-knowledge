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


# ---- one completion rule: the NEWEST key decides (engine.shared.session_signals)
#
# The session has ended when the newest key on the row is an end signal.
# Nothing is ever consumed, so these pin both directions: an end signal on top
# ends the session, and anything landing after it makes the session live again.
# The second half is what stops a resumed session from re-mining per batch now
# that the worker no longer deletes finalize keys.


async def _put_batch(store, bucket, customer, session, seq, text="work", *, day="2026/04/29"):
    key = f"raw/claude_code/{customer}/{day}/{session}:{seq}.json"
    await store.put(bucket, key, _envelope(
        session_id=session,
        batch_seq=seq,
        events=[{"line_no": seq, "raw": {"type": "user", "content": text}}],
    ))
    return key


async def _put_client_finalize(store, bucket, customer, session, *, day="2026/04/29"):
    key = f"raw/claude_code/{customer}/{day}/{session}.json"
    await store.put(bucket, key, orjson.dumps({
        "_headers": {},
        "payload": {"finalize": True, "session_id": session, "device_id": "dev-1"},
    }))
    return key


async def _put_marker(store, bucket, customer, session):
    from kb.session_completer import _marker_body

    key = f"raw/claude_code/{customer}/{session}/finalize.marker"
    await store.put(bucket, key, _marker_body(session))
    return key


async def _hydrate(customer, session, keys):
    c = ClaudeCodeConnector(make_default_context())
    return await c.fetch_supplementary(_make_event(customer, session, keys), None)


@pytest.mark.asyncio
async def test_a_batch_after_a_client_finalize_reopens_the_session(stub_store: _StubStore) -> None:
    customer, session = "fs-reopen-cust", "sess-reopen"
    bucket = await stub_store.bucket_for(customer)
    b0 = await _put_batch(stub_store, bucket, customer, session, 0)
    fin = await _put_client_finalize(stub_store, bucket, customer, session)
    b1 = await _put_batch(stub_store, bucket, customer, session, 1, "resumed")

    ended = await _hydrate(customer, session, [b0, fin])
    assert ended["session_complete"] is True
    assert ended["completed_by"] == "v1_client_finalize"

    resumed = await _hydrate(customer, session, [b0, fin, b1])
    assert resumed["session_complete"] is False
    assert resumed["completed_by"] is None
    # The events of every batch still reach the live document.
    assert [e["line_no"] for e in resumed["events"]] == [0, 1]


@pytest.mark.asyncio
async def test_a_marker_on_top_ends_the_session_and_a_later_batch_reopens_it(
    stub_store: _StubStore,
) -> None:
    customer, session = "fs-marker-cust", "sess-marker"
    bucket = await stub_store.bucket_for(customer)
    b0 = await _put_batch(stub_store, bucket, customer, session, 0)
    marker = await _put_marker(stub_store, bucket, customer, session)
    b1 = await _put_batch(stub_store, bucket, customer, session, 1, "back again")

    ended = await _hydrate(customer, session, [b0, marker])
    assert (ended["session_complete"], ended["completed_by"]) == (True, "cron_marker")

    resumed = await _hydrate(customer, session, [b0, marker, b1])
    assert resumed["session_complete"] is False

    re_ended = await _hydrate(customer, session, [b0, marker, b1, marker])
    assert (re_ended["session_complete"], re_ended["completed_by"]) == (True, "cron_marker")


@pytest.mark.asyncio
async def test_a_session_end_event_no_longer_ends_a_session(stub_store: _StubStore) -> None:
    """No producer emits `session_end`, and a sticky in-stream event used to
    certify every later pass as complete. Only end signals on top count."""
    customer, session = "fs-sessend-cust", "sess-sessend"
    bucket = await stub_store.bucket_for(customer)
    key = f"raw/claude_code/{customer}/2026/04/29/{session}:0.json"
    await stub_store.put(bucket, key, _envelope(
        session_id=session,
        batch_seq=0,
        events=[
            {"line_no": 0, "raw": {"type": "user", "content": "hi"}},
            {"line_no": 1, "raw": {"type": "session_end"}},
        ],
    ))
    hydrated = await _hydrate(customer, session, [key])
    assert hydrated["session_complete"] is False


def _v2_payload(session, seq, *, finalize=False, events=None):
    body = {
        "protocol_version": 2,
        "session_id": session,
        "batch_seq": seq,
        "device_id": "dev-1",
        "employee_id": "emp-1",
    }
    if finalize:
        body["finalize"] = True
    else:
        body["events"] = events or [{"line_no": seq, "raw": {"type": "user", "content": f"v2 {seq}"}}]
    return orjson.dumps({"_headers": {}, "payload": body})


@pytest.mark.asyncio
async def test_protocol_2_ends_on_its_sequence_and_on_a_trailing_marker_only(
    stub_store: _StubStore,
) -> None:
    customer, session = "fs-v2-cust", "sess-v2"
    bucket = await stub_store.bucket_for(customer)
    keys = []
    for seq, fin in ((0, False), (1, False)):
        key = f"raw/claude_code/{customer}/sessions-v2/{session}/{seq}-d{seq}.json"
        await stub_store.put(bucket, key, _v2_payload(session, seq, finalize=fin))
        keys.append(key)
    marker = await _put_marker(stub_store, bucket, customer, session)
    fin_key = f"raw/claude_code/{customer}/sessions-v2/{session}/2-d2.json"
    await stub_store.put(bucket, fin_key, _v2_payload(session, 2, finalize=True))

    assert (await _hydrate(customer, session, keys))["session_complete"] is False
    # The sweep's marker ends an idle v2 session while it is the newest key...
    ended = await _hydrate(customer, session, [*keys, marker])
    assert (ended["session_complete"], ended["completed_by"]) == (True, "cron_marker")
    # ...and the client's own finalize ends it by sequence, in any key order.
    finalized = await _hydrate(customer, session, [fin_key, *reversed(keys)])
    assert (finalized["session_complete"], finalized["completed_by"]) == (True, "v2_finalize")
    # A marker that a later v2 batch landed on top of no longer counts.
    late = f"raw/claude_code/{customer}/sessions-v2/{session}/2-late.json"
    await stub_store.put(bucket, late, _v2_payload(session, 2))
    assert (await _hydrate(customer, session, [*keys, marker, late]))["session_complete"] is False


@pytest.mark.asyncio
async def test_an_unreadable_newest_key_does_not_end_the_session(stub_store: _StubStore) -> None:
    customer, session = "fs-garbage-cust", "sess-garbage"
    bucket = await stub_store.bucket_for(customer)
    fin = await _put_client_finalize(stub_store, bucket, customer, session)
    junk = f"raw/claude_code/{customer}/2026/04/29/{session}:9.json"
    await stub_store.put(bucket, junk, b"not json")
    assert (await _hydrate(customer, session, [fin, junk]))["session_complete"] is False
