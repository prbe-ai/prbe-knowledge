"""Unit tests for the polling document sink.

The sink wraps each polled document in the same envelope shape the
webhook-handler HTTP path builds (so the downstream normalizer sees
identical inputs whether the doc arrived via webhook, backfill, or this
sink), uploads it to the customer's R2 bucket, and INSERTs an
ingestion_queue row via the existing _enqueue helper.

These tests stub the ObjectStore + the asyncpg pool so we can verify
shape + dispatch without standing up live R2 or Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import orjson
import pytest

from services.ingestion.polling.sink import (
    PollDocumentSink,
    _resolve_received_at,
    _resolve_source_event_id,
)
from shared.constants import SourceSystem


class _FakeStore:
    """In-memory ObjectStore stub. Records every put so tests can assert
    on the envelope bytes + key shape."""

    def __init__(self) -> None:
        self.puts: list[tuple[str, str, bytes]] = []
        self.buckets_ensured: list[str] = []

    async def bucket_for(self, customer_id: str) -> str:
        return f"probe-{customer_id}"

    async def ensure_bucket(self, bucket: str) -> None:
        self.buckets_ensured.append(bucket)

    async def put(self, bucket: str, key: str, body: bytes) -> None:
        self.puts.append((bucket, key, body))


def test_resolve_source_event_id_prefers_explicit() -> None:
    """When the poller already set source_event_id (the GitHub poller
    does — see services/ingestion/polling/github.py), the sink uses it
    verbatim. Other pollers (Slack/Linear/Notion/Sentry) emit just the
    raw payload and rely on the deterministic fingerprint below."""
    doc = {"source_event_id": "issue:acme/repo:42:opened:2026-05-15T00:00:00Z"}
    assert _resolve_source_event_id(SourceSystem.GITHUB, doc) == doc["source_event_id"]


def test_resolve_source_event_id_falls_back_to_deterministic_fingerprint() -> None:
    """Re-polling the same upstream row must collapse onto the same
    ingestion_queue row (the queue's UNIQUE (customer_id, source_system,
    source_event_id) constraint dedupes). We fingerprint the canonical
    JSON serialization so semantically-identical docs produce the same
    id even if dict-key order differs."""
    doc_a = {"type": "Issue", "data": {"id": "abc"}, "_origin": "poll"}
    doc_b = {"_origin": "poll", "data": {"id": "abc"}, "type": "Issue"}
    id_a = _resolve_source_event_id(SourceSystem.LINEAR, doc_a)
    id_b = _resolve_source_event_id(SourceSystem.LINEAR, doc_b)
    assert id_a == id_b
    assert id_a.startswith("linear:poll:")
    # Different payloads -> different ids.
    other = _resolve_source_event_id(SourceSystem.LINEAR, {"data": {"id": "xyz"}})
    assert other != id_a


def test_resolve_received_at_prefers_upstream_timestamps() -> None:
    """The normalizer's valid_from chain depends on the envelope's
    received_at. When the poller surfaces an upstream timestamp we use
    it so the doc lands in the timeline at the right place; falling
    back to now() only when nothing's available."""
    assert _resolve_received_at({"received_at": "2026-05-15T00:00:00Z"}) == "2026-05-15T00:00:00Z"
    assert _resolve_received_at({"updated_at": "2026-05-14T12:00:00Z"}) == "2026-05-14T12:00:00Z"
    assert _resolve_received_at({"last_edited_time": "2026-05-13T08:00:00Z"}) == "2026-05-13T08:00:00Z"
    assert _resolve_received_at({"dateCreated": "2026-05-12T06:00:00Z"}) == "2026-05-12T06:00:00Z"
    assert _resolve_received_at({"ts": "1700000000.123456"}) == "1700000000.123456"


def test_resolve_received_at_falls_back_to_now() -> None:
    before = datetime.now(UTC)
    result = _resolve_received_at({"unrelated": "field"})
    after = datetime.now(UTC)
    parsed = datetime.fromisoformat(result)
    assert before <= parsed <= after


@pytest.mark.asyncio
async def test_sink_uploads_envelope_and_enqueues(monkeypatch) -> None:
    """End-to-end happy path: one document in, one R2 put + one
    _enqueue call. Envelope shape mirrors the webhook handler so the
    normalizer can't tell poll-sourced from webhook-sourced apart."""
    store = _FakeStore()
    sink = PollDocumentSink(store=store)

    enqueue_calls: list[dict[str, Any]] = []

    async def _fake_enqueue(*, customer_id, source, source_event_id, payload_s3_key):
        enqueue_calls.append(
            {
                "customer_id": customer_id,
                "source": source,
                "source_event_id": source_event_id,
                "payload_s3_key": payload_s3_key,
            }
        )
        return True

    monkeypatch.setattr("services.ingestion.polling.sink._enqueue", _fake_enqueue)

    doc = {
        "source_event_id": "pr:acme/repo:7:opened:2026-05-15T00:00:00Z",
        "received_at": "2026-05-15T00:00:00Z",
        "raw_payload": {"action": "opened", "pull_request": {"number": 7}},
    }
    await sink("cust-xyz", SourceSystem.GITHUB, [doc])

    # R2 side: one put, on the customer's bucket, key shape matches the
    # webhook-handler's `_payload_key(source, customer_id, source_event_id)`.
    assert store.buckets_ensured == ["probe-cust-xyz"]
    assert len(store.puts) == 1
    bucket, key, body = store.puts[0]
    assert bucket == "probe-cust-xyz"
    assert key.startswith("raw/github/cust-xyz/")
    assert key.endswith(".json")

    envelope = orjson.loads(body)
    assert envelope["_headers"] == {}
    assert envelope["payload"] == doc
    assert envelope["received_at"] == "2026-05-15T00:00:00Z"
    assert envelope["trace_id"].startswith("poll-")

    # Queue side: one _enqueue call with the matched args.
    assert len(enqueue_calls) == 1
    call = enqueue_calls[0]
    assert call["customer_id"] == "cust-xyz"
    assert call["source"] == SourceSystem.GITHUB
    assert call["source_event_id"] == doc["source_event_id"]
    assert call["payload_s3_key"] == key


@pytest.mark.asyncio
async def test_sink_skips_dedupe_when_doc_lacks_explicit_id(monkeypatch) -> None:
    """Slack/Linear/Notion/Sentry pollers emit raw payloads without
    source_event_id at the top level — the sink fingerprints them
    deterministically. Re-running the same doc through the sink hits
    the same queue row (the _enqueue ON CONFLICT DO NOTHING dedupe
    handles it), verified here by id stability."""
    sink = PollDocumentSink(store=_FakeStore())
    enqueue_ids: list[str] = []

    async def _fake_enqueue(*, customer_id, source, source_event_id, payload_s3_key):
        enqueue_ids.append(source_event_id)
        return True

    monkeypatch.setattr("services.ingestion.polling.sink._enqueue", _fake_enqueue)

    doc = {"type": "Issue", "data": {"id": "lin-1"}}
    await sink("cust-xyz", SourceSystem.LINEAR, [doc])
    await sink("cust-xyz", SourceSystem.LINEAR, [doc])
    assert enqueue_ids[0] == enqueue_ids[1]
    assert enqueue_ids[0].startswith("linear:poll:")


@pytest.mark.asyncio
async def test_sink_continues_on_r2_failure(monkeypatch) -> None:
    """One bad R2 put must not abort the whole batch. The scheduler
    contract is that a polled batch flushes as much as it can; per-doc
    failures get logged and skipped, the cursor still advances."""

    class _FlakyStore(_FakeStore):
        def __init__(self) -> None:
            super().__init__()
            self._calls = 0

        async def put(self, bucket: str, key: str, body: bytes) -> None:
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("R2 transient")
            await super().put(bucket, key, body)

    store = _FlakyStore()
    sink = PollDocumentSink(store=store)
    accepted: list[str] = []

    async def _fake_enqueue(*, customer_id, source, source_event_id, payload_s3_key):
        accepted.append(source_event_id)
        return True

    monkeypatch.setattr("services.ingestion.polling.sink._enqueue", _fake_enqueue)

    docs = [
        {"source_event_id": "evt-1", "raw_payload": {"x": 1}},
        {"source_event_id": "evt-2", "raw_payload": {"x": 2}},
    ]
    await sink("cust-xyz", SourceSystem.GITHUB, docs)
    # First doc skipped (R2 failure); second doc made it through.
    assert accepted == ["evt-2"]
    assert len(store.puts) == 1


@pytest.mark.asyncio
async def test_sink_continues_on_enqueue_failure(monkeypatch) -> None:
    """Symmetric to R2 failure: an _enqueue raise on one doc shouldn't
    blow up the rest of the batch."""
    sink = PollDocumentSink(store=_FakeStore())
    accepted: list[str] = []

    async def _fake_enqueue(*, customer_id, source, source_event_id, payload_s3_key):
        if source_event_id == "evt-1":
            raise RuntimeError("queue transient")
        accepted.append(source_event_id)
        return True

    monkeypatch.setattr("services.ingestion.polling.sink._enqueue", _fake_enqueue)

    docs = [
        {"source_event_id": "evt-1", "raw_payload": {"x": 1}},
        {"source_event_id": "evt-2", "raw_payload": {"x": 2}},
    ]
    await sink("cust-xyz", SourceSystem.GITHUB, docs)
    assert accepted == ["evt-2"]


@pytest.mark.asyncio
async def test_sink_empty_batch_is_noop(monkeypatch) -> None:
    sink = PollDocumentSink(store=_FakeStore())
    called = False

    async def _fake_enqueue(**_):
        nonlocal called
        called = True
        return True

    monkeypatch.setattr("services.ingestion.polling.sink._enqueue", _fake_enqueue)
    await sink("cust-xyz", SourceSystem.GITHUB, [])
    assert called is False
