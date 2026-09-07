"""Protocol transaction/race tests on a dedicated, local Postgres database.

Set PRBE_RECEIPT_TEST_DATABASE_URL to an isolated database named
session_receipts_test. This fixture never truncates a shared engine database.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest
import pytest_asyncio
from fastapi import HTTPException

from engine.ingest.handlers.base import make_default_context
from engine.shared.constants import SourceSystem
from engine.shared.models import WebhookEvent
from kb import session_receipts as sr
from kb.handlers.claude_code import ClaudeCodeConnector


class Store:
    def __init__(self):
        self.blobs = {}
        self.writes = 0
        self.fail = False

    async def bucket_for(self, customer):
        return customer

    async def ensure_bucket(self, bucket):
        pass

    async def put(self, bucket, key, body):
        self.writes += 1
        self.blobs[bucket, key] = body
        if self.fail:
            raise RuntimeError("crash after blob before transaction commit")

    async def get(self, bucket, key):
        return self.blobs[bucket, key]


@pytest_asyncio.fixture
async def database(monkeypatch):
    dsn = os.environ.get("PRBE_RECEIPT_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("requires a dedicated local session_receipts_test Postgres database")
    from urllib.parse import urlsplit

    parsed = urlsplit(dsn)
    assert parsed.hostname in {"localhost", "127.0.0.1"} and parsed.path == "/session_receipts_test"
    admin = await asyncpg.connect(dsn)
    await admin.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
    await admin.execute("""
        CREATE TABLE customers(customer_id TEXT PRIMARY KEY);
        INSERT INTO customers VALUES('tenant-a'),('tenant-b');
        CREATE TABLE documents(customer_id TEXT,doc_id TEXT);
        CREATE TABLE ingestion_queue(customer_id TEXT,source_system TEXT,source_event_id TEXT,
            payload_s3_key TEXT,payload_s3_keys TEXT[],status TEXT,priority INTEGER,version INTEGER,
            enqueued_at TIMESTAMPTZ,completed_at TIMESTAMPTZ,error TEXT,
            UNIQUE(customer_id,source_system,source_event_id));
        DO $$ BEGIN CREATE ROLE receipt_app NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
    """)
    await admin.execute((Path(__file__).parents[1] / "kb/session_receipts_schema.sql").read_text())
    await admin.execute(
        "GRANT USAGE ON SCHEMA public TO receipt_app; GRANT ALL ON ALL TABLES IN SCHEMA public TO receipt_app"
    )
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4)

    @asynccontextmanager
    async def tenant(customer):
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute("SET LOCAL ROLE receipt_app")
            await conn.execute("SELECT set_config('app.current_customer_id',$1,true)", customer)
            yield conn

    async def connected(*args):
        return True

    monkeypatch.setattr(sr, "with_tenant", tenant)
    monkeypatch.setattr(sr, "is_source_connected", connected)
    yield tenant, admin
    await pool.close()
    await admin.close()


def batch(**changes):
    body = dict(
        protocol_version=2,
        session_id=str(uuid4()),
        stream_id=str(uuid4()),
        batch_seq=0,
        source_byte_start=0,
        source_byte_end=30,
        source_line_start=0,
        source_line_end=3,
        event_start=0,
        event_end=2,
        prefix_sha256=hashlib.sha256(b"source").hexdigest(),
        cwd="/synthetic",
        events=[
            {
                "line_no": i,
                "raw": {"type": "user", "message": {"role": "user", "content": f"turn {i}"}},
            }
            for i in range(2)
        ],
    )
    body.update(changes)
    return body


@pytest.mark.asyncio
async def test_same_key_retry_race_has_one_blob_receipt_and_queue_reference(database):
    tenant, admin = database
    store = Store()
    body = batch()
    first, second = await asyncio.gather(
        *(sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store) for _ in range(2))
    )
    assert {first["status"], second["status"]} == {"accepted", "duplicate"}
    assert first["receipt"] == second["receipt"]
    assert store.writes == 1
    assert await admin.fetchval("SELECT count(*) FROM session_batch_receipts") == 1
    assert await admin.fetchval("SELECT cardinality(payload_s3_keys) FROM ingestion_queue") == 1
    async with tenant("tenant-b") as conn:
        assert await conn.fetchval("SELECT count(*) FROM session_batch_receipts") == 0
    read = await sr.receipts("claude_code", body["session_id"], "tenant-a", -1, 200)
    assert read["stream"]["event_end"] == 2 and read["receipts"][0] == first["receipt"]


@pytest.mark.asyncio
async def test_conflict_precedes_blob_write_and_legacy_is_fenced(database):
    tenant, _admin = database
    store = Store()
    body = batch()
    await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    altered = dict(
        body,
        events=[
            {"line_no": i, "raw": {"type": "user", "message": {"content": "altered"}}}
            for i in range(2)
        ],
    )
    with pytest.raises(HTTPException) as error:
        await sr.accept(altered, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert error.value.status_code == 409 and store.writes == 1
    async with tenant("tenant-a") as conn:
        with pytest.raises(HTTPException) as error:
            await sr.reject_legacy_writer(conn, "tenant-a", "claude_code", body["session_id"])
        assert error.value.status_code == 409
    # Source and tenant are independent namespaces, not global session UUID dedup.
    await sr.accept(body, "tenant-b", SourceSystem.CLAUDE_CODE, store)
    await sr.accept(body, "tenant-a", SourceSystem.PI, store)
    assert store.writes == 3


@pytest.mark.asyncio
async def test_legacy_coverage_cannot_be_blindly_replayed(database):
    _tenant, admin = database
    body = batch()
    store = Store()
    await admin.execute(
        "INSERT INTO documents VALUES('tenant-a',$1)", f"claude_code:tenant-a:{body['session_id']}"
    )
    read = await sr.receipts("claude_code", body["session_id"], "tenant-a", -1, 200)
    assert read["state"] == "legacy"
    with pytest.raises(HTTPException) as error:
        await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert error.value.status_code == 409 and store.writes == 0


@pytest.mark.asyncio
async def test_blob_success_transaction_failure_retries_without_false_receipt(database):
    _tenant, admin = database
    body = batch()
    store = Store()
    store.fail = True
    with pytest.raises(RuntimeError):
        await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert await admin.fetchval("SELECT count(*) FROM session_batch_receipts") == 0
    assert await admin.fetchval("SELECT count(*) FROM ingestion_queue") == 0
    store.fail = False
    await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert len(store.blobs) == 1


@pytest.mark.asyncio
async def test_finalize_pins_accepted_stream_and_retained_event_cursor(database, monkeypatch):
    _tenant, admin = database
    store = Store()
    body = batch()
    await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    final = {k: v for k, v in body.items() if k not in ("events", "cwd")}
    final.update(
        finalize=True, batch_seq=1, source_byte_start=30, source_line_start=3, event_start=2
    )
    for changes in (
        {"source_byte_end": 31},
        {"prefix_sha256": "f" * 64},
        {"stream_id": str(uuid4())},
    ):
        with pytest.raises(HTTPException):
            await sr.accept(dict(final, **changes), "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert store.writes == 1
    accepted = await sr.accept(final, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert accepted["receipt"]["finalized"] is True
    # Exercise the production merge consumer using accepted queue keys, in reverse order.
    from kb.handlers import claude_code

    monkeypatch.setattr(claude_code, "get_store", lambda: store)
    keys = await admin.fetchval("SELECT payload_s3_keys FROM ingestion_queue")
    event = WebhookEvent(
        customer_id="tenant-a",
        source_system=SourceSystem.CLAUDE_CODE,
        source_event_id=body["session_id"],
        received_at=datetime.now(UTC),
        payload_s3_key=keys[0],
        payload_s3_keys=list(reversed(keys)),
        raw_payload=body,
        headers={},
    )
    merged = await ClaudeCodeConnector(make_default_context()).fetch_supplementary(
        event, token=None
    )
    assert [e["raw"]["message"]["content"] for e in merged["events"]] == ["turn 0", "turn 1"]
    assert merged["session_complete"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"event_end": 3},
        {"events": [{"line_no": 0}, {"line_no": 0}]},
        {"batch_seq": True},
        {"source_line_end": -1},
        {"protocol_version": 1},
    ],
)
def test_invalid_intervals_and_codec_never_reach_storage(changes):
    with pytest.raises(HTTPException) as error:
        sr.validate_payload(batch(**changes))
    assert error.value.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["claude_code", "codex", "pi"])
async def test_real_journal_batches_survive_acceptance_and_consumer_order(
    database, monkeypatch, source
):
    """These synthetic fixtures were emitted by the actual shared CLI/tap journal.

    pi's 12 bashExecution source messages expand into 24 retained events, plus
    its header. Source line numbering would silently lose 12 of those events.
    """
    import json

    _tenant, admin = database
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/session_protocol_v2" / f"{source}.json").read_text()
    )
    store = Store()
    for body in fixture["batches"]:
        await sr.accept(body, "tenant-a", SourceSystem(source), store)
        # At-least-once transport must not duplicate queue references.
        await sr.accept(body, "tenant-a", SourceSystem(source), store)
    keys = await admin.fetchval("SELECT payload_s3_keys FROM ingestion_queue")
    assert len(keys) == len(fixture["batches"])
    from kb.handlers import claude_code

    monkeypatch.setattr(claude_code, "get_store", lambda: store)
    first = fixture["batches"][0]
    event = WebhookEvent(
        customer_id="tenant-a",
        source_system=SourceSystem(source),
        source_event_id=first["session_id"],
        received_at=datetime.now(UTC),
        payload_s3_key=keys[0],
        payload_s3_keys=list(reversed(keys)),
        raw_payload=first,
        headers={},
    )
    merged = await ClaudeCodeConnector(make_default_context()).fetch_supplementary(
        event, token=None
    )
    assert len(merged["events"]) == fixture["expected_events"]
    assert [event["line_no"] for event in merged["events"]] == list(
        range(fixture["expected_events"])
    )
    assert merged["events"] == [
        event for body in fixture["batches"] for event in body.get("events", [])
    ]
    assert merged["session_complete"]


@pytest.mark.asyncio
async def test_declared_historical_snapshot_cannot_finalize_an_accepted_prefix(database):
    _tenant, _admin = database
    store = Store()
    body = batch(
        snapshot_byte_end=60, snapshot_sha256=hashlib.sha256(b"whole snapshot").hexdigest()
    )
    await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    final = {k: v for k, v in body.items() if k not in ("events", "cwd")}
    final.update(
        finalize=True, batch_seq=1, source_byte_start=30, source_line_start=3, event_start=2
    )
    with pytest.raises(HTTPException) as error:
        await sr.accept(final, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert error.value.status_code == 409 and "not fully accepted" in error.value.detail
    del final["snapshot_byte_end"]
    del final["snapshot_sha256"]
    with pytest.raises(HTTPException) as error:
        await sr.accept(final, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert error.value.status_code == 409 and "changed before completion" in error.value.detail
    assert store.writes == 1
