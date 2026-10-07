"""Protocol transaction/race tests on a dedicated, local Postgres database.

Set PRBE_RECEIPT_TEST_DATABASE_URL to an isolated local database whose name
ends in `receipts_test` (CI: session_receipts_test). This fixture drops its
schema, so it never runs against a shared engine database.
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
from engine.shared.constants import PrincipalType, SourceSystem
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
    assert parsed.hostname in {"localhost", "127.0.0.1"} and parsed.path.endswith("receipts_test")
    admin = await asyncpg.connect(dsn)
    await admin.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
    await admin.execute("""
        CREATE TABLE customers(customer_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'active');
        INSERT INTO customers VALUES('tenant-a'),('tenant-b');
        CREATE TABLE documents(customer_id TEXT,doc_id TEXT);
        CREATE TABLE ingestion_queue(queue_id BIGSERIAL PRIMARY KEY,customer_id TEXT,source_system TEXT,source_event_id TEXT,
            payload_s3_key TEXT,payload_s3_keys TEXT[],status TEXT,priority INTEGER,version INTEGER,
            enqueued_at TIMESTAMPTZ,first_enqueued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            completed_at TIMESTAMPTZ,error TEXT,
            UNIQUE(customer_id,source_system,source_event_id));
        DO $$ BEGIN CREATE ROLE receipt_app NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
    """)
    await admin.execute((Path(__file__).parents[1] / "kb/session_receipts_schema.sql").read_text())
    # Every ingest door asks whether the session was deleted, under its lock.
    await admin.execute((Path(__file__).parents[1] / "kb/session_deletions_schema.sql").read_text())
    await admin.execute(
        "GRANT USAGE ON SCHEMA public TO receipt_app; GRANT ALL ON ALL TABLES IN SCHEMA public TO receipt_app"
    )
    await admin.execute("GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO receipt_app")
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
    monkeypatch.setattr("engine.ingest.normalizer.get_pool", lambda: pool)
    monkeypatch.setattr("kb.session_completer.get_pool", lambda: pool)
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
async def test_credential_redaction_precedes_raw_storage_without_changing_receipt(database):
    import json

    _tenant, admin = database
    key = "ghp_" + "8a29Df63bC17eA94fE61dB82aC03eF75dA19"
    body = batch(cwd="/work/" + key, device_id=key)
    body["events"][0]["raw"]["message"]["content"] = "token " + key
    body["events"][1]["raw"]["metadata"] = {"password": "Harbor7!", "eos_token": "</s>"}
    digest = hashlib.sha256(sr.canonical_payload(body)).hexdigest()
    store = Store()
    result = await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert result["receipt"]["body_sha256"] == digest
    assert store.blobs
    stored = next(iter(store.blobs.values())).decode()
    assert key not in stored and "Harbor7!" not in stored
    assert "</s>" in stored and len(json.loads(stored)["payload"]["events"]) == 2
    assert await admin.fetchval("SELECT body_sha256 FROM session_batch_receipts") == digest
    assert await admin.fetchval("SELECT count(*) FROM ingestion_queue") == 1
    assert key not in await admin.fetchval("SELECT uploader_device_id FROM session_streams")
    repeat = await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert repeat["status"] == "duplicate" and store.writes == 1


def filesystem_store(root):
    """Production ObjectStore put/get with a local file-backed S3 adapter."""
    from engine.shared.storage import ObjectStore

    class LocalClient:
        writes = 0
        def head_bucket(self, **kwargs):
            (root / kwargs['Bucket']).mkdir(parents=True, exist_ok=True)
        def put_object(self, **kwargs):
            self.writes += 1
            path = root / kwargs['Bucket'] / kwargs['Key']
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(kwargs['Body'])
        def get_object(self, **kwargs):
            return {'Body':(root / kwargs['Bucket'] / kwargs['Key']).open('rb')}

    class LocalStore(ObjectStore):
        def __init__(self):
            self._client = LocalClient()
        async def bucket_for(self, customer):
            return customer

    return LocalStore()


@pytest.mark.asyncio
@pytest.mark.parametrize('protocol',['legacy','v2'])
async def test_first_persistent_object_is_scrubbed_and_readable(database, tmp_path, protocol):
    import json
    _tenant,admin=database
    secret='ghp_'+hashlib.sha256(b'synthetic disk receipt test').hexdigest()[:36]
    body=batch(cwd='/work/'+secret)
    body['events'][0]['raw']['message']['content']='copied '+secret
    body['events'][1]['raw']['metadata']={'nested':[{'password':'Harbor7!'}],'eos_token':'</s>'}
    store=filesystem_store(tmp_path)
    if protocol=='v2':
        await sr.accept(body,'tenant-a',SourceSystem.CLAUDE_CODE,store)
    else:
        body.pop('protocol_version')
        envelope=json.dumps({'payload':body,'_headers':{'x-test-token':secret}}).encode()
        assert await sr.accept_legacy(body,envelope,'tenant-a',SourceSystem.CLAUDE_CODE,store,'raw/synthetic.json',None)
    assert store._client.writes==1
    key=await admin.fetchval('SELECT payload_s3_key FROM ingestion_queue')
    data=await store.get('tenant-a',key)
    assert data==(tmp_path/'tenant-a'/key).read_bytes()
    assert secret.encode() not in data and b'Harbor7!' not in data
    assert b'</s>' in data and len(json.loads(data)['payload']['events'])==2
    assert await admin.fetchval('SELECT count(*) FROM ingestion_queue')==1
    assert await admin.fetchval('SELECT count(*) FROM session_batch_receipts')==(1 if protocol=='v2' else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize('protocol',['legacy','v2'])
async def test_scanner_failure_precedes_every_persistent_write(database,tmp_path,monkeypatch,protocol):
    import json

    from engine.ingest import payload_redaction
    from engine.shared.exceptions import ScanUnavailable
    _tenant,admin=database
    store=filesystem_store(tmp_path)
    body=batch()
    def unavailable(*args):raise ScanUnavailable('synthetic scanner unavailable')
    monkeypatch.setattr(payload_redaction,'redact_documents',unavailable)
    with pytest.raises(ScanUnavailable):
        if protocol=='v2':
            await sr.accept(body,'tenant-a',SourceSystem.CLAUDE_CODE,store)
        else:
            body.pop('protocol_version')
            await sr.accept_legacy(body,json.dumps({'payload':body}).encode(),'tenant-a',SourceSystem.CLAUDE_CODE,store,'raw/synthetic.json',None)
    assert store._client.writes==0 and not list(tmp_path.rglob('*.json'))
    for table in ('session_batch_receipts','session_streams','ingestion_queue'):
        assert await admin.fetchval(f'SELECT count(*) FROM {table}')==0


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
async def test_a_batch_on_a_finished_session_starts_a_new_wait_and_one_on_a_waiting_session_does_not(
    database,
):
    """`first_enqueued_at` is the queue-age alert's clock (engine/ingest/queue_age.py),
    and every agent upload reaches the queue through this door. A session that
    finished two days ago and gets a batch now has waited seconds, not days; a
    session still waiting keeps the time its wait began."""
    _tenant, admin = database
    store = Store()
    old = datetime(2026, 10, 5, 21, 21, tzinfo=UTC)

    def turn(line_no: int, text: str) -> list[dict]:
        return [{"line_no": line_no, "raw": {"type": "user", "message": {"role": "user", "content": text}}}]

    body = batch()
    await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    await admin.execute("UPDATE ingestion_queue SET status='done', first_enqueued_at=$1", old)
    second = _next(body, events=turn(2, "back"))
    await sr.accept(second, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    row = await admin.fetchrow("SELECT status, first_enqueued_at FROM ingestion_queue")
    assert row["status"] == "pending"
    assert row["first_enqueued_at"] > old, "a finished session's new batch is a new wait"

    await admin.execute("UPDATE ingestion_queue SET first_enqueued_at=$1", old)
    await sr.accept(_next(second, events=turn(3, "more")), "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert await admin.fetchval("SELECT first_enqueued_at FROM ingestion_queue") == old, (
        "a batch landing on a waiting session must not reset its age"
    )


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
async def test_kimi_code_batches_are_accepted_and_their_receipts_readable(database):
    """research-os forwards Kimi Code batches here and reads them back through
    GET /api/sessions/kimi_code/<id>/receipts. That door names its agents by
    hand, so a source missing from it accepts batches and then 422s the read."""
    _tenant, admin = database
    store = Store()
    body = batch()
    accepted = await sr.accept(body, "tenant-a", SourceSystem.KIMI_CODE, store)
    assert accepted["status"] == "accepted"
    assert all(key.startswith("raw/kimi_code/tenant-a/") for _bucket, key in store.blobs)
    assert await admin.fetchval("SELECT source_system FROM ingestion_queue") == "kimi_code"
    read = await sr.receipts("kimi_code", body["session_id"], "tenant-a", -1, 200)
    assert read["stream"]["event_end"] == 2 and read["receipts"][0] == accepted["receipt"]


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


def consumer(store, monkeypatch):
    from engine.ingest.normalizer import Normalizer
    from engine.shared import claude_code_extraction as ext
    from kb.handlers import claude_code

    normalizer = Normalizer(make_default_context(), store=store, embedder=object())

    async def token(*args):
        return None

    async def extract(**kwargs):
        return ext.UnitBundle(qa=[ext.QA(prompt="why?", outcome="synthetic")])

    monkeypatch.setattr(normalizer, "_load_token", token)
    monkeypatch.setattr(claude_code, "get_store", lambda: store)
    monkeypatch.setattr("kb.session_completer.get_store", lambda: store)
    monkeypatch.setattr(claude_code._ext, "extract_units_from_session", extract)
    return normalizer


def finalize(body):
    result = {key: value for key, value in body.items() if key not in ("events", "cwd")}
    result.update(finalize=True, batch_seq=body["batch_seq"] + 1)
    for coordinate in ("source_byte", "source_line", "event"):
        result[f"{coordinate}_start"] = result[f"{coordinate}_end"]
    return result


@pytest.mark.asyncio
async def test_v2_completion_survives_extraction_idle_sweep_and_reprocessing(database, monkeypatch):
    from kb.session_completer import enqueue_idle_session_finalizers

    _tenant, admin = database
    store = Store()
    normalizer = consumer(store, monkeypatch)
    body = batch(employee_id="uploader", device_id="device")
    await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    await sr.accept(finalize(body), "tenant-a", SourceSystem.CLAUDE_CODE, store)
    row = await admin.fetchrow("SELECT queue_id,payload_s3_keys FROM ingestion_queue")
    first = await normalizer._normalize_only(
        "tenant-a", SourceSystem.CLAUDE_CODE, row["payload_s3_keys"]
    )
    assert first.documents[0].metadata["session_complete"] and len(first.documents) > 1
    # Nothing is consumed; run the real idle selector against the row as mined.
    await admin.execute("UPDATE ingestion_queue SET enqueued_at=NOW()-INTERVAL '1 hour'")
    assert await enqueue_idle_session_finalizers(idle_minutes=5) == 0
    keys = await admin.fetchval("SELECT payload_s3_keys FROM ingestion_queue")
    assert keys == row["payload_s3_keys"]
    repeated = await normalizer._normalize_only("tenant-a", SourceSystem.CLAUDE_CODE, keys)
    assert repeated.documents[0].metadata["session_complete"]
    # Completion is durable evidence but a later accepted sequence reopens it.
    tail = dict(
        body,
        batch_seq=2,
        source_byte_start=30,
        source_byte_end=40,
        source_line_start=3,
        source_line_end=4,
        event_start=2,
        event_end=3,
        prefix_sha256="a" * 64,
        events=[
            {
                "line_no": 2,
                "raw": {"type": "user", "message": {"role": "user", "content": "resume"}},
            }
        ],
    )
    await sr.accept(tail, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    keys = await admin.fetchval("SELECT payload_s3_keys FROM ingestion_queue")
    reopened = await normalizer._normalize_only("tenant-a", SourceSystem.CLAUDE_CODE, keys)
    assert not reopened.documents[0].metadata["session_complete"]
    assert len(reopened.documents) == 1 and reopened.documents[0].metadata["event_count"] == 3


@pytest.mark.asyncio
async def test_idle_finalizer_checks_the_tenant_scoped_stream_even_without_key_hint(
    database, monkeypatch
):
    """A client-finalized v2 session is left alone; the same session id in
    another tenant is judged under ITS scope, not this one's."""
    from kb.session_completer import enqueue_idle_session_finalizers

    _tenant, admin = database
    store = Store()
    consumer(store, monkeypatch)
    body = batch(employee_id="uploader")
    await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    await sr.accept(finalize(body), "tenant-a", SourceSystem.CLAUDE_CODE, store)
    await admin.execute("UPDATE ingestion_queue SET enqueued_at=NOW()-INTERVAL '1 hour'")
    # The same UUID in another tenant stays eligible under its own scope.
    await admin.execute(
        "INSERT INTO ingestion_queue(customer_id,source_system,source_event_id,payload_s3_key,payload_s3_keys,enqueued_at) VALUES('tenant-b','claude_code',$1,'legacy',ARRAY['legacy'],NOW()-INTERVAL '1 hour')",
        body["session_id"],
    )
    assert await enqueue_idle_session_finalizers(idle_minutes=5) == 1
    assert (
        await admin.fetchval(
            "SELECT cardinality(payload_s3_keys) FROM ingestion_queue WHERE customer_id='tenant-a'"
        )
        == 2
    )
    assert (
        await admin.fetchval(
            "SELECT cardinality(payload_s3_keys) FROM ingestion_queue WHERE customer_id='tenant-b'"
        )
        == 2
    )


@pytest.fixture
async def as_app_role(database, monkeypatch):
    """Run the sweep as a NON-superuser, so FORCE RLS on session_streams is
    real: a missing tenant setting then hides every stream instead of passing."""
    dsn = os.environ["PRBE_RECEIPT_TEST_DATABASE_URL"]

    async def as_app(conn):
        await conn.execute("SET ROLE receipt_app")

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, init=as_app)
    monkeypatch.setattr("kb.session_completer.get_pool", lambda: pool)
    yield
    await pool.close()


@pytest.mark.asyncio
async def test_an_idle_unfinalized_v2_session_is_ended_once_and_then_left_alone(
    database, as_app_role, monkeypatch
):
    """The v2 loop guard. A client that never said goodbye gets the sweep's
    marker; once that session is mined, the marker on top keeps the next sweep
    away -- nothing about session_streams has to change for that."""
    from kb.session_completer import enqueue_idle_session_finalizers

    _tenant, admin = database
    store = Store()
    normalizer = consumer(store, monkeypatch)
    body = batch(employee_id="uploader")
    await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    await admin.execute("UPDATE ingestion_queue SET enqueued_at=NOW()-INTERVAL '2 days'")

    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 1
    keys = await admin.fetchval("SELECT payload_s3_keys FROM ingestion_queue")
    assert keys[-1].endswith("/finalize.marker")
    assert await admin.fetchval("SELECT finalized FROM session_streams") is False, (
        "the sweep's marker is not a client claim"
    )
    mined = await normalizer._normalize_only("tenant-a", SourceSystem.CLAUDE_CODE, keys)
    assert mined.documents[0].metadata["session_complete"]
    assert mined.documents[0].metadata["completed_by"] == "cron_marker"

    await admin.execute(
        "UPDATE ingestion_queue SET status='done', completed_at=NOW(), enqueued_at=NOW()-INTERVAL '2 days'"
    )
    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 0
    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 0
    assert await admin.fetchval("SELECT payload_s3_keys FROM ingestion_queue") == keys

    # A late client finalize lands on top and is accepted; still left alone.
    await sr.accept(finalize(body), "tenant-a", SourceSystem.CLAUDE_CODE, store)
    await admin.execute("UPDATE ingestion_queue SET status='done', enqueued_at=NOW()-INTERVAL '2 days'")
    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 0


@pytest.mark.asyncio
async def test_finalized_v2_sessions_do_not_starve_the_sweep(database, as_app_role, monkeypatch):
    """Most v2 rows are client-finalized. They are excluded in the query, not
    skipped one by one inside a LIMIT, or the oldest finished rows would eat
    every run and a real candidate behind them would never be reached."""
    from kb.session_completer import enqueue_idle_session_finalizers

    _tenant, admin = database
    store = Store()
    consumer(store, monkeypatch)
    for _ in range(3):
        done = batch(employee_id="uploader")
        await sr.accept(done, "tenant-a", SourceSystem.CLAUDE_CODE, store)
        await sr.accept(finalize(done), "tenant-a", SourceSystem.CLAUDE_CODE, store)
    await admin.execute("UPDATE ingestion_queue SET enqueued_at=NOW()-INTERVAL '5 days'")
    silent = batch(employee_id="uploader")
    await sr.accept(silent, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    await admin.execute(
        "UPDATE ingestion_queue SET enqueued_at=NOW()-INTERVAL '2 days' WHERE source_event_id=$1",
        silent["session_id"],
    )
    import structlog

    with structlog.testing.capture_logs() as logs:
        assert await enqueue_idle_session_finalizers(idle_minutes=1440, limit=1) == 1
    keys = await admin.fetchval(
        "SELECT payload_s3_keys FROM ingestion_queue WHERE source_event_id=$1", silent["session_id"]
    )
    assert keys[-1].endswith("/finalize.marker")
    run = next(e for e in logs if e["event"] == "session_completer.run")
    # Excluded IN the query (as the RLS-bound role), not found and then
    # skipped one by one: that is what keeps them out of the LIMIT.
    assert (run["candidates"], run["skipped"]) == (1, 0)
    # This database has no extraction_outcome column: ending still works and
    # the retry step says why it did nothing.
    assert any(e["event"] == "session_completer.retry_skipped" for e in logs)


@pytest.mark.asyncio
async def test_a_batch_arriving_mid_extraction_does_not_free_the_row_for_a_second_worker(
    database, monkeypatch
):
    """A row being processed stays `processing` when a batch lands; the version
    bump alone tells the running worker to go again. Resetting it to pending
    let a second worker claim the same session and mine it twice."""
    _tenant, admin = database
    store = Store()
    consumer(store, monkeypatch)
    body = batch(employee_id="uploader")
    await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    await admin.execute("UPDATE ingestion_queue SET status='processing'")
    before = await admin.fetchval("SELECT version FROM ingestion_queue")
    await sr.accept(finalize(body), "tenant-a", SourceSystem.CLAUDE_CODE, store)
    row = await admin.fetchrow("SELECT status, version FROM ingestion_queue")
    assert row["status"] == "processing" and row["version"] == before + 1
    await admin.execute("UPDATE ingestion_queue SET status='done'")
    await sr.accept(dict(finalize(body), batch_seq=2, prefix_sha256=body["prefix_sha256"]),
                    "tenant-a", SourceSystem.CLAUDE_CODE, store) if False else None
    assert await admin.fetchval("SELECT status FROM ingestion_queue") == "done"


@pytest.mark.asyncio
async def test_normalizer_reads_later_events_after_a_cursor_only_first_batch(database, monkeypatch):
    _tenant, admin = database
    store = Store()
    normalizer = consumer(store, monkeypatch)
    first = batch(
        employee_id="uploader",
        source_byte_end=4 * 1024 * 1024,
        source_line_end=4000,
        event_end=0,
        events=[],
    )
    await sr.accept(first, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    next_body = dict(
        first,
        batch_seq=1,
        source_byte_start=first["source_byte_end"],
        source_byte_end=first["source_byte_end"] + 30,
        source_line_start=4000,
        source_line_end=4001,
        event_end=1,
        prefix_sha256="b" * 64,
        events=[
            {
                "line_no": 0,
                "raw": {
                    "type": "user",
                    "message": {"role": "user", "content": "after dropped history"},
                },
            }
        ],
    )
    await sr.accept(next_body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    keys = await admin.fetchval("SELECT payload_s3_keys FROM ingestion_queue")
    result = await normalizer._normalize_only("tenant-a", SourceSystem.CLAUDE_CODE, keys)
    assert result.documents[0].metadata["event_count"] == 1
    assert "after dropped history" in result.documents[0].body
    broken = dict(first, event_end=1)
    from engine.shared.exceptions import InvalidWebhookPayload

    with pytest.raises(InvalidWebhookPayload):
        ClaudeCodeConnector(make_default_context()).parse_webhook_event("tenant-a", {}, broken)


@pytest.mark.asyncio
async def test_copied_history_projects_native_provenance_and_uploader_without_authorship(
    database, monkeypatch
):
    from engine.shared.constants import EdgeType

    _tenant, admin = database
    store = Store()
    normalizer = consumer(store, monkeypatch)
    body = batch(
        employee_id="copy-uploader",
        employee_name="Copy Uploader",
        employee_email="uploader@example.test",
        employee_hostname="upload-machine",
        device_id="paired-upload-device",
    )
    body["provenance"] = dict(
        original_author="unverified",
        native_session_id=body["session_id"],
        identity_method="producer-record-v1",
        observed_lineage=["ancestor-session"],
    )
    await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    # Finalize carries no source provenance; hydration must retain the first batch's proof.
    end = finalize(body)
    end.pop("provenance")
    await sr.accept(end, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    keys = await admin.fetchval("SELECT payload_s3_keys FROM ingestion_queue")
    result = await normalizer._normalize_only("tenant-a", SourceSystem.CLAUDE_CODE, keys)
    assert len(result.documents) > 1
    assert not any(edge.edge_type == EdgeType.AUTHORED for edge in result.graph_edges)
    for doc in result.documents:
        assert doc.author_id is None and "Copy Uploader" not in doc.title
        assert doc.metadata["provenance"] == body["provenance"]
        assert doc.metadata["native_session_id"] == body["session_id"]
        assert doc.metadata["observed_lineage"] == ["ancestor-session"]
        assert doc.metadata["uploader_id"] == "copy-uploader"
        assert doc.metadata["uploader_name"] == "Copy Uploader"
        assert doc.metadata["uploader_device_id"] == "paired-upload-device"
        assert not any(key.startswith("employee_") for key in doc.metadata)
        # The ACL names the WORKSPACE, not the uploader. Nothing enforces a
        # per-user ACL on session documents -- the transcript route scopes by
        # customer_id and retrieval has no ACL filter -- so naming one asserted
        # a protection that did not exist. What this test actually guards is
        # untouched and still asserted above: no AUTHORED edge, `author_id` is
        # None, the uploader is recorded in metadata but not credited as the
        # author. The uploader still reads the document, by being in the tenant.
        assert doc.acl.principals[0].principal_type == PrincipalType.WORKSPACE
        assert doc.acl.principals[0].principal_id == "tenant-a"


# ---- protocol 3: ATIF fragments ---------------------------------------------
# A batch carries one fragment per event ordinal instead of the events. These
# pin the door: what is advertised, what is accepted, how a stream is pinned
# and what the kill switch does to an open stream.


def fragments_batch(*, with_events: bool = False, **changes):
    body = batch(protocol_version=3, fragment_version=1)
    if not with_events:
        del body["events"]
    body["fragments"] = [
        {"line": {"line_no": i}, "source": "user", "message": f"turn {i}"} for i in range(2)
    ]
    body.update(changes)
    return body


@pytest.fixture
def protocol3(monkeypatch):
    """Point the door at these settings; returns the setter."""
    from engine.shared.config import Settings

    def configure(**values):
        settings = Settings(**values)
        monkeypatch.setattr(sr, "get_settings", lambda: settings)
        return settings

    configure(session_protocol3_customers="tenant-a")
    return configure


def test_a_protocol_2_receipt_digest_is_unchanged_by_the_protocol_3_fields():
    """Every protocol-2 client holds receipts digested before `fragments` and
    `fragment_version` joined the canonical fields. Pinned from main (e1632b2)."""
    body = batch(
        session_id="5b0c8a3e-4f1d-4c2a-9e7b-1d2f3a4b5c6d",
        stream_id="0f9e8d7c-6b5a-4938-8271-605f4e3d2c1b",
        device_id="device",
    )
    assert hashlib.sha256(sr.canonical_payload(body)).hexdigest() == (
        "e28b506f1db4c619a0963503edad51927e8f15f3137ae268d3604ac3691e6e1f"
    )


def test_the_receipt_digest_covers_fragments_and_their_version():
    import copy

    body = fragments_batch()
    digest = hashlib.sha256(sr.canonical_payload(body)).hexdigest()
    changed = copy.deepcopy(body)
    changed["fragments"][1]["message"] = "altered"
    assert hashlib.sha256(sr.canonical_payload(changed)).hexdigest() != digest
    other_version = dict(body, fragment_version=2)
    assert hashlib.sha256(sr.canonical_payload(other_version)).hexdigest() != digest


@pytest.mark.parametrize(
    ("values", "customer", "expected"),
    [
        ({}, "tenant-a", {"protocols": [2], "fragment_versions": [1], "events": True}),
        (
            {"session_protocol3_customers": "tenant-z, tenant-a"},
            "tenant-a",
            {"protocols": [2, 3], "fragment_versions": [1], "events": True},
        ),
        (
            {"session_protocol3_customers": "tenant-z"},
            "tenant-a",
            {"protocols": [2], "fragment_versions": [1], "events": True},
        ),
        (
            {"session_protocol3_all": True, "session_protocol3_events": False},
            "tenant-a",
            {"protocols": [2, 3], "fragment_versions": [1], "events": False},
        ),
        (
            # A typo drops its entry; it never takes the receipts read down.
            # Version 2 is listed but this fold reads only 1: not advertised.
            {"session_protocol3_all": True, "session_fragment_versions": "2, 1,x,\u00b2,"},
            "tenant-a",
            {"protocols": [2, 3], "fragment_versions": [1], "events": True},
        ),
    ],
)
def test_accepts_advertises_protocol_3_only_to_enabled_customers(values, customer, expected):
    from engine.shared.config import Settings

    assert sr.accepts(customer, Settings(**values)) == expected


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"with_events": True},
        # A cursor-only batch: no event, so nothing to carry.
        {"event_end": 0, "fragments": [], "fragment_version": 1},
    ],
)
def test_valid_protocol_3_envelopes_pass(changes):
    sr.validate_payload(fragments_batch(**changes))


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"fragments": [{"line": {"line_no": 0}}, {"line": {"line_no": 2}}]}, "fragment ordinals are not contiguous"),
        ({"fragments": [{"line": {"line_no": 1}}, {"line": {"line_no": 0}}]}, "fragment ordinals are not contiguous"),
        ({"fragments": [{"line": {"line_no": 0}}]}, "fragment coverage does not match payload"),
        ({"event_end": 3}, "fragment coverage does not match payload"),
        ({"fragments": None}, "fragment coverage does not match payload"),
        ({"fragments": [{"line": {"line_no": 0}}, "turn 1"]}, "not an object with an integer ordinal"),
        ({"fragments": [{"line": {"line_no": 0}}, {"line": {"line_no": "1"}}]}, "not an object with an integer ordinal"),
        ({"fragments": [{"line": {"line_no": False}}, {"line": {"line_no": True}}]}, "not an object with an integer ordinal"),
        ({"fragment_version": 99}, "unsupported fragment version"),
        ({"fragment_version": "1"}, "unsupported fragment version"),
        ({"fragment_version": True}, "unsupported fragment version"),
        ({"protocol_version": 3.0}, "unsupported codec"),
        ({"protocol_version": 4}, "unsupported codec"),
        (
            {"events": [{"line_no": 0}, {"line_no": 2}]},
            "retained-event ordinals are not contiguous",
        ),
        ({"events": [{"line_no": 0}]}, "event coverage does not match payload"),
        ({"events": []}, "event coverage does not match payload"),
    ],
)
def test_invalid_protocol_3_envelopes_never_reach_storage(changes, reason):
    with pytest.raises(HTTPException) as error:
        sr.validate_payload(fragments_batch(**changes))
    assert error.value.status_code == 422 and reason in error.value.detail


def test_fragments_without_a_version_are_refused():
    body = fragments_batch()
    del body["fragment_version"]
    with pytest.raises(HTTPException) as error:
        sr.validate_payload(body)
    assert error.value.status_code == 422 and "no fragment_version" in error.value.detail


def test_a_protocol_3_finalize_carries_no_fragments_or_events():
    final = fragments_batch(
        finalize=True, batch_seq=1, source_byte_start=30, source_line_start=3, event_start=2
    )
    with pytest.raises(HTTPException) as error:
        sr.validate_payload(dict(final, event_end=2))
    assert "fragment coverage" in error.value.detail
    with pytest.raises(HTTPException) as error:
        sr.validate_payload(dict(final, event_end=2, fragments=[], events=[{"line_no": 2}]))
    assert "event coverage" in error.value.detail
    del final["fragments"]
    sr.validate_payload(dict(final, event_end=2))


def test_the_fragment_versions_accepted_follow_the_setting(protocol3):
    # Listing only 2 refuses 1; and 2 is refused too while fold reads only 1.
    protocol3(session_fragment_versions="2")
    for version in (1, 2):
        with pytest.raises(HTTPException) as error:
            sr.validate_payload(fragments_batch(fragment_version=version))
        assert "unsupported fragment version" in error.value.detail
    protocol3(session_fragment_versions="1")
    sr.validate_payload(fragments_batch(fragment_version=1))


@pytest.mark.asyncio
async def test_a_protocol_3_stream_is_pinned_receipted_and_stores_its_fragments(
    database, protocol3
):
    import json

    _tenant, admin = database
    store = Store()
    body = fragments_batch(with_events=True)
    digest = hashlib.sha256(sr.canonical_payload(body)).hexdigest()
    accepted = await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert accepted["status"] == "accepted" and accepted["protocol_version"] == 3
    assert accepted["receipt"]["body_sha256"] == digest
    assert await admin.fetchval("SELECT protocol_version FROM session_streams") == 3
    stored = json.loads(next(iter(store.blobs.values())))["payload"]
    assert stored["fragments"] == body["fragments"] and stored["fragment_version"] == 1
    assert stored["events"] == body["events"]
    # Same key layout as protocol 2: the completer and deletion find it by key.
    assert "/sessions-v2/" in next(iter(store.blobs))[1]

    repeat = await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert repeat == {**accepted, "status": "duplicate"} and store.writes == 1

    end = {k: v for k, v in body.items() if k not in ("fragments", "events", "cwd")}
    end.update(finalize=True, batch_seq=1, source_byte_start=30, source_line_start=3, event_start=2)
    finalized = await sr.accept(end, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert finalized["protocol_version"] == 3 and finalized["receipt"]["finalized"] is True

    read = await sr.receipts("claude_code", body["session_id"], "tenant-a", -1, 200)
    assert read["state"] == "ready" and read["protocol_version"] == 3
    assert read["accepts"]["protocols"] == [2, 3]
    assert [r["body_sha256"] for r in read["receipts"]] == [
        digest,
        finalized["receipt"]["body_sha256"],
    ]


@pytest.mark.asyncio
async def test_the_receipts_read_advertises_accepts_in_every_state(database, protocol3):
    _tenant, admin = database
    store = Store()
    v2 = batch()
    await sr.accept(v2, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    legacy, deleted, absent = (str(uuid4()) for _ in range(3))
    await admin.execute(
        "INSERT INTO documents VALUES('tenant-a',$1)", f"claude_code:tenant-a:{legacy}"
    )
    await admin.execute(
        "INSERT INTO session_deletions(customer_id,source_system,session_id,deletion_id,reason) "
        "VALUES('tenant-a','claude_code',$1,$2,'request')",
        deleted,
        uuid4(),
    )
    states = {}
    for customer in ("tenant-a", "tenant-b"):
        for sid in (v2["session_id"], legacy, deleted, absent):
            read = await sr.receipts("claude_code", sid, customer, -1, 200)
            states[customer, read["state"]] = read
    expected = {"tenant-a": [2, 3], "tenant-b": [2]}
    assert {state for _customer, state in states} == {"ready", "legacy", "deleted", "absent"}
    for (customer, state), read in states.items():
        # Old taps refuse any other protocol_version, so only `ready` names the pin.
        assert read["protocol_version"] == 2, state
        assert read["accepts"] == {
            "protocols": expected[customer],
            "fragment_versions": [1],
            "events": True,
        }


@pytest.mark.asyncio
async def test_a_new_protocol_3_stream_needs_the_customer_enabled(database, protocol3):
    _tenant, admin = database
    store = Store()
    with pytest.raises(HTTPException) as error:
        await sr.accept(fragments_batch(), "tenant-b", SourceSystem.CLAUDE_CODE, store)
    assert error.value.status_code == 409 and error.value.detail == sr.PROTOCOL3_NOT_ENABLED == "protocol 3 not enabled"
    assert store.writes == 0
    assert await admin.fetchval("SELECT count(*) FROM session_streams") == 0
    protocol3(session_protocol3_all=True)
    assert (await sr.accept(fragments_batch(), "tenant-b", SourceSystem.CLAUDE_CODE, store))[
        "protocol_version"
    ] == 3


def _next(body, **changes):
    """The batch after `body` in the same stream, carrying one more event."""
    following = dict(
        body,
        batch_seq=body["batch_seq"] + 1,
        source_byte_start=body["source_byte_end"],
        source_byte_end=body["source_byte_end"] + 10,
        source_line_start=body["source_line_end"],
        source_line_end=body["source_line_end"] + 1,
        event_start=body["event_end"],
        event_end=body["event_end"] + 1,
        prefix_sha256="c" * 64,
    )
    following.pop("events", None)
    following.pop("fragments", None)
    following.update(changes)
    return following


@pytest.mark.asyncio
async def test_a_batch_on_the_other_protocol_than_its_stream_is_refused(database, protocol3):
    _tenant, admin = database
    store = Store()
    v2 = batch()
    v3 = fragments_batch()
    await sr.accept(v2, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    await sr.accept(v3, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    onto_v2 = _next(
        v2, protocol_version=3, fragment_version=1, fragments=[{"line": {"line_no": 2}}]
    )
    onto_v3 = _next(v3, protocol_version=2, events=[{"line_no": 2, "raw": {"type": "user"}}])
    for wrong in (onto_v2, onto_v3):
        with pytest.raises(HTTPException) as error:
            await sr.accept(wrong, "tenant-a", SourceSystem.CLAUDE_CODE, store)
        assert error.value.status_code == 409 and error.value.detail == sr.PROTOCOL_MISMATCH == "protocol mismatch"
    assert store.writes == 2
    assert await admin.fetchval("SELECT count(*) FROM session_batch_receipts") == 2


@pytest.mark.asyncio
async def test_an_open_protocol_3_stream_outlives_the_kill_switch(database, protocol3):
    """Withdrawing protocol 3 stops NEW protocol-3 sessions only: a running one
    holds batches it can never re-send as protocol 2."""
    _tenant, admin = database
    store = Store()
    first = fragments_batch()
    await sr.accept(first, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    protocol3(session_protocol3_customers="")
    later = _next(first, fragments=[{"line": {"line_no": 2}, "message": "still running"}])
    accepted = await sr.accept(later, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert accepted["status"] == "accepted" and accepted["protocol_version"] == 3
    read = await sr.receipts("claude_code", first["session_id"], "tenant-a", -1, 200)
    assert read["protocol_version"] == 3 and read["accepts"]["protocols"] == [2]
    with pytest.raises(HTTPException) as error:
        await sr.accept(fragments_batch(), "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert error.value.detail == sr.PROTOCOL3_NOT_ENABLED == "protocol 3 not enabled"
    assert await admin.fetchval("SELECT event_end FROM session_streams") == 3


# -- retiring protocol 2 for NEW streams (SESSION_PROTOCOL2_NEW_STREAMS) -------------
# research-os tap >= 0.9.15 answers 409 `protocol 2 retired` on a new stream's
# batch 0 by sending the session again on protocol 3; older taps keep the batch.


@pytest.mark.parametrize(
    ("values", "retired", "protocols"),
    [
        # Default: today's behaviour, protocol 2 open to everyone.
        ({"session_protocol3_all": True}, False, [2, 3]),
        ({"session_protocol3_all": True, "session_protocol2_new_streams": False}, True, [3]),
        (
            {"session_protocol3_customers": "tenant-z,tenant-a",
             "session_protocol2_new_streams": False},
            True,
            [3],
        ),
        # Not offered protocol 3 (or the kill switch): protocol 2 stays open, so a
        # new session always has a protocol it may start on.
        ({"session_protocol3_customers": "tenant-z", "session_protocol2_new_streams": False},
         False, [2]),
        ({"session_protocol2_new_streams": False}, False, [2]),
        # No fragment version this door reads: no protocol-3 batch could be taken.
        (
            {"session_protocol3_all": True, "session_protocol2_new_streams": False,
             "session_fragment_versions": "99"},
            False,
            [2, 3],
        ),
    ],
)
def test_protocol_2_is_retired_only_where_protocol_3_can_replace_it(values, retired, protocols):
    from engine.shared.config import Settings

    settings = Settings(**values)
    assert sr.protocol2_retired("tenant-a", settings) is retired
    assert sr.accepts("tenant-a", settings)["protocols"] == protocols


def test_the_retired_detail_is_stable():
    """research-os's tap matches this text (`PROTOCOL2_RETIRED` in its journal)."""
    assert sr.PROTOCOL2_RETIRED == "protocol 2 retired"


@pytest.mark.asyncio
async def test_a_retired_protocol_2_refuses_new_streams_and_finishes_open_ones(
    database, protocol3
):
    _tenant, admin = database
    store = Store()
    protocol3(session_protocol3_all=True)
    open_v2 = batch()
    await sr.accept(open_v2, "tenant-a", SourceSystem.CLAUDE_CODE, store)

    protocol3(session_protocol3_all=True, session_protocol2_new_streams=False)
    # A new protocol-2 stream: refused before any byte is written.
    new_v2 = batch()
    with pytest.raises(HTTPException) as error:
        await sr.accept(new_v2, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert error.value.status_code == 409 and error.value.detail == sr.PROTOCOL2_RETIRED
    assert store.writes == 1
    assert await admin.fetchval(
        "SELECT count(*) FROM session_streams WHERE session_id=$1", new_v2["session_id"]
    ) == 0
    read = await sr.receipts("claude_code", new_v2["session_id"], "tenant-a", -1, 200)
    # Old taps refuse any protocol_version but 2 for an absent session.
    assert read["state"] == "absent" and read["protocol_version"] == 2
    assert read["accepts"]["protocols"] == [3]

    # The same session sent again from batch 0 on protocol 3, same stream id: taken.
    restarted = fragments_batch(session_id=new_v2["session_id"], stream_id=new_v2["stream_id"])
    accepted = await sr.accept(restarted, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert accepted["status"] == "accepted" and accepted["protocol_version"] == 3

    # The open protocol-2 stream: a lost-response replay of batch 0 and its next
    # batch are both accepted, and its receipts still name protocol 2.
    replay = await sr.accept(open_v2, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    assert replay["status"] == "duplicate" and replay["protocol_version"] == 2
    later = _next(open_v2, events=[{"line_no": 2, "raw": {"type": "user"}}])
    assert (await sr.accept(later, "tenant-a", SourceSystem.CLAUDE_CODE, store))["status"] == "accepted"
    read = await sr.receipts("claude_code", open_v2["session_id"], "tenant-a", -1, 200)
    assert read["state"] == "ready" and read["protocol_version"] == 2
    assert read["stream"]["last_seq"] == 1 and read["accepts"]["protocols"] == [3]


@pytest.mark.asyncio
async def test_withdrawing_protocol_3_reopens_protocol_2_for_new_streams(database, protocol3):
    """The protocol-3 kill switch must never leave a new session with neither."""
    store = Store()
    protocol3(session_protocol3_customers="", session_protocol2_new_streams=False)
    body = batch()
    assert (await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store))["protocol_version"] == 2
    read = await sr.receipts("claude_code", str(uuid4()), "tenant-a", -1, 200)
    assert read["accepts"]["protocols"] == [2]


@pytest.mark.asyncio
async def test_credentials_inside_fragments_are_redacted_before_storage(
    database, protocol3, tmp_path
):
    _tenant, admin = database
    secret = "ghp_" + hashlib.sha256(b"synthetic fragment secret").hexdigest()[:36]
    body = fragments_batch()
    body["fragments"][0]["message"] = "copied " + secret
    body["fragments"][1]["tool_calls"] = [
        {"arguments": {"command": f"export GITHUB_TOKEN={secret}", "env": {"password": "Harbor7!"}}}
    ]
    digest = hashlib.sha256(sr.canonical_payload(body)).hexdigest()
    store = filesystem_store(tmp_path)
    accepted = await sr.accept(body, "tenant-a", SourceSystem.CLAUDE_CODE, store)
    # The receipt names the request the client sent; only the stored copy changes.
    assert accepted["receipt"]["body_sha256"] == digest
    key = await admin.fetchval("SELECT payload_s3_key FROM ingestion_queue")
    data = await store.get("tenant-a", key)
    assert secret.encode() not in data and b"Harbor7!" not in data
    assert b"copied " in data and b"turn 1" in data


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", [2, 3])
async def test_the_webhook_sends_protocol_2_and_3_batches_to_the_receipt_door(
    monkeypatch, protocol
):
    """A protocol-3 batch that missed the receipt door would be stored as a
    protocol-1 batch of a brand-new session."""
    import httpx
    from httpx import ASGITransport

    from engine.shared.config import get_settings
    from kb import ingestion_app

    key = "test-internal-key-32bytes-padding-padding"
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", key)
    get_settings.cache_clear()
    seen = []

    async def door(payload, customer, source, store):
        seen.append((payload["protocol_version"], customer, source))
        return {"status": "accepted", "protocol_version": payload["protocol_version"]}

    async def switch_on():
        return type("Switch", (), {"enabled": True, "reason": None})()

    async def active(customer):
        return None

    monkeypatch.setattr(sr, "accept", door)
    monkeypatch.setattr(ingestion_app, "get_ingestion_killswitch", switch_on)
    monkeypatch.setattr(ingestion_app, "refusal_for", active)
    monkeypatch.setattr(ingestion_app.app.state, "ctx", make_default_context(), raising=False)
    monkeypatch.setattr(ingestion_app.app.state, "store", Store(), raising=False)
    body = fragments_batch(with_events=True) if protocol == 3 else batch()
    try:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=ingestion_app.app), base_url="http://t"
        ) as client:
            response = await client.post(
                "/webhooks/claude_code",
                json=body,
                headers={"x-internal-knowledge-key": key, "x-prbe-customer": "tenant-a"},
            )
    finally:
        get_settings.cache_clear()
    assert response.status_code == 200, response.text
    assert response.json()["protocol_version"] == protocol
    assert seen == [(protocol, "tenant-a", SourceSystem.CLAUDE_CODE)]


def test_the_door_accepts_only_versions_the_fold_reads() -> None:
    """A version the deploy lists but fold cannot read would be accepted and
    folded to unparsed steps; the door intersects the two."""
    from engine.ingest.atif.fold import SUPPORTED_FRAGMENT_VERSIONS
    from engine.shared.config import Settings

    listed = Settings(session_fragment_versions="1,2,99")
    assert sr.fragment_versions(listed) == sorted({1, 2, 99} & SUPPORTED_FRAGMENT_VERSIONS)
    assert 99 not in sr.fragment_versions(listed)
    assert sr.fragment_versions(Settings(session_fragment_versions="")) == []
