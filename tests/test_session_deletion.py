"""Deleting captured sessions, by id and by author, against real Postgres + MinIO.

Sessions are built through the REAL pipeline wherever one exists -- the ingest
doors (kb/session_receipts.accept / accept_legacy), the worker's
process_queue_row (fetch_supplementary -> normalize -> _persist) with only the
LLM extraction stubbed -- so the inventory is checked against what production
writes, not against a fixture's guess of it. The rows no pipeline step here
produces (ingestion_events, failed_chunks, a parked research-os edge, raw
objects written by the idle sweep and the extraction cache) are added by hand
in the shapes their writers use.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime

import httpx
import orjson
import pytest
import pytest_asyncio
from fastapi import HTTPException

from engine.ingest.handlers.base import make_default_context
from engine.ingest.normalizer import Normalizer
from engine.shared import claude_code_extraction as ext
from engine.shared import db as db_module
from engine.shared.config import get_settings
from engine.shared.constants import SourceSystem, agent_session_canonical_id
from engine.shared.session_suppression import SessionDeleted, session_lock_key
from engine.shared.storage import _reset_bucket_cache_for_tests, get_store, reset_store
from kb import session_deletion as sd
from kb import session_receipts as sr
from kb.session_completer import enqueue_idle_session_finalizers

CC = SourceSystem.CLAUDE_CODE
ALICE = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"
ALICE_EMAIL = "alice@example.test"
BOB_EMAIL = "bob@example.test"


def _sid() -> str:
    return str(uuid.uuid4())


# --- fixtures ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_extraction(monkeypatch):
    """The model call is the only thing stubbed: one unit of each kind."""

    async def fake_extract(*, session_id, events, cwd=None, agent="claude_code", cache=None):
        ref = ext.SegmentRef(index=0, total=1, boundary="size", start_line_no=0, end_line_no=1)
        bundle = ext.UnitBundle(
            qa=[ext.QA(prompt=f"what about {session_id[:8]}?", outcome="answered", tags=["t"])],
            decision=[
                ext.Decision(
                    question="keep or drop?",
                    options_considered=["keep", "drop"],
                    chosen="drop",
                    rationale="the customer asked",
                )
            ],
        )
        for unit in (*bundle.qa, *bundle.decision):
            unit.segment = ref
        return bundle

    monkeypatch.setattr("kb.handlers.claude_code._ext.extract_units_from_session", fake_extract)


@pytest_asyncio.fixture
async def env(live_db, monkeypatch):
    """Two tenants, a fresh store, and cleanup of every bucket they touched."""
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    get_settings.cache_clear()  # type: ignore[attr-defined]
    reset_store()
    _reset_bucket_cache_for_tests()
    tag = uuid.uuid4().hex[:10]
    tenants = [f"sdel-a-{tag}", f"sdel-b-{tag}"]
    async with db_module.raw_conn() as conn:
        for t in tenants:
            await conn.execute(
                "INSERT INTO customers (customer_id, display_name, api_key_hash) "
                "VALUES ($1, 'sdel', 'sdel-hash')",
                t,
            )
    store = get_store()
    for t in tenants:
        await store.ensure_bucket(await store.bucket_for(t))
    try:
        yield tenants, store
    finally:
        for t in tenants:
            await store.delete_bucket_recursive(await store.bucket_for(t))
        get_settings.cache_clear()  # type: ignore[attr-defined]


# --- building sessions through the real pipeline ----------------------------------


def _event(i: int, text: str) -> dict:
    return {"line_no": i, "raw": {"type": "user", "message": {"role": "user", "content": text}}}


def _identity(employee_id: str, email: str) -> dict:
    # What research-os stamps on every forwarded batch (server-authoritative).
    return {
        "employee_id": employee_id,
        "employee_email": email,
        "employee_name": email.split("@")[0],
        "device_id": f"dev-{employee_id[:8]}",
    }


def _v2_batches(sid: str, employee_id: str, email: str, n: int = 2) -> list[dict]:
    stream = str(uuid.uuid4())
    out, byte, line, event = [], 0, 0, 0
    prefix = hashlib.sha256(b"").hexdigest()
    for seq in range(n):
        events = [_event(event, f"v2 turn {seq}"), _event(event + 1, f"v2 reply {seq}")]
        prefix = hashlib.sha256(f"{sid}:{seq}".encode()).hexdigest()
        out.append(
            dict(
                protocol_version=2, session_id=sid, stream_id=stream, batch_seq=seq,
                source_byte_start=byte, source_byte_end=byte + 100,
                source_line_start=line, source_line_end=line + 2,
                event_start=event, event_end=event + 2, prefix_sha256=prefix,
                cwd="/work", events=events, **_identity(employee_id, email),
            )
        )
        byte, line, event = byte + 100, line + 2, event + 2
    out.append(
        dict(
            protocol_version=2, session_id=sid, stream_id=stream, batch_seq=n,
            source_byte_start=byte, source_byte_end=byte, source_line_start=line,
            source_line_end=line, event_start=event, event_end=event,
            prefix_sha256=prefix, finalize=True, events=[], **_identity(employee_id, email),
        )
    )
    return out


async def _mine(customer: str, source: SourceSystem, sid: str) -> None:
    """The worker's pass over the session's queue row, for real."""
    async with db_module.raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT queue_id, payload_s3_keys FROM ingestion_queue "
            "WHERE customer_id = $1 AND source_system = $2 AND source_event_id = $3",
            customer, source.value, sid,
        )
    ctx = make_default_context()
    try:
        await Normalizer(ctx).process_queue_row(
            queue_id=row["queue_id"], customer_id=customer, source_system=source,
            source_event_id=sid, payload_s3_keys=list(row["payload_s3_keys"]),
        )
    finally:
        await ctx.http.aclose()


async def v2_session(customer: str, sid: str, employee_id: str = ALICE, email: str = ALICE_EMAIL,
                     source: SourceSystem = CC) -> None:
    store = get_store()
    for payload in _v2_batches(sid, employee_id, email):
        result = await sr.accept(payload, customer, source, store)
        assert result["status"] == "accepted", result
    await _mine(customer, source, sid)


def _legacy_key(customer: str, sid: str, date: str, seq: int | None) -> str:
    name = f"{sid}:{seq}" if seq is not None else sid
    return f"raw/{CC.value}/{customer}/{date}/{name}.json"


async def v1_session(customer: str, sid: str, employee_id: str = ALICE, email: str = ALICE_EMAIL,
                     date: str = "2025/03/14") -> list[str]:
    """A protocol-1 session: two batches and a client finalize, under an OLD
    date folder, through accept_legacy. Returns its raw keys."""
    store = get_store()
    keys = []
    bodies = [
        {"session_id": sid, "batch_seq": 0, "cwd": "/w", "events": [_event(0, "v1 hello")]},
        {"session_id": sid, "batch_seq": 1, "cwd": "/w", "events": [_event(1, "v1 more")]},
        {"session_id": sid, "finalize": True},
    ]
    for body in bodies:
        payload = {**body, **_identity(employee_id, email)}
        key = _legacy_key(customer, sid, date, body.get("batch_seq"))
        envelope = orjson.dumps({"_headers": {}, "payload": payload,
                                 "received_at": datetime.now(UTC).isoformat(), "trace_id": "t"})
        assert await sr.accept_legacy(payload, envelope, customer, CC, store, key, None)
        keys.append(key)
    await _mine(customer, CC, sid)
    return keys


async def add_side_rows(customer: str, sid: str, *, opaque_date: str = "2025/01/02") -> dict:
    """Rows and objects other writers leave: idle-sweep marker, extraction cache,
    an ingestion_events row pointing at an object whose name does NOT carry the
    session id, a failed chunk, a parked research-os edge, and a research-os Run
    node joined to the AgentSession (to check its degree comes down)."""
    store = get_store()
    bucket = await store.bucket_for(customer)
    marker = f"raw/{CC.value}/{customer}/{sid}/finalize.marker"
    cache = f"raw/{CC.value}/{customer}/{sid}/extraction-cache/{'a' * 64}.json"
    opaque = f"raw/{CC.value}/{customer}/{opaque_date}/opaque-{uuid.uuid4().hex}.json"
    for key in (marker, cache, opaque):
        await store.put(bucket, key, b"{}")
    session_doc = f"{CC.value}:{customer}:{sid}"
    agent_node = agent_session_canonical_id(CC.value, sid)
    async with db_module.with_tenant(customer) as conn:
        await conn.execute(
            "INSERT INTO ingestion_events (customer_id, source_system, event_type, source_event_id, "
            "payload_s3_key, status) VALUES ($1, $2, 'session', $3, $4, 'processed')",
            customer, CC.value, sid, opaque,
        )
        await conn.execute(
            "INSERT INTO failed_chunks (customer_id, doc_id, doc_version, chunk_index, error) "
            "VALUES ($1, $2, 1, 7, 'embed failed')",
            customer, session_doc,
        )
        await conn.execute(
            "INSERT INTO pending_edges (customer_id, missing_label, missing_canonical_id, edge_type, "
            "from_label, from_canonical_id, to_label, to_canonical_id, source_system) "
            "VALUES ($1, 'AgentSession', $2, 'PRODUCED', 'Run', 'run:x', 'AgentSession', $2, 'custom_ingest')",
            customer, agent_node,
        )
        run_node = await conn.fetchval(
            "INSERT INTO graph_nodes (customer_id, label, canonical_id, degree) "
            "VALUES ($1, 'Run', $2, 1) RETURNING node_id",
            customer, f"run:{sid}",
        )
        agent_node_id = await conn.fetchval(
            "SELECT node_id FROM graph_nodes WHERE customer_id = $1 AND label = 'AgentSession' "
            "AND canonical_id = $2",
            customer, agent_node,
        )
        assert agent_node_id is not None, "the pipeline must have written the AgentSession node"
        await conn.execute(
            "INSERT INTO graph_edges (customer_id, edge_type, from_node_id, to_node_id, source_system) "
            "VALUES ($1, 'PRODUCED', $2, $3, 'custom_ingest')",
            customer, run_node, agent_node_id,
        )
    return {"marker": marker, "cache": cache, "opaque": opaque, "run_node": run_node}


# --- observing ----------------------------------------------------------------------


_TABLES = (
    "documents", "chunks", "failed_chunks", "acl_snapshots", "inferred_edges_queue",
    "graph_nodes", "graph_edges", "graph_node_provenance", "node_post_write_queue",
    "pending_edges", "ingestion_queue", "ingestion_events", "session_streams",
    "session_batch_receipts",
)


async def snapshot(customer: str) -> dict:
    """Every row count of the tenant plus every raw object it has."""
    counts = {}
    async with db_module.with_tenant(customer) as conn:
        for table in _TABLES:
            counts[table] = await conn.fetchval(
                f"SELECT count(*) FROM {table} WHERE customer_id = $1", customer
            )
    store = get_store()
    objects = sorted(await store.list_keys(await store.bucket_for(customer), "raw/"))
    return {"rows": counts, "objects": objects}


async def session_rows(customer: str, sid: str) -> dict:
    async with db_module.with_tenant(customer) as conn:
        inv = await sd.inventory(conn, customer, sd.SessionRef(CC.value, sid))
    return {k: v for k, v in inv.counts.items() if v}


async def delete(customer: str, sids: list[str], **kw) -> dict:
    refs = [sd.SessionRef(CC.value, s) for s in sids]
    await sd.record_sessions(customer, refs, deletion_id=str(uuid.uuid4()), reason="test",
                             ticket="T-1", selector={"by": "id"})
    return await sd.run_deletion(customer, refs, grace_s=0, **kw)


# --- tests --------------------------------------------------------------------------


def test_prefix_safety_rejects_ids_that_name_shared_folders() -> None:
    assert sd.valid_session_id(_sid())
    for bad in ("sessions-v2", "2026", "a/b/c/d/e", "", "short", "x" * 201, "has space ok?",
                "abcdefgh:1"):
        assert not sd.valid_session_id(bad), bad


def test_every_writer_serializes_on_the_ingest_lock() -> None:
    # The ingest door, the idle sweep, the worker fence and the deletion all
    # take this one name. If it drifts they stop excluding each other.
    assert session_lock_key("c", "claude_code", "s") == "session-stream:c:claude_code:s"


@pytest.mark.asyncio
async def test_dry_run_counts_everything_and_changes_nothing(env) -> None:
    (a, _b), _store = env
    s1, s2 = _sid(), _sid()
    await v2_session(a, s1)
    await v1_session(a, s2)
    await add_side_rows(a, s1)
    before = await snapshot(a)

    report = await sd.plan(a, [sd.SessionRef(CC.value, s1), sd.SessionRef(CC.value, s2)])

    assert await snapshot(a) == before
    by_sid = {s["session_id"]: s for s in report["sessions"]}
    v2 = by_sid[s1]
    # Session + two units, the parked edge, the stream and its 3 receipts, ...
    for table in ("documents", "chunks", "acl_snapshots", "graph_nodes", "graph_edges",
                  "graph_node_provenance", "pending_edges", "failed_chunks", "ingestion_queue",
                  "ingestion_events", "session_streams", "session_batch_receipts",
                  "inferred_edges_queue"):
        assert v2["rows"].get(table, 0) > 0, (table, v2["rows"])
    assert v2["rows"]["session_batch_receipts"] == 3
    # 3 v2 batches + marker + cache are listed; the opaque object is referenced.
    assert v2["r2_objects"] == 5
    assert v2["r2_referenced_keys"] == 1
    v1 = by_sid[s2]
    assert v1["r2_objects"] == 0 and v1["r2_referenced_keys"] == 3
    assert "session_streams" not in v1["rows"]
    async with db_module.with_tenant(a) as conn:
        assert await conn.fetchval("SELECT count(*) FROM session_deletions") == 0
    assert report["totals"]["sessions"] == 2


@pytest.mark.asyncio
async def test_delete_by_id_removes_v2_and_legacy_everywhere_and_nothing_else(env) -> None:
    (a, b), store = env
    s1, s2, keep = _sid(), _sid(), _sid()
    await v2_session(a, s1)
    legacy_keys = await v1_session(a, s2)
    side = await add_side_rows(a, s1)
    await v2_session(a, keep, BOB, BOB_EMAIL)
    # Tenant B holds a session with THE SAME id as s1.
    await v2_session(b, s1)
    await add_side_rows(b, s1)
    # An unreferenced protocol-1 object of s2 (a crash between upload and
    # enqueue) and a same-date object of another session that must survive.
    bucket = await store.bucket_for(a)
    orphan = _legacy_key(a, s2, "2025/02/03", 7)
    bystander = _legacy_key(a, keep, "2025/02/03", 7)
    for key in (orphan, bystander):
        await store.put(bucket, key, b"{}")
    keep_before = await session_rows(a, keep)
    b_before = await snapshot(b)
    run_degree_before = await _degree(a, side["run_node"])

    outcome = await delete(a, [s1, s2])

    for sid in (s1, s2):
        o = outcome["sessions"][f"claude_code:{sid}"]
        assert o["verified"], o
        assert await session_rows(a, sid) == {}
    # Date-foldered protocol-1 batches and an object whose NAME does not carry
    # the session id go because rows referenced them; nothing listed them.
    for key in (*legacy_keys, side["opaque"], side["marker"], side["cache"]):
        assert not await store.exists(bucket, key), key
    # No row ever referenced the orphan, so only a scan finds it.
    assert await store.exists(bucket, orphan)
    rescan = await delete(a, [s2], deep_scan=True)
    assert rescan["sessions"][f"claude_code:{s2}"]["r2_objects_deleted"] == 1
    remaining = await store.list_keys(bucket, "raw/")
    assert not [k for k in remaining if s1 in k or s2 in k], remaining
    # Everything else is exactly as it was.
    assert await store.exists(bucket, bystander)
    assert await session_rows(a, keep) == keep_before
    assert await snapshot(b) == b_before
    assert await _degree(a, side["run_node"]) == run_degree_before - 1
    async with db_module.with_tenant(a) as conn:
        rows = await conn.fetch(
            "SELECT session_id, status, deleted_at, pending_keys, reason, ticket, selector "
            "FROM session_deletions ORDER BY session_id"
        )
    assert {r["session_id"] for r in rows} == {s1, s2}
    for r in rows:
        assert r["status"] == "done" and r["deleted_at"] is not None
        assert r["pending_keys"] == [], "the key journal is emptied once its deletes are confirmed"
        assert (r["reason"], r["ticket"], json.loads(r["selector"])) == ("test", "T-1", {"by": "id"})


async def _degree(customer: str, node_id: int) -> int:
    async with db_module.with_tenant(customer) as conn:
        return await conn.fetchval("SELECT degree FROM graph_nodes WHERE node_id = $1", node_id)


@pytest.mark.asyncio
async def test_delete_by_author_takes_every_capture_of_that_person_only(env, monkeypatch) -> None:
    (a, _b), store = env
    v1_alice, v2_alice, bob, unmined, unmined2, bob_unmined = (_sid() for _ in range(6))
    await v1_session(a, v1_alice)          # author_id = alice
    await v2_session(a, v2_alice)          # uploader_id = alice, author_id NULL
    await v2_session(a, bob, BOB, BOB_EMAIL)
    # Never processed: only raw batches say whose it is. Paged one row at a
    # time, so a scan that stopped at its first page would miss the second.
    monkeypatch.setattr(sd, "_UNINDEXED_PAGE", 1)
    for sid, who, mail in ((unmined, ALICE, ALICE_EMAIL), (bob_unmined, BOB, BOB_EMAIL),
                           (unmined2, ALICE, ALICE_EMAIL)):
        await sr.accept(_v2_batches(sid, who, mail)[0], a, CC, store)
    bob_before = await session_rows(a, bob)
    bob_unmined_before = await session_rows(a, bob_unmined)

    by_email = await sd.select_by_author(a, [CC.value], employee_id=None, email=ALICE_EMAIL.upper())
    by_id = await sd.select_by_author(a, [CC.value], employee_id=ALICE, email=None)
    expected = {sd.SessionRef(CC.value, s) for s in (v1_alice, v2_alice, unmined, unmined2)}
    assert set(by_email.refs) == expected
    assert set(by_id.refs) == expected
    assert ALICE in by_id.person_ids

    await sd.record_sessions(a, by_id.refs, deletion_id=str(uuid.uuid4()), reason="gdpr",
                             ticket=None, selector={"by": "author"})
    outcome = await sd.run_deletion(a, by_id.refs, person_ids=by_id.person_ids, grace_s=0)

    assert all(o["verified"] for o in outcome["sessions"].values()), outcome
    for sid in (v1_alice, v2_alice, unmined, unmined2):
        assert await session_rows(a, sid) == {}
    assert await session_rows(a, bob) == bob_before
    assert await session_rows(a, bob_unmined) == bob_unmined_before
    async with db_module.with_tenant(a) as conn:
        persons = {
            r["canonical_id"]
            for r in await conn.fetch("SELECT canonical_id FROM graph_nodes WHERE label = 'Person'")
        }
    assert outcome["person_nodes_deleted"] == 1
    assert ALICE not in persons and BOB in persons


@pytest.mark.asyncio
async def test_a_deleted_session_cannot_come_back(env, monkeypatch) -> None:
    (a, _b), store = env
    v2, v1 = _sid(), _sid()
    await v2_session(a, v2)
    await v1_session(a, v1)
    await delete(a, [v2, v1])
    before = await snapshot(a)

    # The client re-sends from batch 0 (a fresh stream, a retry, a reconnect).
    with pytest.raises(HTTPException) as refused:
        await sr.accept(_v2_batches(v2, ALICE, ALICE_EMAIL)[0], a, CC, store)
    assert refused.value.status_code == 410
    assert refused.value.detail["reason"] == "session_deleted"
    payload = {"session_id": v1, "batch_seq": 0, "events": [_event(0, "again")], **_identity(ALICE, ALICE_EMAIL)}
    with pytest.raises(HTTPException) as refused_v1:
        await sr.accept_legacy(payload, orjson.dumps({"payload": payload}), a, CC, store,
                               _legacy_key(a, v1, "2026/09/25", 0), None)
    assert refused_v1.value.status_code == 410
    # The receipts read says so instead of inviting an upload.
    receipts = await sr.receipts(CC.value, v2, x_prbe_customer=a, after=-1, limit=10)
    assert receipts["state"] == "deleted"

    # A worker that read the session before the deletion cannot write it back.
    event_payload = _v2_batches(v2, ALICE, ALICE_EMAIL)[0]
    from engine.shared.models import WebhookEvent
    from kb.handlers.claude_code import ClaudeCodeConnector

    connector = ClaudeCodeConnector(make_default_context())
    result = await connector.normalize(
        WebhookEvent(customer_id=a, source_system=CC, source_event_id=v2,
                     received_at=datetime.now(UTC), payload_s3_key="", payload_s3_keys=[],
                     raw_payload=event_payload, headers={}),
        {"session_id": v2, "events": event_payload["events"], "session_complete": False,
         "cwd": "/w", "employee_id": ALICE},
    )
    with pytest.raises(SessionDeleted):
        await Normalizer(make_default_context())._persist(a, CC, result)
    # Nor can the idle sweep leave a marker for it.
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO ingestion_queue (customer_id, source_system, source_event_id, payload_s3_key, "
            "payload_s3_keys, status, enqueued_at, priority, version) VALUES "
            "($1, 'claude_code', $2, $3, ARRAY[$3], 'done', now() - interval '2 days', 75, 1)",
            a, v1, _legacy_key(a, v1, "2025/03/14", 0),
        )
    assert await enqueue_idle_session_finalizers(idle_minutes=5) == 0
    async with db_module.raw_conn() as conn:
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", a)
    assert await snapshot(a) == before


@pytest.mark.asyncio
async def test_legal_hold_refuses_and_is_rechecked_at_execution(env) -> None:
    (a, _b), _store = env
    sid = _sid()
    await v2_session(a, sid)
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "UPDATE customers SET metadata = metadata || '{\"legal_hold\": \"case-42\"}' "
            "WHERE customer_id = $1",
            a,
        )
    assert await sd.legal_hold(a) == "case-42"
    before = await snapshot(a)

    async with _client() as client:
        dry = await client.post("/api/session-deletions", headers=_headers(a),
                                json={"session_ids": [sid]})
        applied = await client.post("/api/session-deletions", headers=_headers(a),
                                    json={"session_ids": [sid], "dry_run": False, "reason": "r"})
    assert dry.status_code == 200 and dry.json()["legal_hold"] == "case-42"
    assert applied.status_code == 423
    assert applied.json()["detail"]["reason"] == "legal_hold"
    # Held after the request was recorded: the run re-reads the hold and stops.
    ref = sd.SessionRef(CC.value, sid)
    await sd.record_sessions(a, [ref], deletion_id=str(uuid.uuid4()), reason="r",
                             ticket=None, selector={"by": "id"})
    outcome = await sd.erase_session(a, ref, grace_s=0)
    assert not outcome.get("verified")
    async with db_module.with_tenant(a) as conn:
        assert await conn.fetchval("SELECT status FROM session_deletions") == "held"
        await conn.execute("DELETE FROM session_deletions")
    assert await snapshot(a) == before


@pytest.mark.parametrize("value", ["false", "null", '""'])
@pytest.mark.asyncio
async def test_a_cleared_hold_is_not_a_hold(env, value) -> None:
    (a, _b), _store = env
    async with db_module.raw_conn() as conn:
        await conn.execute(
            f"UPDATE customers SET metadata = '{{\"legal_hold\": {value}}}' WHERE customer_id = $1", a
        )
    assert await sd.legal_hold(a) is None


@pytest.mark.asyncio
async def test_rerun_is_idempotent_and_an_unknown_id_is_still_suppressed(env) -> None:
    (a, _b), store = env
    sid, never_uploaded = _sid(), _sid()
    await v2_session(a, sid)
    first = await delete(a, [sid, never_uploaded])
    async with db_module.with_tenant(a) as conn:
        completed = dict(await conn.fetch("SELECT session_id, deleted_at FROM session_deletions"))
    again = await delete(a, [sid, never_uploaded])

    assert all(o["verified"] for o in first["sessions"].values())
    async with db_module.with_tenant(a) as conn:
        # A re-run that found nothing keeps the original completion time.
        assert dict(await conn.fetch("SELECT session_id, deleted_at FROM session_deletions")) == completed
    for o in again["sessions"].values():
        assert o["verified"] and o["rows_deleted"] == {} and o["r2_objects_deleted"] == 0
    with pytest.raises(HTTPException) as refused:
        await sr.accept(_v2_batches(never_uploaded, ALICE, ALICE_EMAIL)[0], a, CC, store)
    assert refused.value.status_code == 410


@pytest.mark.asyncio
async def test_a_session_mid_pass_is_swept_again_after_the_worker_finishes(env, monkeypatch) -> None:
    (a, _b), store = env
    sid = _sid()
    await v2_session(a, sid)
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "UPDATE ingestion_queue SET status = 'processing' WHERE customer_id = $1", a
        )
    bucket = await store.bucket_for(a)
    late = f"raw/{CC.value}/{a}/{sid}/extraction-cache/{'b' * 64}.json"
    async with db_module.with_tenant(a) as conn:
        unit_doc = await conn.fetchval(
            "SELECT doc_id FROM documents WHERE parent_doc_id IS NOT NULL LIMIT 1"
        )
    assert unit_doc

    waits: list[float] = []

    async def worker_saves_cache_while_we_wait(seconds: float) -> None:
        # The pass's post-commit steps: a cache answer and an inferred-edges
        # enqueue naming a unit document whose row is already gone.
        waits.append(seconds)
        await store.put(bucket, late, b"{}")
        async with db_module.raw_conn() as conn:
            await conn.execute(
                "INSERT INTO inferred_edges_queue (customer_id, anchor_doc_id, extractor_id) "
                "VALUES ($1, $2, 'inferred_edges:v1')",
                a, unit_doc,
            )

    monkeypatch.setattr(sd, "_wait_for_in_flight", worker_saves_cache_while_we_wait)
    outcome = (await delete(a, [sid]))["sessions"][f"claude_code:{sid}"]

    assert waits == [0], "a session found mid-pass must be waited on and swept again"
    assert outcome["in_flight"] and outcome["verified"], outcome
    assert not await store.exists(bucket, late)
    async with db_module.raw_conn() as conn:
        assert await conn.fetchval(
            "SELECT count(*) FROM inferred_edges_queue WHERE anchor_doc_id = $1", unit_doc
        ) == 0


@pytest.mark.asyncio
async def test_a_pass_that_outlives_the_deletion_cleans_up_after_itself(env, monkeypatch) -> None:
    """The worker is mid-extraction when the deletion runs AND finishes; its
    cache answer lands after the deletion's last sweep. Its write is refused,
    and it deletes the session folder itself."""
    from engine.ingest.worker import Worker

    (a, _b), store = env
    sid = _sid()
    for payload in _v2_batches(sid, ALICE, ALICE_EMAIL):
        await sr.accept(payload, a, CC, store)
    async with db_module.raw_conn() as conn:
        await conn.execute("UPDATE ingestion_queue SET status = 'processing' WHERE customer_id = $1", a)
        row = await conn.fetchrow("SELECT * FROM ingestion_queue WHERE customer_id = $1", a)
    bucket = await store.bucket_for(a)
    late = f"raw/{CC.value}/{a}/{sid}/extraction-cache/{'c' * 64}.json"
    outcomes: list = []

    async def extraction_during_which_the_session_is_deleted(*, session_id, events, cwd=None,
                                                              agent="claude_code", cache=None):
        outcomes.append(await delete(a, [sid]))
        await store.put(bucket, late, b"{}")  # SegmentCache.save, after the deletion
        return ext.UnitBundle(qa=[ext.QA(prompt="p", outcome="o", tags=[])])

    monkeypatch.setattr("kb.handlers.claude_code._ext.extract_units_from_session",
                        extraction_during_which_the_session_is_deleted)
    monkeypatch.setattr(sd, "_wait_for_in_flight", _no_wait)
    ctx = make_default_context()
    try:
        await Worker(ctx)._process(row)
    finally:
        await ctx.http.aclose()

    assert outcomes and outcomes[0]["sessions"][f"claude_code:{sid}"]["verified"]
    assert not await store.exists(bucket, late)
    assert await session_rows(a, sid) == {}


async def _no_wait(seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_the_worker_does_not_mine_a_deleted_session(env, monkeypatch) -> None:
    """A deleted session's row back in `pending` (the idle sweep's partial-pass
    retry does not read deletions) is skipped before any extraction."""
    from engine.ingest.worker import Worker

    (a, _b), store = env
    sid = _sid()
    for payload in _v2_batches(sid, ALICE, ALICE_EMAIL):
        await sr.accept(payload, a, CC, store)
    await sd.record_sessions(a, [sd.SessionRef(CC.value, sid)], deletion_id=str(uuid.uuid4()),
                             reason="r", ticket=None, selector={"by": "id"})
    async with db_module.raw_conn() as conn:
        await conn.execute("UPDATE ingestion_queue SET status = 'processing' WHERE customer_id = $1", a)
        row = await conn.fetchrow("SELECT * FROM ingestion_queue WHERE customer_id = $1", a)
    mined: list[str] = []

    async def must_not_run(*, session_id, **_kw):
        mined.append(session_id)
        return ext.UnitBundle()

    monkeypatch.setattr("kb.handlers.claude_code._ext.extract_units_from_session", must_not_run)
    ctx = make_default_context()
    try:
        await Worker(ctx)._process(row)
    finally:
        await ctx.http.aclose()
    assert mined == []
    async with db_module.raw_conn() as conn:
        status = await conn.fetchval("SELECT status FROM ingestion_queue WHERE customer_id = $1", a)
        docs = await conn.fetchval("SELECT count(*) FROM documents WHERE customer_id = $1", a)
    assert status == "done" and docs == 0


# --- over HTTP ----------------------------------------------------------------------


def _client() -> httpx.AsyncClient:
    from kb.ingestion_app import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _headers(customer: str) -> dict:
    return {"X-Internal-Knowledge-Key": "test-internal-key", "X-Prbe-Customer": customer}


@pytest.mark.asyncio
async def test_route_contract_dry_run_apply_and_status(env) -> None:
    import asyncio

    (a, _b), _store = env
    sid, other = _sid(), _sid()
    await v2_session(a, sid)
    await v2_session(a, other, BOB, BOB_EMAIL)
    async with _client() as client:
        unauth = await client.post("/api/session-deletions", json={"session_ids": [sid]},
                                   headers={"X-Prbe-Customer": a})
        assert unauth.status_code == 401
        bad = await client.post("/api/session-deletions", headers=_headers(a),
                                json={"session_ids": ["sessions-v2"]})
        assert bad.status_code == 422
        no_reason = await client.post("/api/session-deletions", headers=_headers(a),
                                      json={"session_ids": [sid], "dry_run": False})
        assert no_reason.status_code == 422
        both = await client.post("/api/session-deletions", headers=_headers(a),
                                 json={"session_ids": [sid], "author": {"email": ALICE_EMAIL}})
        assert both.status_code == 422
        unknown = await client.post("/api/session-deletions", headers=_headers("no-such-tenant"),
                                    json={"session_ids": [sid]})
        assert unknown.status_code == 404

        dry = await client.post("/api/session-deletions", headers=_headers(a),
                                json={"author": {"employee_id": ALICE}})
        assert dry.status_code == 200, dry.text
        body = dry.json()
        assert body["dry_run"] and [s["session_id"] for s in body["sessions"]] == [sid]
        assert body["not_covered"] and body["legal_hold"] is None

        applied = await client.post(
            "/api/session-deletions", headers=_headers(a),
            json={"author": {"employee_id": ALICE}, "dry_run": False, "reason": "customer request",
                  "ticket": "prbe-ai/research-os#1"},
        )
        assert applied.status_code == 202, applied.text
        deletion_id = applied.json()["deletion_id"]
        assert applied.json()["sessions"] == [{"source": "claude_code", "session_id": sid}]
        status = None
        for _ in range(200):
            status = (await client.get(f"/api/session-deletions/{deletion_id}", headers=_headers(a))).json()
            if status["status"] != "running":
                break
            await asyncio.sleep(0.05)
        assert status["status"] == "done", status
        assert status["sessions"][0]["deleted_at"] and status["sessions"][0]["result"]["verified"]
        missing = await client.get(f"/api/session-deletions/{uuid.uuid4()}", headers=_headers(a))
        assert missing.status_code == 404
    assert await session_rows(a, sid) == {}
    assert await session_rows(a, other) != {}


@pytest.mark.asyncio
async def test_the_coalesced_write_path_refuses_a_deleted_session_too(env) -> None:
    """persist_batch (INGEST_CLAIM_COALESCE_MAX > 1) writes several sessions in
    one transaction. One deleted session rolls the batch back with a TRANSIENT
    error, so the worker returns the healthy siblings to pending."""
    from engine.shared.models import WebhookEvent
    from kb.handlers.claude_code import ClaudeCodeConnector

    (a, _b), _store = env
    deleted, healthy = _sid(), _sid()
    await delete(a, [deleted])
    connector = ClaudeCodeConnector(make_default_context())
    results = []
    for sid in (deleted, healthy):
        payload = _v2_batches(sid, ALICE, ALICE_EMAIL)[0]
        results.append(
            await connector.normalize(
                WebhookEvent(customer_id=a, source_system=CC, source_event_id=sid,
                             received_at=datetime.now(UTC), payload_s3_key="",
                             payload_s3_keys=[], raw_payload=payload, headers={}),
                {"session_id": sid, "events": payload["events"], "session_complete": False,
                 "cwd": "/w", "employee_id": ALICE},
            )
        )
    before = await snapshot(a)
    with pytest.raises(SessionDeleted) as refused:
        await Normalizer(make_default_context()).persist_batch(
            a, CC, [(results[0], None), (results[1], None)]
        )
    assert refused.value.transient
    assert await snapshot(a) == before


@pytest.mark.asyncio
async def test_recording_a_deletion_stops_a_pending_session_being_mined(env) -> None:
    (a, _b), store = env
    pending, busy = _sid(), _sid()
    for sid in (pending, busy):
        for payload in _v2_batches(sid, ALICE, ALICE_EMAIL)[:1]:
            await sr.accept(payload, a, CC, store)
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "UPDATE ingestion_queue SET status = 'processing' WHERE customer_id = $1 "
            "AND source_event_id = $2", a, busy,
        )
    await sd.record_sessions(a, [sd.SessionRef(CC.value, s) for s in (pending, busy)],
                             deletion_id=str(uuid.uuid4()), reason="r", ticket=None,
                             selector={"by": "id"})
    async with db_module.raw_conn() as conn:
        status = dict(await conn.fetch(
            "SELECT source_event_id, status FROM ingestion_queue WHERE customer_id = $1", a
        ))
    # The pending one is not worth an extraction; the one mid-pass is left for
    # the row phase to see (and wait out).
    assert status == {pending: "done", busy: "processing"}


@pytest.mark.asyncio
async def test_a_stalled_run_is_reported_and_resumed_by_its_id(env, monkeypatch) -> None:
    """A pod restart leaves sessions `pending`. The status says `stalled`, and
    resuming by deletion_id finishes them -- including an AUTHOR request, whose
    selection rows are gone once the row phase has run."""
    import asyncio

    (a, _b), store = env
    sid = _sid()
    await v2_session(a, sid)
    ref = sd.SessionRef(CC.value, sid)
    deletion_id = str(uuid.uuid4())
    await sd.record_sessions(a, [ref], deletion_id=deletion_id, reason="r", ticket=None,
                             selector={"by": "author"})
    # The run died after the rows went and before the objects did.
    await sd._journal_and_delete_rows(a, ref, set())
    bucket = await store.bucket_for(a)
    assert await store.list_keys(bucket, f"raw/{CC.value}/{a}/sessions-v2/{sid}/")
    async with _client() as client:
        fresh = (await client.get(f"/api/session-deletions/{deletion_id}", headers=_headers(a))).json()
        assert fresh["status"] == "running"
        monkeypatch.setattr(sd, "STALLED_AFTER_S", -1)
        assert (await client.get(f"/api/session-deletions/{deletion_id}",
                                 headers=_headers(a))).json()["status"] == "stalled"
        # Re-POSTing the author request cannot find the session any more.
        again = await client.post("/api/session-deletions", headers=_headers(a),
                                  json={"author": {"employee_id": ALICE}, "dry_run": False,
                                        "reason": "r"})
        assert again.json()["status"] == "nothing_to_delete"
        resumed = await client.post(f"/api/session-deletions/{deletion_id}/resume",
                                    headers=_headers(a))
        assert resumed.status_code == 202, resumed.text
        for _ in range(200):
            status = (await client.get(f"/api/session-deletions/{deletion_id}", headers=_headers(a))).json()
            if status["status"] not in ("running", "stalled"):
                break
            await asyncio.sleep(0.05)
        assert status["status"] == "done", status
        assert (await client.post(f"/api/session-deletions/{deletion_id}/resume",
                                  headers=_headers(a))).json()["status"] == "nothing_to_resume"
        assert (await client.post(f"/api/session-deletions/{uuid.uuid4()}/resume",
                                  headers=_headers(a))).status_code == 404
        async with db_module.raw_conn() as conn:
            await conn.execute(
                "UPDATE customers SET metadata = '{\"legal_hold\": true}' WHERE customer_id = $1", a
            )
        held = await client.post(f"/api/session-deletions/{deletion_id}/resume", headers=_headers(a))
        assert held.status_code == 423
    assert not await store.list_keys(bucket, f"raw/{CC.value}/{a}/")
