"""End-to-end: a client finalize turns a captured session into knowledge units.

This is the whole point of the finalize path, and until now nothing tested it.
The gateway forwards batches, the worker coalesces them into a live session
document, and the extracted units — qa, code_change (with its `intent`),
decision (with options considered + rationale), file_ref — are produced ONLY on
the `session_complete=True` branch. If a finalize does not reach that branch,
sessions are captured and never mined, which is exactly what production was
doing.

The extractor's own LLM call is covered by tests/test_claude_code_extraction.py;
here it is stubbed so this test asserts the WIRING — finalize in, unit
documents in Postgres out — without a network call or a paid token.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

import httpx
import pytest
from httpx import ASGITransport

from engine.shared.config import Settings, get_settings
from engine.shared.constants import SourceSystem
from engine.shared.db import close_pool, init_pool, raw_conn
from engine.shared.embeddings import reset_embedder
from engine.shared.models import WebhookEvent
from engine.shared.storage import reset_store
from kb.ingestion_app import app


def _stamped(ref, bundle):
    """Every unit carries where in the session it came from."""
    for unit in (*bundle.qa, *bundle.code_change, *bundle.decision, *bundle.file_ref):
        unit.segment = ref
    return bundle


CUSTOMER = "finalize-e2e-cust"
EMPLOYEE = "emp-finalize-e2e"


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv(
        "TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value()
    )
    get_settings.cache_clear()  # type: ignore[attr-defined]
    reset_embedder()
    reset_store()


@pytest.fixture
def stub_extractor(monkeypatch) -> dict:
    """Replace the LLM extraction with a fixed bundle, recording its input.

    Recording the input matters as much as the output: the extractor must be
    handed the session's REAL events, not the empty event list the finalize
    payload itself carries.
    """
    from engine.shared import claude_code_extraction as ext

    seen: dict = {}

    async def fake_extract(
        *, session_id: str, events: list, cwd: str | None = None,
        agent: str = "claude_code",
    ):
        seen["session_id"] = session_id
        seen["events"] = events
        seen["cwd"] = cwd
        seen["agent"] = agent
        ref = ext.SegmentRef(
            index=2, total=3, boundary="compaction",
            start_line_no=40, end_line_no=90,
        )
        return _stamped(ref, ext.UnitBundle(
            qa=[ext.QA(prompt="why 422?", outcome="tightened the schema", tags=["422"])],
            code_change=[
                ext.CodeChange(
                    summary="Tighten the ingest payload typing",
                    kind="fix",
                    files=["app/schemas/ingest.py"],
                    rationale="validation is the point of the endpoint",
                )
            ],
            decision=[
                ext.Decision(
                    question="loosen schema or fix caller?",
                    options_considered=["loosen", "tighten"],
                    chosen="tighten",
                    rationale="validation is the point",
                )
            ],
            file_ref=[
                ext.FileRef(files=["app/routes/ingest.py"], context="Pydantic v2 fix")
            ],
        ))

    monkeypatch.setattr(
        "kb.handlers.claude_code._ext.extract_units_from_session", fake_extract
    )
    return seen


@pytest.mark.asyncio
async def test_client_finalize_produces_unit_documents(
    live_db: None, settings: Settings, stub_extractor: dict
) -> None:
    device_id = f"dev-finalize-{uuid.uuid4()}"
    session_id = f"sess-finalize-{uuid.uuid4()}"
    token_hash = hashlib.sha256(f"secret-{uuid.uuid4()}".encode()).hexdigest()

    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'finalize-e2e', 'finalize-hash') ON CONFLICT DO NOTHING",
            CUSTOMER,
        )

    from engine.shared.storage import get_store

    store = get_store()
    await store.ensure_bucket(await store.bucket_for(CUSTOMER))

    await close_pool()
    transport = ASGITransport(app=app)
    internal_hdr = {"X-Internal-Knowledge-Key": "test-internal-key"}
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        reg = await client.post(
            "/api/devices/register",
            json={
                "customer_id": CUSTOMER,
                "employee_id": EMPLOYEE,
                "device_id": device_id,
                "token_hash": token_hash,
                "os": "macos",
                "hostname": "finalize-host",
            },
            headers=internal_hdr,
        )
        assert reg.status_code == 200, reg.text

        # One ordinary batch of transcript...
        batch = await client.post(
            "/webhooks/claude_code",
            json={
                "device_id": device_id,
                "session_id": session_id,
                "batch_seq": 0,
                "cwd": "/tmp/finalize-e2e",
                "employee_id": EMPLOYEE,
                "events": [
                    {
                        "line_no": 0,
                        "raw": {
                            "type": "user",
                            "message": {"content": "why is /ingest returning 422?"},
                        },
                    },
                    {
                        "line_no": 1,
                        "raw": {
                            "type": "assistant",
                            "message": {
                                "content": [
                                    {"type": "text", "text": "tightening the schema"}
                                ]
                            },
                        },
                    },
                ],
            },
            headers={**internal_hdr, "X-Prbe-Customer": CUSTOMER},
        )
        assert batch.status_code == 200, batch.text

        # ...then the goodbye the tap now sends on SessionEnd. Shape matches
        # what the gateway rebuilds from SessionFinalizeRequest: finalize +
        # session_id + device_id + server-stamped identity, no events.
        fin = await client.post(
            "/webhooks/claude_code",
            json={
                "finalize": True,
                "session_id": session_id,
                "device_id": device_id,
                "employee_id": EMPLOYEE,
            },
            headers={**internal_hdr, "X-Prbe-Customer": CUSTOMER},
        )
        assert fin.status_code == 200, fin.text
        source_event_id = fin.json()["source_event_id"]
        assert source_event_id == session_id, (
            "finalize must coalesce onto the session's live row, not open a new one"
        )

    await init_pool(settings)

    from engine.ingest.handlers.base import make_default_context
    from engine.ingest.normalizer import Normalizer

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT queue_id, payload_s3_keys FROM ingestion_queue "
            "WHERE customer_id = $1 AND source_event_id = $2",
            CUSTOMER,
            session_id,
        )
    assert row is not None, f"queue row missing for {session_id!r}"
    assert len(row["payload_s3_keys"]) == 2, (
        f"batch + finalize should coalesce into one row's array; got {row['payload_s3_keys']}"
    )

    ctx = make_default_context()
    try:
        normalizer = Normalizer(ctx)
        await normalizer.process_queue_row(
            queue_id=row["queue_id"],
            customer_id=CUSTOMER,
            source_system=SourceSystem.CLAUDE_CODE,
            source_event_id=session_id,
            payload_s3_keys=list(row["payload_s3_keys"]),
        )
    finally:
        await ctx.http.aclose()

    # The extractor must have been handed the real transcript, not the
    # finalize payload's empty event list.
    assert stub_extractor.get("session_id") == session_id, (
        "extraction never ran — the finalize did not mark the session complete"
    )
    assert len(stub_extractor["events"]) == 2, (
        f"extractor got the wrong events: {stub_extractor['events']}"
    )
    # Provenance must reach the prompt: CodexConnector inherits this whole path,
    # and every Codex session used to be introduced to the model as a Claude
    # Code one.
    assert stub_extractor["agent"] == "claude_code"

    async with raw_conn() as conn:
        docs = await conn.fetch(
            "SELECT doc_id, doc_type, parent_doc_id, metadata::text AS metadata "
            "FROM documents WHERE customer_id = $1 ORDER BY doc_type",
            CUSTOMER,
        )
    by_type = {d["doc_type"]: d for d in docs}

    session_doc = by_type.get("claude_code.session")
    assert session_doc is not None, "expected a session document"

    for unit_type in (
        "claude_code.qa",
        "claude_code.code_change",
        "claude_code.decision",
        "claude_code.file_ref",
    ):
        assert unit_type in by_type, (
            f"missing {unit_type}; a finalized session must yield every unit type. "
            f"Got: {sorted(by_type)}"
        )
        assert by_type[unit_type]["parent_doc_id"] == session_doc["doc_id"], (
            f"{unit_type} must hang off the session document — the graph writer "
            "navigates session->units via parent_doc_id, not edges"
        )

    import orjson

    decision_md = orjson.loads(by_type["claude_code.decision"]["metadata"])
    assert decision_md["chosen"] == "tighten"
    assert decision_md["options_considered"] == ["loosen", "tighten"]
    code_md = orjson.loads(by_type["claude_code.code_change"]["metadata"])
    assert code_md["summary"] == "Tighten the ingest payload typing"
    assert code_md["kind"] == "fix"
    assert code_md["files"] == ["app/schemas/ingest.py"]

    # Segment provenance reaches Postgres on EVERY unit type, so a long session's
    # units can be put back in order and located in the transcript rather than
    # read as a bag of free-floating facts.
    for unit_type in (
        "claude_code.qa",
        "claude_code.code_change",
        "claude_code.decision",
        "claude_code.file_ref",
    ):
        md = orjson.loads(by_type[unit_type]["metadata"])
        assert md["segment_index"] == 2, unit_type
        assert md["segment_count"] == 3, unit_type
        assert md["segment_boundary"] == "compaction", unit_type
        assert md["segment_start_line_no"] == 40, unit_type
        assert md["segment_end_line_no"] == 90, unit_type

    # And the session document says how many chapter breaks it has, so a reader
    # sees the shape before opening a single unit.
    session_md = orjson.loads(session_doc["metadata"])
    assert session_md["compaction_count"] == 0

    # The unit's full text must be chunked, not just its 200-char preview —
    # otherwise the extracted intent is stored but not retrievable.
    async with raw_conn() as conn:
        decision_body = await conn.fetchval(
            "SELECT string_agg(content, '' ORDER BY chunk_index) FROM chunks "
            "WHERE customer_id = $1 AND doc_id = $2 AND kind = 'content' "
            "AND valid_to IS NULL",
            CUSTOMER,
            by_type["claude_code.decision"]["doc_id"],
        )
    assert decision_body and "validation is the point" in decision_body, (
        f"decision rationale never reached the index; chunks: {decision_body!r}"
    )


@pytest.mark.asyncio
async def test_a_shrinking_re_extraction_retires_its_orphans(
    live_db: None, settings: Settings, monkeypatch
) -> None:
    """SCD2 handles a document that comes back CHANGED. It has nothing to say
    about one that stops coming back — nothing arrives to trigger it, so the row
    stays live forever.

    That is exactly the shape of re-derived children. A session's units are
    regenerated from scratch each time it is processed, and the count is not
    stable: an extraction yielding one decision where the last yielded three
    used to leave two behind, still searchable, still reading as current.
    """
    from engine.ingest.handlers.base import make_default_context
    from engine.ingest.normalizer import Normalizer
    from engine.shared import claude_code_extraction as ext
    from kb.handlers.claude_code import ClaudeCodeConnector

    customer = "retire-e2e-cust"
    session_id = f"sess-retire-{uuid.uuid4()}"

    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'retire', 'retire-hash') ON CONFLICT DO NOTHING",
            customer,
        )
    from engine.shared.storage import get_store

    store = get_store()
    await store.ensure_bucket(await store.bucket_for(customer))

    def _decision(i: int) -> ext.Decision:
        return ext.Decision(
            question=f"question {i}", options_considered=["a", "b"], chosen="a",
            rationale="r", serves="s",
            segment=ext.SegmentRef(index=1, total=1, boundary="session_start",
                                   position=i),
        )

    async def run_with(count: int) -> set[str]:
        async def fake_extract(**kwargs):
            return ext.UnitBundle(decision=[_decision(i) for i in range(count)])

        monkeypatch.setattr(
            "kb.handlers.claude_code._ext.extract_units_from_session", fake_extract
        )
        ctx = make_default_context()
        try:
            connector = ClaudeCodeConnector(ctx)
            event = WebhookEvent(
                customer_id=customer,
                source_system=SourceSystem.CLAUDE_CODE,
                source_event_id=session_id,
                received_at=datetime.now(UTC),
                payload_s3_key="",
                payload_s3_keys=[],
                raw_payload={"device_id": "d", "session_id": session_id,
                             "employee_id": "emp", "events": []},
                headers={},
            )
            result = await connector.normalize(event, {
                "session_id": session_id,
                "events": [{"line_no": 0, "raw": {"type": "user", "message":
                            {"content": "hi"}}}],
                "session_complete": True,
                "cwd": "/tmp",
                "employee_id": "emp",
            })
            await Normalizer(ctx)._persist(customer, SourceSystem.CLAUDE_CODE, result)
        finally:
            await ctx.http.aclose()

        async with raw_conn() as conn:
            rows = await conn.fetch(
                "SELECT doc_id FROM documents WHERE customer_id = $1 "
                "AND doc_type = 'claude_code.decision' AND valid_to IS NULL",
                customer,
            )
        return {r["doc_id"] for r in rows}

    three = await run_with(3)
    assert len(three) == 3

    one = await run_with(1)
    assert len(one) == 1, f"surplus decisions still live: {one}"
    assert one < three, "the surviving decision keeps its identity"


@pytest.mark.asyncio
async def test_a_resumed_session_does_not_re_extract(
    live_db: None, settings: Settings, monkeypatch
) -> None:
    """`payload_s3_keys` is append-only, which is right for batches and wrong
    for a signal. Leaving the finalize key in the array made every later batch
    of a RESUMED session look complete again and buy another full
    multi-segment extraction — ~4.6 model calls and ~219k input tokens a time,
    bounded only by how fast batches arrive.
    """
    customer = "resume-e2e-cust"
    session_id = f"sess-resume-{uuid.uuid4()}"
    device_id = f"dev-resume-{uuid.uuid4()}"
    calls = {"n": 0}

    async def fake_extract(**kwargs):
        from engine.shared import claude_code_extraction as ext
        calls["n"] += 1
        return ext.UnitBundle(qa=[ext.QA(prompt="p", outcome="o")])

    monkeypatch.setattr(
        "kb.handlers.claude_code._ext.extract_units_from_session", fake_extract
    )

    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'resume', 'resume-hash') ON CONFLICT DO NOTHING",
            customer,
        )
    from engine.shared.storage import get_store

    store = get_store()
    await store.ensure_bucket(await store.bucket_for(customer))

    await close_pool()
    transport = ASGITransport(app=app)
    hdr = {"X-Internal-Knowledge-Key": "test-internal-key", "X-Prbe-Customer": customer}
    internal = {"X-Internal-Knowledge-Key": "test-internal-key"}

    async def post_batch(seq: int) -> None:
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://t") as client,
            app.router.lifespan_context(app),
        ):
            await client.post("/webhooks/claude_code", json={
                "device_id": device_id, "session_id": session_id, "batch_seq": seq,
                "cwd": "/tmp", "employee_id": "emp",
                "events": [{"line_no": seq, "raw": {"type": "user", "message":
                            {"content": f"turn {seq}"}}}],
            }, headers=hdr)

    async def post_finalize() -> None:
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://t") as client,
            app.router.lifespan_context(app),
        ):
            await client.post("/webhooks/claude_code", json={
                "finalize": True, "session_id": session_id,
                "device_id": device_id, "employee_id": "emp",
            }, headers=hdr)

    async def drain() -> None:
        from engine.ingest.handlers.base import make_default_context
        from engine.ingest.normalizer import Normalizer

        async with raw_conn() as conn:
            row = await conn.fetchrow(
                "SELECT queue_id, payload_s3_keys FROM ingestion_queue "
                "WHERE customer_id = $1 AND source_event_id = $2",
                customer, session_id,
            )
        ctx = make_default_context()
        try:
            await Normalizer(ctx).process_queue_row(
                queue_id=row["queue_id"], customer_id=customer,
                source_system=SourceSystem.CLAUDE_CODE,
                source_event_id=session_id,
                payload_s3_keys=list(row["payload_s3_keys"]),
            )
        finally:
            await ctx.http.aclose()

    # register the device
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        await client.post("/api/devices/register", json={
            "customer_id": customer, "employee_id": "emp", "device_id": device_id,
            "token_hash": hashlib.sha256(b"x").hexdigest(),
            "os": "macos", "hostname": "h",
        }, headers=internal)

    await post_batch(0)
    await post_finalize()
    await init_pool(settings)
    await drain()
    assert calls["n"] == 1, "the finalize should mine the session once"

    # The session RESUMES: same id, a new batch. It must not be mined again
    # just because a finalize key from last time is still lying around.
    await close_pool()
    await post_batch(1)
    await init_pool(settings)
    await drain()
    assert calls["n"] == 1, (
        f"a resumed session re-extracted ({calls['n']} passes) — the finalize "
        "key was replayed"
    )
