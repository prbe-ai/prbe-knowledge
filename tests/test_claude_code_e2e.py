"""End-to-end: register device → POST /webhooks/claude_code (gateway shape) → worker → docs.

This test verifies the full Phase 1 path on the post-pivot architecture.
The daemon never reaches prbe-knowledge directly; the gateway forwards a
webhook with X-Internal-Knowledge-Key + X-Prbe-Customer, no bearer.
"""
from __future__ import annotations

import hashlib
import uuid

import httpx
import pytest
from httpx import ASGITransport

from services.ingestion.main import app
from shared.config import Settings, get_settings
from shared.constants import SourceSystem
from shared.db import close_pool, init_pool, raw_conn
from shared.embeddings import reset_embedder
from shared.storage import reset_store

CUSTOMER = "e2e-cust"
EMPLOYEE = "emp-e2e"


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


@pytest.mark.asyncio
async def test_e2e_register_then_gateway_forwarded_webhook_then_docs(
    live_db: None, settings: Settings
) -> None:
    device_id = f"dev-e2e-{uuid.uuid4()}"
    plaintext = f"secret-{uuid.uuid4()}"
    token_hash = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()

    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'e2e', 'e2e-hash') ON CONFLICT DO NOTHING",
            CUSTOMER,
        )

    from shared.storage import get_store

    store = get_store()
    await store.ensure_bucket(store.bucket_for(CUSTOMER))

    source_event_id: str = ""
    await close_pool()
    transport = ASGITransport(app=app)
    internal_hdr = {"X-Internal-Knowledge-Key": "test-internal-key"}
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        # 1. Register device via the internal API (gateway pretends to call this).
        reg = await client.post(
            "/api/devices/register",
            json={
                "customer_id": CUSTOMER,
                "employee_id": EMPLOYEE,
                "device_id": device_id,
                "token_hash": token_hash,
                "os": "macos",
                "hostname": "e2e-host",
            },
            headers=internal_hdr,
        )
        assert reg.status_code == 200, reg.text

        # 2. Gateway-forwarded webhook: X-Internal-Knowledge-Key + X-Prbe-Customer,
        # no client bearer.
        webhook_body = {
            "device_id": device_id,
            "session_id": "e2e-sess-1",
            "batch_seq": 0,
            "cwd": "/tmp/e2e",
            "employee_id": EMPLOYEE,
            "events": [{"line_no": 0, "raw": {"type": "user_prompt", "content": "hi from e2e"}}],
        }
        wh = await client.post(
            "/webhooks/claude_code",
            json=webhook_body,
            headers={**internal_hdr, "X-Prbe-Customer": CUSTOMER},
        )
        assert wh.status_code == 200, wh.text
        source_event_id = wh.json()["source_event_id"]

    await init_pool(settings)

    # 3. Drive the worker for this queue row directly (matches the pattern
    # used by test_session_completer / test_idempotency).
    from services.ingestion.handlers.base import make_default_context
    from services.ingestion.normalizer import Normalizer

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT queue_id, payload_s3_key FROM ingestion_queue WHERE source_event_id = $1",
            source_event_id,
        )
    assert row is not None, f"queue row missing for {source_event_id!r}"

    ctx = make_default_context()
    try:
        normalizer = Normalizer(ctx)
        outcome = await normalizer.process_queue_row(
            queue_id=row["queue_id"],
            customer_id=CUSTOMER,
            source_system=SourceSystem.CLAUDE_CODE,
            source_event_id=source_event_id,
            payload_s3_key=row["payload_s3_key"],
        )
    finally:
        await ctx.http.aclose()

    assert outcome.doc_ids, "normalizer produced no doc_ids"

    # 4. Confirm a claude_code.session Document was persisted AND the live
    # webhook event content actually flowed end-to-end (not just a stub doc).
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT doc_id, doc_type, metadata::text AS metadata, body_size_bytes "
            "FROM documents WHERE customer_id = $1 AND doc_type = 'claude_code.session'",
            CUSTOMER,
        )
    assert rows, "expected a claude_code.session document"
    session_row = rows[0]
    import orjson as _orjson

    meta = _orjson.loads(session_row["metadata"])
    assert meta["event_count"] >= 1, (
        f"session doc was emitted with no events — fetch_supplementary likely "
        f"never picked up the gateway-forwarded body. metadata={meta}"
    )
    assert meta.get("body"), (
        "session doc has no metadata['body'] — chunker won't see the transcript"
    )
    assert "hi from e2e" in meta["body"], (
        f"event content should land in metadata['body']; got: {meta['body']!r}"
    )
