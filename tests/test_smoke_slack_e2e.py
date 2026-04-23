"""Tier 3 gate: end-to-end ingestion → retrieval round-trip.

Flow:
  1. Insert a customer row.
  2. Ensure the tenant R2 bucket exists in MinIO.
  3. POST a signed Slack fixture webhook → verify 202/200 and queue row.
  4. Drive one Normalizer cycle → document + chunks + graph + ACL land.
  5. Hit /query and assert the ingested message is returned.

Gates the Phase 0 architectural pattern. If this passes, Tier 4 (parallel
handlers) is unblocked.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from shared.config import Settings, get_settings
from shared.constants import SourceSystem
from shared.db import raw_conn
from shared.embeddings import reset_embedder
from shared.storage import reset_store

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "slack"
CUSTOMER_ID = "cust-smoke"


def _signed_headers(body: bytes, secret: str) -> dict[str, str]:
    ts = str(int(time.time()))
    sig = (
        "v0="
        + hmac.new(
            secret.encode(), f"v0:{ts}:".encode() + body, hashlib.sha256
        ).hexdigest()
    )
    return {
        "content-type": "application/json",
        "x-prbe-customer": CUSTOMER_ID,
        "x-slack-request-timestamp": ts,
        "x-slack-signature": sig,
    }


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    # Set via env vars so pydantic-settings re-reads on construction. Patching
    # `shared.config.get_settings` alone doesn't propagate — other modules
    # bound the reference at import time.
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_slack_webhook_to_query(live_db, settings: Settings) -> None:
    # 1. customer row so FK constraints pass
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'smoke-test', 'dummy')
            ON CONFLICT DO NOTHING
            """,
            CUSTOMER_ID,
        )

    # 2. ensure bucket
    from shared.storage import get_store

    store = get_store()
    bucket = store.bucket_for(CUSTOMER_ID)
    await store.ensure_bucket(bucket)

    # 3. POST the signed Slack fixture via the FastAPI TestClient
    fixture = json.loads((FIXTURE_DIR / "message_simple.json").read_text())
    body = json.dumps(fixture).encode()

    from services.ingestion.main import app as ingestion_app

    # ASGITransport runs the app in-process on our asyncio loop — TestClient
    # would spin its own loop and collide with our shared asyncpg pool.
    # The ingestion app's lifespan re-initializes the pool; we tear ours down
    # first and recreate after, so the app owns the pool for its span.
    from shared.db import close_pool, init_pool

    await close_pool()
    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        resp = await client.post(
            "/webhooks/slack",
            content=body,
            headers=_signed_headers(body, "test-secret"),
        )
    assert resp.status_code == 200, resp.text
    body_json = resp.json()
    assert body_json["status"] in {"accepted", "duplicate"}
    source_event_id = body_json["source_event_id"]

    # Bring our test-owned pool back up for the assertions below.
    await init_pool(settings)

    # 4. One normalizer pass
    from services.ingestion.handlers.base import make_default_context
    from services.ingestion.normalizer import Normalizer

    ctx = make_default_context()
    try:
        async with raw_conn() as conn:
            row = await conn.fetchrow(
                "SELECT queue_id, payload_s3_key FROM ingestion_queue WHERE source_event_id=$1",
                source_event_id,
            )
        assert row is not None, "queue row missing"

        normalizer = Normalizer(ctx)
        outcome = await normalizer.process_queue_row(
            queue_id=row["queue_id"],
            customer_id=CUSTOMER_ID,
            source_system=SourceSystem.SLACK,
            source_event_id=source_event_id,
            payload_s3_key=row["payload_s3_key"],
        )
        assert outcome.doc_ids, "normalize produced no docs"
        assert outcome.chunk_count >= 1
    finally:
        await ctx.http.aclose()

    # 5. Verify docs + chunks + ACL + graph landed
    async with raw_conn() as conn:
        doc_count = await conn.fetchval(
            "SELECT count(*) FROM documents WHERE customer_id=$1", CUSTOMER_ID
        )
        chunk_count = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE customer_id=$1", CUSTOMER_ID
        )
        acl_count = await conn.fetchval(
            "SELECT count(*) FROM acl_snapshots WHERE customer_id=$1", CUSTOMER_ID
        )
        node_count = await conn.fetchval(
            "SELECT count(*) FROM graph_nodes WHERE customer_id=$1", CUSTOMER_ID
        )
        edge_count = await conn.fetchval(
            "SELECT count(*) FROM graph_edges WHERE customer_id=$1", CUSTOMER_ID
        )
    assert doc_count == 1
    assert chunk_count >= 1
    assert acl_count >= 1
    assert node_count >= 3  # channel, person, document
    assert edge_count >= 2  # member_of, authored

    # 6. /query returns the ingested chunk
    from services.retrieval.main import app as retrieval_app

    await close_pool()
    rtransport = ASGITransport(app=retrieval_app)
    async with (
        httpx.AsyncClient(transport=rtransport, base_url="http://t") as client,
        retrieval_app.router.lifespan_context(retrieval_app),
    ):
        qresp = await client.post(
            "/query",
            json={
                "query": "payments 500s deploy",
                "customer_id": CUSTOMER_ID,
                "top_k": 5,
            },
        )
    assert qresp.status_code == 200, qresp.text
    chunks = qresp.json()["chunks"]
    assert chunks, "retrieval returned no chunks"
    assert any("payments" in c["content"].lower() for c in chunks)
    await init_pool(settings)


# Keep asyncio imported so pytest-asyncio reloads cleanly in IDE plugins.
_ = asyncio
_ = httpx
