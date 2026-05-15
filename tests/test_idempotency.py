"""Replaying the same signed webhook N times must produce exactly one document
version and one set of chunks. UNIQUE(customer_id, source_system, source_event_id)
on ingestion_queue + content_hash dedup in documents combine to guarantee this.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from shared.config import Settings, get_settings
from shared.constants import SourceSystem
from shared.db import close_pool, init_pool, raw_conn
from shared.embeddings import reset_embedder
from shared.encryption import encrypt_token
from shared.storage import reset_store

FIXTURE = Path(__file__).parent.parent / "fixtures" / "slack" / "message_simple.json"
CUSTOMER = "cust-idempotent"
SECRET = "test-secret"


def _signed(body: bytes) -> dict[str, str]:
    """Internal-only headers — gateway is the sole sig verifier in prod, so
    this service trusts X-Internal-Knowledge-Key + X-Prbe-Customer."""
    return {
        "content-type": "application/json",
        "x-internal-knowledge-key": INTERNAL_KEY,
        "x-prbe-customer": CUSTOMER,
    }


INTERNAL_KEY = "test-internal-key-32bytes-padding-padding"


@pytest.fixture(autouse=True)
def _patch(monkeypatch, settings: Settings):
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", INTERNAL_KEY)
    monkeypatch.setenv("ENVIRONMENT", "local")
    get_settings.cache_clear()  # type: ignore[attr-defined]
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_replay_produces_one_doc(live_db, settings: Settings) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1, 't', 'x') ON CONFLICT DO NOTHING",
            CUSTOMER,
        )
        # The pre-enqueue connectedness gate (services/ingestion/connectedness.py)
        # requires an active integration_tokens row for OAuth sources before
        # _enqueue will write to ingestion_queue. Seed slack as "connected"
        # so the webhook replay actually lands rows.
        await conn.execute(
            """
            INSERT INTO integration_tokens
                (customer_id, source_system, access_token_encrypted, status)
            VALUES ($1, $2, $3, 'active')
            ON CONFLICT (customer_id, source_system) DO NOTHING
            """,
            CUSTOMER,
            SourceSystem.SLACK.value,
            encrypt_token("xoxb-test"),
        )

    from shared.storage import get_store

    store = get_store()
    await store.ensure_bucket(await store.bucket_for(CUSTOMER))

    body = json.dumps(json.loads(FIXTURE.read_text())).encode()
    from services.ingestion.main import app as ingestion_app

    await close_pool()
    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        for _ in range(10):
            resp = await client.post("/webhooks/slack", content=body, headers=_signed(body))
            assert resp.status_code == 200
    await init_pool(settings)

    async with raw_conn() as conn:
        queue_count = await conn.fetchval(
            "SELECT count(*) FROM ingestion_queue WHERE customer_id=$1", CUSTOMER
        )
    assert queue_count == 1  # UNIQUE blocked 9 redeliveries

    # Now drive the single queue row through the normalizer 10 times — doc
    # version stays at 1 because content_hash matches.
    from services.ingestion.handlers.base import make_default_context
    from services.ingestion.normalizer import Normalizer

    ctx = make_default_context()
    try:
        async with raw_conn() as conn:
            row = await conn.fetchrow(
                "SELECT queue_id, source_event_id, payload_s3_key FROM ingestion_queue WHERE customer_id=$1",
                CUSTOMER,
            )
        normalizer = Normalizer(ctx)

        for _ in range(10):
            await normalizer.process_queue_row(
                queue_id=row["queue_id"],
                customer_id=CUSTOMER,
                source_system=SourceSystem.SLACK,
                source_event_id=row["source_event_id"],
                payload_s3_keys=[row["payload_s3_key"]],
            )
    finally:
        await ctx.http.aclose()

    async with raw_conn() as conn:
        doc_versions = await conn.fetchval(
            "SELECT count(*) FROM documents WHERE customer_id=$1", CUSTOMER
        )
    assert doc_versions == 1
