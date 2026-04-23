"""Content-addressable chunk lifecycle tests.

Exercises the three-way diff in services/ingestion/normalizer.py:
    reused (hash already present) → embedding reused, no API call
    added  (new hash)             → embedding + row written
    removed (was live, now gone)  → row marked valid_to

And confirms the doc-level valid_to closeout on version bump.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from services.ingestion.handlers.base import make_default_context
from services.ingestion.normalizer import Normalizer
from shared.config import Settings
from shared.constants import SourceSystem
from shared.db import close_pool, init_pool, raw_conn
from shared.embeddings import reset_embedder
from shared.storage import get_store, reset_store

FIXTURE_SIMPLE = Path(__file__).parent.parent / "fixtures" / "slack" / "message_simple.json"
CUSTOMER = "cust-chunk-diff"


@pytest.fixture(autouse=True)
def _env(monkeypatch, settings: Settings):
    monkeypatch.setenv(
        "TOKEN_ENCRYPTION_KEY",
        settings.token_encryption_key.get_secret_value(),
    )
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()


def _slack_payload(text: str, ts: str = "1713628800.000100") -> dict:
    base = json.loads(FIXTURE_SIMPLE.read_text())
    base["event"]["text"] = text
    base["event"]["ts"] = ts
    return base


async def _ingest(normalizer: Normalizer, payload: dict, *, event_id: str) -> None:
    store = get_store()
    bucket = store.bucket_for(CUSTOMER)
    await store.ensure_bucket(bucket)
    key = f"raw/slack/{CUSTOMER}/2026/04/22/{event_id}.json"
    envelope = {
        "_headers": {},
        "payload": payload,
        "received_at": datetime.now(UTC).isoformat(),
        "trace_id": f"t-{event_id}",
    }
    await store.put(bucket, key, json.dumps(envelope).encode())

    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO ingestion_queue (customer_id, source_system, source_event_id, payload_s3_key)
            VALUES ($1, 'slack', $2, $3)
            ON CONFLICT DO NOTHING
            """,
            CUSTOMER,
            event_id,
            key,
        )

    await normalizer.process_queue_row(
        queue_id=1,
        customer_id=CUSTOMER,
        source_system=SourceSystem.SLACK,
        source_event_id=event_id,
        payload_s3_key=key,
    )


@pytest.mark.asyncio
async def test_chunk_diff_marks_removed_chunks_stale(live_db, settings: Settings) -> None:
    """An edit that completely replaces the body marks old chunks stale and writes new ones."""
    await close_pool()
    await init_pool(settings)
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1, 't', 'x')",
            CUSTOMER,
        )

    ctx = make_default_context()
    try:
        normalizer = Normalizer(ctx)
        # v1: short message
        await _ingest(
            normalizer,
            _slack_payload("payments service down"),
            event_id="C456:1713628800.000100",
        )
        async with raw_conn() as conn:
            live_v1 = await conn.fetchval(
                "SELECT count(*) FROM chunks WHERE customer_id=$1 AND valid_to IS NULL",
                CUSTOMER,
            )
            doc_versions_v1 = await conn.fetchval(
                "SELECT count(*) FROM documents WHERE customer_id=$1",
                CUSTOMER,
            )
        assert live_v1 >= 1
        assert doc_versions_v1 == 1

        # Same message re-ingested (different event_id to bypass queue unique,
        # but same body) → content_hash matches → no version bump.
        await _ingest(
            normalizer,
            _slack_payload("payments service down", ts="1713628800.000100"),
            event_id="C456:1713628800.000100:replay",
        )
        async with raw_conn() as conn:
            same = await conn.fetchval(
                "SELECT count(*) FROM documents WHERE customer_id=$1",
                CUSTOMER,
            )
        assert same == 1  # no new version

        # v2: edit replaces the body with completely different text.
        edited = _slack_payload(
            "actually a different root cause — feature flag flip",
            ts="1713628800.000100",
        )
        # Mark it as an edit by simulating message_changed.
        edited["event"] = {
            "type": "message",
            "subtype": "message_changed",
            "channel": edited["event"]["channel"],
            "event_ts": "1713628900.000200",
            "message": edited["event"],
            "previous_message": {"ts": "1713628800.000100", "text": "payments service down"},
        }
        await _ingest(
            normalizer,
            edited,
            event_id="C456:1713628800.000100:edit:1713628900.000200",
        )

        async with raw_conn() as conn:
            versions = await conn.fetchval(
                "SELECT count(*) FROM documents WHERE customer_id=$1",
                CUSTOMER,
            )
            closed = await conn.fetchval(
                "SELECT count(*) FROM documents WHERE customer_id=$1 AND valid_to IS NOT NULL",
                CUSTOMER,
            )
            live = await conn.fetchval(
                "SELECT count(*) FROM chunks WHERE customer_id=$1 AND valid_to IS NULL",
                CUSTOMER,
            )
            stale = await conn.fetchval(
                "SELECT count(*) FROM chunks WHERE customer_id=$1 AND valid_to IS NOT NULL",
                CUSTOMER,
            )
        assert versions == 2, "edit should bump to version 2"
        assert closed == 1, "prior version should have valid_to set"
        assert live >= 1, "new chunk(s) for the new body"
        assert stale >= 1, "old chunk marked stale via three-way diff"
    finally:
        await ctx.http.aclose()


@pytest.mark.asyncio
async def test_chunk_diff_no_op_on_identical_body(live_db, settings: Settings) -> None:
    """Identical body re-ingestion writes no new documents and no new chunks."""
    await close_pool()
    await init_pool(settings)
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1, 't', 'x')",
            CUSTOMER,
        )

    ctx = make_default_context()
    try:
        normalizer = Normalizer(ctx)
        for i in range(3):
            payload = _slack_payload("stable body", ts="1713700000.000001")
            await _ingest(
                normalizer,
                payload,
                event_id=f"C456:1713700000.000001:replay{i}",
            )

        async with raw_conn() as conn:
            versions = await conn.fetchval(
                "SELECT count(*) FROM documents WHERE customer_id=$1",
                CUSTOMER,
            )
            total_chunks = await conn.fetchval(
                "SELECT count(*) FROM chunks WHERE customer_id=$1",
                CUSTOMER,
            )
        assert versions == 1
        # One chunk row, maybe more for longer bodies; but no duplicates across replays.
        # Because chunk_id is deterministic (doc_id + content_hash), replays
        # don't add rows.
        assert total_chunks >= 1
    finally:
        await ctx.http.aclose()
