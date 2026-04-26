"""Verify the session completer enqueues a synthetic 'finalize' event for any
claude_code session whose latest batch is older than the idle threshold and
whose finalize event hasn't already run.
"""
from __future__ import annotations

import pytest

from services.ingestion.session_completer import enqueue_idle_session_finalizers
from shared.db import get_pool


@pytest.mark.asyncio
async def test_idle_session_gets_finalize_enqueued(live_db: None) -> None:
    customer = "completer-test-cust"
    async with get_pool().acquire() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) VALUES ($1, 'c', 'c-hash') ON CONFLICT DO NOTHING",
            customer,
        )
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)

        # An idle batch (10 minutes old)
        await conn.execute("""
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id, payload_s3_key, status, enqueued_at)
            VALUES ($1, 'claude_code', 'sess-idle:0', 'k', 'done', NOW() - INTERVAL '10 minutes')
        """, customer)

        # A fresh batch (1 minute old) — should NOT be finalized
        await conn.execute("""
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id, payload_s3_key, status, enqueued_at)
            VALUES ($1, 'claude_code', 'sess-fresh:0', 'k', 'done', NOW() - INTERVAL '1 minute')
        """, customer)

    n = await enqueue_idle_session_finalizers(idle_minutes=5)
    assert n == 1, f"expected exactly one finalize enqueue, got {n}"

    async with get_pool().acquire() as conn:
        rows = await conn.fetch("""
            SELECT source_event_id FROM ingestion_queue
            WHERE customer_id = $1
              AND source_event_id LIKE '%:finalize'
            ORDER BY source_event_id
        """, customer)
        ids = [r["source_event_id"] for r in rows]
        assert "sess-idle:finalize" in ids
        assert "sess-fresh:finalize" not in ids

        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)


@pytest.mark.asyncio
async def test_finalize_event_processes_through_normalizer(live_db: None, monkeypatch) -> None:
    """End-to-end check that the cron-enqueued finalize event flows through
    parse_webhook_event → fetch_supplementary → normalize without DLQing."""
    import orjson

    from services.ingestion.handlers.base import make_default_context
    from services.ingestion.normalizer import Normalizer
    from services.ingestion.session_completer import enqueue_idle_session_finalizers
    from shared import claude_code_extraction as _ext
    from shared.constants import SourceSystem
    from shared.customer_mapping import record_mapping
    from shared.models import IntegrationToken
    from shared.storage import get_store
    from shared.tokens import save_device_token

    customer = "completer-e2e-cust"
    session_id = "sess-e2e-final"
    employee_id = "emp-final-e2e"

    # Monkeypatch extract_units_from_session so the test doesn't need Anthropic.
    from shared.claude_code_extraction import UnitBundle

    async def _noop_extract(**kwargs):  # type: ignore[no-untyped-def]
        return UnitBundle(qa=[], code_change=[], decision=[], file_ref=[])

    monkeypatch.setattr(_ext, "extract_units_from_session", _noop_extract)

    async with get_pool().acquire() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'cf', 'cf-hash') ON CONFLICT DO NOTHING",
            customer,
        )

    # Pre-stage a device token so the normalizer's load_token call doesn't fail
    # (it returns None for device-scoped sources, which is acceptable).
    # Also record the source mapping so fetch_supplementary can find the session.
    await save_device_token(IntegrationToken(
        customer_id=customer,
        source_system=SourceSystem.CLAUDE_CODE,
        access_token="x",
        webhook_secret="test-secret-hash",
        device_id="cron-finalize-device",
        device_metadata={"hostname": "h"},
    ))
    await record_mapping(
        customer_id=customer,
        source_system=SourceSystem.CLAUDE_CODE,
        external_id="cron-finalize-device",
        external_name="h",
        metadata={},
    )

    # Pre-stage one R2 batch so fetch_supplementary has events to read.
    store = get_store()
    bucket = store.bucket_for(customer)
    await store.ensure_bucket(bucket)
    batch_line = orjson.dumps({
        "line_no": 0,
        "employee_id": employee_id,
        "raw": {"role": "user", "content": "finalize test prompt"},
    })
    await store.put(
        bucket,
        f"raw/claude_code/{customer}/{session_id}/0.jsonl",
        batch_line + b"\n",
    )

    # Pre-insert a done batch row so the cron finds it as idle.
    async with get_pool().acquire() as conn:
        await conn.execute("""
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id, payload_s3_key, status, enqueued_at)
            VALUES ($1, 'claude_code', $2, $3, 'done', NOW() - INTERVAL '10 minutes')
        """, customer, f"{session_id}:0",
            f"raw/claude_code/{customer}/{session_id}/0.jsonl")

    # Trigger the cron.
    n = await enqueue_idle_session_finalizers(idle_minutes=5)
    assert n >= 1, f"expected at least one finalize enqueue, got {n}"

    # Find the finalize queue row.
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT queue_id, source_event_id, payload_s3_key
            FROM ingestion_queue
            WHERE customer_id = $1
              AND source_event_id = $2
            """,
            customer,
            f"{session_id}:finalize",
        )
    assert row is not None, "finalize queue row was not enqueued"

    # Drive the normalizer on the finalize row.
    ctx = make_default_context()
    try:
        normalizer = Normalizer(ctx)
        outcome = await normalizer.process_queue_row(
            queue_id=row["queue_id"],
            customer_id=customer,
            source_system=SourceSystem.CLAUDE_CODE,
            source_event_id=row["source_event_id"],
            payload_s3_key=row["payload_s3_key"],
        )
    finally:
        await ctx.http.aclose()

    assert outcome.doc_ids, "normalizer produced no doc_ids — DLQ would have fired"

    # Confirm a session document was created.
    async with get_pool().acquire() as conn:
        doc_rows = await conn.fetch(
            "SELECT doc_type FROM documents WHERE customer_id = $1",
            customer,
        )
    types = {r["doc_type"] for r in doc_rows}
    assert "claude_code.session" in types, (
        f"expected claude_code.session doc, found: {types}"
    )
