"""Verify the session completer cron upserts a finalize.marker into the
live session row's payload_s3_keys array (post-coalescing semantics).

Pre-coalescing the cron inserted a separate `<session>:finalize` queue
row. Post-coalescing (migration 0026) finalize is just another payload
keyed under the same session_id, appended to the same row's array via
the same UPSERT path the live ingestion uses. The worker detects
`finalize.marker` in the array and forces session_complete=True.
"""
from __future__ import annotations

import pytest

from services.ingestion.session_completer import enqueue_idle_session_finalizers
from shared.db import get_pool


@pytest.mark.asyncio
async def test_idle_session_gets_finalize_marker_appended(live_db: None) -> None:
    """Cron upserts the finalize.marker key into the existing live session row."""
    customer = "completer-test-cust"
    session_id = "sess-idle"
    live_key = f"raw/claude_code/{customer}/2026/04/29/{session_id}:0.json"

    async with get_pool().acquire() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'c', 'c-hash') ON CONFLICT DO NOTHING",
            customer,
        )
        await conn.execute(
            "DELETE FROM ingestion_queue WHERE customer_id = $1", customer
        )
        # An idle live session row (10 minutes old) using the new coalescing
        # shape: source_event_id is the bare session_id, payload_s3_keys is
        # the array of batch keys.
        await conn.execute(
            """
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id,
                 payload_s3_key, payload_s3_keys, status, enqueued_at,
                 priority, version)
            VALUES ($1, 'claude_code', $2, $3, ARRAY[$3], 'done',
                    NOW() - INTERVAL '10 minutes', 75, 1)
            """,
            customer, session_id, live_key,
        )
        # A fresh session (1 minute old) — should NOT be finalized.
        fresh_key = f"raw/claude_code/{customer}/2026/04/29/sess-fresh:0.json"
        await conn.execute(
            """
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id,
                 payload_s3_key, payload_s3_keys, status, enqueued_at,
                 priority, version)
            VALUES ($1, 'claude_code', 'sess-fresh', $2, ARRAY[$2], 'done',
                    NOW() - INTERVAL '1 minute', 75, 1)
            """,
            customer, fresh_key,
        )

    n = await enqueue_idle_session_finalizers(idle_minutes=5)
    assert n == 1, f"expected exactly one finalize enqueue, got {n}"

    async with get_pool().acquire() as conn:
        idle_row = await conn.fetchrow(
            """
            SELECT payload_s3_keys, status, version
            FROM ingestion_queue
            WHERE customer_id = $1 AND source_event_id = $2
            """,
            customer, session_id,
        )
        fresh_row = await conn.fetchrow(
            """
            SELECT payload_s3_keys, status, version
            FROM ingestion_queue
            WHERE customer_id = $1 AND source_event_id = 'sess-fresh'
            """,
            customer,
        )

        await conn.execute(
            "DELETE FROM ingestion_queue WHERE customer_id = $1", customer
        )

    # Idle session: live row got the finalize.marker appended, status reset
    # to pending so the worker re-claims, version bumped.
    assert idle_row is not None
    assert idle_row["status"] == "pending", "idle row should be re-marked pending"
    assert idle_row["version"] == 2, f"version should be bumped from 1 to 2, got {idle_row['version']}"
    assert len(idle_row["payload_s3_keys"]) == 2
    assert any(k.endswith("/finalize.marker") for k in idle_row["payload_s3_keys"]), (
        f"expected finalize.marker in payload_s3_keys, got {idle_row['payload_s3_keys']}"
    )
    assert live_key in idle_row["payload_s3_keys"], (
        "original live batch key must be preserved alongside the marker"
    )

    # Fresh session: untouched.
    assert fresh_row is not None
    assert fresh_row["status"] == "done"
    assert fresh_row["version"] == 1
    assert len(fresh_row["payload_s3_keys"]) == 1
    assert not any(
        k.endswith("/finalize.marker") for k in fresh_row["payload_s3_keys"]
    )


@pytest.mark.asyncio
async def test_finalize_event_processes_through_normalizer(
    live_db: None, monkeypatch
) -> None:
    """End-to-end: cron-injected finalize.marker is detected by
    fetch_supplementary, session_complete=True triggers unit extraction,
    no DLQ.
    """
    import orjson

    from services.ingestion.handlers.base import make_default_context
    from services.ingestion.normalizer import Normalizer
    from shared import claude_code_extraction as _ext
    from shared.claude_code_extraction import UnitBundle
    from shared.constants import SourceSystem
    from shared.customer_mapping import record_mapping
    from shared.models import IntegrationToken
    from shared.storage import get_store
    from shared.tokens import save_device_token

    customer = "completer-e2e-cust"
    session_id = "sess-e2e-final"
    employee_id = "emp-final-e2e"

    async def _noop_extract(**kwargs):  # type: ignore[no-untyped-def]
        return UnitBundle(qa=[], code_change=[], decision=[], file_ref=[])

    monkeypatch.setattr(_ext, "extract_units_from_session", _noop_extract)

    async with get_pool().acquire() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'cf', 'cf-hash') ON CONFLICT DO NOTHING",
            customer,
        )
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

    # Stage one live batch envelope at the date-partitioned key. After the
    # cron runs, payload_s3_keys will be [live_key, finalize.marker] and
    # fetch_supplementary will read both.
    store = get_store()
    bucket = await store.bucket_for(customer)
    await store.ensure_bucket(bucket)
    live_key = f"raw/claude_code/{customer}/2026/04/29/{session_id}:0.json"
    live_envelope = orjson.dumps({
        "_headers": {},
        "payload": {
            "device_id": "cron-finalize-device",
            "session_id": session_id,
            "batch_seq": 0,
            "cwd": None,
            "events": [{
                "line_no": 0,
                "employee_id": employee_id,
                "raw": {"role": "user", "content": "finalize test prompt"},
            }],
        },
    })
    await store.put(bucket, live_key, live_envelope)

    # Insert the live session row (idle for 10 minutes).
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id,
                 payload_s3_key, payload_s3_keys, status, enqueued_at,
                 priority, version)
            VALUES ($1, 'claude_code', $2, $3, ARRAY[$3], 'done',
                    NOW() - INTERVAL '10 minutes', 75, 1)
            """,
            customer, session_id, live_key,
        )

    n = await enqueue_idle_session_finalizers(idle_minutes=5)
    assert n >= 1, f"expected at least one finalize enqueue, got {n}"

    # The cron upserted the marker into the same row — find it.
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT queue_id, source_event_id, payload_s3_key, payload_s3_keys
            FROM ingestion_queue
            WHERE customer_id = $1 AND source_event_id = $2
            """,
            customer, session_id,
        )
    assert row is not None
    assert any(
        k.endswith("/finalize.marker") for k in row["payload_s3_keys"]
    ), "finalize.marker not appended"
    assert live_key in row["payload_s3_keys"]

    # Drive the normalizer with the full coalesced array.
    ctx = make_default_context()
    try:
        normalizer = Normalizer(ctx)
        outcome = await normalizer.process_queue_row(
            queue_id=row["queue_id"],
            customer_id=customer,
            source_system=SourceSystem.CLAUDE_CODE,
            source_event_id=row["source_event_id"],
            payload_s3_keys=list(row["payload_s3_keys"]),
        )
    finally:
        await ctx.http.aclose()

    assert outcome.doc_ids, "normalizer produced no doc_ids — DLQ would have fired"

    async with get_pool().acquire() as conn:
        doc_rows = await conn.fetch(
            "SELECT doc_type FROM documents WHERE customer_id = $1",
            customer,
        )
    types = {r["doc_type"] for r in doc_rows}
    assert "claude_code.session" in types, (
        f"expected claude_code.session doc, found: {types}"
    )
