"""Worker hook: dispatch fires only when requires_investigation, only
after successful normalize, only once per logical incident."""
from __future__ import annotations

import os
import uuid
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from services.investigation.dispatch import DispatchExhausted


pytestmark = pytest.mark.asyncio


def _new_customer_id() -> str:
    return f"worker-dispatch-{uuid.uuid4().hex[:8]}"


async def _seed_customer_and_incident_doc(
    dsn: str, customer_id: str, incident_doc_id: str,
) -> None:
    """Seed a customer + a live INCIDENT document we can dispatch on."""
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash, r2_bucket) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (customer_id) DO NOTHING",
            customer_id, customer_id, "h", f"b-{customer_id}",
        )
        # We just need a documents row matching the doc_id the worker reads.
        # Use a minimal shape — the worker only reads title/metadata/source_system/created_at.
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id, source_system, source_id, source_url,
                doc_class, doc_type, content_type, content_hash, title,
                body_size_bytes, body_token_count, created_at, updated_at,
                valid_from, ingested_at, acl, metadata
            )
            VALUES (
                $1, 1, $2, 'pagerduty', 'PD-INC-001', '', 'raw_source', 'incident',
                'text/markdown', 'h', 'Test', 0, 0,
                '2026-05-17T12:00:00Z', '2026-05-17T12:00:00Z',
                '2026-05-17T12:00:00Z', '2026-05-17T12:00:00Z',
                '{"principals": [], "captured_at": "2026-05-17T12:00:00Z"}'::jsonb,
                '{"service_id": "PSRV001", "urgency": "high", "priority": "P1"}'::jsonb
            )
            """,
            incident_doc_id, customer_id,
        )
    finally:
        await conn.close()


async def _cleanup(dsn: str, customer_id: str) -> None:
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute("DELETE FROM documents WHERE customer_id = $1", customer_id)
        await conn.execute("DELETE FROM customers WHERE customer_id = $1", customer_id)
    finally:
        await conn.close()


@pytest.fixture
def _dsn() -> str:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL not set")
    return dsn


async def test_maybe_dispatch_fires_when_requested(_dsn) -> None:
    """End-to-end: call _maybe_dispatch_investigation with a real
    incident doc; mock the HTTP call; verify the payload shape."""
    from services.ingestion.worker import Worker
    from shared.constants import SourceSystem
    from shared.db import close_pool, init_pool

    cid = _new_customer_id()
    doc_id = "pd:incident:F4-001"
    await close_pool()
    await init_pool()
    try:
        await _seed_customer_and_incident_doc(_dsn, cid, doc_id)
        with patch(
            "services.investigation.dispatch.dispatch_investigation",
            new_callable=AsyncMock,
        ) as fake_dispatch:
            # Construct a Worker just enough to call the helper.
            worker = Worker.__new__(Worker)
            await worker._maybe_dispatch_investigation(
                customer_id=cid,
                incident_doc_id=doc_id,
                source=SourceSystem.PAGERDUTY,
                source_event_id="pd:incident:F4-001:incident.triggered",
            )
        fake_dispatch.assert_awaited_once()
        payload = fake_dispatch.await_args.args[0]
        assert payload["customer_id"] == cid
        assert payload["incident_doc_id"] == doc_id
        assert payload["source"] == "pagerduty"
        assert payload["version"] == 1
        assert payload["incident_signals"]["service"] == "PSRV001"
        assert payload["incident_signals"]["urgency"] == "high"
        assert payload["incident_signals"]["severity"] == "P1"
    finally:
        await _cleanup(_dsn, cid)
        await close_pool()


async def test_maybe_dispatch_marks_failed_on_retry_exhaustion(_dsn) -> None:
    """When dispatch_investigation raises DispatchExhausted, the
    helper stamps metadata.investigation_dispatch_failed=true on the
    INCIDENT doc."""
    from services.ingestion.worker import Worker
    from shared.constants import SourceSystem
    from shared.db import close_pool, init_pool

    cid = _new_customer_id()
    doc_id = "pd:incident:F4-002"
    await close_pool()
    await init_pool()
    try:
        await _seed_customer_and_incident_doc(_dsn, cid, doc_id)
        with patch(
            "services.investigation.dispatch.dispatch_investigation",
            new_callable=AsyncMock,
            side_effect=DispatchExhausted("orchestrator down"),
        ):
            worker = Worker.__new__(Worker)
            await worker._maybe_dispatch_investigation(
                customer_id=cid,
                incident_doc_id=doc_id,
                source=SourceSystem.PAGERDUTY,
                source_event_id="pd:incident:F4-002:incident.triggered",
            )
        # Verify the marker landed on the doc's metadata.
        conn = await asyncpg.connect(dsn=_dsn)
        try:
            row = await conn.fetchrow(
                "SELECT metadata FROM documents "
                "WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL",
                cid, doc_id,
            )
            import json
            md = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            assert md["investigation_dispatch_failed"] is True
        finally:
            await conn.close()
    finally:
        await _cleanup(_dsn, cid)
        await close_pool()


async def test_maybe_dispatch_handles_missing_live_doc(_dsn) -> None:
    """If the incident doc doesn't exist (deleted between normalize
    and dispatch), the helper logs but does NOT crash."""
    from services.ingestion.worker import Worker
    from shared.constants import SourceSystem
    from shared.db import close_pool, init_pool

    cid = _new_customer_id()
    await close_pool()
    await init_pool()
    try:
        conn = await asyncpg.connect(dsn=_dsn)
        try:
            await conn.execute(
                "INSERT INTO customers (customer_id, display_name, api_key_hash, r2_bucket) "
                "VALUES ($1, $2, $3, $4) ON CONFLICT (customer_id) DO NOTHING",
                cid, cid, "h", f"b-{cid}",
            )
        finally:
            await conn.close()

        with patch(
            "services.investigation.dispatch.dispatch_investigation",
            new_callable=AsyncMock,
        ) as fake_dispatch:
            worker = Worker.__new__(Worker)
            await worker._maybe_dispatch_investigation(
                customer_id=cid,
                incident_doc_id="pd:incident:DOES_NOT_EXIST",
                source=SourceSystem.PAGERDUTY,
                source_event_id="pd:incident:DOES_NOT_EXIST:incident.triggered",
            )
        # Dispatch should not be called because the row is missing.
        fake_dispatch.assert_not_called()
    finally:
        await _cleanup(_dsn, cid)
        await close_pool()
