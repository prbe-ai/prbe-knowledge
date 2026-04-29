"""Same-session coalescing for claude_code (migration 0026 + _enqueue
UPSERT path).

Pins the four launch-readiness invariants:

1. **Coalescing**: N batches for the same session collapse into ONE
   queue row whose `payload_s3_keys` array has N entries and `version`
   is bumped to N. Other connectors keep one-row-per-event semantics.

2. **Priority deprioritization**: claude_code rows are inserted at
   priority=75 (vs 100 for live webhooks); the worker's claim ORDER
   BY puts a github row ahead of a CC row when both are pending.

3. **Resurrection**: a session that completed (status='done') gets
   resurrected back to 'pending' when a new batch arrives, with
   payload_s3_keys extended and version bumped.

4. **Data-loss regression** (the silent bug coalescing fixes): with
   N batches coalesced into one row, the session document's body
   contains events from ALL N batches, not just the latest one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
import orjson
import pytest

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.claude_code import (  # noqa: F401 — registers
    ClaudeCodeConnector,
)
from services.ingestion.handlers.slack import SlackConnector  # noqa: F401 — registers
from services.ingestion.normalizer import Normalizer
from shared import claude_code_extraction as _ext
from shared import db as db_module
from shared.claude_code_extraction import UnitBundle
from shared.config import Settings
from shared.constants import EMBEDDING_DIM, SourceSystem
from shared.customer_mapping import record_mapping
from shared.embeddings import EmbeddedChunk, EmbedResult
from shared.models import IntegrationToken
from shared.tokens import save_device_token

# ---- minimal in-memory stubs ------------------------------------------------


@dataclass
class _StubStore:
    blobs: dict[tuple[str, str], bytes]

    def bucket_for(self, customer_id: str) -> str:
        return f"test-bucket-{customer_id}"

    async def get(self, bucket: str, key: str) -> bytes:
        return self.blobs[(bucket, key)]

    async def ensure_bucket(self, bucket: str) -> None:
        return None

    async def put(self, bucket: str, key: str, body: bytes) -> None:
        self.blobs[(bucket, key)] = body

    async def list_keys(self, bucket: str, prefix: str) -> list[str]:
        return [k for (b, k) in self.blobs if b == bucket and k.startswith(prefix)]


class _ZeroEmbedder:
    """Stub embedder returning zero-vector embeddings of the right dim."""

    async def embed_many(self, texts: list[str]) -> EmbedResult:
        return EmbedResult(
            embedded=[
                EmbeddedChunk(chunk_index=i, embedding=[0.0] * EMBEDDING_DIM)
                for i in range(len(texts))
            ],
            failed=[],
        )


def _cc_envelope(
    *, session_id: str, batch_seq: int, employee_id: str, line_no: int, content: str
) -> bytes:
    return orjson.dumps(
        {
            "_headers": {},
            "payload": {
                "device_id": "test-device",
                "session_id": session_id,
                "batch_seq": batch_seq,
                "cwd": None,
                "events": [
                    {
                        "line_no": line_no,
                        "employee_id": employee_id,
                        "raw": {"role": "user", "content": content},
                    }
                ],
            },
            "received_at": datetime.now(UTC).isoformat(),
        }
    )


async def _seed_customer(customer_id: str) -> None:
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'test-hash')
            ON CONFLICT DO NOTHING
            """,
            customer_id,
        )


# ---- 1. coalescing ----------------------------------------------------------


@pytest.mark.asyncio
async def test_three_batches_coalesce_to_one_row(live_db) -> None:
    from services.ingestion.main import _enqueue, _payload_key

    customer = "coalesce-cust-1"
    session = "sess-coalesce-1"
    await _seed_customer(customer)

    keys = []
    for batch_seq in range(3):
        key = _payload_key(SourceSystem.CLAUDE_CODE, customer, session)
        # _payload_key uses a date prefix + safe_event; for this test we
        # just need three distinct strings, so override with batch_seq.
        key = f"raw/claude_code/{customer}/2026/04/29/{session}:{batch_seq}.json"
        keys.append(key)
        await _enqueue(
            customer_id=customer,
            source=SourceSystem.CLAUDE_CODE,
            source_event_id=session,  # bare session_id (parse_webhook_event change)
            payload_s3_key=key,
        )

    async with db_module.raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT queue_id, source_event_id, payload_s3_keys, version, priority "
            "FROM ingestion_queue WHERE customer_id = $1",
            customer,
        )

    assert len(rows) == 1, f"expected 1 coalesced row, got {len(rows)}"
    row = rows[0]
    assert row["source_event_id"] == session
    assert list(row["payload_s3_keys"]) == keys, (
        f"expected exact key array {keys}, got {list(row['payload_s3_keys'])}"
    )
    assert row["version"] == 3, f"expected version=3 after 3 UPSERTs, got {row['version']}"
    assert row["priority"] == 75, "claude_code priority must be 75"


# ---- 2. priority ordering ---------------------------------------------------


@pytest.mark.asyncio
async def test_github_claims_before_cc_at_same_pending_state(live_db) -> None:
    from services.ingestion.main import _enqueue
    from services.ingestion.worker import Worker

    customer = "priority-cust-1"
    await _seed_customer(customer)

    # Insert CC first (older enqueued_at) so without priority it would claim
    # first via the secondary ORDER BY enqueued_at.
    cc_key = f"raw/claude_code/{customer}/2026/04/29/sess-pri:0.json"
    await _enqueue(
        customer_id=customer,
        source=SourceSystem.CLAUDE_CODE,
        source_event_id="sess-pri",
        payload_s3_key=cc_key,
    )
    # Then github with newer enqueued_at but higher priority.
    gh_key = f"raw/github/{customer}/2026/04/29/evt-gh-pri.json"
    await _enqueue(
        customer_id=customer,
        source=SourceSystem.GITHUB,
        source_event_id="evt-gh-pri",
        payload_s3_key=gh_key,
    )

    settings = Settings(environment="local")
    worker = Worker(
        ConnectorContext(settings=settings, http=httpx.AsyncClient()),
        max_attempts=5, concurrency=1,
    )
    claimed = await worker._claim_one()
    assert claimed is not None
    assert claimed["source_system"] == "github", (
        f"github (priority=100) must claim before CC (priority=75); "
        f"got source={claimed['source_system']}"
    )


# ---- 3. resurrection --------------------------------------------------------


@pytest.mark.asyncio
async def test_done_session_resurrects_on_new_batch(live_db) -> None:
    from services.ingestion.main import _enqueue

    customer = "resurrect-cust-1"
    session = "sess-resurrect"
    await _seed_customer(customer)

    # First batch creates the row.
    await _enqueue(
        customer_id=customer,
        source=SourceSystem.CLAUDE_CODE,
        source_event_id=session,
        payload_s3_key=f"raw/claude_code/{customer}/2026/04/29/{session}:0.json",
    )

    # Mark it 'done' as the worker would.
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "UPDATE ingestion_queue SET status='done', completed_at=NOW() "
            "WHERE customer_id=$1 AND source_event_id=$2",
            customer, session,
        )

    # New batch arrives — UPSERT must flip status back to pending and
    # extend the array.
    await _enqueue(
        customer_id=customer,
        source=SourceSystem.CLAUDE_CODE,
        source_event_id=session,
        payload_s3_key=f"raw/claude_code/{customer}/2026/04/29/{session}:1.json",
    )

    async with db_module.raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, version, payload_s3_keys, completed_at FROM ingestion_queue "
            "WHERE customer_id=$1 AND source_event_id=$2",
            customer, session,
        )

    assert row is not None
    assert row["status"] == "pending", "resurrection must flip status back to pending"
    assert row["version"] == 2
    assert len(row["payload_s3_keys"]) == 2
    assert row["completed_at"] is None, "completed_at must be cleared on resurrection"


# ---- 4. data-loss regression ------------------------------------------------


@pytest.mark.asyncio
async def test_full_session_body_has_events_from_all_batches(live_db, monkeypatch) -> None:
    """Pre-coalescing each batch overwrote the session doc with only its
    30-second window. Post-coalescing the document body contains events
    from every batch.

    This test pins the data-loss regression fix.
    """
    customer = "dataloss-cust-1"
    session = "sess-dataloss"
    employee = "emp-dataloss"
    await _seed_customer(customer)

    # Stub extract_units_from_session — Anthropic isn't in test env.
    async def _noop_extract(**kwargs):  # type: ignore[no-untyped-def]
        return UnitBundle(qa=[], code_change=[], decision=[], file_ref=[])

    monkeypatch.setattr(_ext, "extract_units_from_session", _noop_extract)

    # Need a device token + customer mapping for the CC connector to
    # resolve the employee.
    await save_device_token(IntegrationToken(
        customer_id=customer,
        source_system=SourceSystem.CLAUDE_CODE,
        access_token="x",
        webhook_secret="test-hash",
        device_id="test-device",
        device_metadata={"hostname": "h"},
    ))
    await record_mapping(
        customer_id=customer,
        source_system=SourceSystem.CLAUDE_CODE,
        external_id="test-device",
        external_name="h",
        metadata={},
    )

    # Stage 3 distinct batches in the stub store. Each has one unique
    # event whose content is the batch number — easy to verify below.
    store = _StubStore(blobs={})
    bucket = store.bucket_for(customer)
    keys: list[str] = []
    for batch_seq in range(3):
        key = f"raw/claude_code/{customer}/2026/04/29/{session}:{batch_seq}.json"
        keys.append(key)
        store.blobs[(bucket, key)] = _cc_envelope(
            session_id=session,
            batch_seq=batch_seq,
            employee_id=employee,
            line_no=batch_seq,
            content=f"batch-{batch_seq}-marker",
        )

    settings = Settings(environment="local")
    ctx = ConnectorContext(settings=settings, http=httpx.AsyncClient())
    normalizer = Normalizer(ctx, store=store, embedder=_ZeroEmbedder())  # type: ignore[arg-type]

    try:
        outcome = await normalizer.process_queue_row(
            queue_id=1,
            customer_id=customer,
            source_system=SourceSystem.CLAUDE_CODE,
            source_event_id=session,
            payload_s3_keys=keys,
        )
    finally:
        await ctx.http.aclose()

    assert outcome.doc_ids, "expected at least one document"

    # The session document's body (from metadata.body, the chunkable text
    # rendering of merged events) must contain markers from ALL three
    # batches. Pre-coalescing only the latest batch's content survived.
    async with db_module.raw_conn() as conn:
        body = await conn.fetchval(
            "SELECT metadata->>'body' FROM documents "
            "WHERE customer_id=$1 AND doc_type='claude_code.session' "
            "AND valid_to IS NULL",
            customer,
        )

    assert body is not None
    for batch_seq in range(3):
        marker = f"batch-{batch_seq}-marker"
        assert marker in body, (
            f"session doc body missing events from batch {batch_seq} "
            f"(marker {marker!r}); body sample: {body[:300]!r}"
        )


# ---- 5. other connectors unaffected -----------------------------------------


@pytest.mark.asyncio
async def test_slack_still_uses_one_row_per_event(live_db) -> None:
    from services.ingestion.main import _enqueue

    customer = "slack-cust-1"
    await _seed_customer(customer)

    for evt_id in ("evt-A1B2", "evt-D4E5", "evt-F6G7"):
        await _enqueue(
            customer_id=customer,
            source=SourceSystem.SLACK,
            source_event_id=evt_id,
            payload_s3_key=f"raw/slack/{customer}/2026/04/29/{evt_id}.json",
        )

    async with db_module.raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT source_event_id, payload_s3_keys, version, priority FROM ingestion_queue "
            "WHERE customer_id = $1 ORDER BY enqueued_at",
            customer,
        )

    assert len(rows) == 3, f"slack must keep one row per event, got {len(rows)}"
    for r in rows:
        assert r["priority"] == 100, "slack priority stays at 100"
        assert r["version"] == 0, "slack rows don't bump version (no UPSERT)"
        assert len(r["payload_s3_keys"]) == 1, "slack payload_s3_keys is single-element"
