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

from engine.ingest.handlers.base import ConnectorContext
from engine.ingest.normalizer import Normalizer
from engine.shared import claude_code_extraction as _ext
from engine.shared import db as db_module
from engine.shared.claude_code_extraction import UnitBundle
from engine.shared.config import Settings
from engine.shared.constants import EMBEDDING_V2_DIM, SourceSystem
from engine.shared.customer_mapping import record_mapping
from engine.shared.embeddings import EmbeddedChunk, EmbedResult
from engine.shared.encryption import encrypt_token
from engine.shared.models import IntegrationToken
from engine.shared.tokens import save_device_token
from kb.handlers.claude_code import (  # noqa: F401 — registers
    ClaudeCodeConnector,
)
from kb.handlers.slack import SlackConnector  # noqa: F401 — registers

# ---- minimal in-memory stubs ------------------------------------------------


@dataclass
class _StubStore:
    blobs: dict[tuple[str, str], bytes]

    async def bucket_for(self, customer_id: str) -> str:
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
    """Stub embedder returning zero-vector embeddings of the right dim.

    Implements `embed_documents` (what Normalizer calls post-cutover) and
    `embed_many` (compat for older direct callers in this test file).
    """

    async def embed_documents(self, items: list) -> EmbedResult:
        return EmbedResult(
            embedded=[
                EmbeddedChunk(chunk_index=i, embedding=[0.0] * EMBEDDING_V2_DIM)
                for i in range(len(items))
            ],
            failed=[],
        )

    async def embed_many(self, texts: list[str]) -> EmbedResult:
        return EmbedResult(
            embedded=[
                EmbeddedChunk(chunk_index=i, embedding=[0.0] * EMBEDDING_V2_DIM)
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


async def _seed_integration_token(customer_id: str, source: SourceSystem) -> None:
    """Seed an active integration_tokens row so the pre-enqueue connectedness
    gate (services/ingestion/connectedness.py) lets OAuth-source enqueues
    through. Each test gets a freshly-truncated DB (conftest live_db fixture),
    so a plain INSERT is enough."""
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO integration_tokens
                (customer_id, source_system, access_token_encrypted, status)
            VALUES ($1, $2, $3, 'active')
            """,
            customer_id,
            source.value,
            encrypt_token("test-token"),
        )


# ---- 1. coalescing ----------------------------------------------------------


@pytest.mark.asyncio
async def test_three_batches_coalesce_to_one_row(live_db) -> None:
    from kb.ingestion_app import _enqueue, _payload_key

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
    from engine.ingest.worker import Worker
    from kb.ingestion_app import _enqueue

    customer = "priority-cust-1"
    await _seed_customer(customer)
    await _seed_integration_token(customer, SourceSystem.GITHUB)

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
    from kb.ingestion_app import _enqueue

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
    bucket = await store.bucket_for(customer)
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

    # claude_code.fetch_supplementary calls `get_store()` directly (not the
    # injected normalizer store), so monkeypatch the global lookup so the
    # connector's R2 reads land on our stub.
    from kb.handlers import claude_code as _cc_mod
    monkeypatch.setattr(_cc_mod, "get_store", lambda: store)

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

    # The session document's body (joined from live content chunks, the
    # canonical source of truth post-storage-cleanup) must contain markers
    # from ALL three batches. Pre-coalescing only the latest batch's content
    # survived.
    async with db_module.raw_conn() as conn:
        doc_id = await conn.fetchval(
            "SELECT doc_id FROM documents "
            "WHERE customer_id=$1 AND doc_type='claude_code.session' "
            "AND valid_to IS NULL",
            customer,
        )
        body = await conn.fetchval(
            "SELECT string_agg(content, '' ORDER BY chunk_index) "
            "FROM chunks WHERE customer_id=$1 AND doc_id=$2 "
            "AND kind = 'content' AND valid_to IS NULL",
            customer,
            doc_id,
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
    from kb.ingestion_app import _enqueue

    customer = "slack-cust-1"
    await _seed_customer(customer)
    await _seed_integration_token(customer, SourceSystem.SLACK)

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


# ---- 6. R2 key namespace must stay unique per delivery ---------------------


@pytest.mark.parametrize(
    "source",
    [SourceSystem.CLAUDE_CODE, SourceSystem.CODEX],
    ids=["claude_code", "codex"],
)
def test_compose_storage_id_suffixes_batch_seq_per_agent_source(
    source: SourceSystem,
) -> None:
    """Regression: agent-session sources coalesce by bare session_id at the
    queue, so the R2 storage namespace must distinguish batches via a
    `:<batch_seq>` suffix — otherwise batches overwrite each other in
    storage and only the final delivery's envelope survives.

    Calls the real `_compose_storage_id` used by the webhook handler so a
    regression that drops a source from the predicate is caught here. CC
    had this protection; CODEX was originally missing it and lost data
    silently on multi-batch sessions until this fix.
    """
    from kb.ingestion_app import _compose_storage_id, _payload_key

    customer = f"kc-{source.value}-cust"
    session = f"{source.value}-sess-keys-unique"

    keys: set[str] = set()
    storage_ids: set[str] = set()
    for batch_seq in range(5):
        parse_hint = {"session_id": session, "batch_seq": batch_seq}
        storage_id = _compose_storage_id(source, session, parse_hint)
        storage_ids.add(storage_id)
        keys.add(_payload_key(source, customer, storage_id))

    assert len(storage_ids) == 5, (
        f"{source.value}: storage_id collapsed across batches; got "
        f"{storage_ids}. The compose function dropped the batch_seq suffix."
    )
    assert len(keys) == 5, (
        f"{source.value}: expected 5 distinct R2 keys, got {len(keys)}: {keys}. "
        "If only 1 distinct key, multi-batch sessions will overwrite each "
        "other in R2 before the worker reads them."
    )


def test_compose_storage_id_passes_through_for_non_agent_sources() -> None:
    """Non-agent sources (slack, github, etc.) get their own queue row per
    event — source_event_id is already unique per delivery, so the
    storage_id should be the bare event id without a suffix.
    """
    from kb.ingestion_app import _compose_storage_id

    for source in (
        SourceSystem.SLACK,
        SourceSystem.GITHUB,
        SourceSystem.LINEAR,
        SourceSystem.NOTION,
    ):
        # parse_hint may carry whatever; non-agent sources should ignore it.
        out = _compose_storage_id(
            source, "evt-123", {"session_id": "x", "batch_seq": 7}
        )
        assert out == "evt-123", (
            f"{source.value} should pass source_event_id through unchanged; "
            f"got {out!r}"
        )


def test_compose_storage_id_handles_missing_batch_seq() -> None:
    """The pair endpoint and finalize cron use a parse_hint without a
    batch_seq — composition should fall back to bare session_id rather
    than emit `<session>:None` or crash.
    """
    from kb.ingestion_app import _compose_storage_id

    # parse_hint missing batch_seq
    assert _compose_storage_id(
        SourceSystem.CODEX, "sess-1", {"session_id": "sess-1"}
    ) == "sess-1"
    # parse_hint not a dict
    assert _compose_storage_id(
        SourceSystem.CODEX, "sess-1", None
    ) == "sess-1"
    # batch_seq present but not an int (defensive)
    assert _compose_storage_id(
        SourceSystem.CODEX, "sess-1", {"batch_seq": "0"}
    ) == "sess-1"
