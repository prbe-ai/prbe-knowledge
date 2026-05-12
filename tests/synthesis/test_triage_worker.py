"""Integration tests for TriageWorker.

Covers:
- Pending → triaged drain with mocked Anthropic.
- Triage rejection (low score / not important / no targets).
- Customer opt-out gate (preferences.wiki_generation_enabled).
- The post-commit NOTIFY pattern: a listener on wiki_synthesize_triaged
  must see the triaged rows the moment the notify fires.
- Orphan-rejection path: rows whose body fetch returns nothing (doc
  version superseded by valid_to or soft-deleted between enqueue and
  drain) are routed straight to 'rejected', NOT churned through
  mark_for_retry's 3-attempt loop.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncpg
import orjson
import pytest
import pytest_asyncio

from services.ingestion.handlers.base import make_default_context
from services.ingestion.normalizer import Normalizer
from services.synthesis.models import TriageInput
from services.synthesis.triage_worker import (
    TRIAGE_DOC_SUPERSEDED_OR_DELETED_REASON,
    TriageWorker,
)
from shared.config import Settings
from shared.constants import (
    WIKI_TRIAGED_CHANNEL,
    DocClass,
    DocType,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.db import raw_conn
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    ACLSnapshotRow,
    Document,
    NormalizationResult,
)

CUSTOMER = "wiki-triage-cust"


@pytest_asyncio.fixture
async def reset_db(live_db: None, settings: Settings) -> AsyncIterator[None]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash, preferences) "
            "VALUES ($1, 'wiki-triage', 'h', $2::jsonb) "
            "ON CONFLICT (customer_id) DO UPDATE SET preferences = EXCLUDED.preferences",
            CUSTOMER,
            '{"wiki_generation_enabled": true}',
        )
    yield None


def _doc(doc_id: str, body: str) -> Document:
    now = datetime.now(UTC)
    return Document(
        doc_id=doc_id,
        customer_id=CUSTOMER,
        source_system=SourceSystem.GITHUB,
        source_id=doc_id.split(":", 1)[-1],
        source_url=f"https://github.test/{doc_id}",
        doc_class=DocClass.RAW_SOURCE,
        doc_type=DocType.GITHUB_COMMIT,
        content_type="text/markdown",
        content_hash=f"hash-{doc_id}-{len(body)}",
        title=f"Title {doc_id}",
        body_preview=body[:200],
        body_size_bytes=len(body.encode("utf-8")),
        body_token_count=len(body.split()),
        author_id="alice",
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        acl=ACLSnapshot(
            principals=[
                ACLPrincipal(
                    principal_type=PrincipalType.WORKSPACE,
                    principal_id=CUSTOMER,
                    permission=Permission.READ,
                )
            ],
            captured_at=now,
        ),
        metadata={},
        body=body,
    )


def _result(*docs: Document) -> NormalizationResult:
    now = datetime.now(UTC)
    return NormalizationResult(
        documents=list(docs),
        graph_nodes=[],
        graph_edges=[],
        acl_snapshots=[
            ACLSnapshotRow(
                source_system=SourceSystem.GITHUB,
                principal_type=PrincipalType.WORKSPACE,
                principal_id=CUSTOMER,
                resource_type="document",
                resource_id=docs[0].doc_id,
                permission=Permission.READ,
                valid_from=now,
            )
        ],
    )


def _tool_use_response(name: str, payload: dict) -> SimpleNamespace:
    """LiteLLM-shaped response carrying one forced tool call.

    Post-Phase-0b, ``shared.llm_tools.forced_tool_call`` reads
    ``resp.choices[0].message.tool_calls[0].function.{name, arguments}``
    where ``arguments`` is a JSON-encoded string. (Pre-migration this
    returned the Anthropic-native ``content=[{type:'tool_use',...}]``
    shape; the new shape matches what LiteLLM normalizes Anthropic +
    OpenAI + Gemini tool-call responses into.)
    """
    func = SimpleNamespace(name=name, arguments=orjson.dumps(payload).decode("utf-8"))
    call = SimpleNamespace(type="function", function=func)
    message = SimpleNamespace(content=None, tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], usage=None)


# Per-test triage payload, set inside the test body and read by the
# autouse `_patch_acompletion` fixture's fake `acompletion`. Default is
# the empty-verdicts dict — safe for tests that expect zero LLM calls.
_TRIAGE_PAYLOAD: dict = {"verdicts": {}}


def _make_triage_client(triage_payload: dict) -> SimpleNamespace:
    """Stash the per-test triage payload and return an inert sentinel.

    The ``TriageWorker(anthropic_client=...)`` parameter is unused
    post-Phase-0b (kept for caller compat). The real mocking happens at
    ``shared.llm_tools.acompletion`` via the ``_patch_acompletion``
    autouse fixture below, which reads ``_TRIAGE_PAYLOAD`` on every call.
    """
    global _TRIAGE_PAYLOAD
    _TRIAGE_PAYLOAD = triage_payload
    return SimpleNamespace()


@pytest.fixture(autouse=True)
def _patch_acompletion(monkeypatch) -> None:
    """Route every ``shared.llm_tools.acompletion`` call to the
    LiteLLM-shaped response that wraps the current ``_TRIAGE_PAYLOAD``.
    Reset the payload to the empty default between tests so a stale
    payload from a prior test can't leak in.
    """
    global _TRIAGE_PAYLOAD
    _TRIAGE_PAYLOAD = {"verdicts": {}}

    async def _fake(**kwargs):
        return _tool_use_response("record_triage", _TRIAGE_PAYLOAD)

    monkeypatch.setattr("shared.llm_tools.acompletion", _fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)


async def _read_queue_ids(customer_id: str) -> list[int]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT queue_id FROM wiki_synthesis_queue WHERE customer_id = $1 ORDER BY queue_id",
            customer_id,
        )
    return [row["queue_id"] for row in rows]


@pytest.mark.asyncio
async def test_triage_worker_marks_kept_rows_triaged(reset_db: None) -> None:
    normalizer = Normalizer(make_default_context())
    await normalizer._persist(
        CUSTOMER,
        SourceSystem.GITHUB,
        _result(_doc("github:commit:t1", "We adopted pgvector for embeddings.")),
    )

    queue_ids = await _read_queue_ids(CUSTOMER)
    triage_payload = {
        "verdicts": {
            str(qid): {
                "important": True,
                "score": 8.0,
                "targets": [
                    {"wiki_type": "decision", "slug": "adopt-pgvector", "action": "create"},
                ],
                "reason": "decision recorded",
            }
            for qid in queue_ids
        }
    }
    client = _make_triage_client(triage_payload)

    worker = TriageWorker(asyncio.Event(), anthropic_client=client)
    await worker._tick(woken_by_notify=True)

    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT status, triage_score FROM wiki_synthesis_queue WHERE customer_id = $1",
            CUSTOMER,
        )
    assert all(r["status"] == "triaged" for r in rows)
    assert all(r["triage_score"] == 8.0 for r in rows)


@pytest.mark.asyncio
async def test_triage_worker_rejects_low_score(reset_db: None) -> None:
    normalizer = Normalizer(make_default_context())
    await normalizer._persist(
        CUSTOMER,
        SourceSystem.GITHUB,
        _result(_doc("github:commit:noisy", "build green")),
    )

    queue_ids = await _read_queue_ids(CUSTOMER)
    triage_payload = {
        "verdicts": {
            str(queue_ids[0]): {
                "important": False,
                "score": 1.0,
                "targets": [],
                "reason": "noise",
            }
        }
    }
    client = _make_triage_client(triage_payload)

    worker = TriageWorker(asyncio.Event(), anthropic_client=client)
    await worker._tick(woken_by_notify=True)

    async with raw_conn() as conn:
        status = await conn.fetchval(
            "SELECT status FROM wiki_synthesis_queue WHERE customer_id = $1",
            CUSTOMER,
        )
    assert status == "rejected"


@pytest.mark.asyncio
async def test_triage_worker_skips_opted_out_customer(live_db: None, settings: Settings) -> None:
    """A customer with no preferences row (or wiki_generation_enabled !=
    true) must NOT have their pending rows drained by the worker."""
    other_cust = "wiki-triage-opt-out"
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash, preferences) "
            "VALUES ($1, 'wiki-triage-opt-out', 'h', '{}'::jsonb) "
            "ON CONFLICT (customer_id) DO NOTHING",
            other_cust,
        )
        # Force a pending row in directly — simulate a row left behind
        # from when the customer had opted in, then opted out.
        await conn.execute(
            """
            INSERT INTO wiki_synthesis_queue
                (customer_id, doc_id, doc_version, source_system,
                 doc_type, status, enqueued_at)
            VALUES ($1, 'github:commit:left-behind', 1, 'github',
                    'github.commit', 'pending', NOW())
            ON CONFLICT DO NOTHING
            """,
            other_cust,
        )

    client = _make_triage_client({"verdicts": {}})  # no calls expected
    worker = TriageWorker(asyncio.Event(), anthropic_client=client)
    await worker._tick(woken_by_notify=True)

    async with raw_conn() as conn:
        status = await conn.fetchval(
            "SELECT status FROM wiki_synthesis_queue WHERE customer_id = $1",
            other_cust,
        )
    # Row stays pending — the opt-out guard prevented the drain.
    assert status == "pending"


@pytest.mark.asyncio
async def test_triage_worker_fires_notify_after_commit(reset_db: None, settings: Settings) -> None:
    """When the listener on wiki_synthesize_triaged wakes, the triaged
    rows must already be visible. Same-transaction UPDATE+NOTIFY is the
    canonical pattern enforced in `persistence.mark_batch_triaged_and_notify`.
    """
    normalizer = Normalizer(make_default_context())
    await normalizer._persist(
        CUSTOMER,
        SourceSystem.GITHUB,
        _result(_doc("github:commit:nofy", "we picked Y over X for the auth flow.")),
    )
    queue_ids = await _read_queue_ids(CUSTOMER)
    assert len(queue_ids) == 1

    triage_payload = {
        "verdicts": {
            str(queue_ids[0]): {
                "important": True,
                "score": 9.0,
                "targets": [{"wiki_type": "decision", "slug": "auth-y-over-x", "action": "create"}],
                "reason": "decision",
            }
        }
    }
    client = _make_triage_client(triage_payload)

    seen_count_when_notified: list[int] = []
    pending_tasks: list[asyncio.Task] = []
    notify_event = asyncio.Event()
    listener = await asyncpg.connect(settings.database_url)

    def _on_notify(_conn, _pid, _channel, payload) -> None:
        # The query has to happen on the listener side to prove visibility.
        async def _check():
            async with raw_conn() as q:
                count = await q.fetchval(
                    "SELECT count(*) FROM wiki_synthesis_queue "
                    "WHERE customer_id = $1 AND status = 'triaged'",
                    payload,
                )
            seen_count_when_notified.append(count)
            notify_event.set()

        # Hold a reference so the task isn't garbage-collected before completion.
        pending_tasks.append(asyncio.create_task(_check()))

    try:
        await listener.add_listener(WIKI_TRIAGED_CHANNEL, _on_notify)

        worker = TriageWorker(asyncio.Event(), anthropic_client=client)
        await worker._tick(woken_by_notify=True)

        await asyncio.wait_for(notify_event.wait(), timeout=5.0)
        assert seen_count_when_notified == [1], (
            f"listener saw {seen_count_when_notified[0] if seen_count_when_notified else None} "
            "triaged rows when notify fired — expected 1. UPDATE + NOTIFY are no longer in "
            "the same transaction."
        )
    finally:
        with contextlib.suppress(Exception):
            await listener.remove_listener(WIKI_TRIAGED_CHANNEL, _on_notify)
        await listener.close()


# ---------------------------------------------------------------------------
# Orphan-rejection path — DB-free unit tests
# ---------------------------------------------------------------------------
#
# fetch_bodies INNER JOINs `documents` filtered by valid_to IS NULL AND
# deleted_at IS NULL. When a queued doc_version was superseded between
# enqueue and drain (e.g. a GitHub PR gets a new commit before triage
# runs), the queue row drops out of triage_inputs. Without dedicated
# handling, the row falls through to mark_for_retry, churns 3 attempts,
# then dead-letters as 'failed' with the misleading "no verdict from
# triage batch" tombstone. The worker now intercepts those rows after
# fetch_bodies and routes them straight to 'rejected' via
# mark_orphans_rejected, with a categorized reason.


def _ti(qid: int) -> TriageInput:
    """Minimal TriageInput stub for batch-packing assertions."""
    return TriageInput(
        queue_id=qid,
        doc_id=f"doc:{qid}",
        doc_type="github.commit",
        source_system="github",
        title=f"Doc {qid}",
        author_id="alice",
        body="x",
        body_token_count=10,
    )


def _qrow(qid: int, doc_id: str | None = None, version: int = 1) -> dict:
    """Fake claim_pending_batch row. dict-shaped is enough — the worker
    only reads via row['queue_id'] / row['doc_id'] etc."""
    return {
        "queue_id": qid,
        "doc_id": doc_id or f"doc:{qid}",
        "doc_version": version,
        "source_system": "github",
        "doc_type": "github.commit",
        "attempts": 1,
        "triage_score": None,
        "source_ts": datetime.now(UTC),
    }


@pytest.mark.asyncio
async def test_orphan_superseded_row_marked_rejected_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 queued rows; fetch_bodies returns 2 (third's doc was
    superseded). Assert: mark_orphans_rejected is called for the third
    qid; mark_for_retry is NEVER called for it; _apply_verdicts only
    sees the 2 non-orphan rows."""
    from services.synthesis import triage_worker as tw_mod
    from services.synthesis.models import TriageVerdict

    customer = "cust-orphan-1"
    queue_rows = [_qrow(1), _qrow(2), _qrow(3)]
    # fetch_bodies returns inputs only for qid 1 and 2; qid 3 was
    # superseded so its body fetch returned nothing.
    inputs = [_ti(1), _ti(2)]

    # First claim returns rows; second claim returns [] so the drain loop exits.
    claim_calls: list[int] = []

    async def fake_claim(cid: str, *, limit: int) -> list[dict]:
        claim_calls.append(limit)
        if len(claim_calls) == 1:
            return queue_rows
        return []

    fake_fetch_bodies = AsyncMock(return_value=inputs)
    fake_open_run = AsyncMock(return_value=42)
    fake_close_run = AsyncMock(return_value=None)
    fake_mark_orphans = AsyncMock(return_value=None)
    fake_mark_for_retry = AsyncMock(return_value=None)
    fake_mark_rejected = AsyncMock(return_value=None)
    fake_mark_batch_triaged = AsyncMock(return_value=None)
    fake_dlq_customer = AsyncMock(return_value=0)

    monkeypatch.setattr(tw_mod.persistence, "claim_pending_batch", fake_claim)
    monkeypatch.setattr(tw_mod.persistence, "fetch_bodies", fake_fetch_bodies)
    monkeypatch.setattr(tw_mod.persistence, "open_run", fake_open_run)
    monkeypatch.setattr(tw_mod.persistence, "close_run", fake_close_run)
    monkeypatch.setattr(
        tw_mod.persistence, "mark_orphans_rejected", fake_mark_orphans
    )
    monkeypatch.setattr(tw_mod.persistence, "mark_for_retry", fake_mark_for_retry)
    monkeypatch.setattr(tw_mod.persistence, "mark_rejected", fake_mark_rejected)
    monkeypatch.setattr(
        tw_mod.persistence,
        "mark_batch_triaged_and_notify",
        fake_mark_batch_triaged,
    )
    monkeypatch.setattr(
        tw_mod.persistence,
        "dlq_customer_for_triage_failure",
        fake_dlq_customer,
    )

    # Stub out the triage call so each input gets a high-score verdict.
    async def fake_call_batches(self, client, batches, customer_id):
        out: dict[int, TriageVerdict] = {}
        for batch in batches:
            for ev in batch:
                out[ev.queue_id] = TriageVerdict(
                    important=True, score=8.0, reason="ok"
                )
        return out

    monkeypatch.setattr(TriageWorker, "_call_triage_batches", fake_call_batches)

    worker = TriageWorker(asyncio.Event(), anthropic_client=SimpleNamespace())
    await worker._drain_customer(customer, SimpleNamespace(), run_kind="wake")

    # mark_orphans_rejected was called exactly once with qid 3 and the
    # categorized reason; qid 3 never went through mark_for_retry.
    assert fake_mark_orphans.await_count == 1
    args, kwargs = fake_mark_orphans.await_args
    assert args[0] == customer
    assert args[1] == [3]
    assert kwargs["reason"] == TRIAGE_DOC_SUPERSEDED_OR_DELETED_REASON
    fake_mark_for_retry.assert_not_called()

    # The 2 non-orphan rows were marked triaged in a single batch call.
    assert fake_mark_batch_triaged.await_count == 1
    triaged_args, _ = fake_mark_batch_triaged.await_args
    triaged_qids = sorted(qid for qid, _ in triaged_args[1])
    assert triaged_qids == [1, 2]


@pytest.mark.asyncio
async def test_oversized_rows_not_misclassified_as_orphans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An event that goes through pack_into_batches `oversized` MUST be
    routed via _reject_oversized (the existing path) — NOT misclassified
    as an orphan. Otherwise a giant body would yield a generic
    'doc_superseded_or_deleted' tag instead of the precise oversized
    diagnostic."""
    from services.synthesis import triage_worker as tw_mod

    customer = "cust-oversized-1"
    queue_rows = [_qrow(1), _qrow(2)]
    # Both inputs come back from fetch_bodies — but pack_into_batches
    # will move qid 1 to oversized.
    inputs = [_ti(1), _ti(2)]

    claim_calls: list[int] = []

    async def fake_claim(cid: str, *, limit: int) -> list[dict]:
        claim_calls.append(limit)
        if len(claim_calls) == 1:
            return queue_rows
        return []

    # Stub pack_into_batches: qid 1 -> oversized, qid 2 -> normal batch.
    def fake_pack(events):
        oversized = [ev for ev in events if ev.queue_id == 1]
        normal = [ev for ev in events if ev.queue_id != 1]
        batches = [normal] if normal else []
        return batches, oversized

    monkeypatch.setattr(tw_mod, "pack_into_batches", fake_pack)
    monkeypatch.setattr(tw_mod.persistence, "claim_pending_batch", fake_claim)
    monkeypatch.setattr(
        tw_mod.persistence, "fetch_bodies", AsyncMock(return_value=inputs)
    )
    monkeypatch.setattr(tw_mod.persistence, "open_run", AsyncMock(return_value=7))
    monkeypatch.setattr(tw_mod.persistence, "close_run", AsyncMock(return_value=None))
    fake_mark_orphans = AsyncMock(return_value=None)
    fake_mark_rejected = AsyncMock(return_value=None)
    fake_mark_for_retry = AsyncMock(return_value=None)
    monkeypatch.setattr(
        tw_mod.persistence, "mark_orphans_rejected", fake_mark_orphans
    )
    monkeypatch.setattr(tw_mod.persistence, "mark_rejected", fake_mark_rejected)
    monkeypatch.setattr(tw_mod.persistence, "mark_for_retry", fake_mark_for_retry)
    monkeypatch.setattr(
        tw_mod.persistence,
        "mark_batch_triaged_and_notify",
        AsyncMock(return_value=None),
    )

    async def fake_call_batches(self, client, batches, customer_id):
        from services.synthesis.models import TriageVerdict

        return {
            ev.queue_id: TriageVerdict(important=True, score=9.0, reason="ok")
            for batch in batches
            for ev in batch
        }

    monkeypatch.setattr(TriageWorker, "_call_triage_batches", fake_call_batches)

    worker = TriageWorker(asyncio.Event(), anthropic_client=SimpleNamespace())
    await worker._drain_customer(customer, SimpleNamespace(), run_kind="wake")

    # The oversized event went to mark_rejected (via _reject_oversized),
    # NOT to mark_orphans_rejected.
    fake_mark_orphans.assert_not_called()
    assert fake_mark_rejected.await_count == 1
    rejected_args, _ = fake_mark_rejected.await_args
    assert rejected_args[1] == 1  # queue_id
    fake_mark_for_retry.assert_not_called()


@pytest.mark.asyncio
async def test_mark_orphans_rejected_sets_status_and_reason(
    reset_db: None,
) -> None:
    """Persistence-level test: a row at status='triaging' becomes
    'rejected' with the reason set; a row at status='pending' is left
    alone (defensive WHERE clause guards against racing with reclaim or
    a concurrent path that already moved the row)."""
    from services.synthesis import persistence

    async with raw_conn() as conn:
        # Two rows: one currently 'triaging' (the normal post-claim
        # state) and one still 'pending' (a row a sibling worker hasn't
        # claimed yet).
        triaging_qid = await conn.fetchval(
            """
            INSERT INTO wiki_synthesis_queue
                (customer_id, doc_id, doc_version, source_system,
                 doc_type, status, enqueued_at, attempts)
            VALUES ($1, 'github:commit:orph-triaging', 1, 'github',
                    'github.commit', 'triaging', NOW(), 1)
            RETURNING queue_id
            """,
            CUSTOMER,
        )
        pending_qid = await conn.fetchval(
            """
            INSERT INTO wiki_synthesis_queue
                (customer_id, doc_id, doc_version, source_system,
                 doc_type, status, enqueued_at, attempts)
            VALUES ($1, 'github:commit:orph-pending', 1, 'github',
                    'github.commit', 'pending', NOW(), 0)
            RETURNING queue_id
            """,
            CUSTOMER,
        )

    await persistence.mark_orphans_rejected(
        CUSTOMER,
        [int(triaging_qid), int(pending_qid)],
        reason=TRIAGE_DOC_SUPERSEDED_OR_DELETED_REASON,
    )

    async with raw_conn() as conn:
        triaging_after = await conn.fetchrow(
            "SELECT status, triage_score, triage_error, triage_completed_at "
            "FROM wiki_synthesis_queue WHERE queue_id = $1",
            int(triaging_qid),
        )
        pending_after = await conn.fetchrow(
            "SELECT status, triage_score, triage_error, triage_completed_at "
            "FROM wiki_synthesis_queue WHERE queue_id = $1",
            int(pending_qid),
        )

    # The 'triaging' row flipped to 'rejected' with the categorized
    # reason and a zero score (presents like a normal score-rejected
    # row in the dashboard).
    assert triaging_after["status"] == "rejected"
    assert triaging_after["triage_score"] == 0.0
    assert triaging_after["triage_error"] == TRIAGE_DOC_SUPERSEDED_OR_DELETED_REASON
    assert triaging_after["triage_completed_at"] is not None

    # The 'pending' row was NOT touched — defensive WHERE clause is
    # working. (If we removed the `AND status = 'triaging'` guard, this
    # would clobber a sibling worker's still-claimable row.)
    assert pending_after["status"] == "pending"
    assert pending_after["triage_score"] is None
    assert pending_after["triage_error"] is None
    assert pending_after["triage_completed_at"] is None


@pytest.mark.asyncio
async def test_mark_orphans_rejected_empty_list_is_noop() -> None:
    """Empty input -> no SQL fired (the function returns immediately).
    Cheap regression guard: don't accidentally fire an UPDATE that
    matches every customer row."""
    from services.synthesis import persistence

    # No DB needed — empty queue_ids returns before opening a tenant
    # connection. Just confirm it doesn't raise.
    await persistence.mark_orphans_rejected(
        "cust-noop",
        [],
        reason=TRIAGE_DOC_SUPERSEDED_OR_DELETED_REASON,
    )
