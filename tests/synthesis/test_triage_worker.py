"""Integration tests for TriageWorker.

Covers:
- Pending → triaged drain with mocked Anthropic.
- Triage rejection (low score / not important / no targets).
- Customer opt-out gate (preferences.wiki_generation_enabled).
- The post-commit NOTIFY pattern: a listener on wiki_synthesize_triaged
  must see the triaged rows the moment the notify fires.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncpg
import pytest
import pytest_asyncio

from services.ingestion.handlers.base import make_default_context
from services.ingestion.normalizer import Normalizer
from services.synthesis.triage_worker import TriageWorker
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
    block = SimpleNamespace(type="tool_use", name=name, input=payload)
    return SimpleNamespace(content=[block])


def _make_triage_client(triage_payload: dict) -> SimpleNamespace:
    async def create(*, model: str, **kwargs):
        return _tool_use_response("record_triage", triage_payload)

    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(side_effect=create))
    return client


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
