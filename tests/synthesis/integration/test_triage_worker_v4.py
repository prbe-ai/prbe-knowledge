"""Integration test for TriageWorker v4 DLQ-on-batch-failure path.

When every triage batch in a tick fails (e.g. the provider is down), the
worker must DLQ all pending + triaging rows for that customer with a
categorized `dlq_reason` so the dashboard can surface "drain stuck" and
the admin reset endpoint can flip them back to pending.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from engine.ingest.handlers.base import make_default_context
from engine.ingest.normalizer import Normalizer
from engine.shared.config import Settings
from engine.shared.constants import (
    DocClass,
    DocType,
    Permission,
    PrincipalType,
    SourceSystem,
)
from engine.shared.db import raw_conn
from engine.shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    ACLSnapshotRow,
    Document,
    NormalizationResult,
)
from kb.synthesis.triage_worker import TriageWorker

CUSTOMER = "wiki-triage-v4-cust"


@pytest_asyncio.fixture
async def reset_db(live_db: None, settings: Settings) -> AsyncIterator[None]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash, preferences) "
            "VALUES ($1, 'wiki-triage-v4', 'h', $2::jsonb) "
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


@pytest.mark.asyncio
async def test_batch_failure_dlqs_all_pending_and_triaging_for_customer(
    reset_db: None,
) -> None:
    """Every triage batch fails -> all in-flight rows DLQ, dlq_reason set."""
    normalizer = Normalizer(make_default_context())
    for i in range(3):
        await normalizer._persist(
            CUSTOMER,
            SourceSystem.GITHUB,
            _result(_doc(f"github:commit:dlq-{i}", f"body {i}")),
        )

    # Mock client that always raises on .messages.create — simulates a
    # persistent provider outage. Every batch in the drain fails.
    failing_client = SimpleNamespace()
    failing_client.messages = SimpleNamespace(
        create=AsyncMock(side_effect=RuntimeError("anthropic-down"))
    )

    worker = TriageWorker(asyncio.Event(), anthropic_client=failing_client)
    await worker._tick(woken_by_notify=True)

    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT status, dlq_reason FROM wiki_synthesis_queue "
            "WHERE customer_id = $1 ORDER BY queue_id",
            CUSTOMER,
        )
    assert len(rows) == 3
    for row in rows:
        assert row["status"] == "dlq", f"expected dlq, got {row['status']}"
        assert row["dlq_reason"] is not None
        assert row["dlq_reason"].startswith("triage."), row["dlq_reason"]
