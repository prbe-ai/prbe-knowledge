"""Regression test: main ingestion worker still acks even if inferred_edges
queue insert fails.

CRITICAL: The inferred_edges_queue insert is best-effort. Any exception
from _enqueue_inferred_edges must be caught by normalizer._persist so that
the main ingestion pipeline still commits and the queue row still gets
marked done.

This test mocks _enqueue_inferred_edges to raise and verifies that
_persist completes successfully (no exception propagated).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
import pytest_asyncio

from engine.ingest.handlers.base import make_default_context
from engine.ingest.normalizer import Normalizer
from engine.shared.constants import DocClass, DocType, Permission, PrincipalType, SourceSystem
from engine.shared.db import raw_conn
from engine.shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    ACLSnapshotRow,
    Document,
    NormalizationResult,
)

CUSTOMER = "cust-ie-regression"


@pytest_asyncio.fixture
async def db_setup(live_db) -> AsyncIterator[None]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'ie-regression', 'h') "
            "ON CONFLICT (customer_id) DO NOTHING",
            CUSTOMER,
        )
    yield None


def _doc(doc_id: str) -> Document:
    now = datetime.now(UTC)
    body = "Some test content for regression test."
    return Document(
        doc_id=doc_id,
        customer_id=CUSTOMER,
        source_system=SourceSystem.SLACK,
        source_id=doc_id,
        source_url=f"https://example.test/{doc_id}",
        doc_class=DocClass.RAW_SOURCE,
        doc_type=DocType.SLACK_THREAD,
        content_type="text/plain",
        content_hash=f"hash-{doc_id}",
        title=f"Title {doc_id}",
        body_preview=body[:120],
        body_size_bytes=len(body.encode()),
        body_token_count=10,
        author_id=None,
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


def _result(doc: Document) -> NormalizationResult:
    now = datetime.now(UTC)
    return NormalizationResult(
        documents=[doc],
        graph_nodes=[],
        graph_edges=[],
        acl_snapshots=[
            ACLSnapshotRow(
                source_system=doc.source_system,
                principal_type=PrincipalType.WORKSPACE,
                principal_id=CUSTOMER,
                resource_type="document",
                resource_id=doc.doc_id,
                permission=Permission.READ,
                valid_from=now,
            )
        ],
    )


@pytest.mark.asyncio
async def test_persist_succeeds_even_when_inferred_edges_enqueue_raises(
    db_setup: None,
) -> None:
    """REGRESSION: _enqueue_inferred_edges failure must not propagate.

    The main ingestion pipeline must still commit and the doc must be
    persisted even if the inferred_edges_queue insert raises an exception.
    """
    normalizer = Normalizer(make_default_context())

    doc = _doc("slack:thread:regression-001")

    # Patch _enqueue_inferred_edges to raise a DB error
    with patch.object(
        normalizer,
        "_enqueue_inferred_edges",
        side_effect=RuntimeError("Simulated DB failure on queue insert"),
    ):
        # This must NOT raise -- the exception must be swallowed
        outcome = await normalizer._persist(CUSTOMER, SourceSystem.SLACK, _result(doc))

    # Ingestion succeeded: doc_id is in the outcome
    assert "slack:thread:regression-001" in outcome.doc_ids

    # The document was actually persisted to the DB
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT doc_id FROM documents WHERE customer_id = $1 AND doc_id = $2",
            CUSTOMER,
            "slack:thread:regression-001",
        )
    assert row is not None, "Document was not persisted despite queue insert failure"


@pytest.mark.asyncio
async def test_persist_enqueues_inferred_edges_on_success(db_setup: None) -> None:
    """Verify _enqueue_inferred_edges IS called on successful persist."""
    normalizer = Normalizer(make_default_context())
    doc = _doc("slack:thread:ie-enqueue-001")

    enqueue_called_with: list = []

    async def _spy(customer_id, doc_ids):
        enqueue_called_with.append((customer_id, doc_ids))
        # Don't call original to avoid needing the table in this test

    with patch.object(normalizer, "_enqueue_inferred_edges", side_effect=_spy):
        outcome = await normalizer._persist(CUSTOMER, SourceSystem.SLACK, _result(doc))

    assert "slack:thread:ie-enqueue-001" in outcome.doc_ids
    # The enqueue method was called with the correct customer and doc_ids
    assert len(enqueue_called_with) == 1
    customer_arg, doc_ids_arg = enqueue_called_with[0]
    assert customer_arg == CUSTOMER
    assert "slack:thread:ie-enqueue-001" in doc_ids_arg


@pytest.mark.asyncio
async def test_persist_does_not_enqueue_inferred_edges_for_wiki(db_setup: None) -> None:
    """Wiki source writes must NOT be enqueued for inferred-edge extraction."""
    normalizer = Normalizer(make_default_context())
    now = datetime.now(UTC)
    body = "Wiki compiled content."
    doc = Document(
        doc_id="wiki:page:auth-runbook",
        customer_id=CUSTOMER,
        source_system=SourceSystem.WIKI,
        source_id="wiki:page:auth-runbook",
        source_url="https://example.test/wiki-page",
        doc_class=DocClass.COMPILED_WIKI,
        doc_type="wiki.runbook",
        content_type="text/markdown",
        content_hash="hash-wiki-001",
        title="Auth Runbook",
        body_preview=body[:120],
        body_size_bytes=len(body.encode()),
        body_token_count=10,
        author_id=None,
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
    result = NormalizationResult(
        documents=[doc],
        graph_nodes=[],
        graph_edges=[],
        acl_snapshots=[
            ACLSnapshotRow(
                source_system=SourceSystem.WIKI,
                principal_type=PrincipalType.WORKSPACE,
                principal_id=CUSTOMER,
                resource_type="document",
                resource_id=doc.doc_id,
                permission=Permission.READ,
                valid_from=now,
            )
        ],
    )

    enqueue_called = False

    async def _spy(customer_id, doc_ids):
        nonlocal enqueue_called
        enqueue_called = True

    with patch.object(normalizer, "_enqueue_inferred_edges", side_effect=_spy):
        await normalizer._persist(CUSTOMER, SourceSystem.WIKI, result)

    assert not enqueue_called, (
        "Wiki source writes must NOT be enqueued for inferred-edge extraction"
    )
