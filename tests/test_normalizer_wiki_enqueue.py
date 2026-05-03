"""Verify Normalizer._persist enqueues the right wiki_synthesis_queue rows
and emits a pg_notify on the WIKI_SYNTHESIZE_CHANNEL.

The hot path itself is just an INSERT + a notify; the LLM stages run
elsewhere and are tested in tests/synthesis/test_*.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import asyncpg
import pytest
import pytest_asyncio

from services.ingestion.handlers.base import make_default_context
from services.ingestion.normalizer import Normalizer
from shared.config import Settings
from shared.constants import (
    WIKI_SYNTHESIZE_CHANNEL,
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

CUSTOMER = "wiki-enqueue-cust"


@pytest_asyncio.fixture
async def reset_db(live_db: None, settings: Settings) -> AsyncIterator[None]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'wiki-enqueue', 'h') ON CONFLICT DO NOTHING",
            CUSTOMER,
        )
    yield None


def _doc(doc_id: str, *, source_system: SourceSystem, doc_type: DocType) -> Document:
    now = datetime.now(UTC)
    body = "Body content for synthesis triage."
    return Document(
        doc_id=doc_id,
        customer_id=CUSTOMER,
        source_system=source_system,
        source_id=doc_id.split(":", 1)[-1],
        source_url=f"https://example.test/{doc_id}",
        doc_class=DocClass.RAW_SOURCE,
        doc_type=doc_type,
        content_type="text/markdown",
        content_hash=f"hash-{doc_id}",
        title=f"Title {doc_id}",
        body_preview=body[:120],
        body_size_bytes=len(body.encode("utf-8")),
        body_token_count=20,
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
                source_system=docs[0].source_system,
                principal_type=PrincipalType.WORKSPACE,
                principal_id=CUSTOMER,
                resource_type="document",
                resource_id=docs[0].doc_id,
                permission=Permission.READ,
                valid_from=now,
            )
        ],
    )


@pytest.fixture
def normalizer() -> Normalizer:
    """Real normalizer; embedder is the in-process stub when ANTHROPIC_API_KEY=''."""
    return Normalizer(make_default_context())


@pytest.mark.asyncio
async def test_persist_enqueues_wiki_synthesis_row(reset_db: None, normalizer: Normalizer) -> None:
    doc = _doc(
        "github:commit:abc",
        source_system=SourceSystem.GITHUB,
        doc_type=DocType.GITHUB_COMMIT,
    )
    await normalizer._persist(CUSTOMER, SourceSystem.GITHUB, _result(doc))

    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT doc_id, doc_version, source_system, doc_type, status "
            "FROM wiki_synthesis_queue WHERE customer_id = $1",
            CUSTOMER,
        )
    assert len(rows) == 1
    row = rows[0]
    assert row["doc_id"] == "github:commit:abc"
    assert row["doc_version"] == 1
    assert row["source_system"] == "github"
    assert row["doc_type"] == DocType.GITHUB_COMMIT.value
    assert row["status"] == "pending"


@pytest.mark.asyncio
async def test_persist_does_not_enqueue_wiki_self_writes(
    reset_db: None, normalizer: Normalizer
) -> None:
    """Wiki connector writes (Phase 1 manual upload, Phase 2 cron compile)
    must NOT feed back into the synthesis queue — otherwise the cron
    triages its own outputs."""
    doc = _doc(
        "wiki:runbook:auth",
        source_system=SourceSystem.WIKI,
        doc_type=DocType.WIKI_RUNBOOK,
    )
    await normalizer._persist(CUSTOMER, SourceSystem.WIKI, _result(doc))

    async with raw_conn() as conn:
        count = await conn.fetchval(
            "SELECT count(*) FROM wiki_synthesis_queue WHERE customer_id = $1",
            CUSTOMER,
        )
    assert count == 0


@pytest.mark.asyncio
async def test_persist_emits_pg_notify(
    reset_db: None, normalizer: Normalizer, settings: Settings
) -> None:
    """Open a separate LISTEN connection, persist, assert we received the notify
    on `wiki_synthesize` with the customer_id payload."""
    received: list[str] = []
    notify_event = asyncio.Event()

    listener = await asyncpg.connect(settings.database_url)
    try:

        def _on_notify(_conn, _pid, _channel, payload) -> None:
            received.append(payload)
            notify_event.set()

        await listener.add_listener(WIKI_SYNTHESIZE_CHANNEL, _on_notify)

        doc = _doc(
            "github:commit:notify",
            source_system=SourceSystem.GITHUB,
            doc_type=DocType.GITHUB_COMMIT,
        )
        await normalizer._persist(CUSTOMER, SourceSystem.GITHUB, _result(doc))

        # Notify is asynchronous — wait briefly.
        await asyncio.wait_for(notify_event.wait(), timeout=5.0)
        assert CUSTOMER in received
    finally:
        with contextlib_suppress():
            await listener.remove_listener(WIKI_SYNTHESIZE_CHANNEL, _on_notify)
        await listener.close()


@pytest.mark.asyncio
async def test_persist_enqueue_idempotent_on_redelivery(
    reset_db: None, normalizer: Normalizer
) -> None:
    """Re-persisting the same doc version (idempotent webhook) must not
    create a second queue row."""
    doc = _doc(
        "github:commit:idem",
        source_system=SourceSystem.GITHUB,
        doc_type=DocType.GITHUB_COMMIT,
    )
    await normalizer._persist(CUSTOMER, SourceSystem.GITHUB, _result(doc))
    await normalizer._persist(CUSTOMER, SourceSystem.GITHUB, _result(doc))

    async with raw_conn() as conn:
        count = await conn.fetchval(
            "SELECT count(*) FROM wiki_synthesis_queue WHERE customer_id = $1",
            CUSTOMER,
        )
    assert count == 1


# Tiny shim so the suppress idiom doesn't import contextlib at module top
def contextlib_suppress():
    import contextlib

    return contextlib.suppress(Exception)
