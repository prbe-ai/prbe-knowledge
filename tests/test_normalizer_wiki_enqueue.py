"""Verify Normalizer._persist enqueues the right wiki_synthesis_queue rows.

The hot path is now INSERT-only — the redesign moved daytime synthesis
to a nightly batch fired by the wiki-cron fly app, so Normalizer no
longer fires `pg_notify` on every webhook. This file pins both the
positive (rows are inserted, idempotent on redelivery, opt-in gated)
and the negative (NO pg_notify on either the legacy `wiki_synthesize`
channel or the new `wiki_synthesize_pending` channel).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import asyncpg
import pytest
import pytest_asyncio

from services.ingestion.handlers.base import make_default_context
from services.ingestion.normalizer import Normalizer
from shared.config import Settings
from shared.constants import (
    WIKI_PENDING_CHANNEL,
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
            "INSERT INTO customers(customer_id, display_name, api_key_hash, preferences) "
            "VALUES ($1, 'wiki-enqueue', 'h', $2::jsonb) "
            "ON CONFLICT (customer_id) DO UPDATE SET preferences = EXCLUDED.preferences",
            CUSTOMER,
            '{"wiki_generation_enabled": true}',
        )
    yield None


@pytest_asyncio.fixture
async def reset_db_opt_out(live_db: None, settings: Settings) -> AsyncIterator[None]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash, preferences) "
            "VALUES ($1, 'wiki-enqueue-off', 'h', '{}'::jsonb) "
            "ON CONFLICT (customer_id) DO UPDATE SET preferences = EXCLUDED.preferences",
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
    """Wiki connector writes (Phase 1 manual upload, synthesis worker
    compile) must NOT feed back into the synthesis queue — otherwise
    the cron triages its own outputs."""
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
async def test_persist_does_not_emit_pg_notify(
    reset_db: None, normalizer: Normalizer, settings: Settings
) -> None:
    """REGRESSION GATE: Normalizer._persist must NOT fire a pg_notify on
    either the new `wiki_synthesize_pending` channel or the legacy
    `wiki_synthesize` channel.

    Synthesis is now nightly-batch — the wiki-cron fly app fires NOTIFY
    at 02:00 UTC, plus a manual trigger endpoint for the dashboard
    button. Re-introducing daytime NOTIFY here breaks the slow-moving
    knowledge-base scope of the wiki.
    """
    received: list[tuple[str, str]] = []
    notify_event = asyncio.Event()

    listener = await asyncpg.connect(settings.database_url)
    try:

        def _on_notify(_conn, _pid, channel, payload) -> None:
            received.append((channel, payload))
            notify_event.set()

        # Listen on both the new channel and the legacy one — either
        # firing should fail this test.
        await listener.add_listener(WIKI_PENDING_CHANNEL, _on_notify)
        await listener.add_listener("wiki_synthesize", _on_notify)

        doc = _doc(
            "github:commit:no-notify",
            source_system=SourceSystem.GITHUB,
            doc_type=DocType.GITHUB_COMMIT,
        )
        await normalizer._persist(CUSTOMER, SourceSystem.GITHUB, _result(doc))

        # Wait briefly — a real enqueue+notify would arrive well under
        # this; absence after the wait is the assertion.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(notify_event.wait(), timeout=0.5)
        assert received == [], f"Normalizer._persist fired unexpected pg_notify: {received}"

        # The row must still have been INSERTed.
        async with raw_conn() as conn:
            count = await conn.fetchval(
                "SELECT count(*) FROM wiki_synthesis_queue WHERE customer_id = $1",
                CUSTOMER,
            )
        assert count == 1
    finally:
        with contextlib.suppress(Exception):
            await listener.remove_listener(WIKI_PENDING_CHANNEL, _on_notify)
            await listener.remove_listener("wiki_synthesize", _on_notify)
        await listener.close()


@pytest.mark.asyncio
async def test_persist_enqueue_idempotent_on_redelivery(
    reset_db: None, normalizer: Normalizer
) -> None:
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


@pytest.mark.asyncio
async def test_persist_skips_enqueue_when_wiki_generation_disabled(
    reset_db_opt_out: None, normalizer: Normalizer, settings: Settings
) -> None:
    """OPT-IN CONTRACT regression gate.

    A tenant whose `customers.preferences.wiki_generation_enabled` is
    missing or false must NOT have queue rows appended.
    """
    doc = _doc(
        "github:commit:opt-out",
        source_system=SourceSystem.GITHUB,
        doc_type=DocType.GITHUB_COMMIT,
    )
    await normalizer._persist(CUSTOMER, SourceSystem.GITHUB, _result(doc))

    async with raw_conn() as conn:
        count = await conn.fetchval(
            "SELECT count(*) FROM wiki_synthesis_queue WHERE customer_id = $1",
            CUSTOMER,
        )
    assert count == 0
