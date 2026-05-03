"""End-to-end integration test for the wiki synthesis cron.

Seeds two raw documents through `Normalizer._persist` (which also
auto-populates `wiki_synthesis_queue` via the hot-path hook), runs one cron
tick with mocked LLM responses, and asserts:

- queue rows transitioned to `done`
- a `wiki.runbook` doc was persisted with `doc_class=COMPILED_WIKI`
- the `wiki.index` doc was regenerated
- a `wiki_synthesis_runs` row was opened, counted, and closed

LLM calls are mocked at the AsyncAnthropic boundary (passed in via the
constructor). No network, no API credits.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from services.ingestion.handlers.base import make_default_context
from services.ingestion.normalizer import Normalizer
from services.synthesis.wiki_cron import WikiSynthesisCron
from shared.config import Settings
from shared.constants import (
    HAIKU_MODEL,
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
from shared.storage import get_store

CUSTOMER = "wiki-cron-cust"


@pytest_asyncio.fixture
async def reset_db(live_db: None, settings: Settings) -> AsyncIterator[None]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'wiki-cron', 'h') ON CONFLICT DO NOTHING",
            CUSTOMER,
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
        metadata={"body": body},
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


def _haiku_response(payload: dict) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", name="record_triage", input=payload)
    return SimpleNamespace(content=[block])


def _sonnet_response(payload: dict) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", name="render_wiki_page", input=payload)
    return SimpleNamespace(content=[block])


def _make_mock_client(triage_payload: dict, synthesis_payload: dict) -> SimpleNamespace:
    """Routes Haiku calls -> triage, Sonnet calls -> synthesis."""

    async def create(*, model: str, **kwargs):
        if model == HAIKU_MODEL:
            return _haiku_response(triage_payload)
        return _sonnet_response(synthesis_payload)

    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(side_effect=create))
    return client


@pytest.mark.asyncio
async def test_cron_drains_seeds_and_writes_wiki_page(
    reset_db: None,
) -> None:
    # ---- arrange: seed 2 raw docs (auto-enqueue via _persist hook).
    normalizer = Normalizer(make_default_context())
    body_a = "We refactored the auth flow to use OAuth client credentials."
    body_b = "Also rotated the auth signing key on 2026-05-02 incident."
    await normalizer._persist(
        CUSTOMER,
        SourceSystem.GITHUB,
        _result(_doc("github:commit:aaa", body_a)),
    )
    await normalizer._persist(
        CUSTOMER,
        SourceSystem.GITHUB,
        _result(_doc("github:commit:bbb", body_b)),
    )

    # Confirm both queue rows landed.
    async with raw_conn() as conn:
        pending = await conn.fetchval(
            "SELECT count(*) FROM wiki_synthesis_queue "
            "WHERE customer_id = $1 AND status = 'pending'",
            CUSTOMER,
        )
    assert pending == 2

    # ---- arrange: mock LLMs.
    queue_rows = await _read_queue_ids(CUSTOMER)
    triage_payload = {
        "verdicts": {
            str(qid): {
                "important": True,
                "score": 8.0,
                "targets": [
                    {
                        "wiki_type": "runbook",
                        "slug": "auth-flow",
                        "action": "create",
                    }
                ],
                "reason": "auth refactor",
            }
            for qid in queue_rows
        }
    }
    synthesis_payload = {
        "title": "Auth flow runbook",
        "body_markdown": (
            "We refactored auth to OAuth client credentials.\n\n"
            "Signing key rotated [[Decision: rotate-auth-key]]."
        ),
        "summary": "How the auth flow is wired and what to do when it breaks.",
        "frontmatter": {"owner": "alice"},
        "commit_message": "Initial compilation from 2 commits.",
    }
    client = _make_mock_client(triage_payload, synthesis_payload)

    cron = WikiSynthesisCron(
        ctx=make_default_context(),
        store=get_store(),
        wake_event=asyncio.Event(),
        anthropic_client=client,
    )

    # ---- act: one drain.
    await cron._tick(woken_by_notify=True)

    # ---- assert: queue rows marked done.
    async with raw_conn() as conn:
        statuses = await conn.fetch(
            "SELECT status FROM wiki_synthesis_queue WHERE customer_id = $1",
            CUSTOMER,
        )
        assert {row["status"] for row in statuses} == {"done"}

        # ---- assert: wiki:runbook:auth-flow exists with COMPILED_WIKI.
        page = await conn.fetchrow(
            """
            SELECT doc_class, doc_type, title, metadata
            FROM documents
            WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL
            """,
            CUSTOMER,
            "wiki:runbook:auth-flow",
        )
        assert page is not None
        assert page["doc_class"] == DocClass.COMPILED_WIKI.value
        assert page["doc_type"] == DocType.WIKI_RUNBOOK.value
        assert page["title"] == "Auth flow runbook"

        # ---- assert: index regenerated.
        index_row = await conn.fetchrow(
            """
            SELECT title, doc_type
            FROM documents
            WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL
            """,
            CUSTOMER,
            "wiki:index:contents",
        )
        assert index_row is not None
        assert index_row["doc_type"] == DocType.WIKI_INDEX.value

        # ---- assert: synthesis run row closed.
        run_rows = await conn.fetch(
            "SELECT status, events_total, events_kept, pages_created, "
            "pages_updated, finished_at "
            "FROM wiki_synthesis_runs WHERE customer_id = $1",
            CUSTOMER,
        )
        assert len(run_rows) == 1
        run = run_rows[0]
        assert run["status"] == "complete"
        assert run["events_total"] == 2
        assert run["events_kept"] == 2
        assert run["pages_created"] == 1
        assert run["pages_updated"] == 0
        assert run["finished_at"] is not None


@pytest.mark.asyncio
async def test_cron_rejects_low_score_events(reset_db: None) -> None:
    """Events scored below the threshold land in 'rejected' and never
    trigger synthesis."""
    normalizer = Normalizer(make_default_context())
    body = "Routine bot ack."
    await normalizer._persist(
        CUSTOMER,
        SourceSystem.GITHUB,
        _result(_doc("github:commit:noisy", body)),
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
    client = _make_mock_client(triage_payload, {})  # no synthesis call expected

    cron = WikiSynthesisCron(
        ctx=make_default_context(),
        store=get_store(),
        wake_event=asyncio.Event(),
        anthropic_client=client,
    )
    await cron._tick(woken_by_notify=True)

    async with raw_conn() as conn:
        status = await conn.fetchval(
            "SELECT status FROM wiki_synthesis_queue WHERE customer_id = $1",
            CUSTOMER,
        )
    assert status == "rejected"

    async with raw_conn() as conn:
        wiki_count = await conn.fetchval(
            """
            SELECT count(*) FROM documents
            WHERE customer_id = $1 AND source_system = 'wiki'
              AND doc_type = ANY(ARRAY[
                'wiki.service_card','wiki.decision','wiki.feature','wiki.runbook'
              ])
            """,
            CUSTOMER,
        )
    assert wiki_count == 0


async def _read_queue_ids(customer_id: str) -> list[int]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT queue_id FROM wiki_synthesis_queue WHERE customer_id = $1 ORDER BY queue_id",
            customer_id,
        )
    return [row["queue_id"] for row in rows]
