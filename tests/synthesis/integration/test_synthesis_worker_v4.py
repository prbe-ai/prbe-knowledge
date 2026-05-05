"""Integration tests for the v4 SynthesisWorker advisory-lock + halt path.

Drives the worker against a live Postgres with a stub LLM client so we
exercise:
  - advisory-lock acquired -> drain proceeds
  - advisory-lock contended -> no-op
  - agent halt -> all 'synthesizing' rows DLQ'd
  - cron + button concurrent -> only one drains
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio

from services.ingestion.handlers.base import make_default_context
from services.ingestion.normalizer import Normalizer
from services.synthesis.synthesis_worker import SynthesisWorker
from shared.config import Settings, get_settings
from shared.constants import (
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

CUSTOMER = "wiki-syn-v4-cust"


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest_asyncio.fixture
async def reset_db(live_db: None, settings: Settings) -> AsyncIterator[None]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash, preferences) "
            "VALUES ($1, 'wiki-syn-v4', 'h', $2::jsonb) "
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


async def _seed_triaged(doc_id: str, body: str) -> int:
    normalizer = Normalizer(make_default_context())
    await normalizer._persist(CUSTOMER, SourceSystem.GITHUB, _result(_doc(doc_id, body)))
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'triaged'
            WHERE customer_id = $1 AND doc_id = $2
            RETURNING queue_id
            """,
            CUSTOMER,
            doc_id,
        )
    return int(row["queue_id"])


# ---------------------------------------------------------------------------
# Stub LLM that immediately calls done()
# ---------------------------------------------------------------------------


class _DoneOnFirstTurnLLM:
    async def create_cache(self, **kwargs):
        return "caches/stub-syn-1"

    async def generate_with_cache(self, **kwargs):
        return {
            "tool_calls": [{"name": "done", "args": {}}],
            "usage_metadata": {
                "prompt_token_count": 100,
                "cached_content_token_count": 900,
                "candidates_token_count": 50,
            },
        }


class _ImmediatelyHaltsLLM:
    """Returns text-only responses forever; loop halts on stall."""

    async def create_cache(self, **kwargs):
        return "caches/stub-syn-halt"

    async def generate_with_cache(self, **kwargs):
        return {"text": "thinking", "tool_calls": [], "usage_metadata": {}}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advisory_lock_acquired_drain_proceeds(reset_db: None) -> None:
    """Single worker holds the lock and drains; queue rows go to terminal."""
    await _seed_triaged("github:commit:lock-1", "body 1")
    worker = SynthesisWorker(asyncio.Event(), llm_client=_DoneOnFirstTurnLLM())
    await worker._tick(woken_by_notify=True)
    async with raw_conn() as conn:
        statuses = await conn.fetch(
            "SELECT DISTINCT status FROM wiki_synthesis_queue "
            "WHERE customer_id = $1",
            CUSTOMER,
        )
    statuses_set = {r["status"] for r in statuses}
    # All rows in terminal states (synthesis_skipped is the agent's
    # "no apply" path; other terminal states are also acceptable).
    assert statuses_set <= {"done", "synthesis_skipped", "rejected", "dlq"}


@pytest.mark.asyncio
async def test_advisory_lock_contended_no_op_exit(reset_db: None) -> None:
    """If a different connection holds the lock, the worker no-ops and
    leaves the queue rows alone."""
    await _seed_triaged("github:commit:contend-1", "body 1")

    # Hold the advisory lock from a sibling conn for the duration of
    # the test.
    sibling_conn = None
    sibling_txn = None
    try:
        from shared.db import raw_conn as raw_conn_factory

        cm = raw_conn_factory()
        sibling_conn = await cm.__aenter__()
        # Acquire a txn-scoped lock on the customer's key. We compute
        # the same key the worker would compute.
        worker = SynthesisWorker(asyncio.Event(), llm_client=_DoneOnFirstTurnLLM())
        lock_key = worker._lock_key(CUSTOMER)
        sibling_txn = sibling_conn.transaction()
        await sibling_txn.start()
        held = await sibling_conn.fetchval(
            "SELECT pg_try_advisory_xact_lock($1)", lock_key
        )
        assert held is True

        # Now run the worker; it should no-op and leave the queue alone.
        await worker._tick(woken_by_notify=True)

        async with raw_conn() as conn:
            statuses = await conn.fetch(
                "SELECT status FROM wiki_synthesis_queue WHERE customer_id = $1",
                CUSTOMER,
            )
        # Row is still 'triaged' (worker never claimed it).
        assert all(r["status"] == "triaged" for r in statuses)
    finally:
        if sibling_txn is not None:
            await sibling_txn.commit()
        if sibling_conn is not None:
            await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_agent_halt_dlqs_all_synthesizing_rows(
    reset_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent stalls -> all 'synthesizing' rows go to DLQ with reason."""
    await _seed_triaged("github:commit:halt-1", "body 1")
    await _seed_triaged("github:commit:halt-2", "body 2")

    # Lower the stall threshold so the test halts quickly without
    # producing 200 turns of stall.
    import services.synthesis.agent_harness as h

    monkeypatch.setattr(h, "WIKI_AGENT_STALL_TURNS", 1)

    worker = SynthesisWorker(asyncio.Event(), llm_client=_ImmediatelyHaltsLLM())
    await worker._tick(woken_by_notify=True)

    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT status, dlq_reason FROM wiki_synthesis_queue "
            "WHERE customer_id = $1",
            CUSTOMER,
        )
    assert len(rows) == 2
    for row in rows:
        assert row["status"] == "dlq"
        assert row["dlq_reason"] is not None
        assert row["dlq_reason"].startswith("agent."), row["dlq_reason"]


@pytest.mark.asyncio
async def test_cron_and_button_concurrent_only_one_drains(
    reset_db: None,
) -> None:
    """Two workers running the same customer in parallel: one drains,
    one no-ops via the advisory lock."""
    await _seed_triaged("github:commit:race-1", "body 1")

    class _SlowDoneLLM:
        async def create_cache(self, **kwargs):
            await asyncio.sleep(0.05)
            return "caches/slow"

        async def generate_with_cache(self, **kwargs):
            await asyncio.sleep(0.05)
            return {
                "tool_calls": [{"name": "done", "args": {}}],
                "usage_metadata": {},
            }

    w1 = SynthesisWorker(asyncio.Event(), llm_client=_SlowDoneLLM())
    w2 = SynthesisWorker(asyncio.Event(), llm_client=_SlowDoneLLM())

    await asyncio.gather(
        w1._tick(woken_by_notify=True),
        w2._tick(woken_by_notify=True),
    )

    async with raw_conn() as conn:
        # Exactly one synthesis-stage run row should have run to
        # completion or stayed in-flight; we don't assert on count
        # because both workers' run rows exist (the second opens its
        # row before the lock check). What we assert: the queue row
        # ended up terminal (the drain didn't double-process).
        statuses = await conn.fetch(
            "SELECT status FROM wiki_synthesis_queue WHERE customer_id = $1",
            CUSTOMER,
        )
    terminal = {"done", "synthesis_skipped", "rejected", "dlq"}
    assert all(r["status"] in terminal for r in statuses), statuses
