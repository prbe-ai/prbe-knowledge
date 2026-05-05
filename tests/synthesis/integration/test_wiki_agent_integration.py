"""Integration tests for WikiAgentRuntime tool handlers + commit flow.

Exercise the runtime against a live Postgres + the in-process embedder
stub. Mocks Gemini SDK; runtime tool handlers are pure DB / state, no
LLM calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from services.ingestion.handlers.base import make_default_context
from services.ingestion.normalizer import Normalizer
from services.synthesis.wiki_agent import WikiAgentRuntime
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

CUSTOMER = "wiki-agent-int-cust"


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest_asyncio.fixture
async def reset_db(live_db: None, settings: Settings) -> AsyncIterator[None]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash, preferences) "
            "VALUES ($1, 'wiki-agent-int', 'h', $2::jsonb) "
            "ON CONFLICT (customer_id) DO UPDATE SET preferences = EXCLUDED.preferences",
            CUSTOMER,
            '{"wiki_generation_enabled": true}',
        )
    yield None


def _doc(doc_id: str, body: str, *, source_ts: datetime | None = None) -> Document:
    now = source_ts or datetime.now(UTC)
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
        metadata={"created_at": now.isoformat()},
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


async def _seed_synthesizing(
    doc_id: str,
    body: str,
    *,
    source_ts: datetime | None = None,
) -> int:
    """Persist a doc + flip its queue row to status='synthesizing'."""
    normalizer = Normalizer(make_default_context())
    await normalizer._persist(
        CUSTOMER, SourceSystem.GITHUB, _result(_doc(doc_id, body, source_ts=source_ts))
    )
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'synthesizing'
            WHERE customer_id = $1 AND doc_id = $2
            RETURNING queue_id
            """,
            CUSTOMER,
            doc_id,
        )
    return int(row["queue_id"])


async def _open_run() -> int:
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO wiki_synthesis_runs (customer_id, kind, stage)
            VALUES ($1, 'wake', 'synthesis')
            RETURNING run_id
            """,
            CUSTOMER,
        )
    return int(row["run_id"])


def _make_runtime(run_id: int) -> WikiAgentRuntime:
    return WikiAgentRuntime(
        CUSTOMER,
        agent_run_id="rt-1",
        run_id=run_id,
        run_kind="wake",
    )


# ---------------------------------------------------------------------------
# next_events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_next_events_empty_queue(reset_db: None) -> None:
    run_id = await _open_run()
    rt = _make_runtime(run_id)
    out = await rt.dispatch_tool("next_events", {"count": 50})
    assert out["events"] == []
    assert out["remaining"] == 0
    assert out["drain_complete"] is True


@pytest.mark.asyncio
async def test_tool_next_events_partial_fill(reset_db: None) -> None:
    for i in range(3):
        await _seed_synthesizing(f"github:commit:p-{i}", f"body {i}")
    run_id = await _open_run()
    rt = _make_runtime(run_id)
    out = await rt.dispatch_tool("next_events", {"count": 2})
    assert len(out["events"]) == 2
    assert out["remaining"] == 1
    assert out["drain_complete"] is False


@pytest.mark.asyncio
async def test_tool_next_events_excludes_applied_and_skipped(reset_db: None) -> None:
    qids = []
    for i in range(3):
        qid = await _seed_synthesizing(f"github:commit:e-{i}", f"body {i}")
        qids.append(qid)
    run_id = await _open_run()
    rt = _make_runtime(run_id)

    # Stage an update that applies one event.
    await rt.dispatch_tool(
        "update_page",
        {
            "wiki_type": "decision",
            "slug": "x",
            "body_markdown": "b",
            "summary": "s",
            "commit_message": "m",
            "applied_queue_ids": [qids[0]],
        },
    )
    # Skip another.
    await rt.dispatch_tool(
        "skip_events", {"queue_ids": [qids[1]], "reason": "noise"}
    )
    out = await rt.dispatch_tool("next_events", {"count": 50})
    returned_qids = {ev["queue_id"] for ev in out["events"]}
    assert qids[2] in returned_qids
    assert qids[0] not in returned_qids
    assert qids[1] not in returned_qids


@pytest.mark.asyncio
async def test_tool_next_events_ordered_by_source_ts_asc(reset_db: None) -> None:
    base = datetime(2026, 5, 4, 8, 0, tzinfo=UTC)
    qids = []
    # Insert out of order; assert manifest sorts ASC.
    for offset in [60, 0, 30]:
        qid = await _seed_synthesizing(
            f"github:commit:s-{offset}",
            "body",
            source_ts=base + timedelta(minutes=offset),
        )
        qids.append(qid)
    # Force source_ts on the queue rows since the doc fixture uses NOW;
    # the queue_row.source_ts came from documents.created_at which we
    # do set above.
    async with raw_conn() as conn:
        for qid, offset in zip(qids, [60, 0, 30], strict=True):
            await conn.execute(
                "UPDATE wiki_synthesis_queue SET source_ts = $2 "
                "WHERE customer_id = $1 AND queue_id = $3",
                CUSTOMER,
                base + timedelta(minutes=offset),
                qid,
            )
    run_id = await _open_run()
    rt = _make_runtime(run_id)
    out = await rt.dispatch_tool("next_events", {"count": 50})
    timestamps = [ev["source_ts"] for ev in out["events"]]
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# read_page
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_read_page_on_disk_returns_db_row(reset_db: None) -> None:
    """Seed a manual-entry wiki page directly via Normalizer, then
    read_page should return its body."""
    from services.ingestion.handlers.wiki import build_normalization_result
    from shared.models import WebhookEvent

    received_at = datetime.now(UTC)
    payload = {
        "wiki": {
            "wiki_type": "decision",
            "slug": "ondisk",
            "title": "On Disk",
            "body": "this is on disk",
            "frontmatter": {},
            "doc_class": DocClass.MANUAL_ENTRY.value,
            "is_delete": False,
            "updated_at": received_at.isoformat(),
            "summary": "exists on disk",
        }
    }
    event = WebhookEvent(
        customer_id=CUSTOMER,
        source_system=SourceSystem.WIKI,
        source_event_id="decision:ondisk:edit",
        received_at=received_at,
        payload_s3_key="",
        payload_s3_keys=[],
        raw_payload=payload,
        headers={},
    )
    norm = build_normalization_result(event)
    normalizer = Normalizer(make_default_context())
    await normalizer._persist(CUSTOMER, SourceSystem.WIKI, norm)

    run_id = await _open_run()
    rt = _make_runtime(run_id)
    out = await rt.dispatch_tool(
        "read_page", {"wiki_type": "decision", "slug": "ondisk"}
    )
    assert out["is_staged"] is False
    assert "on disk" in (out["body_markdown"] or "")


@pytest.mark.asyncio
async def test_tool_read_page_staged_for_update_returns_staged(reset_db: None) -> None:
    run_id = await _open_run()
    rt = _make_runtime(run_id)
    await rt.dispatch_tool(
        "update_page",
        {
            "wiki_type": "decision",
            "slug": "stagedu",
            "body_markdown": "STAGED BODY",
            "summary": "s",
            "commit_message": "m",
            "applied_queue_ids": [],
        },
    )
    out = await rt.dispatch_tool(
        "read_page", {"wiki_type": "decision", "slug": "stagedu"}
    )
    assert out["is_staged"] is True
    assert out["stage_kind"] == "update"
    assert out["body_markdown"] == "STAGED BODY"


@pytest.mark.asyncio
async def test_tool_read_page_staged_for_create_returns_staged(reset_db: None) -> None:
    run_id = await _open_run()
    rt = _make_runtime(run_id)
    await rt.dispatch_tool(
        "create_page",
        {
            "wiki_type": "decision",
            "slug": "stagedc",
            "title": "T",
            "body_markdown": "STAGED CREATE",
            "summary": "s",
            "commit_message": "m",
            "applied_queue_ids": [],
        },
    )
    out = await rt.dispatch_tool(
        "read_page", {"wiki_type": "decision", "slug": "stagedc"}
    )
    assert out["is_staged"] is True
    assert out["stage_kind"] == "create"
    assert out["title"] == "T"


# ---------------------------------------------------------------------------
# get_event_body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_get_event_body_fits_one_page(reset_db: None) -> None:
    qid = await _seed_synthesizing("github:commit:short", "Short body content.")
    run_id = await _open_run()
    rt = _make_runtime(run_id)
    out = await rt.dispatch_tool("get_event_body", {"queue_id": qid})
    assert out["page"] == 1
    assert out["total_pages"] == 1
    assert out["truncated"] is False


@pytest.mark.asyncio
async def test_tool_get_event_body_needs_pagination(reset_db: None) -> None:
    big = "x" * 14000
    qid = await _seed_synthesizing("github:commit:big", big)
    run_id = await _open_run()
    rt = _make_runtime(run_id)
    out = await rt.dispatch_tool("get_event_body", {"queue_id": qid})
    assert out["total_pages"] >= 2
    assert out["truncated"] is True


# ---------------------------------------------------------------------------
# done() commit + index regen
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_done_atomic_commit_one_version_per_slug(
    reset_db: None,
) -> None:
    qid = await _seed_synthesizing(
        "github:commit:q1", "We chose pgvector inside Neon."
    )
    run_id = await _open_run()
    rt = _make_runtime(run_id)
    await rt.dispatch_tool(
        "update_page",
        {
            "wiki_type": "decision",
            "slug": "adopt-pg",
            "body_markdown": "We chose pgvector inside Neon for embeddings.",
            "summary": "Embeddings store choice.",
            "commit_message": "Initial decision",
            "applied_queue_ids": [qid],
        },
    )
    # No on-disk page exists yet; the runtime treats update for a missing
    # page as a no-op (logs warning). So convert to create_page for this
    # smoke test.
    rt._pending_updates.clear()
    await rt.dispatch_tool(
        "create_page",
        {
            "wiki_type": "decision",
            "slug": "adopt-pg",
            "title": "Adopt pgvector",
            "body_markdown": "We chose pgvector for embeddings.",
            "summary": "Choice.",
            "frontmatter": {},
            "commit_message": "Initial",
            "applied_queue_ids": [qid],
        },
    )
    await rt.dispatch_tool("done", {})

    async with raw_conn() as conn:
        page_versions = await conn.fetch(
            "SELECT version FROM documents "
            "WHERE customer_id = $1 AND doc_id = $2 ORDER BY version",
            CUSTOMER,
            "wiki:decision:adopt-pg",
        )
        # Exactly one page version (the agent staged a single update).
        assert len(page_versions) == 1
        # The queue row marked done.
        statuses = await conn.fetch(
            "SELECT status FROM wiki_synthesis_queue "
            "WHERE customer_id = $1",
            CUSTOMER,
        )
        # All of the customer's queue rows should be in a terminal state
        # (done / synthesis_skipped / etc.).
        terminal = {"done", "synthesis_skipped", "rejected"}
        assert all(r["status"] in terminal for r in statuses)


@pytest.mark.asyncio
async def test_tool_done_marks_applied_and_skipped_correctly(
    reset_db: None,
) -> None:
    qid_apply = await _seed_synthesizing("github:commit:apply", "applied body")
    qid_skip = await _seed_synthesizing("github:commit:skip", "skipped body")
    run_id = await _open_run()
    rt = _make_runtime(run_id)
    await rt.dispatch_tool(
        "create_page",
        {
            "wiki_type": "decision",
            "slug": "marked",
            "title": "T",
            "body_markdown": "b",
            "summary": "s",
            "commit_message": "m",
            "applied_queue_ids": [qid_apply],
        },
    )
    await rt.dispatch_tool(
        "skip_events", {"queue_ids": [qid_skip], "reason": "noise"}
    )
    await rt.dispatch_tool("done", {})

    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT queue_id, status FROM wiki_synthesis_queue "
            "WHERE customer_id = $1",
            CUSTOMER,
        )
    by_qid = {int(r["queue_id"]): r["status"] for r in rows}
    assert by_qid[qid_apply] == "done"
    assert by_qid[qid_skip] == "synthesis_skipped"


@pytest.mark.asyncio
async def test_tool_done_regenerates_wiki_index(reset_db: None) -> None:
    qid = await _seed_synthesizing("github:commit:idx-1", "body")
    run_id = await _open_run()
    rt = _make_runtime(run_id)
    await rt.dispatch_tool(
        "create_page",
        {
            "wiki_type": "decision",
            "slug": "indexed",
            "title": "Indexed Page",
            "body_markdown": "b",
            "summary": "indexed summary",
            "commit_message": "m",
            "applied_queue_ids": [qid],
        },
    )
    await rt.dispatch_tool("done", {})

    async with raw_conn() as conn:
        index_row = await conn.fetchrow(
            "SELECT title FROM documents "
            "WHERE customer_id = $1 AND doc_id = 'wiki:index:contents' "
            "AND valid_to IS NULL",
            CUSTOMER,
        )
    assert index_row is not None


# ---------------------------------------------------------------------------
# discard() DLQs synthesizing rows (via persistence helper)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discard_dlqs_triaged_rows_with_reason(reset_db: None) -> None:
    """The runtime's discard() drops in-memory state; the worker calls
    dlq_agent_synthesizing_rows separately to flip rows to DLQ. This
    test exercises both halves end-to-end."""
    from services.synthesis import persistence

    qid = await _seed_synthesizing("github:commit:dlq", "body")
    run_id = await _open_run()
    rt = _make_runtime(run_id)
    await rt.dispatch_tool(
        "update_page",
        {
            "wiki_type": "decision",
            "slug": "x",
            "body_markdown": "b",
            "summary": "s",
            "commit_message": "m",
            "applied_queue_ids": [qid],
        },
    )
    await rt.discard()
    n = await persistence.dlq_agent_synthesizing_rows(
        CUSTOMER, reason="agent.test_halt"
    )
    assert n >= 1
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, dlq_reason FROM wiki_synthesis_queue "
            "WHERE customer_id = $1 AND queue_id = $2",
            CUSTOMER,
            qid,
        )
    assert row["status"] == "dlq"
    assert row["dlq_reason"] == "agent.test_halt"
