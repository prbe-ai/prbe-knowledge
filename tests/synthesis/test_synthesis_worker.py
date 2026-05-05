"""Integration tests for SynthesisWorker.

Drives queue rows from status='triaged' through verifier + synthesize
to status='done' (or verifier_rejected) and asserts:
  - happy path writes a wiki page + regenerates the index
  - empty kept_doc_ids → verifier_rejected (not 'done')
  - non-empty kept_doc_ids → SynthesisInput filtered to kept set
  - cluster cap drops oldest events with a 'capped' synthesis_error
  - MANUAL_ENTRY pages are not clobbered
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from services.ingestion.handlers.base import make_default_context
from services.ingestion.main import app
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
from shared.db import close_pool, raw_conn
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    ACLSnapshotRow,
    Document,
    NormalizationResult,
)

CUSTOMER = "wiki-synth-cust"


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest_asyncio.fixture
async def reset_db(live_db: None, settings: Settings) -> AsyncIterator[None]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash, preferences) "
            "VALUES ($1, 'wiki-synth', 'h', $2::jsonb) "
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


def _make_client(
    synthesis_payload: dict,
    *,
    verifier_payload: dict | None = None,
) -> SimpleNamespace:
    """Mock dispatches by tool name. Verifier defaults to keeping every
    doc_id present in the cluster (extracted from the user message)."""

    def _extract_doc_ids(user_msg: str) -> list[str]:
        out: list[str] = []
        for line in user_msg.splitlines():
            if line.strip().startswith("doc_id: "):
                out.append(line.strip()[len("doc_id: ") :])
        return out

    async def create(*, model: str, **kwargs):
        tool_name = (kwargs.get("tools") or [{}])[0].get("name", "")
        if tool_name == "record_verifier_verdict":
            if verifier_payload is not None:
                return _tool_use_response("record_verifier_verdict", verifier_payload)
            user_msg = kwargs.get("messages", [{}])[0].get("content", "")
            kept = _extract_doc_ids(user_msg)
            return _tool_use_response(
                "record_verifier_verdict",
                {"kept_doc_ids": kept, "rationale_per_doc": {}},
            )
        return _tool_use_response("render_wiki_page", synthesis_payload)

    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(side_effect=create))
    return client


async def _seed_triaged(
    customer_id: str,
    doc_id: str,
    body: str,
    *,
    wiki_type: str,
    slug: str,
    action: str = "create",
    score: float = 8.0,
) -> int:
    """Persist a doc + flip its queue row directly to status='triaged'
    with the requested target. Returns the queue_id.
    """
    normalizer = Normalizer(make_default_context())
    await normalizer._persist(customer_id, SourceSystem.GITHUB, _result(_doc(doc_id, body)))
    targets_json = json.dumps(
        {
            "important": True,
            "score": score,
            "reason": "test",
            "targets": [{"wiki_type": wiki_type, "slug": slug, "action": action}],
        }
    )
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'triaged',
                triage_score = $2,
                triage_targets = $3::jsonb,
                triage_completed_at = NOW()
            WHERE customer_id = $1 AND doc_id = $4
              AND status IN ('pending', 'triaging')
            RETURNING queue_id
            """,
            customer_id,
            score,
            targets_json,
            doc_id,
        )
    return int(row["queue_id"])


@pytest.mark.asyncio
async def test_synthesis_worker_writes_wiki_page_and_regenerates_index(
    reset_db: None,
) -> None:
    await _seed_triaged(
        CUSTOMER,
        "github:commit:adopt-pgvector",
        "We chose pgvector inside Neon for embeddings.",
        wiki_type="decision",
        slug="adopt-pgvector",
    )
    synthesis_payload = {
        "title": "Adopt pgvector",
        "body_markdown": "We chose pgvector for embeddings inside Neon.",
        "summary": "Embedding store decision.",
        "commit_message": "Initial decision.",
    }
    client = _make_client(synthesis_payload)

    worker = SynthesisWorker(asyncio.Event(), anthropic_client=client)
    await worker._tick(woken_by_notify=True)

    async with raw_conn() as conn:
        statuses = await conn.fetch(
            "SELECT status FROM wiki_synthesis_queue WHERE customer_id = $1",
            CUSTOMER,
        )
        assert {r["status"] for r in statuses} == {"done"}

        page = await conn.fetchrow(
            """
            SELECT title, doc_class FROM documents
            WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL
            """,
            CUSTOMER,
            "wiki:decision:adopt-pgvector",
        )
        assert page is not None
        assert page["doc_class"] == DocClass.COMPILED_WIKI.value
        assert page["title"] == "Adopt pgvector"

        index_row = await conn.fetchrow(
            """
            SELECT title FROM documents
            WHERE customer_id = $1 AND doc_id = 'wiki:index:contents'
              AND valid_to IS NULL
            """,
            CUSTOMER,
        )
        assert index_row is not None


@pytest.mark.asyncio
async def test_synthesis_worker_marks_verifier_rejected_when_kept_empty(
    reset_db: None,
) -> None:
    await _seed_triaged(
        CUSTOMER,
        "github:commit:vr-discuss",
        "Some discussion that didn't lead to a decision.",
        wiki_type="decision",
        slug="auth-discussion",
    )
    forbidden_synthesis = {
        "title": "SHOULD NOT APPEAR",
        "body_markdown": "x",
        "summary": "x",
        "commit_message": "x",
    }
    verifier_rejection = {
        "kept_doc_ids": [],
        "drop_reason": "discussion did not produce a decision",
    }
    client = _make_client(forbidden_synthesis, verifier_payload=verifier_rejection)

    worker = SynthesisWorker(asyncio.Event(), anthropic_client=client)
    await worker._tick(woken_by_notify=True)

    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT status, synthesis_error FROM wiki_synthesis_queue WHERE customer_id = $1",
            CUSTOMER,
        )
    assert all(r["status"] == "verifier_rejected" for r in rows)
    assert all(r["synthesis_error"] and "did not produce" in r["synthesis_error"] for r in rows)

    async with raw_conn() as conn:
        page_count = await conn.fetchval(
            "SELECT count(*) FROM documents WHERE customer_id = $1 AND doc_id = $2",
            CUSTOMER,
            "wiki:decision:auth-discussion",
        )
    assert page_count == 0


@pytest.mark.asyncio
async def test_synthesis_worker_filters_to_kept_doc_ids(reset_db: None) -> None:
    await _seed_triaged(
        CUSTOMER,
        "github:commit:keep-1",
        "We adopted pgvector for embeddings.",
        wiki_type="decision",
        slug="adopt-pgvector",
    )
    await _seed_triaged(
        CUSTOMER,
        "github:commit:drop-1",
        "Unrelated CSS tweak.",
        wiki_type="decision",
        slug="adopt-pgvector",
    )
    synthesis_payload = {
        "title": "Adopt pgvector",
        "body_markdown": "Decision.",
        "summary": "Embedding store decision.",
        "commit_message": "Initial.",
    }
    verifier_payload = {
        "kept_doc_ids": ["github:commit:keep-1"],
        "rationale_per_doc": {
            "github:commit:keep-1": "states the decision",
            "github:commit:drop-1": "unrelated",
        },
    }
    client = _make_client(synthesis_payload, verifier_payload=verifier_payload)

    worker = SynthesisWorker(asyncio.Event(), anthropic_client=client)
    await worker._tick(woken_by_notify=True)

    create_calls = client.messages.create.await_args_list
    synth_calls = [
        c
        for c in create_calls
        if (c.kwargs.get("tools") or [{}])[0].get("name") == "render_wiki_page"
    ]
    assert len(synth_calls) == 1
    user_msg = synth_calls[0].kwargs["messages"][0]["content"]
    assert "github:commit:keep-1" in user_msg
    assert "github:commit:drop-1" not in user_msg

    async with raw_conn() as conn:
        statuses = await conn.fetch(
            "SELECT doc_id, status FROM wiki_synthesis_queue "
            "WHERE customer_id = $1 ORDER BY doc_id",
            CUSTOMER,
        )
    by_doc = {r["doc_id"]: r["status"] for r in statuses}
    # Both queue rows mark done — the cluster as a whole completed even
    # though only one event fed the synthesis call.
    assert by_doc["github:commit:keep-1"] == "done"
    assert by_doc["github:commit:drop-1"] == "done"


@pytest.mark.asyncio
async def test_synthesis_worker_truncates_cluster_above_max(
    reset_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import services.synthesis.synthesis_worker as sw

    monkeypatch.setattr(sw, "WIKI_SYNTHESIS_CLUSTER_MAX_EVENTS", 2)

    for i in range(4):
        await _seed_triaged(
            CUSTOMER,
            f"github:commit:trunc-{i}",
            f"event {i}",
            wiki_type="runbook",
            slug="trunc-runbook",
        )

    synthesis_payload = {
        "title": "Trunc runbook",
        "body_markdown": "body",
        "summary": "summary",
        "commit_message": "msg",
    }
    client = _make_client(synthesis_payload)

    worker = SynthesisWorker(asyncio.Event(), anthropic_client=client)
    await worker._tick(woken_by_notify=True)

    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT synthesis_error FROM wiki_synthesis_queue WHERE customer_id = $1",
            CUSTOMER,
        )
    errors = [r["synthesis_error"] for r in rows]
    capped = sum(1 for e in errors if e and "capped" in e)
    not_capped = sum(1 for e in errors if not e or "capped" not in e)
    assert capped == 2, f"expected 2 capped rows, got {capped}: {errors}"
    assert not_capped == 2, f"expected 2 non-capped rows, got {not_capped}: {errors}"


@pytest.mark.asyncio
async def test_synthesis_worker_does_not_clobber_manual_entry(reset_db: None) -> None:
    """When a MANUAL_ENTRY page already exists for a target slug, the
    synthesis worker must NOT regenerate the body. Queue rows are marked
    done with a skip reason; the human-authored page stays exactly as it
    was."""
    httpx_client = httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t")
    await close_pool()
    async with app.router.lifespan_context(app), httpx_client as c:
        await c.put(
            "/api/wiki/pages/runbook/auth-flow",
            json={
                "title": "Auth flow runbook (human authored)",
                "body": "Hand-written instructions.",
                "author_id": "richard@prbe.ai",
            },
            headers={
                "X-Internal-Knowledge-Key": "test-internal-key",
                "X-Prbe-Customer": CUSTOMER,
            },
        )

    await _seed_triaged(
        CUSTOMER,
        "github:commit:auth-update",
        "We updated the auth flow today.",
        wiki_type="runbook",
        slug="auth-flow",
        action="update",
    )

    forbidden_synthesis = {
        "title": "SHOULD NOT APPEAR",
        "body_markdown": "SHOULD NOT APPEAR",
        "summary": "x",
        "commit_message": "x",
    }
    client = _make_client(forbidden_synthesis)

    worker = SynthesisWorker(asyncio.Event(), anthropic_client=client)
    await worker._tick(woken_by_notify=True)

    async with raw_conn() as conn:
        page = await conn.fetchrow(
            """
            SELECT title, doc_class FROM documents
            WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL
            """,
            CUSTOMER,
            "wiki:runbook:auth-flow",
        )
        assert page is not None
        assert page["doc_class"] == DocClass.MANUAL_ENTRY.value
        assert page["title"] == "Auth flow runbook (human authored)"

        skip_rows = await conn.fetch(
            """
            SELECT status, synthesis_error FROM wiki_synthesis_queue
            WHERE customer_id = $1 AND doc_id = 'github:commit:auth-update'
            """,
            CUSTOMER,
        )
        assert skip_rows
        assert all(r["status"] == "done" for r in skip_rows)
        assert all(
            r["synthesis_error"] and "MANUAL_ENTRY" in r["synthesis_error"] for r in skip_rows
        )
