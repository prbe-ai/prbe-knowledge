"""POST /api/wiki-artifacts route tests.

Asserts the full Plan A Component 5 writeback contract:

- documents row lands at visibility='draft' (and chunks alongside),
  not 'approved' — that flip is gated on the review approve action.
- Target-doc validation for corrections runs against documents with
  visibility='approved'.
- artifact_doc_id is stable across redeliveries; re-POSTs short-
  circuit with duplicate=True.
- prior_artifact_doc_id seeds the next version's artifact_doc_id +
  the wiki_review_queue row's parent_artifact_doc_id link.
- Stub-mode artifacts land in failed_pending_review instead of
  pending_review.
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from services.ingestion.main import app
from shared.config import Settings, get_settings
from shared.db import close_pool, init_pool, raw_conn

pytestmark = pytest.mark.asyncio


_INTERNAL_KEY = "test-wiki-writeback-key"
_CUSTOMER = f"wiki-wb-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", _INTERNAL_KEY)
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def client(live_db: None, settings: Settings) -> AsyncClient:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash, r2_bucket) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (customer_id) DO NOTHING",
            _CUSTOMER, f"wiki-wb {_CUSTOMER}", "h", f"b-{_CUSTOMER}",
        )

    await close_pool()
    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac,
        app.router.lifespan_context(app),
    ):
        yield ac
    await init_pool(settings)


def _hdrs() -> dict:
    return {
        "x-internal-knowledge-key": _INTERNAL_KEY,
        "content-type": "application/json",
    }


def _payload(
    *,
    artifact_kind: str = "postmortem",
    target_doc_id: str | None = None,
    incident_doc_id: str = "pd:incident:PD-INC-001",
    mode: str = "full",
    prior_artifact_doc_id: str | None = None,
    title: str = "Postmortem: Checkout DB pool exhausted",
    body_markdown: str = (
        "## Summary\n\nCheckout-svc DB pool exhausted at 10:00 UTC.\n\n"
        "## Root cause\n\nRunaway query in v1.42 deploy.\n\n"
        "## Timeline\n\n- 10:00 alert fired\n- 10:30 pool restarted\n"
    ),
    **overrides,
) -> dict:
    md = {
        "mode": mode,
        "evidence_refs": ["evidence-1"],
        "rationale": "first draft",
        "tool_trace_run_id": "trace-run-001",
    }
    if prior_artifact_doc_id is not None:
        md["prior_artifact_doc_id"] = prior_artifact_doc_id
    base = {
        "customer_id": _CUSTOMER,
        "incident_doc_id": incident_doc_id,
        "investigation_doc_id": "pd:investigation:PD-INC-001:v1",
        "artifact_kind": artifact_kind,
        "target_doc_id": target_doc_id,
        "title": title,
        "body_markdown": body_markdown,
        "metadata": md,
    }
    base.update(overrides)
    return base


async def _seed_target_doc(
    customer_id: str,
    doc_id: str,
    *,
    visibility: str = "approved",
) -> None:
    """Insert a minimal documents row for correction-target validation.

    The correction writeback only checks documents (not chunks), so a
    bare row is enough. Uses an INSERT bypassing the Normalizer — that
    keeps the test from depending on the chunker's tokenization.
    """
    now = datetime.now(UTC)
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at,
                acl, metadata, entities, attachments, doc_references,
                normalizer_version, visibility
            )
            VALUES (
                $1, 1, $2,
                'wiki', $1, '',
                'compiled_wiki', 'wiki.runbook', 'text/markdown',
                $3, 0, 0,
                $4, $4, $4, $4,
                '{"principals":[],"captured_at":"2026-05-17T00:00:00Z"}'::jsonb,
                '{}'::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb,
                'v1', $5
            )
            ON CONFLICT (customer_id, doc_id, version) DO NOTHING
            """,
            doc_id, customer_id, f"hash-{doc_id}", now, visibility,
        )


async def test_postmortem_writeback_creates_draft_doc_and_queue_row(
    client,
) -> None:
    r = await client.post(
        "/api/wiki-artifacts", headers=_hdrs(), json=_payload(),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["artifact_doc_id"] == "pd:wiki.postmortem:PD-INC-001:v1"
    assert data["state"] == "pending_review"
    assert data["duplicate"] is False

    async with raw_conn() as conn:
        doc = await conn.fetchrow(
            "SELECT visibility, doc_type, doc_class, parent_doc_id, source_system "
            "FROM documents WHERE customer_id = $1 AND doc_id = $2 "
            "AND valid_to IS NULL",
            _CUSTOMER, data["artifact_doc_id"],
        )
        chunk_rows = await conn.fetch(
            "SELECT visibility FROM chunks "
            "WHERE customer_id = $1 AND doc_id = $2 "
            "AND valid_to IS NULL",
            _CUSTOMER, data["artifact_doc_id"],
        )
        queue = await conn.fetchrow(
            "SELECT state FROM wiki_review_queue "
            "WHERE customer_id = $1 AND artifact_doc_id = $2",
            _CUSTOMER, data["artifact_doc_id"],
        )
    assert doc is not None
    assert doc["visibility"] == "draft"
    assert doc["doc_type"] == "wiki.postmortem"
    assert doc["doc_class"] == "agent_artifact"
    assert doc["parent_doc_id"] == "pd:incident:PD-INC-001"
    assert doc["source_system"] == "pagerduty"
    assert len(chunk_rows) > 0
    assert all(c["visibility"] == "draft" for c in chunk_rows)
    assert queue is not None
    assert queue["state"] == "pending_review"


async def test_correction_requires_existing_target(client) -> None:
    """Correction writeback against a phantom target_doc_id → 422.

    The check runs before the document INSERT so no draft lands.
    """
    r = await client.post(
        "/api/wiki-artifacts", headers=_hdrs(),
        json=_payload(
            artifact_kind="correction",
            target_doc_id="wiki:runbook:does-not-exist",
            title="Correction: outdated runbook",
            body_markdown="### Correction\n\nUpdated playbook step.\n",
        ),
    )
    assert r.status_code == 422, r.text
    assert "not exist or is not readable" in r.json()["detail"]


async def test_correction_with_existing_target_sets_parent_doc_id(
    client,
) -> None:
    target = "wiki:runbook:checkout-pool"
    await _seed_target_doc(_CUSTOMER, target, visibility="approved")

    r = await client.post(
        "/api/wiki-artifacts", headers=_hdrs(),
        json=_payload(
            artifact_kind="correction",
            target_doc_id=target,
            title="Correction: stale pool size guidance",
            body_markdown="### Correction\n\nPool size is 25, not 10.\n",
        ),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    # Correction artifact_doc_id carries the target-hash segment.
    assert data["artifact_doc_id"].startswith("pd:wiki.correction:PD-INC-001:")
    assert data["artifact_doc_id"].endswith(":v1")

    async with raw_conn() as conn:
        doc = await conn.fetchrow(
            "SELECT parent_doc_id, doc_type, metadata FROM documents "
            "WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL",
            _CUSTOMER, data["artifact_doc_id"],
        )
    assert doc is not None
    assert doc["parent_doc_id"] == target
    assert doc["doc_type"] == "wiki.correction"
    md = json.loads(doc["metadata"]) if isinstance(doc["metadata"], str) else doc["metadata"]
    assert md["target_doc_id"] == target


async def test_knowledge_page_has_null_parent_doc_id(client) -> None:
    r = await client.post(
        "/api/wiki-artifacts", headers=_hdrs(),
        json=_payload(
            artifact_kind="knowledge_page",
            incident_doc_id="iio:incident:01ABCDEFGHIJ",
            title="Knowledge: db pool sizing",
            body_markdown="### Pool sizing\n\nDefault is 25 connections.\n",
        ),
    )
    assert r.status_code == 200, r.text
    artifact_doc_id = r.json()["artifact_doc_id"]
    assert artifact_doc_id == "iio:wiki.knowledge_page:01ABCDEFGHIJ:v1"

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT parent_doc_id, source_system FROM documents "
            "WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL",
            _CUSTOMER, artifact_doc_id,
        )
    assert row is not None
    assert row["parent_doc_id"] is None
    assert row["source_system"] == "incident_io"


async def test_writeback_idempotency(client) -> None:
    body = _payload(incident_doc_id="pd:incident:PD-INC-IDEM")
    r1 = await client.post("/api/wiki-artifacts", headers=_hdrs(), json=body)
    r2 = await client.post("/api/wiki-artifacts", headers=_hdrs(), json=body)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["duplicate"] is False
    assert r2.json()["duplicate"] is True
    assert r1.json()["artifact_doc_id"] == r2.json()["artifact_doc_id"]

    async with raw_conn() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM wiki_review_queue "
            "WHERE customer_id = $1 AND artifact_doc_id = $2",
            _CUSTOMER, r1.json()["artifact_doc_id"],
        )
    assert n == 1


async def test_stub_mode_sets_failed_pending_review(client) -> None:
    r = await client.post(
        "/api/wiki-artifacts", headers=_hdrs(),
        json=_payload(
            incident_doc_id="pd:incident:PD-INC-STUB",
            mode="stub",
        ),
    )
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "failed_pending_review"

    async with raw_conn() as conn:
        queue = await conn.fetchrow(
            "SELECT state FROM wiki_review_queue "
            "WHERE customer_id = $1 AND artifact_doc_id = $2",
            _CUSTOMER, r.json()["artifact_doc_id"],
        )
    assert queue["state"] == "failed_pending_review"


async def test_rerun_writeback_links_to_prior(client) -> None:
    """v2 writeback with prior_artifact_doc_id should:
       - bump the version segment to :v2
       - link the queue row's parent_artifact_doc_id to v1

    We simulate the reject (without going through the route) by calling
    mark_rejected directly, which mirrors the lifecycle the orchestrator
    sees in production.
    """
    incident = "pd:incident:PD-INC-RERUN"
    r1 = await client.post(
        "/api/wiki-artifacts", headers=_hdrs(),
        json=_payload(incident_doc_id=incident),
    )
    assert r1.status_code == 200
    v1_doc_id = r1.json()["artifact_doc_id"]
    assert v1_doc_id == "pd:wiki.postmortem:PD-INC-RERUN:v1"

    # Simulate a reject so the state machine is in rejected before the
    # re-run lands. Direct call to the state layer keeps the test
    # focused on the writeback path's version-linking behavior.
    from services.post_approval.wiki_review_state import mark_rejected
    await mark_rejected(
        customer_id=_CUSTOMER,
        artifact_doc_id=v1_doc_id,
        reviewer_id="reviewer-1",
        feedback="please expand the timeline",
    )

    r2 = await client.post(
        "/api/wiki-artifacts", headers=_hdrs(),
        json=_payload(
            incident_doc_id=incident,
            prior_artifact_doc_id=v1_doc_id,
            title="Postmortem v2",
            body_markdown="### v2 body with expanded timeline\n",
        ),
    )
    assert r2.status_code == 200, r2.text
    v2_doc_id = r2.json()["artifact_doc_id"]
    assert v2_doc_id == "pd:wiki.postmortem:PD-INC-RERUN:v2"
    assert r2.json()["duplicate"] is False

    async with raw_conn() as conn:
        v2_row = await conn.fetchrow(
            "SELECT parent_artifact_doc_id FROM wiki_review_queue "
            "WHERE customer_id = $1 AND artifact_doc_id = $2",
            _CUSTOMER, v2_doc_id,
        )
    assert v2_row is not None
    assert v2_row["parent_artifact_doc_id"] == v1_doc_id


async def test_writeback_malformed_prior_artifact_doc_id_returns_422(
    client,
) -> None:
    """A malformed prior_artifact_doc_id (no :vN suffix) must 422,
    not silently collide with v1 via the idempotency probe."""
    r = await client.post(
        "/api/wiki-artifacts", headers=_hdrs(),
        json=_payload(
            incident_doc_id="pd:incident:PD-INC-MALFORMED",
            prior_artifact_doc_id="pd:wiki.postmortem:PD-INC-X-NO-VERSION-SUFFIX",
        ),
    )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"].lower()
    assert "malformed" in detail or "missing" in detail


async def test_writeback_non_integer_version_suffix_returns_422(
    client,
) -> None:
    """prior_artifact_doc_id with :vXX where XX isn't an int must 422."""
    r = await client.post(
        "/api/wiki-artifacts", headers=_hdrs(),
        json=_payload(
            incident_doc_id="pd:incident:PD-INC-BADVER",
            prior_artifact_doc_id="pd:wiki.postmortem:PD-INC-X:vWHAT",
        ),
    )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"].lower()
    assert "non-integer" in detail or "version" in detail


async def test_bad_internal_key_returns_401(client) -> None:
    r = await client.post(
        "/api/wiki-artifacts",
        headers={"x-internal-knowledge-key": "wrong", "content-type": "application/json"},
        json=_payload(),
    )
    assert r.status_code == 401
