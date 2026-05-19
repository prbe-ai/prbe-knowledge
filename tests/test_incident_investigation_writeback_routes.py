"""POST /api/incident-investigations route tests.

Asserts: 401 without internal key, source_system stamped correctly,
doc_type/doc_class correct, idempotency on redelivery, state row landed
in pending_review, parent_doc_id linked to incident doc.
"""
from __future__ import annotations

import json
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from services.ingestion.main import app
from shared.config import Settings, get_settings
from shared.db import close_pool, init_pool, raw_conn

pytestmark = pytest.mark.asyncio


_INTERNAL_KEY = "test-writeback-key"
_CUSTOMER = f"writeback-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", _INTERNAL_KEY)
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def client(live_db: None, settings: Settings) -> AsyncClient:
    """HTTP client with app lifespan running. live_db owns pool init/teardown."""
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash, r2_bucket) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (customer_id) DO NOTHING",
            _CUSTOMER, f"writeback test {_CUSTOMER}", "h", f"b-{_CUSTOMER}",
        )

    # Close the test-fixture pool so the app lifespan can init its own.
    await close_pool()
    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac,
        app.router.lifespan_context(app),
    ):
        yield ac
    # Re-init pool so live_db's teardown TRUNCATE can run.
    await init_pool(settings)


def _payload(
    *,
    version: int = 1,
    source: str = "pagerduty",
    incident_doc_id: str = "pd:incident:PD-INC-001",
    **overrides,
) -> dict:
    base = {
        "customer_id": _CUSTOMER,
        "incident_doc_id": incident_doc_id,
        "source_system": source,
        "source_event_id": f"{incident_doc_id}:incident.triggered",
        "version": version,
        "mode": "full",
        "title": "Investigation: Checkout DB pool exhausted",
        "body_markdown": (
            "## Hypothesis\n\nThe checkout-svc DB pool exhausted because "
            "of a runaway query in checkout-svc introduced in the v1.42 "
            "deploy.\n\n## Evidence\n\n- 2 deploys in last 24h\n- 18 open "
            "Sentry issues\n"
        ),
        "evidence": [
            {
                "source": "knowledge",
                "query": "checkout-svc recent deploys",
                "result_summary": "Two deploys in last 24h.",
                "linked_doc_ids": ["github:pr:1234"],
            }
        ],
        "narrative": "Pool exhausted; recent deploy correlated.",
        "tool_trace_run_id": "run-abc",
    }
    base.update(overrides)
    return base


def _hdrs() -> dict:
    return {
        "x-internal-knowledge-key": _INTERNAL_KEY,
        "content-type": "application/json",
    }


async def test_writeback_persists_doc_with_correct_source_system(
    client,
) -> None:
    body = _payload()
    r = await client.post("/api/incident-investigations", headers=_hdrs(), json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["state"] == "pending_review"
    assert data["duplicate"] is False
    assert data["report_doc_id"] == "pd:investigation:PD-INC-001:v1"

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT source_system::text, doc_type::text, doc_class::text, "
            "       parent_doc_id, metadata "
            "FROM documents WHERE customer_id = $1 AND doc_id = $2 "
            "AND valid_to IS NULL",
            _CUSTOMER, data["report_doc_id"],
        )
    assert row is not None
    assert row["source_system"] == "pagerduty"
    assert row["doc_type"] == "incident.investigation"
    assert row["doc_class"] == "agent_artifact"
    assert row["parent_doc_id"] == "pd:incident:PD-INC-001"


async def test_writeback_stamps_incident_io_source_system(
    client,
) -> None:
    body = _payload(
        source="incident_io",
        incident_doc_id="iio:incident:01ABCDEFGHIJ",
    )
    r = await client.post("/api/incident-investigations", headers=_hdrs(), json=body)
    assert r.status_code == 200
    assert r.json()["report_doc_id"] == "iio:investigation:01ABCDEFGHIJ:v1"

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT source_system::text FROM documents "
            "WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL",
            _CUSTOMER, r.json()["report_doc_id"],
        )
    assert row["source_system"] == "incident_io"


async def test_writeback_idempotent_on_redelivery(client) -> None:
    body = _payload()
    r1 = await client.post("/api/incident-investigations", headers=_hdrs(), json=body)
    r2 = await client.post("/api/incident-investigations", headers=_hdrs(), json=body)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["duplicate"] is False
    assert r2.json()["duplicate"] is True
    assert r1.json()["report_doc_id"] == r2.json()["report_doc_id"]


async def test_writeback_v2_appends_version(client) -> None:
    await client.post(
        "/api/incident-investigations",
        headers=_hdrs(),
        json=_payload(version=1),
    )
    r = await client.post(
        "/api/incident-investigations",
        headers=_hdrs(),
        json=_payload(
            version=2,
            prior_report_doc_id="pd:investigation:PD-INC-001:v1",
            reviewer_feedback="missed the recent deploy",
        ),
    )
    assert r.status_code == 200
    assert r.json()["report_doc_id"] == "pd:investigation:PD-INC-001:v2"
    assert r.json()["duplicate"] is False


async def test_writeback_creates_state_row_in_pending_review(
    client,
) -> None:
    body = _payload()
    await client.post("/api/incident-investigations", headers=_hdrs(), json=body)

    from services.ingestion.investigation_state import get_detail
    detail = await get_detail(_CUSTOMER, "pd:incident:PD-INC-001")
    assert detail is not None
    assert detail.state == "pending_review"
    assert detail.current_report_doc_id == "pd:investigation:PD-INC-001:v1"
    assert len(detail.versions) == 1


async def test_writeback_rejects_missing_internal_key(client) -> None:
    r = await client.post(
        "/api/incident-investigations",
        headers={"content-type": "application/json"},
        json=_payload(),
    )
    assert r.status_code == 401


async def test_writeback_rejects_wrong_internal_key(client) -> None:
    r = await client.post(
        "/api/incident-investigations",
        headers={"x-internal-knowledge-key": "wrong", "content-type": "application/json"},
        json=_payload(),
    )
    assert r.status_code == 401


async def test_writeback_rejects_missing_required_field(client, customer_id=_CUSTOMER) -> None:
    """Pydantic validation: missing `incident_doc_id` returns 422."""
    bad_payload = _payload()
    del bad_payload["incident_doc_id"]
    r = await client.post(
        "/api/incident-investigations", headers=_hdrs(), json=bad_payload,
    )
    assert r.status_code == 422


async def test_writeback_metadata_carries_evidence_and_narrative(
    client,
) -> None:
    """Structured evidence + narrative live in metadata jsonb (not body)
    so embeddings don't see JSON noise. Verify the round-trip."""
    body = _payload()
    r = await client.post("/api/incident-investigations", headers=_hdrs(), json=body)
    assert r.status_code == 200

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT metadata FROM documents WHERE customer_id = $1 AND doc_id = $2 "
            "AND valid_to IS NULL",
            _CUSTOMER, r.json()["report_doc_id"],
        )
    md = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
    assert md["mode"] == "full"
    assert md["version"] == 1
    assert md["narrative"] == "Pool exhausted; recent deploy correlated."
    assert md["tool_trace_run_id"] == "run-abc"
    assert len(md["evidence"]) == 1
    assert md["evidence"][0]["source"] == "knowledge"
    assert md["evidence"][0]["linked_doc_ids"] == ["github:pr:1234"]
