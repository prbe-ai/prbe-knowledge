"""Review endpoint tests: list, detail, approve, reject.

Each test seeds an investigation via the writeback route, then drives
the review endpoints. Verifies state machine transitions, feedback
required on reject, 404 on missing rows, and authorization via the
internal key.
"""
from __future__ import annotations

import os
import uuid

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

from services.ingestion.main import app


pytestmark = pytest.mark.asyncio


_INTERNAL_KEY = "test-review-key"


def _new_customer_id() -> str:
    return f"review-test-{uuid.uuid4().hex[:8]}"


async def _seed_customer(customer_id: str) -> None:
    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash, r2_bucket) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (customer_id) DO NOTHING",
            customer_id, f"review test {customer_id}", "h", f"b-{customer_id}",
        )
    finally:
        await conn.close()


async def _cleanup_customer(customer_id: str) -> None:
    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute("DELETE FROM chunks WHERE customer_id = $1", customer_id)
        await conn.execute(
            "DELETE FROM incident_investigations WHERE customer_id = $1", customer_id,
        )
        await conn.execute("DELETE FROM documents WHERE customer_id = $1", customer_id)
        await conn.execute("DELETE FROM customers WHERE customer_id = $1", customer_id)
    finally:
        await conn.close()


@pytest.fixture
async def customer_id(monkeypatch, live_db):
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", _INTERNAL_KEY)
    from shared.config import get_settings
    get_settings.cache_clear()
    cid = _new_customer_id()
    await _seed_customer(cid)
    try:
        yield cid
    finally:
        await _cleanup_customer(cid)
        get_settings.cache_clear()


@pytest.fixture
async def client():
    # Match the D.3 writeback test fixture pattern: open the lifespan so
    # app.state.normalizer is initialized + pools are bound. The lifespan
    # context-manager wraps the test.
    from shared.db import close_pool, init_pool
    await close_pool()
    await init_pool()
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as ac:
            yield ac


_INTERNAL_HEADERS = {
    "x-internal-knowledge-key": _INTERNAL_KEY,
    "content-type": "application/json",
}


def _writeback_payload(
    customer_id: str,
    *,
    incident_doc_id: str = "pd:incident:R-001",
    version: int = 1,
) -> dict:
    return {
        "customer_id": customer_id,
        "incident_doc_id": incident_doc_id,
        "source_system": "pagerduty",
        "source_event_id": f"{incident_doc_id}:incident.triggered",
        "version": version,
        "mode": "full",
        "title": "Investigation report",
        "body_markdown": "## Hypothesis\n\nThing broke.\n",
        "evidence": [],
    }


async def _seed_investigation(
    client: AsyncClient,
    customer_id: str,
    *,
    incident_doc_id: str = "pd:incident:R-001",
    version: int = 1,
) -> None:
    r = await client.post(
        "/api/incident-investigations",
        headers=_INTERNAL_HEADERS,
        json=_writeback_payload(customer_id, incident_doc_id=incident_doc_id, version=version),
    )
    assert r.status_code == 200, r.text


async def test_list_returns_seeded_investigation(client, customer_id) -> None:
    await _seed_investigation(client, customer_id)
    r = await client.get(
        "/api/incident-investigations",
        headers={"x-internal-knowledge-key": _INTERNAL_KEY},
        params={"customer_id": customer_id},
    )
    assert r.status_code == 200
    items = r.json()
    assert any(i["incident_doc_id"] == "pd:incident:R-001" for i in items)
    assert any(i["state"] == "pending_review" for i in items)


async def test_list_filters_by_state(client, customer_id) -> None:
    await _seed_investigation(client, customer_id, incident_doc_id="pd:incident:R-002")
    await _seed_investigation(client, customer_id, incident_doc_id="pd:incident:R-003")
    # Approve one
    await client.post(
        "/api/incident-investigations/pd:incident:R-002/approve",
        headers=_INTERNAL_HEADERS,
        params={"customer_id": customer_id},
        json={"reviewer_id": "user-42"},
    )
    r_pending = await client.get(
        "/api/incident-investigations",
        headers={"x-internal-knowledge-key": _INTERNAL_KEY},
        params={"customer_id": customer_id, "state": "pending_review"},
    )
    r_approved = await client.get(
        "/api/incident-investigations",
        headers={"x-internal-knowledge-key": _INTERNAL_KEY},
        params={"customer_id": customer_id, "state": "approved"},
    )
    pending_ids = {i["incident_doc_id"] for i in r_pending.json()}
    approved_ids = {i["incident_doc_id"] for i in r_approved.json()}
    assert "pd:incident:R-003" in pending_ids
    assert "pd:incident:R-002" not in pending_ids
    assert "pd:incident:R-002" in approved_ids


async def test_detail_returns_full_row(client, customer_id) -> None:
    await _seed_investigation(client, customer_id)
    r = await client.get(
        "/api/incident-investigations/pd:incident:R-001",
        headers={"x-internal-knowledge-key": _INTERNAL_KEY},
        params={"customer_id": customer_id},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "pending_review"
    assert body["incident_doc_id"] == "pd:incident:R-001"
    assert len(body["versions"]) == 1
    assert body["versions"][0]["version"] == 1


async def test_detail_returns_404_when_missing(client, customer_id) -> None:
    r = await client.get(
        "/api/incident-investigations/pd:incident:DOES_NOT_EXIST",
        headers={"x-internal-knowledge-key": _INTERNAL_KEY},
        params={"customer_id": customer_id},
    )
    assert r.status_code == 404


async def test_approve_flips_state_and_records_reviewer(client, customer_id) -> None:
    await _seed_investigation(client, customer_id)
    r = await client.post(
        "/api/incident-investigations/pd:incident:R-001/approve",
        headers=_INTERNAL_HEADERS,
        params={"customer_id": customer_id},
        json={"reviewer_id": "user-42"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "approved"
    assert body["reviewer_id"] == "user-42"
    assert body["versions"][-1]["decision"] == "approved"
    assert body["versions"][-1]["reviewed_by"] == "user-42"


async def test_approve_returns_404_when_missing(client, customer_id) -> None:
    r = await client.post(
        "/api/incident-investigations/pd:incident:DOES_NOT_EXIST/approve",
        headers=_INTERNAL_HEADERS,
        params={"customer_id": customer_id},
        json={"reviewer_id": "user-42"},
    )
    assert r.status_code == 404


async def test_reject_requires_nonempty_feedback(client, customer_id) -> None:
    await _seed_investigation(client, customer_id)
    r = await client.post(
        "/api/incident-investigations/pd:incident:R-001/reject",
        headers=_INTERNAL_HEADERS,
        params={"customer_id": customer_id},
        json={"reviewer_id": "user-42", "feedback": ""},
    )
    assert r.status_code == 422


async def test_reject_records_feedback_and_flips_state(client, customer_id) -> None:
    await _seed_investigation(client, customer_id)
    r = await client.post(
        "/api/incident-investigations/pd:incident:R-001/reject",
        headers=_INTERNAL_HEADERS,
        params={"customer_id": customer_id},
        json={"reviewer_id": "user-42", "feedback": "missed the recent deploy"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "rejected"
    assert body["versions"][-1]["decision"] == "rejected"
    assert body["versions"][-1]["feedback"] == "missed the recent deploy"


async def test_reject_returns_404_when_missing(client, customer_id) -> None:
    r = await client.post(
        "/api/incident-investigations/pd:incident:DOES_NOT_EXIST/reject",
        headers=_INTERNAL_HEADERS,
        params={"customer_id": customer_id},
        json={"reviewer_id": "user-42", "feedback": "anything"},
    )
    assert r.status_code == 404


async def test_endpoints_reject_missing_internal_key(client, customer_id) -> None:
    """All four review endpoints must require X-Internal-Knowledge-Key."""
    routes = [
        ("GET", "/api/incident-investigations", None),
        ("GET", "/api/incident-investigations/pd:incident:R-001", None),
        ("POST", "/api/incident-investigations/pd:incident:R-001/approve",
         {"reviewer_id": "u"}),
        ("POST", "/api/incident-investigations/pd:incident:R-001/reject",
         {"reviewer_id": "u", "feedback": "x"}),
    ]
    for method, url, json_body in routes:
        r = await client.request(
            method, url,
            params={"customer_id": customer_id},
            headers={"content-type": "application/json"},
            json=json_body,
        )
        assert r.status_code == 401, f"{method} {url} should require key"


async def test_customer_id_query_param_required(client) -> None:
    """Missing customer_id query param → 422 from FastAPI validation."""
    r = await client.get(
        "/api/incident-investigations",
        headers={"x-internal-knowledge-key": _INTERNAL_KEY},
    )
    assert r.status_code == 422
