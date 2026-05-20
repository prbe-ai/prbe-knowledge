"""POST / GET /api/incident-evidence-packs route tests.

Lifecycle:
- POST writes the EvidencePack to incident_investigations.evidence_pack
  (jsonb). Idempotent — second POST returns duplicate=True.
- GET reads the cached pack back. 404 when the row is missing OR the
  column is NULL.
- Both endpoints require X-Internal-Knowledge-Key.

Live Postgres + the ingestion app lifespan required (mirrors
test_incident_investigation_writeback_routes.py).
"""
from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from services.ingestion.main import app
from shared.config import Settings, get_settings
from shared.db import close_pool, init_pool, raw_conn

pytestmark = pytest.mark.asyncio


_INTERNAL_KEY = "test-evidence-pack-key"
_CUSTOMER = f"evpack-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", _INTERNAL_KEY)
    get_settings.cache_clear()


async def _seed_incident_row(customer_id: str, incident_doc_id: str) -> None:
    """The evidence-pack writeback UPDATES an existing investigation
    row; we seed a minimal one rather than going through the full
    investigation writeback path (which would call the embedder).
    """
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO incident_investigations "
            "(customer_id, incident_doc_id, state) "
            "VALUES ($1, $2, 'pending_review') "
            "ON CONFLICT (customer_id, incident_doc_id) DO NOTHING",
            customer_id, incident_doc_id,
        )


@pytest_asyncio.fixture
async def client(live_db: None, settings: Settings) -> AsyncClient:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash, r2_bucket) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (customer_id) DO NOTHING",
            _CUSTOMER, f"evpack {_CUSTOMER}", "h", f"b-{_CUSTOMER}",
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


def _pack(mode: str = "full") -> dict:
    return {
        "timeline_events": [
            {"at": "2026-05-17T10:00:00Z", "event": "incident triggered", "source": "pd"},
        ],
        "resolution_actions": [
            {"at": "2026-05-17T10:30:00Z", "actor": "alice", "action": "restarted pool", "source": "slack"},
        ],
        "recovery_signals": [],
        "post_resolution_discussion": ["thread excerpt"],
        "related_doc_ids": ["github:pr:1234"],
        "deploys_in_window": [],
        "similar_past_incidents": [],
        "free_form_findings": "checkout pool exhaustion correlated to v1.42 deploy",
        "mode": mode,
    }


def _payload(*, incident_doc_id: str = "pd:incident:PD-EVP-001", **overrides) -> dict:
    base = {
        "customer_id": _CUSTOMER,
        "incident_doc_id": incident_doc_id,
        "evidence_pack": _pack(),
    }
    base.update(overrides)
    return base


async def test_writeback_persists_pack(client) -> None:
    incident = "pd:incident:PD-EVP-001"
    await _seed_incident_row(_CUSTOMER, incident)

    r = await client.post(
        "/api/incident-evidence-packs", headers=_hdrs(),
        json=_payload(incident_doc_id=incident),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["incident_doc_id"] == incident
    assert data["duplicate"] is False

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT evidence_pack::text AS pack FROM incident_investigations "
            "WHERE customer_id = $1 AND incident_doc_id = $2",
            _CUSTOMER, incident,
        )
    assert row is not None and row["pack"] is not None
    import json
    pack = json.loads(row["pack"])
    assert pack["mode"] == "full"
    assert pack["free_form_findings"].startswith("checkout pool")


async def test_writeback_marks_duplicate_on_redelivery(client) -> None:
    incident = "pd:incident:PD-EVP-002"
    await _seed_incident_row(_CUSTOMER, incident)

    r1 = await client.post(
        "/api/incident-evidence-packs", headers=_hdrs(),
        json=_payload(incident_doc_id=incident),
    )
    r2 = await client.post(
        "/api/incident-evidence-packs", headers=_hdrs(),
        json=_payload(incident_doc_id=incident),
    )
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["duplicate"] is False
    assert r2.json()["duplicate"] is True


async def test_writeback_unknown_incident_returns_404(client) -> None:
    r = await client.post(
        "/api/incident-evidence-packs", headers=_hdrs(),
        json=_payload(incident_doc_id="pd:incident:DOES-NOT-EXIST"),
    )
    assert r.status_code == 404, r.text


async def test_writeback_rejects_bad_internal_key(client) -> None:
    r = await client.post(
        "/api/incident-evidence-packs",
        headers={"x-internal-knowledge-key": "wrong", "content-type": "application/json"},
        json=_payload(),
    )
    assert r.status_code == 401


async def test_get_returns_pack_when_present(client) -> None:
    incident = "pd:incident:PD-EVP-GET-1"
    await _seed_incident_row(_CUSTOMER, incident)
    await client.post(
        "/api/incident-evidence-packs", headers=_hdrs(),
        json=_payload(incident_doc_id=incident),
    )

    r = await client.get(
        "/api/incident-evidence-packs",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER, "incident_doc_id": incident},
    )
    assert r.status_code == 200, r.text
    pack = r.json()
    assert pack["mode"] == "full"
    assert len(pack["timeline_events"]) == 1


async def test_get_returns_404_when_empty(client) -> None:
    """Row exists but evidence_pack column is NULL."""
    incident = "pd:incident:PD-EVP-GET-2"
    await _seed_incident_row(_CUSTOMER, incident)

    r = await client.get(
        "/api/incident-evidence-packs",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER, "incident_doc_id": incident},
    )
    assert r.status_code == 404


async def test_get_returns_404_when_row_missing(client) -> None:
    r = await client.get(
        "/api/incident-evidence-packs",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER, "incident_doc_id": "pd:incident:NO-ROW"},
    )
    assert r.status_code == 404
