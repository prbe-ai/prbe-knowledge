"""Tests for /api/wiki/synthesize/status response shape.

The dashboard's status badge consumes this; the BFF will codegen
against it. Pin the field set + counts so adding new states later
doesn't silently break the contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from services.ingestion.main import app
from shared.config import Settings, get_settings
from shared.db import close_pool, raw_conn

CUSTOMER = "wiki-status-cust"


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest_asyncio.fixture
async def reset_db(live_db: None, settings: Settings) -> AsyncIterator[None]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash, preferences) "
            "VALUES ($1, 'wiki-status', 'h', $2::jsonb) "
            "ON CONFLICT (customer_id) DO UPDATE SET preferences = EXCLUDED.preferences",
            CUSTOMER,
            '{"wiki_generation_enabled": true}',
        )
    yield None


async def _seed_queue_row(*, doc_id: str, status: str, attempts: int = 0) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO wiki_synthesis_queue (
                customer_id, doc_id, doc_version, source_system, doc_type,
                status, attempts
            )
            VALUES ($1, $2, 1, 'github', 'github.commit', $3, $4)
            """,
            CUSTOMER,
            doc_id,
            status,
            attempts,
        )


async def _seed_run(*, stage: str, pages_updated: int, pages_created: int) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO wiki_synthesis_runs (
                customer_id, kind, stage, status,
                pages_updated, pages_created, finished_at
            )
            VALUES ($1, 'wake', $2, 'complete', $3, $4, NOW())
            """,
            CUSTOMER,
            stage,
            pages_updated,
            pages_created,
        )


@pytest.mark.asyncio
async def test_status_endpoint_exposes_all_typed_fields(reset_db: None) -> None:
    # Seed a row in each state we surface.
    await _seed_queue_row(doc_id="github:commit:p1", status="pending")
    await _seed_queue_row(doc_id="github:commit:p2", status="pending")
    await _seed_queue_row(doc_id="github:commit:t1", status="triaged")
    await _seed_queue_row(doc_id="github:commit:tg1", status="triaging")
    await _seed_queue_row(doc_id="github:commit:s1", status="synthesizing")
    await _seed_queue_row(doc_id="github:commit:f1", status="failed")
    await _seed_queue_row(doc_id="github:commit:vr1", status="verifier_rejected")
    await _seed_queue_row(doc_id="github:commit:vr2", status="verifier_rejected")
    # done / rejected are deliberately not surfaced — confirm by adding
    # one and asserting it doesn't bleed into any of the typed fields.
    await _seed_queue_row(doc_id="github:commit:d1", status="done")
    await _seed_queue_row(doc_id="github:commit:r1", status="rejected")

    httpx_client = httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t")
    await close_pool()
    async with app.router.lifespan_context(app), httpx_client as c:
        resp = await c.get(
            "/api/wiki/synthesize/status",
            headers={
                "X-Internal-Knowledge-Key": "test-internal-key",
                "X-Prbe-Customer": CUSTOMER,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pending_events"] == 2
    assert body["triaged_events"] == 1
    assert body["in_flight_events"] == 2  # triaging + synthesizing
    assert body["failed_events"] == 1
    assert body["verifier_rejected_events"] == 2


@pytest.mark.asyncio
async def test_status_last_run_filters_to_synthesis_stage(reset_db: None) -> None:
    """The triage worker opens a run row with stage='triage' and
    pages_*=0; the synthesis worker opens stage='synthesis' with the
    real pages_* counts. Status must surface the synthesis row, not
    the triage row, so the dashboard never shows '0 pages updated'
    while a synthesis run is in flight or has just completed.
    """
    # Triage run — newer started_at, but pages_*=0 by definition.
    await _seed_run(stage="triage", pages_updated=0, pages_created=0)
    # Synthesis run — older started_at, but real pages_*.
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO wiki_synthesis_runs (
                customer_id, kind, stage, status,
                pages_updated, pages_created,
                started_at, finished_at
            )
            VALUES ($1, 'wake', 'synthesis', 'complete', 3, 1,
                    NOW() - INTERVAL '1 minute', NOW())
            """,
            CUSTOMER,
        )

    httpx_client = httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t")
    await close_pool()
    async with app.router.lifespan_context(app), httpx_client as c:
        resp = await c.get(
            "/api/wiki/synthesize/status",
            headers={
                "X-Internal-Knowledge-Key": "test-internal-key",
                "X-Prbe-Customer": CUSTOMER,
            },
        )
    body = resp.json()
    # Picked the synthesis row (older but real), not the triage row
    # (newer, all zeros).
    assert body["last_run_pages_updated"] == 3
    assert body["last_run_pages_created"] == 1
    assert body["last_run_status"] == "complete"
