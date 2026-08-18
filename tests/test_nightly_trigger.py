"""Nightly trigger step B0: the queue-seed reconcile.

The reconcile is the guarantee behind every best-effort seeding path
(Normalizer enqueue swallow, settings-toggle BackgroundTasks, SQL flag
flips), so what is pinned here is the guarantee itself: every enabled
active tenant gets seeded, disabled and inactive tenants are skipped,
and one tenant's failure does not starve the rest.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engine.shared.db import raw_conn, with_tenant
from kb.synthesis import nightly_trigger

_NOW = datetime(2026, 8, 1, tzinfo=UTC)


async def _customer(customer_id: str, *, enabled: bool, status: str = "active") -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash, "
            "preferences, status) VALUES ($1, $2, 'h', $3::jsonb, $4)",
            customer_id,
            customer_id,
            '{"wiki_generation_enabled": true}' if enabled else "{}",
            status,
        )


async def _doc(customer_id: str, doc_id: str) -> None:
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            INSERT INTO documents
                (doc_id, version, customer_id, source_system, source_id,
                 source_url, doc_type, content_hash, created_at, updated_at,
                 valid_from, acl)
            VALUES ($1, 1, $2, 'slack', $1, 'https://example.test',
                    'slack.message', $3, $4, $4, $4, '{}'::jsonb)
            """,
            doc_id,
            customer_id,
            f"hash-{doc_id}",
            _NOW,
        )


async def _pending_count(customer_id: str) -> int:
    async with raw_conn() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM wiki_synthesis_queue "
            "WHERE customer_id = $1 AND status = 'pending'",
            customer_id,
        )


@pytest.mark.asyncio
async def test_reconcile_seeds_enabled_and_skips_disabled(live_db: None) -> None:
    await _customer("recon-on", enabled=True)
    await _customer("recon-off", enabled=False)
    await _customer("recon-inactive", enabled=True, status="suspended")
    await _doc("recon-on", "doc:on")
    await _doc("recon-off", "doc:off")
    await _doc("recon-inactive", "doc:inactive")

    summary = await nightly_trigger.reconcile_missing_queue_rows()

    assert summary == {"customers": 1, "seeded": 1, "failures": 0}
    assert await _pending_count("recon-on") == 1
    assert await _pending_count("recon-off") == 0
    assert await _pending_count("recon-inactive") == 0


@pytest.mark.asyncio
async def test_reconcile_is_idempotent(live_db: None) -> None:
    await _customer("recon-idem", enabled=True)
    await _doc("recon-idem", "doc:a")

    first = await nightly_trigger.reconcile_missing_queue_rows()
    second = await nightly_trigger.reconcile_missing_queue_rows()

    assert first["seeded"] == 1
    assert second["seeded"] == 0
    assert await _pending_count("recon-idem") == 1


@pytest.mark.asyncio
async def test_one_broken_tenant_does_not_starve_the_rest(
    live_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _customer("recon-bad", enabled=True)
    await _customer("recon-good", enabled=True)
    await _doc("recon-bad", "doc:bad")
    await _doc("recon-good", "doc:good")

    real_seed = nightly_trigger.seed_missing_docs

    async def exploding_seed(conn, customer_id):
        if customer_id == "recon-bad":
            raise RuntimeError("boom")
        return await real_seed(conn, customer_id)

    monkeypatch.setattr(nightly_trigger, "seed_missing_docs", exploding_seed)

    summary = await nightly_trigger.reconcile_missing_queue_rows()

    assert summary == {"customers": 2, "seeded": 1, "failures": 1}
    assert await _pending_count("recon-good") == 1
    assert await _pending_count("recon-bad") == 0
