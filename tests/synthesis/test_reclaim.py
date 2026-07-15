"""Tests for the wiki-queue reclaim path.

Covers:
- Stale 'triaging' row with attempts < cap → reset to 'pending'.
- Stale 'synthesizing' row with attempts < cap → reset to 'triaged'.
- Stale row with attempts >= cap → terminal 'failed' (poison-pill cap).
- Fresh heartbeat (within threshold) → untouched.
- Rows in 'pending'/'triaged'/'done' → never swept.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from engine.shared.config import Settings
from engine.shared.db import raw_conn
from kb.synthesis import persistence

CUSTOMER = "wiki-reclaim-cust"


@pytest_asyncio.fixture
async def reset_db(live_db: None, settings: Settings) -> AsyncIterator[None]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash, preferences) "
            "VALUES ($1, 'wiki-reclaim', 'h', $2::jsonb) "
            "ON CONFLICT (customer_id) DO UPDATE SET preferences = EXCLUDED.preferences",
            CUSTOMER,
            '{"wiki_generation_enabled": true}',
        )
    yield None


async def _seed_queue_row(
    *,
    customer_id: str,
    doc_id: str,
    status: str,
    attempts: int,
    heartbeat_age_seconds: int | None,
) -> int:
    """Direct INSERT into wiki_synthesis_queue with the requested state.

    Bypasses the normal claim path so each test can pin status,
    attempts, and heartbeat_at independently.
    """
    heartbeat_at = (
        datetime.now(UTC) - timedelta(seconds=heartbeat_age_seconds)
        if heartbeat_age_seconds is not None
        else None
    )
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO wiki_synthesis_queue (
                customer_id, doc_id, doc_version, source_system, doc_type,
                status, attempts, heartbeat_at
            )
            VALUES ($1, $2, 1, 'github', 'github.commit', $3, $4, $5)
            RETURNING queue_id
            """,
            customer_id,
            doc_id,
            status,
            attempts,
            heartbeat_at,
        )
    return int(row["queue_id"])


@pytest.mark.asyncio
async def test_reclaim_resets_stale_triaging_to_pending(reset_db: None) -> None:
    qid = await _seed_queue_row(
        customer_id=CUSTOMER,
        doc_id="github:commit:stale-triage",
        status="triaging",
        attempts=1,
        heartbeat_age_seconds=900,  # 15 min stale, threshold 600s
    )
    retried, failed = await persistence.reclaim_stuck_rows(
        threshold_seconds=600,
        max_attempts=3,
    )
    assert retried == 1
    assert failed == 0

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, attempts, heartbeat_at, triage_error "
            "FROM wiki_synthesis_queue WHERE queue_id = $1",
            qid,
        )
    assert row["status"] == "pending"
    # Reclaim doesn't increment attempts directly; the next claim does.
    assert row["attempts"] == 1
    assert row["heartbeat_at"] is None
    assert row["triage_error"] and "reclaimed: heartbeat stale" in row["triage_error"]


@pytest.mark.asyncio
async def test_reclaim_resets_stale_synthesizing_to_triaged(reset_db: None) -> None:
    qid = await _seed_queue_row(
        customer_id=CUSTOMER,
        doc_id="github:commit:stale-synth",
        status="synthesizing",
        attempts=2,
        heartbeat_age_seconds=900,
    )
    retried, failed = await persistence.reclaim_stuck_rows(
        threshold_seconds=600,
        max_attempts=3,
    )
    assert retried == 1
    assert failed == 0

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, synthesis_error FROM wiki_synthesis_queue WHERE queue_id = $1",
            qid,
        )
    assert row["status"] == "triaged"
    assert row["synthesis_error"] and "reclaimed: heartbeat stale" in row["synthesis_error"]


@pytest.mark.asyncio
async def test_reclaim_dead_letters_when_attempts_cap_reached(reset_db: None) -> None:
    """attempts >= max_attempts → terminal 'failed' so ops can investigate.

    Prevents poison rows from looping forever and burning LLM spend.
    """
    qid = await _seed_queue_row(
        customer_id=CUSTOMER,
        doc_id="github:commit:poison",
        status="synthesizing",
        attempts=3,  # at cap
        heartbeat_age_seconds=900,
    )
    retried, failed = await persistence.reclaim_stuck_rows(
        threshold_seconds=600,
        max_attempts=3,
    )
    assert retried == 0
    assert failed == 1

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status, synthesis_error FROM wiki_synthesis_queue WHERE queue_id = $1",
            qid,
        )
    assert row["status"] == "failed"
    assert row["synthesis_error"] and "attempts exhausted" in row["synthesis_error"]


@pytest.mark.asyncio
async def test_reclaim_skips_fresh_heartbeat(reset_db: None) -> None:
    qid = await _seed_queue_row(
        customer_id=CUSTOMER,
        doc_id="github:commit:fresh",
        status="triaging",
        attempts=1,
        heartbeat_age_seconds=60,  # well within 600s threshold
    )
    retried, failed = await persistence.reclaim_stuck_rows(
        threshold_seconds=600,
        max_attempts=3,
    )
    assert retried == 0
    assert failed == 0

    async with raw_conn() as conn:
        status = await conn.fetchval(
            "SELECT status FROM wiki_synthesis_queue WHERE queue_id = $1",
            qid,
        )
    assert status == "triaging"


@pytest.mark.asyncio
async def test_reclaim_does_not_touch_terminal_or_pending_states(
    reset_db: None,
) -> None:
    # Seed one each of pending / triaged / done with stale heartbeat —
    # reclaim must ignore them (its WHERE filter is status IN
    # ('triaging','synthesizing')).
    pending_qid = await _seed_queue_row(
        customer_id=CUSTOMER,
        doc_id="github:commit:pending",
        status="pending",
        attempts=0,
        heartbeat_age_seconds=900,
    )
    triaged_qid = await _seed_queue_row(
        customer_id=CUSTOMER,
        doc_id="github:commit:triaged",
        status="triaged",
        attempts=1,
        heartbeat_age_seconds=900,
    )
    done_qid = await _seed_queue_row(
        customer_id=CUSTOMER,
        doc_id="github:commit:done",
        status="done",
        attempts=1,
        heartbeat_age_seconds=900,
    )
    retried, failed = await persistence.reclaim_stuck_rows(
        threshold_seconds=600,
        max_attempts=3,
    )
    assert retried == 0
    assert failed == 0

    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT queue_id, status FROM wiki_synthesis_queue WHERE queue_id = ANY($1::bigint[])",
            [pending_qid, triaged_qid, done_qid],
        )
    by_qid = {r["queue_id"]: r["status"] for r in rows}
    assert by_qid[pending_qid] == "pending"
    assert by_qid[triaged_qid] == "triaged"
    assert by_qid[done_qid] == "done"
