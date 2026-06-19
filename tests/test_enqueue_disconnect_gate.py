"""Pre-insert gate: _enqueue must refuse when the source is disconnected.

Covers the acme/github incident (2026-05-15): a webhook handler
raced the disconnect_integration DB transaction by ~180ms and wrote 9
ingestion_queue rows for a source whose `integration_tokens` row had
already been deleted. The new gate calls `is_source_connected` before
any INSERT and drops the enqueue if the integration is gone.
"""

from __future__ import annotations

import pytest

from services.ingestion.connectedness import (
    _OAUTH_SOURCES,
    _UNGATED_SOURCES,
    is_source_connected,
)
from services.ingestion.main import _enqueue
from shared.constants import SourceSystem
from shared.db import raw_conn
from shared.encryption import encrypt_token

CUSTOMER_ID = "test-gate-customer"
SOURCE = SourceSystem.GITHUB


async def _seed_customer() -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'gate-test', 'gate-hash') ON CONFLICT DO NOTHING",
            CUSTOMER_ID,
        )


async def _seed_token(status: str = "active") -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO integration_tokens
                (customer_id, source_system, access_token_encrypted, status)
            VALUES ($1, $2, $3, $4)
            """,
            CUSTOMER_ID,
            SOURCE.value,
            encrypt_token("dummy-token"),
            status,
        )


async def _delete_token() -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "DELETE FROM integration_tokens WHERE customer_id=$1 AND source_system=$2",
            CUSTOMER_ID,
            SOURCE.value,
        )


async def _queue_count() -> int:
    async with raw_conn() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM ingestion_queue "
            "WHERE customer_id=$1 AND source_system=$2",
            CUSTOMER_ID,
            SOURCE.value,
        )


@pytest.mark.asyncio
async def test_enqueue_dropped_when_token_missing(live_db) -> None:
    """No integration_tokens row → enqueue returns False, no queue write."""
    await _seed_customer()
    # No token seeded.

    result = await _enqueue(
        customer_id=CUSTOMER_ID,
        source=SOURCE,
        source_event_id="evt-1",
        payload_s3_key="raw/github/test/evt-1.json",
    )

    assert result is False
    assert await _queue_count() == 0


@pytest.mark.asyncio
async def test_enqueue_dropped_when_token_disconnected_post_check(live_db) -> None:
    """Token row deleted after the connector handed the event off → bail."""
    await _seed_customer()
    await _seed_token(status="active")
    # Simulate the race: token is deleted between the webhook arriving and
    # the _enqueue running. The gate must catch this.
    await _delete_token()

    result = await _enqueue(
        customer_id=CUSTOMER_ID,
        source=SOURCE,
        source_event_id="evt-race",
        payload_s3_key="raw/github/test/evt-race.json",
    )

    assert result is False
    assert await _queue_count() == 0


@pytest.mark.asyncio
async def test_enqueue_dropped_when_token_auth_failed(live_db) -> None:
    """Token exists but status != 'active' → drop."""
    await _seed_customer()
    await _seed_token(status="auth_failed")

    result = await _enqueue(
        customer_id=CUSTOMER_ID,
        source=SOURCE,
        source_event_id="evt-2",
        payload_s3_key="raw/github/test/evt-2.json",
    )

    assert result is False
    assert await _queue_count() == 0


@pytest.mark.asyncio
async def test_enqueue_succeeds_when_token_active(live_db) -> None:
    """Happy path: token active → INSERT happens, returns True."""
    await _seed_customer()
    await _seed_token(status="active")

    result = await _enqueue(
        customer_id=CUSTOMER_ID,
        source=SOURCE,
        source_event_id="evt-ok",
        payload_s3_key="raw/github/test/evt-ok.json",
    )

    assert result is True
    assert await _queue_count() == 1


@pytest.mark.asyncio
async def test_gate_skipped_for_non_oauth_sources(live_db) -> None:
    """Sources without integration_tokens (CC, CODEX, etc.) bypass the gate."""
    await _seed_customer()
    # No token seeded for claude_code.

    assert await is_source_connected(CUSTOMER_ID, SourceSystem.CLAUDE_CODE) is True
    assert await is_source_connected(CUSTOMER_ID, SourceSystem.CODEX) is True
    assert await is_source_connected(CUSTOMER_ID, SourceSystem.MANUAL_UPLOAD) is True
    assert await is_source_connected(CUSTOMER_ID, SourceSystem.CODE_GRAPH) is True


def test_all_sources_classified() -> None:
    """Tripwire: every SourceSystem value must be in either the gated
    (_OAUTH_SOURCES) or explicitly-ungated (_UNGATED_SOURCES) set. When
    someone adds a new SourceSystem (e.g. JIRA), this test forces them
    to think about the disconnect-time gate instead of silently inheriting
    "ungated by omission".
    """
    classified = _OAUTH_SOURCES | _UNGATED_SOURCES
    missing = set(SourceSystem) - classified
    assert not missing, (
        f"SourceSystem values not classified for disconnect gate: {missing}. "
        f"Add them to _OAUTH_SOURCES or _UNGATED_SOURCES in "
        f"services/ingestion/connectedness.py."
    )
    # And no overlap.
    overlap = _OAUTH_SOURCES & _UNGATED_SOURCES
    assert not overlap, f"Source can't be both gated and ungated: {overlap}"
