"""CRUD tests for incident_investigations state layer.

Live Postgres required (DATABASE_URL must point at a running instance
with migration 0080 applied). The tests insert + delete their own rows
using a unique customer_id per test, skip cleanly if no DB is configured.
"""
from __future__ import annotations

import os
import uuid

import pytest

from services.ingestion.investigation_state import (
    get_detail,
    list_for_customer,
    mark_approved,
    mark_rejected,
    upsert_pending_review,
)
from shared import db as db_module
from shared.exceptions import InvestigationNotFound

pytestmark = pytest.mark.asyncio


def _skip_if_no_db() -> None:
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")


def _new_customer_id() -> str:
    return f"state-test-{uuid.uuid4().hex[:8]}"


async def _seed_customer(customer_id: str) -> None:
    """Create a minimal customers row so the FK is satisfied."""
    import asyncpg
    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash, r2_bucket) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (customer_id) DO NOTHING",
            customer_id, f"test {customer_id}", "h", f"b-{customer_id}",
        )
    finally:
        await conn.close()


async def _cleanup_customer(customer_id: str) -> None:
    import asyncpg
    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute(
            "DELETE FROM incident_investigations WHERE customer_id = $1",
            customer_id,
        )
        await conn.execute(
            "DELETE FROM customers WHERE customer_id = $1", customer_id,
        )
    finally:
        await conn.close()


@pytest.fixture
async def customer_id():
    _skip_if_no_db()
    db_module.reset_pool()
    await db_module.init_pool()
    cid = _new_customer_id()
    await _seed_customer(cid)
    try:
        yield cid
    finally:
        await _cleanup_customer(cid)
        await db_module.close_pool()


async def test_upsert_creates_row_in_pending_review(customer_id: str) -> None:
    detail = await upsert_pending_review(
        customer_id=customer_id,
        incident_doc_id="pd:incident:T-001",
        report_doc_id="pd:investigation:T-001:v1",
        version=1, mode="full",
    )
    assert detail.state == "pending_review"
    assert detail.current_report_doc_id == "pd:investigation:T-001:v1"
    assert len(detail.versions) == 1
    v = detail.versions[0]
    assert v.version == 1 and v.doc_id == "pd:investigation:T-001:v1"
    assert v.mode == "full" and v.decision == "pending"
    assert v.reviewed_by is None and v.feedback is None


async def test_upsert_second_call_appends_version_and_resets_state(
    customer_id: str,
) -> None:
    await upsert_pending_review(
        customer_id=customer_id, incident_doc_id="pd:incident:T-002",
        report_doc_id="pd:investigation:T-002:v1", version=1, mode="full",
    )
    detail = await upsert_pending_review(
        customer_id=customer_id, incident_doc_id="pd:incident:T-002",
        report_doc_id="pd:investigation:T-002:v2", version=2, mode="playbook_only",
    )
    assert detail.state == "pending_review"
    assert detail.current_report_doc_id == "pd:investigation:T-002:v2"
    assert len(detail.versions) == 2
    assert detail.versions[0].version == 1
    assert detail.versions[1].version == 2
    assert detail.versions[1].mode == "playbook_only"


async def test_mark_approved_sets_state_and_decision_on_latest_version(
    customer_id: str,
) -> None:
    await upsert_pending_review(
        customer_id=customer_id, incident_doc_id="pd:incident:T-003",
        report_doc_id="pd:investigation:T-003:v1", version=1, mode="full",
    )
    detail = await mark_approved(
        customer_id=customer_id, incident_doc_id="pd:incident:T-003",
        reviewer_id="user-42",
    )
    assert detail.state == "approved"
    assert detail.reviewer_id == "user-42"
    assert detail.versions[-1].decision == "approved"
    assert detail.versions[-1].reviewed_by == "user-42"
    assert detail.versions[-1].reviewed_at is not None


async def test_mark_rejected_sets_state_and_records_feedback(
    customer_id: str,
) -> None:
    await upsert_pending_review(
        customer_id=customer_id, incident_doc_id="pd:incident:T-004",
        report_doc_id="pd:investigation:T-004:v1", version=1, mode="full",
    )
    detail = await mark_rejected(
        customer_id=customer_id, incident_doc_id="pd:incident:T-004",
        reviewer_id="user-42", feedback="missed the recent deploy",
    )
    assert detail.state == "rejected"
    assert detail.reviewer_id == "user-42"
    assert detail.versions[-1].decision == "rejected"
    assert detail.versions[-1].feedback == "missed the recent deploy"


async def test_mark_approved_only_affects_latest_version(
    customer_id: str,
) -> None:
    """v1 should retain its original 'pending' decision even after v2 is approved."""
    await upsert_pending_review(
        customer_id=customer_id, incident_doc_id="pd:incident:T-005",
        report_doc_id="pd:investigation:T-005:v1", version=1, mode="full",
    )
    await upsert_pending_review(
        customer_id=customer_id, incident_doc_id="pd:incident:T-005",
        report_doc_id="pd:investigation:T-005:v2", version=2, mode="full",
    )
    detail = await mark_approved(
        customer_id=customer_id, incident_doc_id="pd:incident:T-005",
        reviewer_id="user-42",
    )
    assert detail.versions[0].decision == "pending"
    assert detail.versions[1].decision == "approved"


async def test_mark_approved_raises_when_missing(
    customer_id: str,
) -> None:
    with pytest.raises(InvestigationNotFound):
        await mark_approved(
            customer_id=customer_id,
            incident_doc_id="pd:incident:DOES_NOT_EXIST",
            reviewer_id="u",
        )


async def test_mark_rejected_raises_when_missing(
    customer_id: str,
) -> None:
    with pytest.raises(InvestigationNotFound):
        await mark_rejected(
            customer_id=customer_id,
            incident_doc_id="pd:incident:DOES_NOT_EXIST",
            reviewer_id="u",
            feedback="anything",
        )


async def test_get_detail_returns_none_when_missing(customer_id: str) -> None:
    assert await get_detail(customer_id, "pd:incident:NONE") is None


async def test_list_returns_seeded_rows(customer_id: str) -> None:
    await upsert_pending_review(
        customer_id=customer_id, incident_doc_id="pd:incident:T-006",
        report_doc_id="pd:investigation:T-006:v1", version=1, mode="full",
    )
    await upsert_pending_review(
        customer_id=customer_id, incident_doc_id="pd:incident:T-007",
        report_doc_id="pd:investigation:T-007:v1", version=1, mode="stub",
    )
    items = await list_for_customer(customer_id)
    incident_ids = {item.incident_doc_id for item in items}
    assert "pd:incident:T-006" in incident_ids
    assert "pd:incident:T-007" in incident_ids


async def test_list_filters_by_state(customer_id: str) -> None:
    await upsert_pending_review(
        customer_id=customer_id, incident_doc_id="pd:incident:T-008",
        report_doc_id="pd:investigation:T-008:v1", version=1, mode="full",
    )
    await upsert_pending_review(
        customer_id=customer_id, incident_doc_id="pd:incident:T-009",
        report_doc_id="pd:investigation:T-009:v1", version=1, mode="full",
    )
    await mark_approved(
        customer_id=customer_id, incident_doc_id="pd:incident:T-008",
        reviewer_id="u",
    )
    pending = await list_for_customer(customer_id, state="pending_review")
    approved = await list_for_customer(customer_id, state="approved")
    pending_ids = {i.incident_doc_id for i in pending}
    approved_ids = {i.incident_doc_id for i in approved}
    assert "pd:incident:T-009" in pending_ids
    assert "pd:incident:T-008" not in pending_ids
    assert "pd:incident:T-008" in approved_ids
