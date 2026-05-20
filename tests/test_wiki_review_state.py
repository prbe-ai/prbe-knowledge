"""CRUD tests for the wiki_review_queue state layer.

Live Postgres required (DATABASE_URL must point at a running instance
with migrations through 0084 applied). Each test allocates a unique
customer_id and cleans up after itself, mirroring the
``test_investigation_state.py`` pattern.
"""
from __future__ import annotations

import os
import uuid

import pytest

from services.post_approval.wiki_review_state import (
    get_detail,
    list_for_customer,
    mark_approved,
    mark_rejected,
    upsert_pending_review,
)
from shared import db as db_module

pytestmark = pytest.mark.asyncio


def _skip_if_no_db() -> None:
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")


def _new_customer_id() -> str:
    return f"wiki-state-test-{uuid.uuid4().hex[:8]}"


async def _seed_customer(customer_id: str) -> None:
    """Insert a minimal customers row. r2_bucket is filled by the trigger."""
    import asyncpg
    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, $2, $3) ON CONFLICT (customer_id) DO NOTHING",
            customer_id, f"test {customer_id}", "h",
        )
    finally:
        await conn.close()


async def _cleanup_customer(customer_id: str) -> None:
    import asyncpg
    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute(
            "DELETE FROM wiki_review_queue WHERE customer_id = $1",
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


async def test_upsert_creates_row_pending_review(customer_id: str) -> None:
    detail = await upsert_pending_review(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:T-001:v1",
        incident_doc_id="pd:incident:T-001",
        artifact_kind="postmortem",
        target_doc_id=None,
        parent_artifact_doc_id=None,
        initial_state="pending_review",
        metadata={"mode": "full", "rationale": "first draft"},
    )
    assert detail.state == "pending_review"
    assert detail.artifact_kind == "postmortem"
    assert detail.target_doc_id is None
    assert detail.parent_artifact_doc_id is None
    assert detail.metadata.get("rationale") == "first draft"
    # The detail returned from upsert is the bare row (no lineage fetch).
    assert detail.versions == []


async def test_upsert_correction_requires_target(customer_id: str) -> None:
    with pytest.raises(ValueError, match="correction artifact requires target_doc_id"):
        await upsert_pending_review(
            customer_id=customer_id,
            artifact_doc_id="pd:correction:T-001:v1",
            incident_doc_id="pd:incident:T-001",
            artifact_kind="correction",
            target_doc_id=None,
            parent_artifact_doc_id=None,
            initial_state="pending_review",
            metadata={},
        )


async def test_upsert_non_correction_forbids_target(customer_id: str) -> None:
    with pytest.raises(ValueError, match="must have null target_doc_id"):
        await upsert_pending_review(
            customer_id=customer_id,
            artifact_doc_id="pd:postmortem:T-002:v1",
            incident_doc_id="pd:incident:T-002",
            artifact_kind="postmortem",
            target_doc_id="pd:knowledge_page:other",
            parent_artifact_doc_id=None,
            initial_state="pending_review",
            metadata={},
        )
    with pytest.raises(ValueError, match="must have null target_doc_id"):
        await upsert_pending_review(
            customer_id=customer_id,
            artifact_doc_id="pd:knowledge_page:T-003:v1",
            incident_doc_id="pd:incident:T-003",
            artifact_kind="knowledge_page",
            target_doc_id="pd:knowledge_page:other",
            parent_artifact_doc_id=None,
            initial_state="pending_review",
            metadata={},
        )


async def test_mark_approved_sets_state(customer_id: str) -> None:
    await upsert_pending_review(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:A-001:v1",
        incident_doc_id="pd:incident:A-001",
        artifact_kind="postmortem",
        target_doc_id=None,
        parent_artifact_doc_id=None,
        initial_state="pending_review",
        metadata={},
    )
    detail = await mark_approved(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:A-001:v1",
        reviewer_id="user-42",
    )
    assert detail.state == "approved"
    assert detail.reviewer_id == "user-42"
    assert detail.reviewed_at is not None
    assert len(detail.versions) == 1
    assert detail.versions[0].decision == "approved"
    assert detail.versions[0].reviewed_by == "user-42"


async def test_mark_approved_idempotent_if_already_approved(
    customer_id: str,
) -> None:
    await upsert_pending_review(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:A-002:v1",
        incident_doc_id="pd:incident:A-002",
        artifact_kind="postmortem",
        target_doc_id=None,
        parent_artifact_doc_id=None,
        initial_state="pending_review",
        metadata={},
    )
    first = await mark_approved(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:A-002:v1",
        reviewer_id="user-1",
    )
    second = await mark_approved(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:A-002:v1",
        reviewer_id="user-2",
    )
    # Idempotent: second call returns the existing row (reviewer_id stays
    # at the original 'user-1', not overwritten to 'user-2').
    assert second.state == "approved"
    assert second.reviewer_id == "user-1"
    assert second.reviewed_at == first.reviewed_at


async def test_mark_approved_raises_on_missing_row(customer_id: str) -> None:
    with pytest.raises(LookupError):
        await mark_approved(
            customer_id=customer_id,
            artifact_doc_id="pd:postmortem:DOES_NOT_EXIST",
            reviewer_id="user-1",
        )


async def test_mark_approved_raises_on_rejected_row(customer_id: str) -> None:
    await upsert_pending_review(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:R-001:v1",
        incident_doc_id="pd:incident:R-001",
        artifact_kind="postmortem",
        target_doc_id=None,
        parent_artifact_doc_id=None,
        initial_state="pending_review",
        metadata={},
    )
    await mark_rejected(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:R-001:v1",
        reviewer_id="user-1",
        feedback="not good",
    )
    with pytest.raises(ValueError, match="cannot approve from terminal state rejected"):
        await mark_approved(
            customer_id=customer_id,
            artifact_doc_id="pd:postmortem:R-001:v1",
            reviewer_id="user-2",
        )


async def test_mark_rejected_records_feedback(customer_id: str) -> None:
    await upsert_pending_review(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:RF-001:v1",
        incident_doc_id="pd:incident:RF-001",
        artifact_kind="postmortem",
        target_doc_id=None,
        parent_artifact_doc_id=None,
        initial_state="pending_review",
        metadata={"mode": "full"},
    )
    detail = await mark_rejected(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:RF-001:v1",
        reviewer_id="user-42",
        feedback="missed the upstream alert",
    )
    assert detail.state == "rejected"
    assert detail.reviewer_id == "user-42"
    assert detail.metadata.get("last_feedback") == "missed the upstream alert"
    # mode stayed; jsonb || preserves existing keys.
    assert detail.metadata.get("mode") == "full"
    assert len(detail.versions) == 1
    assert detail.versions[0].decision == "rejected"
    assert detail.versions[0].feedback == "missed the upstream alert"


async def test_list_filters_by_state_and_kind(customer_id: str) -> None:
    # Seed two postmortems and one knowledge_page.
    await upsert_pending_review(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:LIST-1:v1",
        incident_doc_id="pd:incident:LIST-1",
        artifact_kind="postmortem",
        target_doc_id=None,
        parent_artifact_doc_id=None,
        initial_state="pending_review",
        metadata={},
    )
    await upsert_pending_review(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:LIST-2:v1",
        incident_doc_id="pd:incident:LIST-2",
        artifact_kind="postmortem",
        target_doc_id=None,
        parent_artifact_doc_id=None,
        initial_state="pending_review",
        metadata={},
    )
    await upsert_pending_review(
        customer_id=customer_id,
        artifact_doc_id="pd:knowledge_page:LIST-3:v1",
        incident_doc_id="pd:incident:LIST-3",
        artifact_kind="knowledge_page",
        target_doc_id=None,
        parent_artifact_doc_id=None,
        initial_state="pending_review",
        metadata={},
    )
    # Approve one postmortem.
    await mark_approved(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:LIST-1:v1",
        reviewer_id="u",
    )
    pending_postmortems = await list_for_customer(
        customer_id,
        state="pending_review",
        artifact_kind="postmortem",
    )
    ids = {item.artifact_doc_id for item in pending_postmortems}
    assert ids == {"pd:postmortem:LIST-2:v1"}


async def test_list_filters_by_incident(customer_id: str) -> None:
    await upsert_pending_review(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:INC-1:v1",
        incident_doc_id="pd:incident:INC-1",
        artifact_kind="postmortem",
        target_doc_id=None,
        parent_artifact_doc_id=None,
        initial_state="pending_review",
        metadata={},
    )
    await upsert_pending_review(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:INC-2:v1",
        incident_doc_id="pd:incident:INC-2",
        artifact_kind="postmortem",
        target_doc_id=None,
        parent_artifact_doc_id=None,
        initial_state="pending_review",
        metadata={},
    )
    items = await list_for_customer(
        customer_id, incident_doc_id="pd:incident:INC-1",
    )
    ids = {item.artifact_doc_id for item in items}
    assert ids == {"pd:postmortem:INC-1:v1"}


async def test_get_detail_returns_versions_in_lineage_order(
    customer_id: str,
) -> None:
    # v1: root.
    await upsert_pending_review(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:V:v1",
        incident_doc_id="pd:incident:V",
        artifact_kind="postmortem",
        target_doc_id=None,
        parent_artifact_doc_id=None,
        initial_state="pending_review",
        metadata={},
    )
    await mark_rejected(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:V:v1",
        reviewer_id="u1",
        feedback="needs more depth",
    )
    # v2: child of v1.
    await upsert_pending_review(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:V:v2",
        incident_doc_id="pd:incident:V",
        artifact_kind="postmortem",
        target_doc_id=None,
        parent_artifact_doc_id="pd:postmortem:V:v1",
        initial_state="pending_review",
        metadata={},
    )
    await mark_rejected(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:V:v2",
        reviewer_id="u1",
        feedback="still missing the root cause",
    )
    # v3: child of v2 (current pending).
    await upsert_pending_review(
        customer_id=customer_id,
        artifact_doc_id="pd:postmortem:V:v3",
        incident_doc_id="pd:incident:V",
        artifact_kind="postmortem",
        target_doc_id=None,
        parent_artifact_doc_id="pd:postmortem:V:v2",
        initial_state="pending_review",
        metadata={},
    )

    detail = await get_detail(customer_id, "pd:postmortem:V:v3")
    assert detail is not None
    assert detail.state == "pending_review"
    assert len(detail.versions) == 3
    # Ordered by created_at ASC.
    assert detail.versions[0].artifact_doc_id == "pd:postmortem:V:v1"
    assert detail.versions[0].decision == "rejected"
    assert detail.versions[0].feedback == "needs more depth"
    assert detail.versions[1].artifact_doc_id == "pd:postmortem:V:v2"
    assert detail.versions[1].decision == "rejected"
    assert detail.versions[1].feedback == "still missing the root cause"
    assert detail.versions[2].artifact_doc_id == "pd:postmortem:V:v3"
    assert detail.versions[2].decision == "pending"
    assert detail.versions[2].feedback is None

    # get_detail(v1) — walking up from v1 stays at v1 (no parent), and
    # the recursive CTE picks up v2 and v3 as descendants.
    detail_from_root = await get_detail(customer_id, "pd:postmortem:V:v1")
    assert detail_from_root is not None
    assert [v.artifact_doc_id for v in detail_from_root.versions] == [
        "pd:postmortem:V:v1",
        "pd:postmortem:V:v2",
        "pd:postmortem:V:v3",
    ]


async def test_get_detail_returns_none_for_missing(customer_id: str) -> None:
    assert await get_detail(customer_id, "pd:postmortem:NEVER") is None
