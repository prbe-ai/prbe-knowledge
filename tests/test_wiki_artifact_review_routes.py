"""GET/POST review-route tests for wiki artifacts.

Covers:
- list: state / kind / incident filters
- detail: 404 vs versions list
- approve: atomic flip of documents+chunks visibility AND queue state,
  idempotency, 409 on rejected->approve
- reject: pydantic-validated feedback, durable state flip when the
  orchestrator re-dispatch fails (metadata.re_dispatch_failed=true),
  409 on approved->reject
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from services.ingestion.main import app
from shared.config import Settings, get_settings
from shared.db import close_pool, init_pool, raw_conn

pytestmark = pytest.mark.asyncio


_INTERNAL_KEY = "test-wiki-review-key"
_CUSTOMER = f"wiki-rv-test-{uuid.uuid4().hex[:8]}"


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
            _CUSTOMER, f"wiki-rv {_CUSTOMER}", "h", f"b-{_CUSTOMER}",
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


def _wb_payload(
    *,
    artifact_kind: str = "postmortem",
    incident_doc_id: str = "pd:incident:PD-RV-001",
    body_markdown: str = "## Body\n\nSome body text for the artifact.\n",
    mode: str = "full",
) -> dict:
    return {
        "customer_id": _CUSTOMER,
        "incident_doc_id": incident_doc_id,
        "investigation_doc_id": "pd:investigation:PD-RV-001:v1",
        "artifact_kind": artifact_kind,
        "target_doc_id": None,
        "title": f"Artifact {incident_doc_id}",
        "body_markdown": body_markdown,
        "metadata": {
            "mode": mode,
            "evidence_refs": [],
            "rationale": "test",
            "tool_trace_run_id": "trace-rv-001",
        },
    }


async def _seed_via_writeback(
    client: AsyncClient,
    *,
    incident_doc_id: str,
    artifact_kind: str = "postmortem",
) -> str:
    r = await client.post(
        "/api/wiki-artifacts", headers=_hdrs(),
        json=_wb_payload(
            incident_doc_id=incident_doc_id,
            artifact_kind=artifact_kind,
        ),
    )
    assert r.status_code == 200, r.text
    return r.json()["artifact_doc_id"]


# ---- list ---------------------------------------------------------------


async def test_list_returns_pending(client) -> None:
    artifact_id = await _seed_via_writeback(
        client, incident_doc_id="pd:incident:PD-RV-LIST-1",
    )
    r = await client.get(
        "/api/wiki-artifacts",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER, "state": "pending_review"},
    )
    assert r.status_code == 200
    items = r.json()
    ids = {it["artifact_doc_id"] for it in items}
    assert artifact_id in ids
    for it in items:
        if it["artifact_doc_id"] == artifact_id:
            assert it["state"] == "pending_review"


async def test_list_filters_by_incident(client) -> None:
    incident_a = "pd:incident:PD-RV-INC-A"
    incident_b = "pd:incident:PD-RV-INC-B"
    a_id = await _seed_via_writeback(client, incident_doc_id=incident_a)
    b_id = await _seed_via_writeback(client, incident_doc_id=incident_b)
    r = await client.get(
        "/api/wiki-artifacts",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER, "incident_doc_id": incident_a},
    )
    assert r.status_code == 200
    ids = {it["artifact_doc_id"] for it in r.json()}
    assert a_id in ids
    assert b_id not in ids


async def test_list_filters_by_kind(client) -> None:
    pm_id = await _seed_via_writeback(
        client, incident_doc_id="pd:incident:PD-RV-K-PM",
        artifact_kind="postmortem",
    )
    kp_id = await _seed_via_writeback(
        client, incident_doc_id="pd:incident:PD-RV-K-KP",
        artifact_kind="knowledge_page",
    )
    r = await client.get(
        "/api/wiki-artifacts",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER, "artifact_kind": "knowledge_page"},
    )
    assert r.status_code == 200
    ids = {it["artifact_doc_id"] for it in r.json()}
    assert kp_id in ids
    assert pm_id not in ids


# ---- detail -------------------------------------------------------------


async def test_detail_returns_versions(client) -> None:
    artifact_id = await _seed_via_writeback(
        client, incident_doc_id="pd:incident:PD-RV-DET",
    )
    r = await client.get(
        f"/api/wiki-artifacts/{artifact_id}",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["artifact_doc_id"] == artifact_id
    assert data["state"] == "pending_review"
    assert isinstance(data["versions"], list)
    assert len(data["versions"]) >= 1


async def test_detail_returns_404_when_missing(client) -> None:
    r = await client.get(
        "/api/wiki-artifacts/pd:wiki.postmortem:NONE:v1",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER},
    )
    assert r.status_code == 404


# ---- approve ------------------------------------------------------------


async def test_approve_flips_visibility_atomically(client) -> None:
    artifact_id = await _seed_via_writeback(
        client, incident_doc_id="pd:incident:PD-RV-AP",
    )

    r = await client.post(
        f"/api/wiki-artifacts/{artifact_id}/approve",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER},
        json={"reviewer_id": "reviewer-1"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "approved"
    assert r.json()["reviewer_id"] == "reviewer-1"

    async with raw_conn() as conn:
        doc = await conn.fetchrow(
            "SELECT visibility FROM documents "
            "WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL",
            _CUSTOMER, artifact_id,
        )
        chunk_rows = await conn.fetch(
            "SELECT visibility FROM chunks "
            "WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL",
            _CUSTOMER, artifact_id,
        )
        queue = await conn.fetchrow(
            "SELECT state, reviewer_id FROM wiki_review_queue "
            "WHERE customer_id = $1 AND artifact_doc_id = $2",
            _CUSTOMER, artifact_id,
        )
    assert doc["visibility"] == "approved"
    assert len(chunk_rows) > 0
    assert all(c["visibility"] == "approved" for c in chunk_rows)
    assert queue["state"] == "approved"
    assert queue["reviewer_id"] == "reviewer-1"


async def test_approve_idempotent(client) -> None:
    artifact_id = await _seed_via_writeback(
        client, incident_doc_id="pd:incident:PD-RV-AP-IDM",
    )

    r1 = await client.post(
        f"/api/wiki-artifacts/{artifact_id}/approve",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER},
        json={"reviewer_id": "reviewer-1"},
    )
    r2 = await client.post(
        f"/api/wiki-artifacts/{artifact_id}/approve",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER},
        json={"reviewer_id": "reviewer-2"},
    )
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["state"] == "approved"
    assert r2.json()["state"] == "approved"
    # First reviewer's id is canonical — idempotent re-approve doesn't
    # overwrite it.
    assert r2.json()["reviewer_id"] == "reviewer-1"


async def test_approve_rejected_artifact_returns_409(client) -> None:
    artifact_id = await _seed_via_writeback(
        client, incident_doc_id="pd:incident:PD-RV-AP-REJ",
    )
    # Bypass the route to mark it rejected without firing the
    # orchestrator dispatch.
    from services.post_approval.wiki_review_state import mark_rejected
    await mark_rejected(
        customer_id=_CUSTOMER,
        artifact_doc_id=artifact_id,
        reviewer_id="reviewer-1",
        feedback="no thanks",
    )

    r = await client.post(
        f"/api/wiki-artifacts/{artifact_id}/approve",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER},
        json={"reviewer_id": "reviewer-2"},
    )
    assert r.status_code == 409


# ---- reject -------------------------------------------------------------


async def test_reject_requires_feedback(client) -> None:
    artifact_id = await _seed_via_writeback(
        client, incident_doc_id="pd:incident:PD-RV-RJ-FB",
    )
    r = await client.post(
        f"/api/wiki-artifacts/{artifact_id}/reject",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER},
        json={"reviewer_id": "reviewer-1", "feedback": ""},
    )
    assert r.status_code == 422


async def test_reject_persists_even_if_redispatch_fails(client) -> None:
    """State flip is durable; metadata.re_dispatch_failed=true marks
    the row for ops recovery."""
    artifact_id = await _seed_via_writeback(
        client, incident_doc_id="pd:incident:PD-RV-RJ-REDISP",
    )
    with patch(
        "services.ingestion.wiki_artifact_review_routes._post_rerun_dispatch",
        new=AsyncMock(return_value=False),
    ):
        r = await client.post(
            f"/api/wiki-artifacts/{artifact_id}/reject",
            headers=_hdrs(),
            params={"customer_id": _CUSTOMER},
            json={
                "reviewer_id": "reviewer-1",
                "feedback": "missed the deploy correlation",
            },
        )
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "rejected"

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT state, metadata FROM wiki_review_queue "
            "WHERE customer_id = $1 AND artifact_doc_id = $2",
            _CUSTOMER, artifact_id,
        )
    assert row["state"] == "rejected"
    import json
    md = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
    assert md.get("re_dispatch_failed") is True


async def test_reject_concurrent_race_does_not_double_dispatch(
    client,
) -> None:
    """Inner-race: two callers both see ``pending_review`` at ``get_detail``
    time (the outer idempotent ``existing.state == "rejected"`` short-
    circuit does NOT fire), but only one wins ``mark_rejected``; the
    loser must NOT fire a second orchestrator re-run dispatch."""
    artifact_id = await _seed_via_writeback(
        client, incident_doc_id="pd:incident:PD-RV-RJ-RACE",
    )

    from services.post_approval.wiki_review_state import (
        get_detail as real_get_detail,
    )
    from services.post_approval.wiki_review_state import (
        mark_rejected as real_mark_rejected,
    )

    # Capture the pre-race detail so both callers see pending_review at
    # the outer-idempotency check. After mark_rejected lands the second
    # invocation will still get the (real, post-state) detail at the
    # final refresh, so we only need to override the FIRST get_detail
    # per request.
    pre_race_detail = await real_get_detail(_CUSTOMER, artifact_id)
    assert pre_race_detail is not None
    assert pre_race_detail.state == "pending_review"

    get_detail_calls = 0

    async def faked_get_detail(customer_id, artifact_doc_id):
        nonlocal get_detail_calls
        get_detail_calls += 1
        # The outer-idempotency call is the 1st call per request — both
        # requests' outer calls must see pending_review to force them
        # both into the try block. Sequence per request:
        #   request 1: call 1 = outer (faked) -> winner; call 2 = refetch
        #   request 2: call 3 = outer (faked) -> loser; call 4 = refetch
        # So calls 1 and 3 must return the pre-race snapshot; calls 2
        # and 4 (the post-race refetches) must reflect the real state.
        if get_detail_calls in (1, 3):
            return pre_race_detail
        return await real_get_detail(customer_id, artifact_doc_id)

    mark_rejected_calls = 0

    async def racing_mark_rejected(*args, **kwargs):
        nonlocal mark_rejected_calls
        mark_rejected_calls += 1
        if mark_rejected_calls == 1:
            return await real_mark_rejected(*args, **kwargs)
        # The loser sees the terminal-state guard fire.
        raise ValueError(
            "cannot reject wiki artifact from terminal state rejected"
        )

    dispatch_calls: list[dict] = []

    async def fake_post(payload):
        dispatch_calls.append(payload)
        return True

    with (
        patch(
            "services.ingestion.wiki_artifact_review_routes.get_detail",
            side_effect=faked_get_detail,
        ),
        patch(
            "services.ingestion.wiki_artifact_review_routes.mark_rejected",
            side_effect=racing_mark_rejected,
        ),
        patch(
            "services.ingestion.wiki_artifact_review_routes._post_rerun_dispatch",
            new=fake_post,
        ),
    ):
        r1 = await client.post(
            f"/api/wiki-artifacts/{artifact_id}/reject",
            headers=_hdrs(),
            params={"customer_id": _CUSTOMER},
            json={"reviewer_id": "reviewer-1", "feedback": "bad draft"},
        )
        r2 = await client.post(
            f"/api/wiki-artifacts/{artifact_id}/reject",
            headers=_hdrs(),
            params={"customer_id": _CUSTOMER},
            json={"reviewer_id": "reviewer-2", "feedback": "bad draft"},
        )

    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    assert r1.json()["state"] == "rejected"
    assert r2.json()["state"] == "rejected"
    # Critical invariant: only ONE dispatch fired, not two. Without the
    # early-return in the ValueError branch the loser would also fire,
    # double-dispatching the v2 re-run to the orchestrator.
    assert len(dispatch_calls) == 1





async def test_reject_approved_artifact_returns_409(client) -> None:
    artifact_id = await _seed_via_writeback(
        client, incident_doc_id="pd:incident:PD-RV-RJ-AP",
    )
    # Approve first.
    await client.post(
        f"/api/wiki-artifacts/{artifact_id}/approve",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER},
        json={"reviewer_id": "reviewer-1"},
    )
    # Then attempt to reject — should be 409.
    r = await client.post(
        f"/api/wiki-artifacts/{artifact_id}/reject",
        headers=_hdrs(),
        params={"customer_id": _CUSTOMER},
        json={
            "reviewer_id": "reviewer-2",
            "feedback": "actually nevermind",
        },
    )
    assert r.status_code == 409
