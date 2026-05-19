"""Tests for the (approved ∧ resolved) post-approval dispatch seam.

Live Postgres required (DATABASE_URL must point at a running instance
with migration 0086 applied). Each test allocates a unique customer_id
and cleans up after itself, mirroring the
``test_investigation_state.py`` / ``test_wiki_review_state.py`` pattern.

The HTTP boundary is always mocked — these are state-machine tests, not
orchestrator-protocol tests.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from services.post_approval.dispatch import (
    fire_post_approval_dispatch,
    on_approval,
    on_resolution_event,
)
from shared import db as db_module
from shared.db import with_tenant

pytestmark = pytest.mark.asyncio


def _skip_if_no_db() -> None:
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")


def _new_customer_id() -> str:
    return f"dispatch-test-{uuid.uuid4().hex[:8]}"


async def _seed_customer(customer_id: str) -> None:
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
            "DELETE FROM incident_investigations WHERE customer_id = $1",
            customer_id,
        )
        await conn.execute(
            "DELETE FROM customers WHERE customer_id = $1", customer_id,
        )
    finally:
        await conn.close()


async def _seed_pending_review_row(
    customer_id: str,
    incident_doc_id: str,
    report_doc_id: str = "pd:investigation:PD-INC-001:v1",
) -> None:
    """Insert a baseline ``incident_investigations`` row in
    state='pending_review' with a current_report_doc_id. Mirrors what
    Plan 4's investigation pipeline produces before reviewer approves.
    """
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            INSERT INTO incident_investigations
                (customer_id, incident_doc_id, current_report_doc_id, state)
            VALUES ($1, $2, $3, 'pending_review')
            ON CONFLICT (customer_id, incident_doc_id) DO NOTHING;
            """,
            customer_id, incident_doc_id, report_doc_id,
        )


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


_INCIDENT_ID = "pd:incident:PD-INC-001"


async def test_approval_first_then_resolution_dispatches_once(
    customer_id: str,
) -> None:
    await _seed_pending_review_row(customer_id, _INCIDENT_ID)
    with patch(
        "services.post_approval.dispatch._post_dispatch",
        new_callable=AsyncMock,
        return_value=True,
    ) as fake_post:
        await on_approval(
            customer_id=customer_id, incident_doc_id=_INCIDENT_ID,
        )
        # No dispatch yet — resolution not set.
        fake_post.assert_not_awaited()
        await on_resolution_event(
            customer_id=customer_id, incident_doc_id=_INCIDENT_ID,
        )
        fake_post.assert_awaited_once()


async def test_resolution_first_then_approval_dispatches_once(
    customer_id: str,
) -> None:
    await _seed_pending_review_row(customer_id, _INCIDENT_ID)
    with patch(
        "services.post_approval.dispatch._post_dispatch",
        new_callable=AsyncMock,
        return_value=True,
    ) as fake_post:
        await on_resolution_event(
            customer_id=customer_id, incident_doc_id=_INCIDENT_ID,
        )
        fake_post.assert_not_awaited()
        await on_approval(
            customer_id=customer_id, incident_doc_id=_INCIDENT_ID,
        )
        fake_post.assert_awaited_once()


async def test_resolution_before_row_exists_creates_partial_row(
    customer_id: str,
) -> None:
    """``on_resolution_event`` is robust to the resolution-before-create
    edge: it UPSERTs a partial row so the (approved ∧ resolved) flip
    fires later when the investigation is approved."""
    new_doc_id = "pd:incident:NEW-INC"
    with patch(
        "services.post_approval.dispatch._post_dispatch",
        new_callable=AsyncMock,
        return_value=True,
    ) as fake_post:
        await on_resolution_event(
            customer_id=customer_id, incident_doc_id=new_doc_id,
        )
        # Resolution alone doesn't dispatch.
        fake_post.assert_not_awaited()
        async with with_tenant(customer_id) as conn:
            row = await conn.fetchrow(
                "SELECT resolved_at, approved_at FROM incident_investigations "
                "WHERE customer_id = $1 AND incident_doc_id = $2",
                customer_id, new_doc_id,
            )
    assert row is not None
    assert row["resolved_at"] is not None
    assert row["approved_at"] is None


async def test_dispatch_only_fires_once_even_on_double_call(
    customer_id: str,
) -> None:
    await _seed_pending_review_row(customer_id, _INCIDENT_ID)
    with patch(
        "services.post_approval.dispatch._post_dispatch",
        new_callable=AsyncMock,
        return_value=True,
    ) as fake_post:
        await on_approval(customer_id, _INCIDENT_ID)
        await on_resolution_event(customer_id, _INCIDENT_ID)
        # Re-arrive of the same events should NOT re-dispatch.
        await on_approval(customer_id, _INCIDENT_ID)
        await on_resolution_event(customer_id, _INCIDENT_ID)
    fake_post.assert_awaited_once()


async def test_concurrent_approve_and_resolve_dispatches_exactly_once(
    customer_id: str,
) -> None:
    """Two coroutines calling ``on_approval`` and ``on_resolution_event``
    simultaneously must dispatch exactly once. The FOR UPDATE row lock
    serializes them; the conditional UPDATE on
    ``post_approval_dispatched_at IS NULL`` closes the residual window
    between the lock release and the guard flip.
    """
    await _seed_pending_review_row(customer_id, _INCIDENT_ID)
    with patch(
        "services.post_approval.dispatch._post_dispatch",
        new_callable=AsyncMock,
        return_value=True,
    ) as fake_post:
        await asyncio.gather(
            on_approval(customer_id, _INCIDENT_ID),
            on_resolution_event(customer_id, _INCIDENT_ID),
        )
    fake_post.assert_awaited_once()


async def test_dispatch_5xx_leaves_timestamp_null(
    customer_id: str,
) -> None:
    """Failed dispatch leaves the guard NULL and stamps
    ``metadata.post_approval_dispatch_failed=true`` so the dashboard
    re-trigger flow can find the row."""
    await _seed_pending_review_row(customer_id, _INCIDENT_ID)
    with patch(
        "services.post_approval.dispatch._post_dispatch",
        new_callable=AsyncMock,
        return_value=False,
    ):
        await on_approval(customer_id, _INCIDENT_ID)
        await on_resolution_event(customer_id, _INCIDENT_ID)
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            "SELECT post_approval_dispatched_at, metadata "
            "FROM incident_investigations "
            "WHERE customer_id = $1 AND incident_doc_id = $2",
            customer_id, _INCIDENT_ID,
        )
    assert row is not None
    assert row["post_approval_dispatched_at"] is None
    # Asyncpg may return jsonb as str or dict depending on codec.
    md = row["metadata"]
    if isinstance(md, str):
        import json
        md = json.loads(md)
    assert md.get("post_approval_dispatch_failed") is True


async def test_fire_post_approval_dispatch_clears_guard_and_retries(
    customer_id: str,
) -> None:
    """The dashboard recovery entrypoint clears the guard + failure flag
    and re-runs the dispatch check."""
    await _seed_pending_review_row(customer_id, _INCIDENT_ID)
    # First: force a failed dispatch.
    with patch(
        "services.post_approval.dispatch._post_dispatch",
        new_callable=AsyncMock,
        return_value=False,
    ):
        await on_approval(customer_id, _INCIDENT_ID)
        await on_resolution_event(customer_id, _INCIDENT_ID)
    # Confirm the failure flag is set.
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            "SELECT post_approval_dispatched_at, metadata "
            "FROM incident_investigations "
            "WHERE customer_id = $1 AND incident_doc_id = $2",
            customer_id, _INCIDENT_ID,
        )
    md = row["metadata"]
    if isinstance(md, str):
        import json
        md = json.loads(md)
    assert md.get("post_approval_dispatch_failed") is True
    assert row["post_approval_dispatched_at"] is None

    # Now hit the recovery button with a successful dispatch.
    with patch(
        "services.post_approval.dispatch._post_dispatch",
        new_callable=AsyncMock,
        return_value=True,
    ) as fake_post:
        await fire_post_approval_dispatch(customer_id, _INCIDENT_ID)
    fake_post.assert_awaited_once()

    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            "SELECT post_approval_dispatched_at, metadata "
            "FROM incident_investigations "
            "WHERE customer_id = $1 AND incident_doc_id = $2",
            customer_id, _INCIDENT_ID,
        )
    assert row["post_approval_dispatched_at"] is not None
    md = row["metadata"]
    if isinstance(md, str):
        import json
        md = json.loads(md)
    # Failure flag should be cleared.
    assert "post_approval_dispatch_failed" not in md


async def test_source_derivation_pd_prefix(customer_id: str) -> None:
    """Smoke test: a pd:* incident_doc_id surfaces source='pagerduty' in
    the dispatch payload."""
    await _seed_pending_review_row(customer_id, _INCIDENT_ID)
    captured_payloads: list[dict] = []

    async def _fake_post(payload: dict) -> bool:
        captured_payloads.append(payload)
        return True

    with patch(
        "services.post_approval.dispatch._post_dispatch",
        side_effect=_fake_post,
    ):
        await on_approval(customer_id, _INCIDENT_ID)
        await on_resolution_event(customer_id, _INCIDENT_ID)
    assert len(captured_payloads) == 1
    assert captured_payloads[0]["source"] == "pagerduty"
    assert captured_payloads[0]["incident_doc_id"] == _INCIDENT_ID
    assert captured_payloads[0]["customer_id"] == customer_id


async def test_source_derivation_iio_prefix(customer_id: str) -> None:
    """Same smoke test for incident.io's iio:* prefix."""
    iio_doc_id = "iio:incident:01XYZ"
    await _seed_pending_review_row(
        customer_id, iio_doc_id,
        report_doc_id="iio:investigation:01XYZ:v1",
    )
    captured_payloads: list[dict] = []

    async def _fake_post(payload: dict) -> bool:
        captured_payloads.append(payload)
        return True

    with patch(
        "services.post_approval.dispatch._post_dispatch",
        side_effect=_fake_post,
    ):
        await on_approval(customer_id, iio_doc_id)
        await on_resolution_event(customer_id, iio_doc_id)
    assert len(captured_payloads) == 1
    assert captured_payloads[0]["source"] == "incident_io"


async def test_dispatch_rollback_skipped_when_guard_changed_by_recovery(
    customer_id: str,
) -> None:
    """CAS guard prevents a stale failure handler from stomping a
    concurrent recovery dispatch's successful state.

    Scenario: the initial dispatch's HTTP is slow and ultimately
    fails. Mid-flight, a recovery dispatch (simulated here by a direct
    UPDATE that re-stamps the guard with a different timestamp) marks
    the row as freshly-dispatched. The initial dispatch's rollback
    fires AFTER the recovery has succeeded, so its
    ``_mark_dispatch_failed`` UPDATE must hit the CAS predicate and
    affect zero rows — leaving the recovery's state intact.
    """
    from datetime import UTC, datetime

    await _seed_pending_review_row(customer_id, _INCIDENT_ID)

    async def _slow_then_fail(_payload: dict) -> bool:
        # Long enough for the in-band recovery to land and stamp.
        await asyncio.sleep(0.2)
        return False

    async def _recovery_after_delay() -> None:
        # Wait until the initial dispatch has stamped its guard and
        # entered ``_post_dispatch``, then simulate the recovery flow
        # (clear failure flag + re-stamp guard with a NEW timestamp).
        await asyncio.sleep(0.05)
        async with with_tenant(customer_id) as conn:
            await conn.execute(
                """
                UPDATE incident_investigations
                SET post_approval_dispatched_at = $3,
                    metadata = (COALESCE(metadata, '{}'::jsonb)
                                - 'post_approval_dispatch_failed'),
                    updated_at = $3
                WHERE customer_id = $1 AND incident_doc_id = $2;
                """,
                customer_id, _INCIDENT_ID, datetime.now(UTC),
            )

    with patch(
        "services.post_approval.dispatch._post_dispatch",
        side_effect=_slow_then_fail,
    ):
        # The initial dispatch stamps the guard, sleeps inside
        # ``_post_dispatch``, then returns False → ``_mark_dispatch_failed``
        # tries to roll back but CAS misses because the recovery has
        # already re-stamped.
        await asyncio.gather(
            on_approval(customer_id, _INCIDENT_ID),
            on_resolution_event(customer_id, _INCIDENT_ID),
            _recovery_after_delay(),
        )

    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            "SELECT post_approval_dispatched_at, metadata "
            "FROM incident_investigations "
            "WHERE customer_id = $1 AND incident_doc_id = $2",
            customer_id, _INCIDENT_ID,
        )
    assert row is not None
    # The recovery's stamp survives — initial dispatch's rollback was
    # a CAS miss.
    assert row["post_approval_dispatched_at"] is not None
    md = row["metadata"]
    if isinstance(md, str):
        import json
        md = json.loads(md)
    md = md or {}
    # The failure flag must NOT be re-set by the stale rollback.
    assert md.get("post_approval_dispatch_failed") is not True


async def test_dispatch_non_http_exception_triggers_rollback(
    customer_id: str,
) -> None:
    """A non-HTTP exception inside ``_post_dispatch`` (e.g. RuntimeError,
    OSError, JSON serialization bug) must NOT propagate up to
    ``_check_and_dispatch``: the catch-all converts it to a False
    return so the caller's ``_mark_dispatch_failed`` branch fires and
    the guard is cleared + failure flag is set."""
    await _seed_pending_review_row(customer_id, _INCIDENT_ID)

    async def _boom(_payload: dict) -> bool:
        raise RuntimeError("simulated non-HTTP failure")

    # We patch the inner httpx call indirectly: the simplest way to
    # exercise the catch-all is to make the whole ``_post_dispatch``
    # raise. But the catch-all lives INSIDE ``_post_dispatch``, so
    # patching it would bypass the very code we're testing. Instead,
    # patch ``httpx.AsyncClient`` to raise a non-HTTPError on entry
    # so the real ``_post_dispatch`` body runs.
    class _BoomClient:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            raise RuntimeError("simulated non-HTTP failure")

        async def __aenter__(self) -> Any:  # pragma: no cover
            raise RuntimeError("simulated non-HTTP failure")

        async def __aexit__(self, *_exc: Any) -> None:  # pragma: no cover
            return None

    with patch(
        "services.post_approval.dispatch.httpx.AsyncClient", _BoomClient,
    ):
        # If the catch-all is missing, this call raises and the
        # rollback never runs.
        await on_approval(customer_id, _INCIDENT_ID)
        await on_resolution_event(customer_id, _INCIDENT_ID)

    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            "SELECT post_approval_dispatched_at, metadata "
            "FROM incident_investigations "
            "WHERE customer_id = $1 AND incident_doc_id = $2",
            customer_id, _INCIDENT_ID,
        )
    assert row is not None
    # Rollback ran: guard cleared + failure flag set.
    assert row["post_approval_dispatched_at"] is None
    md = row["metadata"]
    if isinstance(md, str):
        import json
        md = json.loads(md)
    assert md is not None
    assert md.get("post_approval_dispatch_failed") is True


async def test_post_dispatch_sends_x_prbe_customer_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator's /internal/post-approval-actions route uses
    Depends(require_customer_id) and 400s without ``x-prbe-customer``.
    The header must be on every dispatch HTTP request, sourced from the
    payload's customer_id."""
    import httpx
    import respx
    from pydantic import SecretStr

    from services.post_approval.dispatch import _post_dispatch
    from shared.config import Settings, get_settings

    get_settings.cache_clear()

    def _settings() -> Settings:
        return Settings(
            orchestrator_base_url="http://orchestrator.internal:8080",
            internal_backend_api_key=SecretStr("test-internal-key"),
        )

    monkeypatch.setattr(
        "services.post_approval.dispatch.get_settings", _settings,
    )

    payload = {
        "customer_id": "cust-1",
        "incident_doc_id": "pd:incident:PD-INC-001",
        "investigation_doc_id": "pd:investigation:PD-INC-001:v1",
        "source": "pagerduty",
        "approved_at": "2026-01-01T00:00:00+00:00",
        "resolved_at": "2026-01-01T00:00:00+00:00",
    }

    expected_url = (
        "http://orchestrator.internal:8080/internal/post-approval-actions"
    )
    with respx.mock(assert_all_called=True) as router:
        route = router.post(expected_url).mock(
            return_value=httpx.Response(200, json={"ok": True}),
        )
        ok = await _post_dispatch(payload)

    assert ok is True
    request = route.calls[0].request
    assert request.headers["x-prbe-customer"] == "cust-1"
    assert request.headers["x-internal-backend-key"] == "test-internal-key"
    get_settings.cache_clear()
