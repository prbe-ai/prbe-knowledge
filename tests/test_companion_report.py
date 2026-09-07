"""Fault-catalog readback: truncation on the row readback, per-seam aggregates.

Imports the HTTP harness and fixtures from the endpoints test module so the two
files share one tenant/enable/request setup.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from tests.test_companion_endpoints import (  # noqa: F401  (fixtures by name)
    ALICE,
    _enqueue_body,
    _internal_key,
    _request,
    enabled,
    tenant,
)


@pytest.mark.asyncio
async def test_deliveries_reports_truncation(enabled: str) -> None:  # noqa: F811
    card = (await _request("POST", "/companion/enqueue", _enqueue_body())).json()["card"]
    for _ in range(3):
        await _request(
            "POST",
            "/companion/ack",
            {
                "mailbox_id": card["mailbox_id"],
                "attempt_id": str(uuid4()),
                "seam": "stop",
                "outcome": "emitted",
                "receiving_instance": "x",
            },
        )
    two = await _request("GET", "/companion/deliveries?session_id=sess-http&limit=2")
    assert len(two.json()["deliveries"]) == 2 and two.json()["truncated"] is True
    all_ = await _request("GET", "/companion/deliveries?session_id=sess-http&limit=10")
    assert len(all_.json()["deliveries"]) == 3 and all_.json()["truncated"] is False


@pytest.mark.asyncio
async def test_report_aggregates_per_seam(enabled: str) -> None:  # noqa: F811
    """The fault catalog's numbers: per seam x outcome counts, the monotonic
    latency distribution, and the two evidence rates (`harness_accepted`,
    `observed_in_context`) counted from the evidence keys the actuators write."""
    card = (await _request("POST", "/companion/enqueue", _enqueue_body())).json()["card"]
    for ms, ev in (
        (10, {"harness_accepted": True, "observed_in_context": True}),
        (30, {"harness_accepted": True}),
        (50, {}),
    ):
        await _request(
            "POST",
            "/companion/ack",
            {
                "mailbox_id": card["mailbox_id"],
                "attempt_id": str(uuid4()),
                "seam": "stop",
                "outcome": "emitted",
                "receiving_instance": "x",
                "receipt_to_emission_ms": ms,
                "evidence": ev,
            },
        )
    await _request(
        "POST",
        "/companion/ack",
        {
            "mailbox_id": card["mailbox_id"],
            "attempt_id": str(uuid4()),
            "seam": "user-prompt",
            "outcome": "canceled",
            "receiving_instance": "x",
        },
    )
    resp = await _request("GET", "/companion/report?session_id=sess-http")
    assert resp.status_code == 200, resp.text
    rows = {(r["seam"], r["outcome"]): r for r in resp.json()["seams"]}
    stop = rows[("stop", "emitted")]
    assert (
        stop["attempts"] == 3 and stop["harness_accepted"] == 2 and stop["observed_in_context"] == 1
    )
    assert (
        stop["latency_ms"]["p50"] == 30
        and stop["latency_ms"]["max"] == 50
        and stop["latency_ms"]["n"] == 3
    )
    assert (
        rows[("user-prompt", "canceled")]["attempts"] == 1
        and rows[("user-prompt", "canceled")]["latency_ms"]["n"] == 0
    )
    assert resp.json()["capability"]["enabled"] is True
