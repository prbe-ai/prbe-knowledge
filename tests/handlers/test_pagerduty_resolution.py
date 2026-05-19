"""PD connector detects `incident.resolved` and sets the
`requires_resolution_check` flag — the signal the post-approval dispatch
seam listens for.

These tests reuse the existing PD fixtures from tests/handlers/test_pagerduty.py;
the resolution flag is orthogonal to every other connector behavior, so
the tests are small and focused on the flag itself.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.pagerduty import PagerDutyConnector
from services.ingestion.handlers.registry import build_connector
from shared.config import Settings
from shared.constants import SourceSystem
from shared.models import WebhookEvent

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pagerduty"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _build() -> PagerDutyConnector:
    settings = Settings(environment="local")
    ctx = ConnectorContext(settings=settings, http=httpx.AsyncClient())
    return build_connector(SourceSystem.PAGERDUTY, ctx)  # type: ignore[return-value]


def _make_event(payload: dict, source_event_id: str) -> WebhookEvent:
    return WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.PAGERDUTY,
        source_event_id=source_event_id,
        received_at=datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC),
        raw_payload=payload,
        headers={},
    )


@pytest.mark.asyncio
async def test_triggered_does_not_set_resolution_flag() -> None:
    pd = _build()
    result = await pd.normalize(
        _make_event(
            _load("incident_triggered.json"),
            "pd:incident:PD-INC-001:incident.triggered",
        ),
        {},
    )
    assert result.requires_resolution_check is False


@pytest.mark.asyncio
async def test_resolved_sets_resolution_flag() -> None:
    pd = _build()
    result = await pd.normalize(
        _make_event(
            _load("incident_resolved.json"),
            "pd:incident:PD-INC-001:incident.resolved",
        ),
        {},
    )
    assert result.requires_resolution_check is True


@pytest.mark.asyncio
async def test_resolved_does_not_set_investigation_flag() -> None:
    """`incident.resolved` must NOT set requires_investigation — the
    investigation pipeline runs once per logical incident, on triggered.
    """
    pd = _build()
    result = await pd.normalize(
        _make_event(
            _load("incident_resolved.json"),
            "pd:incident:PD-INC-001:incident.resolved",
        ),
        {},
    )
    assert result.requires_investigation is False


@pytest.mark.asyncio
async def test_acknowledged_does_not_set_resolution_flag() -> None:
    """A lifecycle event that isn't resolved (e.g. acknowledged) must
    leave the flag False — only resolved is the resolution signal.
    """
    pd = _build()
    result = await pd.normalize(
        _make_event(
            _load("incident_acknowledged.json"),
            "pd:incident:PD-INC-001:incident.acknowledged",
        ),
        {},
    )
    assert result.requires_resolution_check is False
