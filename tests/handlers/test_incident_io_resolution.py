"""incident.io connector detects `public_incident.incident_closed_v2`
and sets the `requires_resolution_check` flag — the signal the
post-approval dispatch seam listens for.

The event-kind string was confirmed against the existing
tests/fixtures/incident_io/incident_closed.json fixture (an incident
transitioning into status category 'closed'). incident.io fires this
event exactly once per incident close; we deliberately do NOT also
listen for status_changed_v2 with category=closed — that would
double-trigger.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.incident_io import IncidentIoConnector
from services.ingestion.handlers.registry import build_connector
from shared.config import Settings
from shared.constants import SourceSystem
from shared.models import WebhookEvent

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "incident_io"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _build() -> IncidentIoConnector:
    settings = Settings(environment="local")
    ctx = ConnectorContext(settings=settings, http=httpx.AsyncClient())
    return build_connector(SourceSystem.INCIDENT_IO, ctx)  # type: ignore[return-value]


def _make_event(payload: dict, source_event_id: str) -> WebhookEvent:
    return WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.INCIDENT_IO,
        source_event_id=source_event_id,
        received_at=datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC),
        raw_payload=payload,
        headers={},
    )


@pytest.mark.asyncio
async def test_created_does_not_set_resolution_flag() -> None:
    iio = _build()
    result = await iio.normalize(
        _make_event(
            _load("incident_created.json"),
            "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX:"
            "public_incident.incident_created_v2",
        ),
        {},
    )
    assert result.requires_resolution_check is False


@pytest.mark.asyncio
async def test_closed_sets_resolution_flag() -> None:
    iio = _build()
    result = await iio.normalize(
        _make_event(
            _load("incident_closed.json"),
            "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX:"
            "public_incident.incident_closed_v2",
        ),
        {},
    )
    assert result.requires_resolution_check is True


@pytest.mark.asyncio
async def test_closed_does_not_set_investigation_flag() -> None:
    """`incident_closed_v2` must NOT set requires_investigation — the
    investigation pipeline runs once per logical incident, on created.
    """
    iio = _build()
    result = await iio.normalize(
        _make_event(
            _load("incident_closed.json"),
            "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX:"
            "public_incident.incident_closed_v2",
        ),
        {},
    )
    assert result.requires_investigation is False


@pytest.mark.asyncio
async def test_status_changed_does_not_set_resolution_flag() -> None:
    """Intermediate status transitions (e.g. Investigating -> Fixing)
    must NOT trigger post-approval. Only the dedicated closed event.
    """
    iio = _build()
    result = await iio.normalize(
        _make_event(
            _load("incident_status_changed.json"),
            "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX:"
            "public_incident.incident_status_changed_v2",
        ),
        {},
    )
    assert result.requires_resolution_check is False
