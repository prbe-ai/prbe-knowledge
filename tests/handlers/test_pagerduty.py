"""PagerDuty connector parse_webhook_event tests."""
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
from shared.constants import IngestionEventType, SourceSystem
from shared.exceptions import InvalidWebhookPayload

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pagerduty"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _build() -> PagerDutyConnector:
    settings = Settings(environment="local")
    ctx = ConnectorContext(settings=settings, http=httpx.AsyncClient())
    return build_connector(SourceSystem.PAGERDUTY, ctx)  # type: ignore[return-value]


def test_parse_triggered_event_id_is_lifecycle_scoped() -> None:
    pd = _build()
    payload = _load("incident_triggered.json")
    result = pd.parse_webhook_event("cust-1", {}, payload)
    assert result is not None
    assert result.source_event_id == "pd:incident:PD-INC-001:incident.triggered"
    assert result.event_kind == IngestionEventType.WEBHOOK
    assert result.parse_hint["event_type"] == "incident.triggered"
    assert result.parse_hint["incident_id"] == "PD-INC-001"
    assert result.parse_hint["service_id"] == "PSRV001"
    assert result.received_at == datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC)


def test_parse_acknowledged_returns_lifecycle_scoped_id() -> None:
    pd = _build()
    result = pd.parse_webhook_event("cust-1", {}, _load("incident_acknowledged.json"))
    assert result is not None
    assert result.source_event_id == "pd:incident:PD-INC-001:incident.acknowledged"


def test_parse_resolved_returns_lifecycle_scoped_id() -> None:
    pd = _build()
    result = pd.parse_webhook_event("cust-1", {}, _load("incident_resolved.json"))
    assert result is not None
    assert result.source_event_id == "pd:incident:PD-INC-001:incident.resolved"


def test_parse_non_incident_event_type_returns_none() -> None:
    pd = _build()
    payload = _load("incident_triggered.json")
    payload["event"]["event_type"] = "service.created"  # outside incident.* namespace
    assert pd.parse_webhook_event("cust-1", {}, payload) is None


def test_parse_missing_event_block_raises() -> None:
    pd = _build()
    with pytest.raises(InvalidWebhookPayload, match="missing 'event' dict"):
        pd.parse_webhook_event("cust-1", {}, {})


def test_parse_missing_data_block_raises() -> None:
    pd = _build()
    with pytest.raises(InvalidWebhookPayload, match="missing 'event.data'"):
        pd.parse_webhook_event(
            "cust-1", {}, {"event": {"event_type": "incident.triggered"}},
        )


def test_parse_missing_incident_id_raises() -> None:
    pd = _build()
    with pytest.raises(InvalidWebhookPayload, match="missing data.id"):
        pd.parse_webhook_event(
            "cust-1", {},
            {"event": {"event_type": "incident.triggered", "data": {}}},
        )


def test_verify_signature_always_true_gateway_owns_it() -> None:
    pd = _build()
    assert pd.verify_signature({}, b"anything") is True


@pytest.mark.parametrize(
    "event_type",
    [
        "incident.priority_updated",
        "incident.escalated",
        "incident.reassigned",
        "incident.annotated",
        "incident.delegated",
        "incident.unacknowledged",
    ],
)
def test_parse_handles_any_incident_event_type(event_type: str) -> None:
    """The startswith('incident.') guard must accept every documented
    PD v3 incident lifecycle event_type, not just the three we hand-
    fixtured. parse_hint should propagate event_type so downstream
    normalize can branch on it."""
    pd = _build()
    payload = _load("incident_triggered.json")
    payload["event"]["event_type"] = event_type
    payload["event"]["id"] = f"01EVENT_{event_type}"
    result = pd.parse_webhook_event("cust-1", {}, payload)
    assert result is not None
    assert result.source_event_id == f"pd:incident:PD-INC-001:{event_type}"
    assert result.parse_hint["event_type"] == event_type


def test_parse_is_idempotent_for_same_input() -> None:
    """Re-deliveries of the same lifecycle event must produce the same
    source_event_id so the ingestion_queue UNIQUE constraint dedupes."""
    pd = _build()
    payload = _load("incident_triggered.json")
    r1 = pd.parse_webhook_event("cust-1", {}, payload)
    r2 = pd.parse_webhook_event("cust-1", {}, payload)
    assert r1 is not None and r2 is not None
    assert r1.source_event_id == r2.source_event_id


@pytest.mark.asyncio
async def test_normalize_stub_returns_skipped() -> None:
    from shared.models import WebhookEvent
    pd = _build()
    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.PAGERDUTY,
        source_event_id="pd:incident:X:incident.triggered",
        received_at=datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC),
        raw_payload={}, headers={},
    )
    result = await pd.normalize(event, {})
    assert result.skipped_reason == "pagerduty normalize not yet implemented (B.2)"
    assert result.documents == []
