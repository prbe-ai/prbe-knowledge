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
from shared.constants import (
    DocClass,
    DocType,
    IngestionEventType,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.exceptions import InvalidWebhookPayload
from shared.models import WebhookEvent

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
async def test_normalize_triggered_emits_incident_doc_and_flag() -> None:
    pd = _build()
    payload = _load("incident_triggered.json")
    result = await pd.normalize(
        _make_event(payload, "pd:incident:PD-INC-001:incident.triggered"), {},
    )
    assert result.skipped_reason is None
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.doc_id == "pd:incident:PD-INC-001"
    assert doc.source_system == SourceSystem.PAGERDUTY
    assert doc.doc_type == DocType.INCIDENT
    assert doc.doc_class == DocClass.RAW_SOURCE
    assert doc.content_type == "text/markdown"
    assert doc.title == "Checkout service: database connection pool exhausted"
    assert "triggered" in (doc.body or "").lower()
    assert doc.metadata["current_status"] == "triggered"
    assert doc.metadata["urgency"] == "high"
    assert doc.metadata["priority"] == "P1"
    assert doc.metadata["service_id"] == "PSRV001"
    assert doc.metadata["last_event_type"] == "incident.triggered"
    assert doc.metadata["incident_key"] == "checkout-svc/db-conn-pool-exhaust"
    assert doc.coalesce_into_live is True
    assert result.requires_investigation is True


@pytest.mark.asyncio
async def test_normalize_acknowledged_updates_same_doc_id_no_flag() -> None:
    pd = _build()
    payload = _load("incident_acknowledged.json")
    result = await pd.normalize(
        _make_event(payload, "pd:incident:PD-INC-001:incident.acknowledged"), {},
    )
    assert len(result.documents) == 1
    doc = result.documents[0]
    # Same logical incident → same stable doc_id
    assert doc.doc_id == "pd:incident:PD-INC-001"
    assert doc.metadata["current_status"] == "acknowledged"
    assert doc.metadata["last_event_type"] == "incident.acknowledged"
    assert result.requires_investigation is False


@pytest.mark.asyncio
async def test_normalize_resolved_updates_same_doc_id_no_flag() -> None:
    pd = _build()
    result = await pd.normalize(
        _make_event(_load("incident_resolved.json"),
                    "pd:incident:PD-INC-001:incident.resolved"),
        {},
    )
    assert result.documents[0].doc_id == "pd:incident:PD-INC-001"
    assert result.documents[0].metadata["current_status"] == "resolved"
    assert result.requires_investigation is False


@pytest.mark.asyncio
async def test_normalize_acl_workspace_plus_service_group() -> None:
    pd = _build()
    result = await pd.normalize(
        _make_event(_load("incident_triggered.json"),
                    "pd:incident:PD-INC-001:incident.triggered"),
        {},
    )
    principals = {
        (p.principal_type, p.principal_id)
        for p in result.documents[0].acl.principals
    }
    assert (PrincipalType.WORKSPACE, "cust-1") in principals
    assert (PrincipalType.GROUP, "incident-service:PSRV001") in principals


@pytest.mark.asyncio
async def test_normalize_acl_workspace_only_when_service_id_missing() -> None:
    pd = _build()
    payload = _load("incident_triggered.json")
    del payload["event"]["data"]["service"]["id"]
    result = await pd.normalize(
        _make_event(payload, "pd:incident:PD-INC-001:incident.triggered"), {},
    )
    principals = {
        (p.principal_type, p.principal_id)
        for p in result.documents[0].acl.principals
    }
    assert (PrincipalType.WORKSPACE, "cust-1") in principals
    # No service group when service id is absent
    assert not any(
        pt == PrincipalType.GROUP for (pt, _) in principals
    )


@pytest.mark.asyncio
async def test_normalize_skipped_when_data_id_missing() -> None:
    pd = _build()
    payload = _load("incident_triggered.json")
    del payload["event"]["data"]["id"]
    result = await pd.normalize(
        _make_event(payload, "pd:incident:X:incident.triggered"), {},
    )
    assert result.skipped_reason == "missing event.data.id"
    assert result.documents == []


@pytest.mark.asyncio
async def test_normalize_content_hash_stable_across_redeliveries() -> None:
    """Same payload twice should produce identical content_hash so SCD2
    coalesce treats redeliveries as a no-op."""
    pd = _build()
    payload = _load("incident_triggered.json")
    r1 = await pd.normalize(_make_event(payload, "e1"), {})
    r2 = await pd.normalize(_make_event(payload, "e2"), {})
    assert r1.documents[0].content_hash == r2.documents[0].content_hash


@pytest.mark.asyncio
async def test_normalize_content_hash_differs_across_lifecycle() -> None:
    """A triggered → acknowledged → resolved transition must mutate the
    content_hash so the normalizer's content-hash dedup recognizes the
    update and rewrites the coalesced live row instead of treating the
    re-delivery as a no-op."""
    pd = _build()
    triggered = await pd.normalize(
        _make_event(_load("incident_triggered.json"),
                    "pd:incident:PD-INC-001:incident.triggered"),
        {},
    )
    ack = await pd.normalize(
        _make_event(_load("incident_acknowledged.json"),
                    "pd:incident:PD-INC-001:incident.acknowledged"),
        {},
    )
    resolved = await pd.normalize(
        _make_event(_load("incident_resolved.json"),
                    "pd:incident:PD-INC-001:incident.resolved"),
        {},
    )
    hashes = {triggered.documents[0].content_hash,
              ack.documents[0].content_hash,
              resolved.documents[0].content_hash}
    assert len(hashes) == 3
