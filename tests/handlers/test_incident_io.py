"""incident.io connector tests."""
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
from shared.constants import (
    DocClass,
    DocType,
    IngestionEventType,
    PrincipalType,
    SourceSystem,
)
from shared.exceptions import InvalidWebhookPayload
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


# ---------------- parse_webhook_event ----------------


def test_parse_created_event_id_is_lifecycle_scoped() -> None:
    iio = _build()
    result = iio.parse_webhook_event("cust-1", {}, _load("incident_created.json"))
    assert result is not None
    assert result.source_event_id == (
        "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX:public_incident.incident_created_v2"
    )
    assert result.event_kind == IngestionEventType.WEBHOOK
    assert result.parse_hint["incident_id"] == "01ABCDEFGHIJKLMNOPQRSTUVWX"
    assert result.parse_hint["event_type"] == "public_incident.incident_created_v2"
    assert result.parse_hint["slack_team_id"] == "T01XYZ"


def test_parse_status_changed_event_id_is_lifecycle_scoped() -> None:
    iio = _build()
    result = iio.parse_webhook_event(
        "cust-1", {}, _load("incident_status_changed.json")
    )
    assert result is not None
    assert "incident_status_changed_v2" in result.source_event_id


def test_parse_closed_event_id_is_lifecycle_scoped() -> None:
    iio = _build()
    result = iio.parse_webhook_event("cust-1", {}, _load("incident_closed.json"))
    assert result is not None
    assert "incident_closed_v2" in result.source_event_id


@pytest.mark.parametrize(
    "event_type",
    [
        "public_incident.incident_updated_v2",
        "public_incident.incident_severity_changed_v2",
        "public_incident.incident_role_assigned_v2",
    ],
)
def test_parse_handles_any_public_incident_event_type(event_type: str) -> None:
    """The startswith('public_incident.') guard must accept every
    documented incident.io lifecycle event_type."""
    iio = _build()
    payload = _load("incident_created.json")
    # Move the incident object to the new event_type key (real envelope shape)
    incident_obj = payload.pop("public_incident.incident_created_v2")
    payload["event_type"] = event_type
    payload[event_type] = incident_obj
    result = iio.parse_webhook_event("cust-1", {}, payload)
    assert result is not None
    assert event_type in result.source_event_id


def test_parse_non_incident_event_returns_none() -> None:
    iio = _build()
    result = iio.parse_webhook_event(
        "cust-1", {},
        {"event_type": "public_action.action_created_v2", "data": {}},
    )
    assert result is None


def test_parse_missing_event_type_key_raises() -> None:
    iio = _build()
    with pytest.raises(
        InvalidWebhookPayload,
        match=r"missing 'public_incident\.incident_created_v2' object",
    ):
        iio.parse_webhook_event(
            "cust-1", {},
            {"event_type": "public_incident.incident_created_v2"},
        )


def test_parse_missing_incident_id_raises() -> None:
    iio = _build()
    with pytest.raises(InvalidWebhookPayload, match="missing id"):
        iio.parse_webhook_event(
            "cust-1", {},
            {
                "event_type": "public_incident.incident_created_v2",
                "public_incident.incident_created_v2": {},
            },
        )


def test_parse_is_idempotent_for_same_input() -> None:
    iio = _build()
    payload = _load("incident_created.json")
    r1 = iio.parse_webhook_event("cust-1", {}, payload)
    r2 = iio.parse_webhook_event("cust-1", {}, payload)
    assert r1 is not None and r2 is not None
    assert r1.source_event_id == r2.source_event_id


def test_verify_signature_always_true_gateway_owns_it() -> None:
    iio = _build()
    assert iio.verify_signature({}, b"anything") is True


# ---------------- normalize ----------------


@pytest.mark.asyncio
async def test_normalize_created_emits_incident_doc_and_flag() -> None:
    iio = _build()
    payload = _load("incident_created.json")
    event_id = (
        "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX:public_incident.incident_created_v2"
    )
    result = await iio.normalize(_make_event(payload, event_id), {})
    assert result.skipped_reason is None
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.doc_id == "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX"
    assert doc.source_system == SourceSystem.INCIDENT_IO
    assert doc.doc_type == DocType.INCIDENT
    assert doc.doc_class == DocClass.RAW_SOURCE
    assert doc.content_type == "text/markdown"
    assert doc.title == "Checkout payments degraded"
    assert "investigating" in (doc.body or "").lower()
    assert doc.metadata["current_status"] == "Investigating"
    assert doc.metadata["severity"] == "Major"
    assert doc.metadata["slack_team_id"] == "T01XYZ"
    assert doc.metadata["reference"] == "INC-42"
    assert doc.metadata["service_tag"] == "checkout-svc"
    assert doc.metadata["last_event_type"] == "public_incident.incident_created_v2"
    assert doc.coalesce_into_live is True
    assert result.requires_investigation is True


@pytest.mark.asyncio
async def test_normalize_status_changed_updates_same_doc_id_no_flag() -> None:
    iio = _build()
    payload = _load("incident_status_changed.json")
    event_id = (
        "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX:"
        "public_incident.incident_status_changed_v2"
    )
    result = await iio.normalize(_make_event(payload, event_id), {})
    assert result.documents[0].doc_id == "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX"
    assert result.documents[0].metadata["current_status"] == "Fixing"
    assert result.requires_investigation is False


@pytest.mark.asyncio
async def test_normalize_closed_updates_same_doc_id_no_flag() -> None:
    iio = _build()
    payload = _load("incident_closed.json")
    event_id = (
        "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX:public_incident.incident_closed_v2"
    )
    result = await iio.normalize(_make_event(payload, event_id), {})
    assert result.documents[0].metadata["current_status"] == "Closed"
    assert result.requires_investigation is False


@pytest.mark.asyncio
async def test_normalize_acl_workspace_plus_service_group() -> None:
    iio = _build()
    payload = _load("incident_created.json")
    event_id = (
        "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX:public_incident.incident_created_v2"
    )
    result = await iio.normalize(_make_event(payload, event_id), {})
    principals = {
        (p.principal_type, p.principal_id)
        for p in result.documents[0].acl.principals
    }
    assert (PrincipalType.WORKSPACE, "cust-1") in principals
    assert (PrincipalType.GROUP, "incident-service:checkout-svc") in principals


@pytest.mark.asyncio
async def test_normalize_acl_workspace_only_when_service_field_missing() -> None:
    iio = _build()
    payload = _load("incident_created.json")
    payload["public_incident.incident_created_v2"]["custom_field_entries"] = []
    event_id = (
        "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX:public_incident.incident_created_v2"
    )
    result = await iio.normalize(_make_event(payload, event_id), {})
    principals = {
        (p.principal_type, p.principal_id)
        for p in result.documents[0].acl.principals
    }
    assert (PrincipalType.WORKSPACE, "cust-1") in principals
    assert not any(pt == PrincipalType.GROUP for (pt, _) in principals)


@pytest.mark.asyncio
async def test_normalize_skipped_when_incident_id_missing() -> None:
    iio = _build()
    payload = _load("incident_created.json")
    del payload["public_incident.incident_created_v2"]["id"]
    event_id = "iio:incident:X:public_incident.incident_created_v2"
    result = await iio.normalize(_make_event(payload, event_id), {})
    assert result.skipped_reason == "missing data.incident.id"
    assert result.documents == []


@pytest.mark.asyncio
async def test_normalize_content_hash_stable_across_redeliveries() -> None:
    iio = _build()
    payload = _load("incident_created.json")
    r1 = await iio.normalize(_make_event(payload, "e1"), {})
    r2 = await iio.normalize(_make_event(payload, "e2"), {})
    assert r1.documents[0].content_hash == r2.documents[0].content_hash


@pytest.mark.asyncio
async def test_normalize_content_hash_differs_across_lifecycle() -> None:
    """A created → status_changed → closed transition must mutate the
    content_hash so the normalizer's content-hash dedup recognizes the
    update and rewrites the coalesced live row."""
    iio = _build()
    created = await iio.normalize(
        _make_event(
            _load("incident_created.json"),
            "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX:public_incident.incident_created_v2",
        ),
        {},
    )
    changed = await iio.normalize(
        _make_event(
            _load("incident_status_changed.json"),
            "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX:public_incident.incident_status_changed_v2",
        ),
        {},
    )
    closed = await iio.normalize(
        _make_event(
            _load("incident_closed.json"),
            "iio:incident:01ABCDEFGHIJKLMNOPQRSTUVWX:public_incident.incident_closed_v2",
        ),
        {},
    )
    hashes = {created.documents[0].content_hash,
              changed.documents[0].content_hash,
              closed.documents[0].content_hash}
    assert len(hashes) == 3
