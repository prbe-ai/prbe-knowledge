"""incident.io connector — incident-response source.

One INCIDENT document per logical incident.io incident, updated across
lifecycle events (incident_created / status_changed / closed / etc.) via
SCD2 coalesce. `requires_investigation` fires only on
`public_incident.incident_created_v2` so the investigation pipeline runs
exactly once per logical incident.

Connector-level `verify_signature` is intentionally a stub — incident.io
uses Svix-format signatures (HMAC over `id.timestamp.body`) verified by
the per-tenant secret resolved from `webhook_secrets` at the prbe-backend
gateway. See the method docstring for the rationale.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar

from services.ingestion.chunker import count_tokens
from services.ingestion.handlers.base import Connector
from services.ingestion.handlers.registry import register_connector
from shared.constants import (
    DocClass,
    DocType,
    IngestionEventType,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.exceptions import InvalidWebhookPayload
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    Document,
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)

_INCIDENT_EVENT_PREFIX = "public_incident."
_CREATED_EVENT_TYPE = "public_incident.incident_created_v2"


def _parse_iso8601(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _content_hash(event_type: str, status: str, body: str) -> str:
    return hashlib.sha256(f"{event_type}|{status}|{body}".encode()).hexdigest()


@register_connector(SourceSystem.INCIDENT_IO)
class IncidentIoConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.INCIDENT_IO
    display_name: ClassVar[str] = "incident.io"

    def verify_signature(
        self, headers: Mapping[str, str], raw_body: bytes
    ) -> bool:
        """Connector-level signature verification — intentionally a stub.

        Deliberate deviation from the env-secret + HMAC pattern used by
        sentry/github/notion/slack/linear. incident.io webhooks use
        Svix-format signatures (HMAC-SHA256 over `id.timestamp.body`,
        base64-encoded, with replay-defense via `webhook-timestamp`).
        The per-endpoint secret lives in `webhook_secrets` keyed by the
        URL-path tenant token, and is verified at the prbe-backend
        gateway BEFORE the body reaches the knowledge ingestion service.

        `services/ingestion/main.py`'s webhook endpoint does not invoke
        `verify_signature` at runtime; this method exists only to
        satisfy the abstract base contract.
        """
        return True

    def parse_webhook_event(
        self,
        customer_id: str,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        event_type = raw_payload.get("event_type")
        if not isinstance(event_type, str) or not event_type.startswith(
            _INCIDENT_EVENT_PREFIX
        ):
            return None
        data = raw_payload.get("data")
        if not isinstance(data, dict):
            raise InvalidWebhookPayload("incident_io payload missing 'data' dict")
        incident = data.get("incident")
        if not isinstance(incident, dict):
            raise InvalidWebhookPayload(
                "incident_io payload missing 'data.incident'"
            )
        incident_id = incident.get("id")
        if not incident_id:
            raise InvalidWebhookPayload("incident_io incident missing id")
        received_at = (
            _parse_iso8601(raw_payload.get("delivered_at"))
            or _parse_iso8601(incident.get("created_at"))
            or datetime.now(UTC)
        )
        return WebhookParseResult(
            source_event_id=f"iio:incident:{incident_id}:{event_type}",
            received_at=received_at,
            event_kind=IngestionEventType.WEBHOOK,
            parse_hint={
                "incident_id": incident_id,
                "event_type": event_type,
                "workspace_id": incident.get("workspace_id"),
                "reference": incident.get("reference"),
            },
        )

    async def normalize(
        self, event: WebhookEvent, hydrated: Mapping[str, Any]
    ) -> NormalizationResult:
        payload = event.raw_payload
        event_type = payload.get("event_type") or ""
        data = payload.get("data") or {}
        incident = data.get("incident") or {}
        incident_id = incident.get("id")
        if not incident_id:
            return NormalizationResult(skipped_reason="missing data.incident.id")

        name = incident.get("name") or f"incident.io {incident_id}"
        summary = incident.get("summary") or ""
        status_obj = (
            incident.get("incident_status")
            if isinstance(incident.get("incident_status"), dict)
            else {}
        )
        status = status_obj.get("name") or ""
        severity_obj = (
            incident.get("severity")
            if isinstance(incident.get("severity"), dict)
            else {}
        )
        severity = severity_obj.get("name") or ""
        reference = incident.get("reference") or ""

        created_at = _parse_iso8601(incident.get("created_at")) or event.received_at
        occurred_at = event.received_at

        body_lines = [
            f"# {name}",
            "",
            summary or "_no summary provided_",
            "",
            f"- Status: **{status}**",
            f"- Severity: {severity or 'unset'}",
            f"- Reference: {reference or 'n/a'}",
            f"- Created at: {created_at.isoformat()}",
            f"- Last event: {event_type} at {occurred_at.isoformat()}",
        ]
        body = "\n".join(body_lines)
        content_hash = _content_hash(event_type, status, body)

        doc_id = f"iio:incident:{incident_id}"
        source_url = incident.get("permalink") or ""

        # Extract a service tag from custom_field_entries — used as ACL
        # group principal when present. Not all incident.io workspaces
        # populate this field, so it's optional.
        service_tag: str | None = None
        for entry in incident.get("custom_field_entries") or []:
            if not isinstance(entry, dict):
                continue
            cf = entry.get("custom_field") if isinstance(entry.get("custom_field"), dict) else {}
            if (cf.get("name") or "").lower() in {"affected service", "service"}:
                for v in entry.get("values") or []:
                    if isinstance(v, dict) and (text := v.get("value_text")):
                        service_tag = text
                        break
                if service_tag:
                    break

        acl_principals = [
            ACLPrincipal(
                principal_type=PrincipalType.WORKSPACE,
                principal_id=event.customer_id,
                permission=Permission.READ,
            ),
        ]
        if service_tag:
            acl_principals.append(
                ACLPrincipal(
                    principal_type=PrincipalType.GROUP,
                    principal_id=f"incident-service:{service_tag}",
                    permission=Permission.READ,
                )
            )

        doc = Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.INCIDENT_IO,
            source_id=incident_id,
            source_url=source_url,
            doc_class=DocClass.RAW_SOURCE,
            doc_type=DocType.INCIDENT,
            content_type="text/markdown",
            content_hash=content_hash,
            title=name[:240],
            body=body,
            body_preview=body[:280],
            body_size_bytes=len(body.encode("utf-8")),
            body_token_count=count_tokens(body),
            created_at=created_at,
            updated_at=occurred_at,
            valid_from=occurred_at,
            ingested_at=datetime.now(UTC),
            acl=ACLSnapshot(principals=acl_principals, captured_at=event.received_at),
            metadata={
                "incident_id": incident_id,
                "current_status": status,
                "severity": severity,
                "last_event_type": event_type,
                "workspace_id": incident.get("workspace_id"),
                "reference": reference,
                "service_tag": service_tag,
                "permalink": source_url,
            },
            coalesce_into_live=True,
        )

        return NormalizationResult(
            documents=[doc],
            requires_investigation=event_type == _CREATED_EVENT_TYPE,
        )


__all__ = ["IncidentIoConnector"]
