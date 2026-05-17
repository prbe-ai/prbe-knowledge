"""PagerDuty connector — incident-pager source.

One INCIDENT document per logical PD incident, updated across lifecycle
events (triggered / acknowledged / resolved / etc.).
`requires_investigation` fires only on `incident.triggered` so the
investigation pipeline runs once per logical incident — see B.2 for the
normalize implementation that owns that flag.

Signature verification is a no-op at the connector level: the gateway
(prbe-backend `apps/data_plane/routers/webhooks/sources/pagerduty.py`)
is the trust boundary, matching every other handler in this package.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar

from services.ingestion.handlers.base import Connector
from services.ingestion.handlers.registry import register_connector
from shared.constants import IngestionEventType, SourceSystem
from shared.exceptions import InvalidWebhookPayload
from shared.models import (
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@register_connector(SourceSystem.PAGERDUTY)
class PagerDutyConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.PAGERDUTY
    display_name: ClassVar[str] = "PagerDuty"

    def verify_signature(
        self, headers: Mapping[str, str], raw_body: bytes
    ) -> bool:
        """Connector-level signature verification — intentionally a stub.

        Deliberate deviation from the env-secret + HMAC pattern used by
        sentry/github/notion/slack/linear in this package. PagerDuty does
        not have a Probe-owned default app, so there is no shared secret to
        verify against at the connector layer; the per-subscription secret
        lives in `webhook_secrets` keyed by the URL-path tenant token, and
        is verified at the prbe-backend gateway BEFORE the body reaches the
        knowledge ingestion service.

        `services/ingestion/main.py`'s webhook endpoint does not invoke
        `verify_signature` at runtime today (trust is conveyed via the
        `X-Internal-Knowledge-Key` header); this method exists only to
        satisfy the abstract base contract.
        """
        return True

    def parse_webhook_event(
        self,
        customer_id: str,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        event = raw_payload.get("event")
        if not isinstance(event, dict):
            raise InvalidWebhookPayload("pagerduty payload missing 'event' dict")

        event_type = event.get("event_type") or ""
        if not isinstance(event_type, str) or not event_type.startswith("incident."):
            return None

        data = event.get("data")
        if not isinstance(data, dict):
            raise InvalidWebhookPayload("pagerduty payload missing 'event.data'")

        incident_id = data.get("id")
        if not incident_id:
            raise InvalidWebhookPayload("pagerduty incident missing data.id")

        received_at = _parse_iso8601(event.get("occurred_at")) or datetime.now(UTC)
        service = data.get("service") if isinstance(data.get("service"), dict) else {}

        return WebhookParseResult(
            source_event_id=f"pd:incident:{incident_id}:{event_type}",
            received_at=received_at,
            event_kind=IngestionEventType.WEBHOOK,
            parse_hint={
                "incident_id": incident_id,
                "event_type": event_type,
                "service_id": service.get("id"),
                "service_summary": service.get("summary"),
                "service_url": service.get("html_url"),
            },
        )

    async def normalize(
        self, event: WebhookEvent, hydrated: Mapping[str, Any]
    ) -> NormalizationResult:
        # B.2 owns the real normalize. Returning a non-empty skipped_reason
        # so that any accidental routing through this stub is loud rather
        # than silently producing zero-doc outcomes.
        return NormalizationResult(
            skipped_reason="pagerduty normalize not yet implemented (B.2)"
        )


__all__ = ["PagerDutyConnector"]
