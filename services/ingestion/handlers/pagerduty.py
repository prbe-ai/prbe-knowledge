"""PagerDuty connector — incident-pager source.

One INCIDENT document per logical PD incident, updated across lifecycle
events (triggered / acknowledged / resolved / etc.) via SCD2 coalesce.
`requires_investigation` fires only on `incident.triggered` so the
investigation pipeline runs exactly once per logical incident.
`requires_resolution_check` fires only on `incident.resolved` so the
post-approval dispatch seam (services/post_approval/dispatch.py) can
detect the (approved ∧ resolved) edge.

Connector-level `verify_signature` is intentionally a stub — see the
method docstring for the rationale. PD's per-subscription secret is
verified at the prbe-backend gateway BEFORE this code is reached.
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
    EdgeType,
    IngestionEventType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.exceptions import InvalidWebhookPayload
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    Document,
    GraphEdgeSpec,
    GraphNodeSpec,
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)


def _pd_person_id(obj: dict[str, Any] | None) -> str | None:
    """Extract a PagerDuty user id from an embedded user/assignee object.

    PD nests person references one of two ways:
      - direct user object: ``{"id": "PXXXXXX", "type": "user_reference", ...}``
      - assignment wrapper: ``{"assignee": {"id": "PXXXXXX", ...}, ...}``
    Caller passes whichever shape they have; we accept both.
    """
    if not isinstance(obj, dict):
        return None
    assignee = obj.get("assignee")
    if isinstance(assignee, dict):
        obj = assignee
    pid = obj.get("id")
    return pid if isinstance(pid, str) and pid else None


def _pd_person_props(obj: dict[str, Any] | None) -> dict[str, Any]:
    """Properties dict for a PD Person graph node — name + source tag."""
    props: dict[str, Any] = {"source_system": SourceSystem.PAGERDUTY.value}
    if not isinstance(obj, dict):
        return props
    inner = obj.get("assignee") if isinstance(obj.get("assignee"), dict) else obj
    summary = (inner.get("summary") or "").strip() if isinstance(inner, dict) else ""
    if summary:
        props["name"] = summary
    return props


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

        # Sub-resource events (notes/status updates/responders/etc.) put the
        # sub-resource at event.data, with the parent incident referenced via
        # event.data.incident. The lifecycle events (triggered/acknowledged/
        # resolved/escalated/reassigned/priority_updated/reopened/unacknowledged/
        # delegated) put the incident itself at event.data with data.type =
        # "incident".
        data_type = data.get("type")
        if data_type == "incident":
            incident_id = data.get("id")
            service = data.get("service") if isinstance(data.get("service"), dict) else {}
        else:
            # Sub-resource event — the incident is referenced under data.incident.
            incident = data.get("incident") if isinstance(data.get("incident"), dict) else {}
            incident_id = incident.get("id")
            service = incident.get("service") if isinstance(incident.get("service"), dict) else {}

        if not incident_id:
            raise InvalidWebhookPayload("pagerduty event has no resolvable incident id")

        received_at = _parse_iso8601(event.get("occurred_at")) or datetime.now(UTC)

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
        raw = event.raw_payload
        pd_event = raw.get("event") or {}
        event_type: str = pd_event.get("event_type") or ""
        occurred_at_str: str | None = pd_event.get("occurred_at")
        data: dict[str, Any] = pd_event.get("data") or {}

        # Resolve the incident object — lifecycle events have data.type="incident"
        # and the incident fields directly on data; sub-resource events (annotated,
        # status_update_published, responder.added, etc.) have the incident
        # referenced under data.incident.
        data_type = data.get("type")
        if data_type == "incident":
            incident: dict[str, Any] = data
        else:
            incident = data.get("incident") if isinstance(data.get("incident"), dict) else {}

        service: dict[str, Any] = incident.get("service") if isinstance(incident.get("service"), dict) else {}
        priority: dict[str, Any] = incident.get("priority") if isinstance(incident.get("priority"), dict) else {}
        escalation_policy: dict[str, Any] = incident.get("escalation_policy") if isinstance(incident.get("escalation_policy"), dict) else {}

        # Person attribution: PD nests "who's currently paged" and "who last
        # changed status" separately. assignments[0].assignee is the current
        # responder; last_status_change_by is whoever ack'd / resolved /
        # reassigned the incident. We mirror the sentry pattern of
        # `author_id = assignee or actor_id` so the Document points at one of
        # them for the recency / author-id filter, and emit Person nodes for
        # both (deduped when they're the same person).
        assignments = incident.get("assignments")
        first_assignment = assignments[0] if isinstance(assignments, list) and assignments else None
        assignee_obj = first_assignment if isinstance(first_assignment, dict) else None
        actor_obj = incident.get("last_status_change_by")
        if not isinstance(actor_obj, dict):
            actor_obj = None

        assignee_id = _pd_person_id(assignee_obj)
        actor_id = _pd_person_id(actor_obj)
        author_id = assignee_id or actor_id

        incident_id: str | None = incident.get("id")
        if not incident_id:
            return NormalizationResult(skipped_reason="missing event.data.id")

        doc_id = f"pd:incident:{incident_id}"
        status: str = incident.get("status") or ""
        urgency: str | None = incident.get("urgency")
        incident_key: str | None = incident.get("incident_key")
        title_raw: str = incident.get("title") or f"PagerDuty incident {incident_id}"
        service_id: str | None = service.get("id")
        service_summary: str | None = service.get("summary")
        service_url: str | None = service.get("html_url")
        ep_summary: str | None = escalation_policy.get("summary")
        priority_name: str | None = priority.get("name")
        created_at_str: str | None = incident.get("created_at")

        # Build human-readable markdown body
        body_lines: list[str] = [
            f"# {title_raw}",
            "",
            f"**Status:** {status}",
            f"**Urgency:** {urgency or 'unknown'}",
            f"**Priority:** {priority_name or 'none'}",
            f"**Service:** {service_summary or service_id or 'unknown'}",
            f"**Escalation policy:** {ep_summary or 'unknown'}",
            f"**Created at:** {created_at_str or 'unknown'}",
            f"**Last event:** {event_type} at {occurred_at_str or 'unknown'}",
        ]
        if incident_key:
            body_lines.append(f"**Incident key:** {incident_key}")
        body = "\n".join(body_lines)

        content_hash = hashlib.sha256(
            f"{event_type}|{status}|{body}".encode()
        ).hexdigest()

        # ACL: workspace principal + optional service group
        acl_principals: list[ACLPrincipal] = [
            ACLPrincipal(
                principal_type=PrincipalType.WORKSPACE,
                principal_id=event.customer_id,
                permission=Permission.READ,
            ),
        ]
        if service_id:
            acl_principals.append(
                ACLPrincipal(
                    principal_type=PrincipalType.GROUP,
                    principal_id=f"incident-service:{service_id}",
                    permission=Permission.READ,
                )
            )

        created_at = _parse_iso8601(created_at_str) or event.received_at

        doc = Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.PAGERDUTY,
            source_id=incident_id,
            source_url=incident.get("html_url") or "",
            doc_class=DocClass.RAW_SOURCE,
            doc_type=DocType.INCIDENT,
            content_type="text/markdown",
            content_hash=content_hash,
            title=title_raw[:240],
            body_preview=body[:280],
            body_size_bytes=len(body.encode("utf-8")),
            body_token_count=count_tokens(body),
            author_id=author_id,
            created_at=created_at,
            updated_at=event.received_at,
            valid_from=event.received_at,
            ingested_at=datetime.now(UTC),
            parent_doc_id=None,
            acl=ACLSnapshot(principals=acl_principals, captured_at=event.received_at),
            metadata={
                "incident_id": incident_id,
                "current_status": status,
                "last_event_type": event_type,
                "urgency": urgency,
                "priority": priority_name,
                "service_id": service_id,
                "service_url": service_url,
                "escalation_policy": ep_summary,
                "incident_key": incident_key,
            },
            body=body,
            coalesce_into_live=True,
        )

        # Graph emission: Document + Person nodes for assignee/actor + edges.
        graph_nodes: list[GraphNodeSpec] = [
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=doc_id,
                properties={"doc_type": DocType.INCIDENT.value},
            ),
        ]
        graph_edges: list[GraphEdgeSpec] = []

        if assignee_id:
            graph_nodes.append(
                GraphNodeSpec(
                    label=NodeLabel.PERSON,
                    canonical_id=assignee_id,
                    properties=_pd_person_props(assignee_obj),
                )
            )
            graph_edges.append(
                GraphEdgeSpec(
                    edge_type=EdgeType.ASSIGNED_TO,
                    from_label=NodeLabel.DOCUMENT,
                    from_canonical_id=doc_id,
                    to_label=NodeLabel.PERSON,
                    to_canonical_id=assignee_id,
                    valid_from=event.received_at,
                )
            )

        if actor_id and actor_id != assignee_id:
            graph_nodes.append(
                GraphNodeSpec(
                    label=NodeLabel.PERSON,
                    canonical_id=actor_id,
                    properties=_pd_person_props(actor_obj),
                )
            )

        # AUTHORED edge from the human who most recently touched the
        # incident (actor preferred; falls back to assignee). Mirrors the
        # sentry pattern where status changes are attributed to whoever
        # triggered the webhook event.
        author_node_id = actor_id or assignee_id
        if author_node_id:
            graph_edges.append(
                GraphEdgeSpec(
                    edge_type=EdgeType.AUTHORED,
                    from_label=NodeLabel.PERSON,
                    from_canonical_id=author_node_id,
                    to_label=NodeLabel.DOCUMENT,
                    to_canonical_id=doc_id,
                    valid_from=event.received_at,
                )
            )

        return NormalizationResult(
            documents=[doc],
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            requires_investigation=(event_type == "incident.triggered"),
            requires_resolution_check=(event_type == "incident.resolved"),
        )


__all__ = ["PagerDutyConnector"]
