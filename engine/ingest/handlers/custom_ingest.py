"""Connector for customer-supplied text/structured documents."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar

from engine.ingest.chunker import count_tokens
from engine.ingest.handlers.base import Connector
from engine.ingest.handlers.registry import register_connector
from engine.shared.constants import (
    DocClass,
    DocType,
    EdgeType,
    IngestionEventType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from engine.shared.custom_ingest import CustomIngestEnvelope, custom_ingest_doc_id
from engine.shared.exceptions import InvalidWebhookPayload
from engine.shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    ACLSnapshotRow,
    Document,
    GraphEdgeSpec,
    GraphNodeSpec,
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)


@register_connector(SourceSystem.CUSTOM_INGEST)
class CustomIngestConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.CUSTOM_INGEST
    display_name: ClassVar[str] = "Custom ingest"
    doc_type_prefix: ClassVar[str] = "custom."
    # Queue priority 75: customer batches are bursty and search-indexable,
    # not user-blocking — same tier as agent-session sources so a large
    # custom push can't preempt interactive webhooks (100).
    ingestion_priority: ClassVar[int] = 75

    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> bool:
        # No public webhook surface — documents land via the authenticated
        # /api/custom-ingest/documents route (X-Internal-Knowledge-Key in
        # hosted mode, KNOWLEDGE_API_TOKEN bearer in standalone; see
        # custom_ingest_routes.py). Returning False keeps standalone
        # /webhooks/custom_ingest a hard 401. Mirrors handlers/wiki.py.
        return False

    def parse_webhook_event(
        self,
        customer_id: str,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        source_event_id = str(raw_payload.get("source_event_id") or "").strip()
        if not source_event_id:
            raise InvalidWebhookPayload("custom ingest payload missing source_event_id")
        return WebhookParseResult(
            source_event_id=source_event_id,
            received_at=_parse_iso(raw_payload.get("received_at")) or datetime.now(UTC),
            event_kind=IngestionEventType.MANUAL,
        )

    async def normalize(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        payload = event.raw_payload
        source_key = str(payload.get("source_key") or "").strip()
        document_payload = payload.get("document")
        if not source_key or not isinstance(document_payload, Mapping):
            raise InvalidWebhookPayload("custom ingest payload missing document")

        try:
            envelope = CustomIngestEnvelope.model_validate(
                {
                    "source_key": source_key,
                    "batch_id": payload.get("batch_id"),
                    "documents": [dict(document_payload)],
                }
            )
        except Exception as exc:
            raise InvalidWebhookPayload(f"invalid custom ingest payload: {exc}") from exc

        document = envelope.documents[0]
        received_at = event.received_at
        created_at = document.created_at or received_at
        updated_at = document.updated_at or created_at
        # Injective composition -- ':' in source_key is percent-encoded so
        # (source_key, id) pairs can never collide. See custom_ingest_doc_id.
        doc_id = custom_ingest_doc_id(event.customer_id, source_key, document.id)
        source_id = f"{source_key}:{document.id}"
        source_url = document.url or _dashboard_source_url(event.customer_id, source_key)
        title = document.title or f"{source_key}/{document.id}"
        is_delete = document.deleted
        # For a delete, body is empty and we write a tombstone. The
        # content_hash must differ from the prior live version so the
        # normalizer bumps the version and the chunk diff marks all previous
        # chunks stale — same shape as the linear/github connector deletes.
        deleted_at = received_at if is_delete else None
        body = "" if is_delete else document.body
        if is_delete:
            content_hash = _sha256(
                f"{doc_id}|__deleted__|{payload.get('content_hash') or ''}"
            )
        else:
            content_hash = _sha256(
                f"{doc_id}|{document.type or ''}|{title}|{document.body}|"
                f"{payload.get('content_hash') or ''}"
            )
        author = document.author
        author_id = author.id if author and author.id else author.email if author else None

        acl_principals = _workspace_read_write_principals(event.customer_id)
        acl_rows = [
            ACLSnapshotRow(
                source_system=SourceSystem.CUSTOM_INGEST,
                principal_type=principal.principal_type,
                principal_id=principal.principal_id,
                resource_type="custom.document",
                resource_id=source_id,
                permission=principal.permission,
                valid_from=updated_at,
                metadata={"source_key": source_key},
            )
            for principal in acl_principals
        ]

        metadata: dict[str, Any] = dict(document.metadata)
        metadata.update(
            {
                "source_key": source_key,
                "custom_document_id": document.id,
                "custom_document_type": document.type,
                "batch_id": payload.get("batch_id"),
                "provided_url": document.url,
                "author": author.model_dump(mode="json") if author else None,
                # Envelope-level content hash (shared.custom_ingest.
                # document_content_hash) — surfaced by the enumeration
                # endpoint so a consumer-side reconciler can diff its source
                # of truth against the index without re-pushing everything.
                "content_hash": payload.get("content_hash"),
            }
        )

        doc = Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.CUSTOM_INGEST,
            source_id=source_id,
            source_url=source_url,
            doc_class=DocClass.RAW_SOURCE,
            doc_type=DocType.CUSTOM_DOCUMENT,
            content_type="text/plain",
            content_hash=content_hash,
            title=title[:240],
            body_preview=body[:280] if body else None,
            body_size_bytes=len(body.encode("utf-8")),
            body_token_count=count_tokens(body),
            author_id=author_id,
            created_at=created_at,
            updated_at=updated_at,
            valid_from=updated_at,
            deleted_at=deleted_at,
            ingested_at=datetime.now(UTC),
            acl=ACLSnapshot(principals=acl_principals, captured_at=received_at),
            metadata=metadata,
            body=body,
        )

        # Tombstones carry no graph payload: there is nothing new to assert
        # about a deleted document, and the normalizer's chunk diff already
        # removes every live chunk (deleted_at is set), which drops the doc
        # from retrieval.
        if is_delete:
            return NormalizationResult(documents=[doc], acl_snapshots=acl_rows)

        graph_nodes = [
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=doc_id,
                properties={
                    "title": title,
                    "source_system": SourceSystem.CUSTOM_INGEST.value,
                    "source_key": source_key,
                    "custom_document_id": document.id,
                },
            )
        ]
        graph_edges: list[GraphEdgeSpec] = []
        if author_id:
            author_node_id = f"person:{author_id}"
            graph_nodes.append(
                GraphNodeSpec(
                    label=NodeLabel.PERSON,
                    canonical_id=author_node_id,
                    properties={
                        "name": author.name if author else None,
                        "email": author.email if author else None,
                        "source_system": SourceSystem.CUSTOM_INGEST.value,
                    },
                )
            )
            graph_edges.append(
                GraphEdgeSpec(
                    edge_type=EdgeType.AUTHORED,
                    from_label=NodeLabel.PERSON,
                    from_canonical_id=author_node_id,
                    to_label=NodeLabel.DOCUMENT,
                    to_canonical_id=doc_id,
                    properties={"source_key": source_key},
                    valid_from=updated_at,
                )
            )

        return NormalizationResult(
            documents=[doc],
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            acl_snapshots=acl_rows,
        )


def _workspace_read_write_principals(customer_id: str) -> list[ACLPrincipal]:
    return [
        ACLPrincipal(
            principal_type=PrincipalType.WORKSPACE,
            principal_id=customer_id,
            permission=Permission.READ,
        ),
        ACLPrincipal(
            principal_type=PrincipalType.WORKSPACE,
            principal_id=customer_id,
            permission=Permission.WRITE,
        ),
    ]


def _dashboard_source_url(customer_id: str, source_key: str) -> str:
    return f"https://prbe.ai/dashboard/ingestion?customer={customer_id}&source_key={source_key}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
