"""Manual upload connector.

Dashboard users upload arbitrary files; the ingestion API stages the original
bytes, extracts text into the queued payload, and this connector turns that
payload into the same canonical Document/chunks shape as every source.
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
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.exceptions import InvalidWebhookPayload
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    ACLSnapshotRow,
    Document,
    GraphNodeSpec,
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)


@register_connector(SourceSystem.MANUAL_UPLOAD)
class ManualUploadConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.MANUAL_UPLOAD
    display_name: ClassVar[str] = "Manual uploads"

    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> bool:
        return True

    def parse_webhook_event(
        self,
        customer_id: str,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        upload_id = str(raw_payload.get("upload_id") or "").strip()
        if not upload_id:
            raise InvalidWebhookPayload("manual upload payload missing upload_id")
        return WebhookParseResult(
            source_event_id=upload_id,
            received_at=_parse_iso(raw_payload.get("uploaded_at")) or datetime.now(UTC),
            event_kind=IngestionEventType.MANUAL,
        )

    async def normalize(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        payload = event.raw_payload
        upload_id = str(payload.get("upload_id") or event.source_event_id).strip()
        filename = str(payload.get("filename") or "Manual upload").strip()
        body = str(payload.get("extracted_text") or "").strip()
        if not upload_id:
            raise InvalidWebhookPayload("manual upload payload missing upload_id")
        if not body:
            raise InvalidWebhookPayload("manual upload payload missing extracted_text")

        uploaded_at = _parse_iso(payload.get("uploaded_at")) or event.received_at
        doc_type = _parse_doc_type(payload.get("doc_type"))
        content_type = str(payload.get("content_type") or "text/plain")
        uploaded_by = _string_or_none(payload.get("uploaded_by"))
        doc_id = str(payload.get("doc_id") or f"manual_upload:{upload_id}")
        content_hash = _sha256(f"{doc_id}|{body}|{payload.get('file_sha256', '')}")

        acl_principals = [
            ACLPrincipal(
                principal_type=PrincipalType.WORKSPACE,
                principal_id=event.customer_id,
                permission=Permission.READ,
            )
        ]
        acl_rows = [
            ACLSnapshotRow(
                source_system=SourceSystem.MANUAL_UPLOAD,
                principal_type=PrincipalType.WORKSPACE,
                principal_id=event.customer_id,
                resource_type="manual_upload.file",
                resource_id=upload_id,
                permission=Permission.READ,
                valid_from=uploaded_at,
                metadata={"filename": filename},
            )
        ]

        doc = Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.MANUAL_UPLOAD,
            source_id=upload_id,
            source_url=f"manual-upload://{upload_id}",
            doc_class=DocClass.MANUAL_ENTRY,
            doc_type=doc_type,
            content_type="text/markdown" if doc_type == DocType.MANUAL_UPLOAD_MARKDOWN else "text/plain",
            content_hash=content_hash,
            title=filename[:240],
            body_preview=body[:280],
            body_size_bytes=len(body.encode("utf-8")),
            body_token_count=count_tokens(body),
            author_id=uploaded_by,
            created_at=uploaded_at,
            updated_at=uploaded_at,
            valid_from=uploaded_at,
            ingested_at=datetime.now(UTC),
            acl=ACLSnapshot(principals=acl_principals, captured_at=event.received_at),
            metadata={
                "body": body,
                "upload_id": upload_id,
                "filename": filename,
                "content_type": content_type,
                "file_size_bytes": payload.get("file_size_bytes"),
                "file_sha256": payload.get("file_sha256"),
                "uploaded_by": uploaded_by,
                "parse_engine": payload.get("parse_engine"),
                "original_object_key": payload.get("original_object_key"),
                "original_deleted_after_ingest": True,
            },
        )

        return NormalizationResult(
            documents=[doc],
            graph_nodes=[
                GraphNodeSpec(
                    label=NodeLabel.DOCUMENT,
                    canonical_id=doc_id,
                    properties={
                        "title": filename,
                        "source_system": SourceSystem.MANUAL_UPLOAD.value,
                        "upload_id": upload_id,
                    },
                )
            ],
            acl_snapshots=acl_rows,
        )


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


def _parse_doc_type(value: object) -> DocType:
    try:
        return DocType(str(value))
    except ValueError:
        return DocType.MANUAL_UPLOAD_FILE


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
