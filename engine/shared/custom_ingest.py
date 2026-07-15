"""Validation helpers for the customer-facing custom ingest document API."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Colon allowed (after the first char) so consumers can namespace dynamic
# source keys, e.g. research-os pushes workspace files under
# `workspace:<uuid>`. The key is treated as opaque everywhere it flows
# (R2 payload path, doc_id, source_event_id), so widening the charset is
# additive-only.
_SOURCE_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9:_-]{0,127}$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def is_valid_source_key(value: str) -> bool:
    """Shared source_key validation for the envelope and query params."""
    return bool(_SOURCE_KEY_RE.match(value))


class CustomIngestAuthor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, min_length=1, max_length=256)
    name: str | None = Field(default=None, min_length=1, max_length=256)
    email: str | None = Field(default=None, min_length=1, max_length=320)


class CustomIngestDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=256)
    type: str | None = Field(default=None, min_length=1, max_length=128)
    title: str | None = Field(default=None, min_length=1, max_length=512)
    # Optional ONLY for delete entries (`deleted: true`); enforced non-empty
    # for upserts by _require_body_unless_deleted below.
    body: str = ""
    url: str | None = Field(default=None, min_length=1, max_length=2048)
    author: CustomIngestAuthor | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Tombstone flag: `{"id": ..., "deleted": true}` (body optional) closes
    # the document's live version + chunks with the same semantics as a
    # connector-originated delete (linear/github/slack tombstones).
    deleted: bool = False
    # Accepted for backward compatibility with the original v1 payload shape,
    # but intentionally ignored until document-level permissions are built.
    acl: list[dict[str, Any]] = Field(default_factory=list, max_length=100)

    @field_validator("id", "type", "title", "url", mode="before")
    @classmethod
    def _strip_optional_strings(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("body", mode="before")
    @classmethod
    def _validate_body(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("body must be text")
        body = value.strip()
        if _CONTROL_CHAR_RE.search(body):
            raise ValueError("body contains binary/control characters")
        return body

    @model_validator(mode="after")
    def _require_body_unless_deleted(self) -> CustomIngestDocument:
        if not self.deleted and not self.body:
            raise ValueError("body is required")
        return self

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> object:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("metadata must be an object")
        if "body" in value:
            raise ValueError("metadata.body is reserved; send document text in body")
        return value


class CustomIngestEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_key: str = Field(min_length=1, max_length=128)
    batch_id: str | None = Field(default=None, min_length=1, max_length=256)
    documents: list[CustomIngestDocument] = Field(min_length=1, max_length=100)

    @field_validator("source_key")
    @classmethod
    def _validate_source_key(cls, value: str) -> str:
        source_key = value.strip()
        if not _SOURCE_KEY_RE.match(source_key):
            raise ValueError("source_key must match ^[a-z0-9][a-z0-9:_-]{0,127}$")
        return source_key

    @field_validator("batch_id", mode="before")
    @classmethod
    def _strip_batch_id(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


def json_size(value: object) -> int:
    return len(orjson.dumps(value))


def document_content_hash(source_key: str, document: CustomIngestDocument) -> str:
    """Stable digest of the source-visible document content.

    This is the hash the enumeration endpoint reports back per document, so
    a consumer-side reconciler can recompute it from its own source of truth
    and diff. The `deleted` key participates only when True — keeping every
    pre-existing upsert hash byte-identical while a delete for the same id
    always hashes differently from the upsert that preceded it.
    """
    payload: dict[str, Any] = {
        "source_key": source_key,
        "id": document.id,
        "type": document.type,
        "title": document.title,
        "body": document.body,
        "url": document.url,
        "author": document.author.model_dump(mode="json") if document.author else None,
        "created_at": _iso_or_none(document.created_at),
        "updated_at": _iso_or_none(document.updated_at),
        "metadata": document.metadata,
    }
    if document.deleted:
        payload["deleted"] = True
    return hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def document_payload_key(
    customer_id: str,
    source_key: str,
    document_id: str,
    content_hash: str,
) -> str:
    doc_hash = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:16]
    return f"raw/custom_ingest/{customer_id}/{source_key}/{doc_hash}/{content_hash}.json"


def source_event_id(
    envelope: CustomIngestEnvelope,
    document: CustomIngestDocument,
    content_hash: str,
) -> str:
    """Queue idempotency key.

    The content hash is intentionally included so unchanged retries dedupe at
    the queue, while changed bodies for the same document id still reach the
    normalizer and open a new bitemporal version. `batch_id` is stored in the
    raw payload but deliberately excluded from this key: a customer changing
    batch ids should not force duplicate queue rows for identical content.
    """
    doc_hash = hashlib.sha256(document.id.encode("utf-8")).hexdigest()[:16]
    return f"{envelope.source_key}:{doc_hash}:{content_hash[:16]}"

