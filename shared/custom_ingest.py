"""Validation helpers for the customer-facing custom ingest document API."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator

_SOURCE_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


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
    body: str = Field(min_length=1)
    url: str | None = Field(default=None, min_length=1, max_length=2048)
    author: CustomIngestAuthor | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
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
        if not body:
            raise ValueError("body is required")
        if _CONTROL_CHAR_RE.search(body):
            raise ValueError("body contains binary/control characters")
        return body

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
            raise ValueError("source_key must match ^[a-z0-9][a-z0-9_-]{0,127}$")
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
    """Stable digest of the source-visible document content."""
    payload = {
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

