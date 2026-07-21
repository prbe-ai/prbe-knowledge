"""Validation helpers for the customer-facing custom ingest document API."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from engine.shared.constants import EdgeType, NodeLabel

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


class CustomIngestNode(BaseModel):
    """An entity node a client asserts alongside a document.

    Lets a client project its own domain graph (experiment -> run ->
    artifact, say) instead of the flat Document + Person + AUTHORED shape
    custom-ingest emitted before. The node is an ANCHOR: the graph
    retriever's neighbour join is `AND n.label = 'Document'`, so an entity
    node is traversed FROM, never returned as a hit. Reaching a document
    means asserting an edge to it.

    `name` is required. Grounding resolves query tokens by fuzzy-matching
    `coalesce(properties->>'name', canonical_id)`, so a nameless node with an
    opaque canonical_id (a UUID) is silently unfindable -- it exists, ingest
    succeeds, and no query ever reaches it.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=64)
    canonical_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=512)
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        label = value.strip()
        try:
            NodeLabel(label)
        except ValueError:
            allowed = ", ".join(sorted(nl.value for nl in NodeLabel))
            raise ValueError(
                f"unknown node label {label!r}; expected one of: {allowed}"
            ) from None
        return label

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("node name is required and cannot be blank")
        return name


class CustomIngestEdge(BaseModel):
    """An edge a client asserts between two nodes in the same document push.

    Edges are BIDIRECTIONAL in retrieval -- the graph walk matches on
    `(from_node_id = anchor OR to_node_id = anchor)` and takes whichever end
    is opposite -- so assert each relationship ONCE. Emitting both directions
    just doubles the write cost and the degree of both endpoints.
    """

    model_config = ConfigDict(extra="forbid")

    edge_type: str = Field(min_length=1, max_length=64)
    from_label: str = Field(min_length=1, max_length=64)
    from_canonical_id: str = Field(min_length=1, max_length=256)
    to_label: str = Field(min_length=1, max_length=64)
    to_canonical_id: str = Field(min_length=1, max_length=256)
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("edge_type")
    @classmethod
    def _validate_edge_type(cls, value: str) -> str:
        edge_type = value.strip()
        try:
            EdgeType(edge_type)
        except ValueError:
            allowed = ", ".join(sorted(et.value for et in EdgeType))
            raise ValueError(
                f"unknown edge type {edge_type!r}; expected one of: {allowed}"
            ) from None
        return edge_type

    @field_validator("from_label", "to_label")
    @classmethod
    def _validate_endpoint_label(cls, value: str) -> str:
        label = value.strip()
        try:
            NodeLabel(label)
        except ValueError:
            allowed = ", ".join(sorted(nl.value for nl in NodeLabel))
            raise ValueError(
                f"unknown node label {label!r}; expected one of: {allowed}"
            ) from None
        return label


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

    # Client-asserted graph payload. Optional and empty by default, so every
    # existing caller is byte-identical. Caps bound a single document's write
    # amplification -- each node is an upsert and each edge a row, all landing
    # on shared hot rows (a project node is touched by every run beneath it).
    nodes: list[CustomIngestNode] = Field(default_factory=list, max_length=64)
    edges: list[CustomIngestEdge] = Field(default_factory=list, max_length=128)

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


def encode_source_key_for_doc_id(source_key: str) -> str:
    """Percent-encode ':' in a source_key for use as a doc-id segment.

    Internal doc ids are colon-delimited
    (``custom_ingest:{customer}:{source_key}:{document_id}``). Since the
    source_key charset admits ':' (workspace:<uuid> namespacing) and the
    caller's document id is unrestricted, naive concatenation is not
    injective: (source_key='workspace:abc', id='doc1') would collide with
    (source_key='workspace', id='abc:doc1'). Encoding ':' -> '%3A' in the
    source_key segment restores injectivity: the encoded segment contains
    no ':' so ``doc_id.split(':', 3)`` recovers (system, customer,
    encoded_key, document_id) unambiguously, and the map is reversible
    because '%' is outside the source_key charset (_SOURCE_KEY_RE), so
    '%3A' can never occur in a raw key. Keys without ':' are byte-
    identical under this encoding -- zero impact on existing doc ids.
    """
    return source_key.replace(":", "%3A")


def custom_ingest_doc_id(customer_id: str, source_key: str, document_id: str) -> str:
    """Compose the internal documents.doc_id for a custom-ingest document.

    Format (the ONLY blessed shape -- consumers parse source_key back out
    of doc_id client-side, so any change here is a cross-repo contract
    change):

        custom_ingest:{customer_id}:{encoded_source_key}:{document_id}

    where encoded_source_key = source_key with ':' -> '%3A' (see
    encode_source_key_for_doc_id). document_id is the caller's raw id and
    is the final segment, so it may itself contain ':'.
    """
    encoded = encode_source_key_for_doc_id(source_key)
    return f"custom_ingest:{customer_id}:{encoded}:{document_id}"


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

