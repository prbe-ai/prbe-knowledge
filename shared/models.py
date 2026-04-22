"""Canonical Pydantic schemas. The contract every handler produces."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from shared.constants import (
    AttachmentKind,
    CompileTrigger,
    DocClass,
    DocType,
    EntityType,
    Permission,
    PrincipalType,
    RefType,
    SourceSystem,
)


class EntityRef(BaseModel):
    model_config = ConfigDict(frozen=False)

    entity_type: EntityType
    canonical_id: str
    external_id: str | None = None
    display_name: str
    confidence: float = 1.0
    span: tuple[int, int] | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class AttachmentRef(BaseModel):
    kind: AttachmentKind
    url: str
    s3_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocRef(BaseModel):
    doc_id: str | None = None
    external_url: str
    ref_type: RefType


class ACLPrincipal(BaseModel):
    principal_type: PrincipalType
    principal_id: str
    name: str | None = None
    permission: Permission = Permission.READ


class ACLSnapshot(BaseModel):
    principals: list[ACLPrincipal]
    captured_at: datetime


class Document(BaseModel):
    """Canonical normalized document. One instance per version.

    doc_id is stable by (source_system, source_id, customer_id). Content edits
    bump `version` only. A source-side delete + recreate with a new source_id
    produces a new doc_id (separate history).
    """

    doc_id: str
    customer_id: str
    version: int = 1

    source_system: SourceSystem
    source_id: str
    source_url: str

    doc_class: DocClass = DocClass.RAW_SOURCE
    doc_type: DocType
    content_type: str = "text/plain"
    language: str | None = None

    content_hash: str
    title: str | None = None
    body_preview: str | None = None
    body_size_bytes: int = 0
    body_token_count: int = 0
    author_id: str | None = None

    created_at: datetime
    updated_at: datetime
    valid_from: datetime
    valid_to: datetime | None = None
    deleted_at: datetime | None = None
    ingested_at: datetime

    parent_doc_id: str | None = None
    supersedes_doc_id: str | None = None

    acl: ACLSnapshot
    metadata: dict[str, Any] = Field(default_factory=dict)
    entities: list[EntityRef] = Field(default_factory=list)
    attachments: list[AttachmentRef] = Field(default_factory=list)
    doc_references: list[DocRef] = Field(default_factory=list)

    ingestion_event_id: int | None = None
    normalizer_version: str = "v1"

    compiled_from_doc_ids: list[str] | None = None
    compilation_model: str | None = None
    compiled_at: datetime | None = None
    compile_trigger: CompileTrigger | None = None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_version: int
    customer_id: str
    chunk_index: int
    content: str
    token_count: int
    embedding: list[float]
    embedding_model: str
    embedding_dim: int
    chunker_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebhookEvent(BaseModel):
    """Uniform shape that every handler's parse_webhook_event produces."""

    customer_id: str
    source_system: SourceSystem
    source_event_id: str
    received_at: datetime
    payload_s3_key: str
    raw_payload: dict[str, Any]
    headers: dict[str, str] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    customer_id: str
    top_k: int = 20
    sources: list[SourceSystem] | None = None
    requesting_user_id: str | None = None
    trace_id: str | None = None


class QueryChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_version: int
    source_system: SourceSystem
    source_url: str
    title: str | None
    content: str
    score: float
    rank: int
    retriever_scores: dict[str, float] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    chunks: list[QueryChunk]
    total_candidates: int
    router_hit_cache: bool
    timing_ms: dict[str, float] = Field(default_factory=dict)
    trace_id: str


class BootstrapConfig(BaseModel):
    customer_id: str
    display_name: str
    r2_bucket_name: str
    api_key: str
    oauth_urls: dict[SourceSystem, str]
