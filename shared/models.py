"""Canonical Pydantic schemas. The contract every handler produces."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.constants import (
    AttachmentKind,
    CompileTrigger,
    DocClass,
    DocType,
    EdgeType,
    EntityType,
    IngestionEventType,
    NodeLabel,
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
    """Content-addressable chunk row.

    Identity is (doc_id, content_hash). When a doc edits and a chunk with the
    same content still exists, the existing row's `last_seen_version` is bumped
    instead of writing a new row — so embedding cost is proportional to what
    actually changed, not to total chunk count.
    """

    chunk_id: str
    doc_id: str
    customer_id: str
    chunk_index: int
    content: str
    content_hash: str
    token_count: int
    embedding: list[float]
    embedding_model: str
    embedding_dim: int
    chunker_version: str
    first_seen_version: int
    last_seen_version: int
    valid_from: datetime
    valid_to: datetime | None = None
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


class TemporalMode(StrEnum):
    """How retrieval should interpret document/chunk versions.

    latest          — only chunks from each doc's live version (valid_to IS NULL).
                      Default for agent queries about current state.
    as_of           — snapshot of what was live at a point in time.
                      Set TemporalSpec.as_of.
    changed_between — docs whose latest version landed in [since, until).
                      Useful for "what moved during the incident" queries.
    all             — no temporal filter. Returns every version ever ingested.
                      Escape hatch; not a common default.
    """

    LATEST = "latest"
    AS_OF = "as_of"
    CHANGED_BETWEEN = "changed_between"
    ALL = "all"


class TemporalSpec(BaseModel):
    """Temporal predicate bundle passed to retrievers.

    `time_basis` controls whether time-bounded modes use the source-side clock
    (`updated_at` — "what changed in Linear") or our ingest-side clock
    (`ingested_at` — "what our pipeline learned about"). Source is the default
    because agents almost always want source time when asking about an incident.
    """

    mode: TemporalMode = TemporalMode.LATEST
    as_of: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
    time_basis: Literal["source", "ingest"] = "source"

    @model_validator(mode="after")
    def _check_fields_match_mode(self) -> "TemporalSpec":
        if self.mode == TemporalMode.AS_OF and self.as_of is None:
            raise ValueError("as_of mode requires `as_of` timestamp")
        if self.mode == TemporalMode.CHANGED_BETWEEN and (
            self.since is None or self.until is None
        ):
            raise ValueError("changed_between mode requires `since` and `until`")
        if (
            self.mode == TemporalMode.CHANGED_BETWEEN
            and self.until is not None
            and self.since is not None
            and self.until <= self.since
        ):
            raise ValueError("changed_between requires `until` > `since`")
        return self


class QueryRequest(BaseModel):
    query: str
    customer_id: str
    top_k: int = 20
    sources: list[SourceSystem] | None = None
    requesting_user_id: str | None = None
    trace_id: str | None = None
    temporal: TemporalSpec = Field(default_factory=TemporalSpec)


class QueryChunk(BaseModel):
    chunk_id: str
    doc_id: str
    # Which document version this chunk's content came from when first seen.
    # Under content-addressable chunks the same chunk can span many versions;
    # the `first_seen_version` is the stable provenance value we return.
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


# ---------------------------------------------------------------------------
# Connector contract — shared output schema.
#
# Every connector (Slack, GitHub, Linear, ...) returns a NormalizationResult
# from its .normalize() method. The normalizer/worker persists these to the
# canonical tables (documents, chunks, graph_nodes, graph_edges, acl_snapshots)
# regardless of source system.
#
# To add a new connector, subclass Connector (in services/ingestion/handlers/base.py),
# return one of these result shapes, and register with @register_connector.
# ---------------------------------------------------------------------------


class GraphNodeSpec(BaseModel):
    """One graph node to upsert. Resolved to a node_id by graph_writer."""

    label: NodeLabel
    canonical_id: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdgeSpec(BaseModel):
    """One graph edge to upsert. Endpoints reference (label, canonical_id) pairs
    and are resolved to node_ids after their GraphNodeSpec peers are upserted."""

    edge_type: EdgeType
    from_label: NodeLabel
    from_canonical_id: str
    to_label: NodeLabel
    to_canonical_id: str
    properties: dict[str, Any] = Field(default_factory=dict)
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class ACLSnapshotRow(BaseModel):
    """One row for acl_snapshots. Separate from ACLPrincipal (embedded in Document.acl)
    because the temporal ACL table is a wider row shape."""

    source_system: SourceSystem
    principal_type: PrincipalType
    principal_id: str
    resource_type: str
    resource_id: str
    permission: Permission
    valid_from: datetime
    valid_to: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizationResult(BaseModel):
    """Uniform output shape produced by every connector's .normalize().

    The normalizer persists each field into its canonical table. Empty lists
    are fine — a connector that never touches the graph just returns `graph_nodes=[]`.
    """

    documents: list[Document] = Field(default_factory=list)
    graph_nodes: list[GraphNodeSpec] = Field(default_factory=list)
    graph_edges: list[GraphEdgeSpec] = Field(default_factory=list)
    acl_snapshots: list[ACLSnapshotRow] = Field(default_factory=list)
    # Non-fatal reason this event produced no documents (e.g. "slack edit of deleted msg").
    skipped_reason: str | None = None

    @property
    def is_empty(self) -> bool:
        return not (
            self.documents or self.graph_nodes or self.graph_edges or self.acl_snapshots
        )


class WebhookParseResult(BaseModel):
    """What parse_webhook_event() returns. None means: ignore this webhook."""

    source_event_id: str
    received_at: datetime
    event_kind: IngestionEventType = IngestionEventType.WEBHOOK
    # Connector-specific hint the normalizer can pass through to fetch_supplementary
    # without re-parsing the payload.
    parse_hint: dict[str, Any] = Field(default_factory=dict)


class IntegrationToken(BaseModel):
    """Decrypted OAuth credentials passed into fetch_supplementary / backfill."""

    customer_id: str
    source_system: SourceSystem
    access_token: str
    refresh_token: str | None = None
    expires_at: datetime | None = None
    scope: str | None = None
    webhook_secret: str | None = None


class ExternalWorkspaceRef(BaseModel):
    """A source-side workspace/team/org identifier linked to a PRBE customer.

    Returned by `Connector.identify_workspaces` after OAuth token exchange.
    Stored in `customer_source_mapping` so incoming webhooks can resolve
    their owning customer from the payload alone (no X-Prbe-Customer header).
    """

    external_id: str
    external_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
