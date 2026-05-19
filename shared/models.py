"""Canonical Pydantic schemas. The contract every handler produces."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.constants import (
    AttachmentKind,
    CompileTrigger,
    DocClass,
    DocType,  # noqa: F401  (re-exported; other modules import via shared.models)
    EdgeType,
    EntityType,
    IngestionEventType,
    NodeLabel,
    Permission,
    PrincipalType,
    RefType,
    SourceSystem,
)

# Sentinel chunk_index for the synthetic per-Document metadata chunk
# (the kind='metadata' chunk that carries identifying key:value text for
# search anchoring). Negative so it can't collide with any real chunker
# output index. Lives here so cross-module callers (chunker, normalizer,
# code_graph pipeline) all reach for the same constant.
METADATA_CHUNK_INDEX = -1


@dataclass(slots=True)
class ChunkPiece:
    """One unit of embeddable text within a Document.

    Lives here in `shared.models` (not chunker.py) so cross-module
    contracts — like `NormalizationResult.documents_with_chunks` —
    can reference it without dragging chunker's tiktoken dependency
    or violating the shared/services layering. Re-exported from
    `services.ingestion.chunker` for backwards-compatible imports.
    """

    chunk_index: int
    content: str
    token_count: int


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
    # doc_type is a free-form string at the model level so connectors that
    # mint dynamic types (e.g. wiki pages, where the LLM picks `wiki.repo`,
    # `wiki.runbook`, etc.) can pass them through. Closed-set sources still
    # pass `DocType.<MEMBER>` enum values — StrEnum members are str
    # subclasses so the existing call sites keep working unchanged.
    doc_type: str
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

    # Transient body text. Excluded from model_dump so it never serializes
    # into the documents row's metadata jsonb. The normalizer reads `doc.body`
    # to feed the chunker; chunks.content is the persisted source of truth.
    # Historically connectors stuffed the full body into metadata["body"],
    # which doubled storage on every doc. See migration 0035.
    body: str | None = Field(default=None, exclude=True, repr=False)

    # When True, the normalizer coalesces re-ingests of this doc into the
    # current live SCD2 version (UPDATE in place) instead of opening a new
    # version per content edit. Cleared when a final state is reached.
    # Used by claude_code to fold per-batch session updates into one row
    # until session_complete=True. See migration 0036.
    coalesce_into_live: bool = Field(default=False, exclude=True, repr=False)

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
    """Uniform shape that every handler's parse_webhook_event produces.

    `payload_s3_keys` carries every R2 path coalesced into the queue row.
    For non-CC connectors this is always a single-element list (one
    payload per row). For claude_code it grows as batches arrive under
    the same session_id. The connector decides how to consume the list
    in its `fetch_supplementary` method.

    `payload_s3_key` is a back-compat alias for `payload_s3_keys[0]`.
    Older code paths that only know about a single payload (most connector
    backfill generators) still set this; new code paths read the array.

    `raw_payload` is the parsed contents of the *first* key in
    `payload_s3_keys` (the most relevant single payload — e.g. for non-CC
    it's literally THE payload). Connectors that need the full set
    iterate `payload_s3_keys` themselves.
    """

    customer_id: str
    source_system: SourceSystem
    source_event_id: str
    received_at: datetime
    payload_s3_key: str = ""
    payload_s3_keys: list[str] = Field(default_factory=list)
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
        if self.mode == TemporalMode.CHANGED_BETWEEN and (self.since is None or self.until is None):
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
    top_k: int = 20
    sources: list[SourceSystem] | None = None
    doc_types: list[str] | None = Field(
        default=None,
        description=(
            "Optional caller-provided doc_type filter. Values are dotted "
            "DocType strings (e.g. 'github.commit'). When set, overrides "
            "any doc_type the router would have inferred from the query. "
            "Hard filter on list mode; soft RRF boost on search mode."
        ),
    )
    requesting_user_id: str | None = None
    trace_id: str | None = None
    temporal: TemporalSpec = Field(default_factory=TemporalSpec)
    recency_half_life_days: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Override the half-life used for recency decay. Each fused chunk "
            "is multiplied by exp(-ln2 * age_days / half_life), where age uses "
            "documents.updated_at vs. now (UTC). When unset, fusion falls back "
            "to per-source overrides (e.g. claude_code/codex at 7d) or the "
            "universal baseline (DEFAULT_RECENCY_HALF_LIFE_DAYS); decay is "
            "always-on so backfilled tenants don't surface stale content."
        ),
    )
    entity_must_match: bool = Field(
        default=False,
        description=(
            "If true and the router extracts at least one high-confidence "
            "entity, drop fused candidates whose content/title doesn't "
            "textually contain the entity's canonical_id or display_name. "
            "Filters obvious vector-similarity false positives (queries like "
            "'whats going on with klavis' otherwise match generic Slack "
            "greetings on conversational shape). When true, ALSO applies "
            "hard entity filters in list mode: `author_id = ANY(...)` from "
            "`person` entities and graph_nodes membership from narrowing "
            "entities (`service`/`repo`/`ticket`/`pr`/`channel`). When false "
            "(default), list mode skips entity-based narrowing entirely and "
            "relies on sort + temporal + source + doc_type only — preferred "
            "for broad-recall callers (e.g. the MCP) where a router-extracted "
            "entity that has no matching graph_node would otherwise zero out "
            "the SQL result."
        ),
    )
    entity_match_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence threshold for entities to qualify as filter needles "
            "when `entity_must_match` is true. Entities below this confidence "
            "are ignored by the filter. Lower (e.g. 0.5) is more aggressive; "
            "higher (e.g. 0.9) only filters on dead-certain entities."
        ),
    )
    min_confidence: str | None = Field(
        default="INFERRED",
        description=(
            "Floor for graph-edge confidence tier when joining graph "
            "neighbors into the result set. Values: 'EXTRACTED' (only "
            "deterministic edges), 'INFERRED' (default — drops AMBIGUOUS), "
            "'AMBIGUOUS' or null (include everything; debug mode). "
            "Filters edges produced by code-graph (PR-A) and Graphify "
            "(PR-B); has no effect on retrievers that don't traverse "
            "the graph (vector / BM25)."
        ),
    )
    top_k_related: int = Field(
        default=10,
        ge=0,
        le=50,
        description=(
            "How many `related_entities` to return on the response — "
            "non-Document graph nodes attached to the result-set docs, "
            "surfaced as crawl candidates for an LLM doing knowledge-"
            "graph BFS. Set 0 to skip the post-fusion walk entirely "
            "(token-sensitive flows). Default 10. Cap 50."
        ),
    )
    discovery: bool = Field(
        default=False,
        description=(
            "When true, weight graph-retriever hits by their surprise score "
            "in fusion's RRF math (multiplier capped at 2.0). Surfaces "
            "cross-source / cross-community / inferred-edge neighbors that "
            "default focus mode buries beneath dominant vector/BM25 hits. "
            "Use for conceptual queries ('how should we approach X', "
            "'anything else I should know about Y') where the agent wants "
            "less-obvious connections. Skip for direct lookups (PR#, ticket, "
            "error message) where the canonical answer is wanted. "
            "When false (default), graph hits contribute flat 1/(k+rank) RRF."
        ),
    )


def normalize_author_id(value: str | None) -> str | None:
    """Map ingestion-side sentinel `"unknown"` back to None at the response boundary.

    Some handlers (github, notion) store the literal `"unknown"` when no
    author resolved at ingestion time. Surfacing that string to agents would
    look like a real identity. We strip it here, on the way out, so the field
    is genuinely nullable without rewriting `documents.author_id` rows
    (which would invalidate hashes and break callers passing
    `author_ids=["unknown"]`).
    """
    if value == "unknown":
        return None
    return value


class GraphEvidence(BaseModel):
    """Per-chunk hint that a graph traversal anchored this chunk's surfacing.

    A chunk reached via N distinct seed entities carries N GraphEvidence
    entries on QueryChunk.graph_evidence. Empty list when the chunk
    surfaced via vector / BM25 / id_lookup alone. Lets MCP / dashboard
    consumers filter on confidence tier without re-running the retrieval.

    `via_entity_title` is an optional human-readable title for the
    `via_entity` doc, populated only by the adapter's post-hoc
    enrichment step (LEFT JOIN documents). Empty / null when the title
    isn't known (entity nodes without docs, or pre-enrichment paths).
    Lets the dashboard's chain-of-reasoning graph viz render the
    OTHER endpoint of an edge with a readable label even when that doc
    isn't itself in the curated result set.
    """

    edge_type: str
    confidence: str  # EXTRACTED | INFERRED | AMBIGUOUS
    via_entity: str
    reason: str | None = None
    via_entity_title: str | None = None


class QueryChunk(BaseModel):
    """One body chunk inside a `QueryDocumentResult`.

    Doc-level fields (doc_id, source_system, title, source_url, author_id,
    created_at, updated_at, doc_version) live on the parent
    `QueryDocumentResult`, NOT on the chunk -- they are identical for every
    chunk of a given document. The chunk only carries identity + content +
    rank-within-doc + per-retriever scores + graph-walk provenance.
    """

    chunk_id: str
    content: str
    score: float
    # 1-indexed position within its parent document's `chunks` list
    # (chunks are sorted by score desc per doc).
    rank_in_doc: int
    retriever_scores: dict[str, float] = Field(default_factory=dict)
    # Multiple seeding entities can produce multiple GraphEvidence entries
    # for the same chunk -- e.g. the chunk reached via Repo:foo AND Person:bar
    # carries two entries. Empty list when the chunk surfaced via vector /
    # BM25 alone.
    graph_evidence: list[GraphEvidence] = Field(default_factory=list)


class RelatedEntity(BaseModel):
    """A non-Document graph node attached to >=1 doc in the result set.

    Surfaced to MCP consumers as crawl candidates: the LLM can drop the
    canonical_id into the next search_knowledge query bag to BFS the
    knowledge graph. Excludes any entity already in extracted_entities
    (the LLM has those handles already).
    """

    canonical_id: str
    label: str  # NodeLabel.value (Service, Repo, Person, Ticket, ...)
    display_name: str | None = None  # from properties->>'name' or entity_cluster_metadata.display_name override
    edge_types: list[str] = Field(default_factory=list)  # MENTIONS, AUTHORED, ...
    max_confidence: str  # EXTRACTED | INFERRED | AMBIGUOUS
    doc_count: int  # # of result-set docs adjacent to this entity (BFS priority)
    # IDF-adjusted score used for ranking. score = doc_count / log(1 +
    # global_doc_count). Generic high-degree entities (e.g.
    # Channel:#engineering attached to 10k docs) get crushed; specific
    # entities surface. Surfaced so LLMs can see the ranking signal.
    score: float
    # Up to 3 doc IDs the entity is attached to, ordered by result rank
    # (strongest first). Caps at 3 even when doc_count > 3 -- the LLM uses
    # these to ground/audit the doc_count claim against its visible chunks
    # list, not to enumerate every attached doc. DISTINCT -- multi-edge
    # docs do not duplicate.
    associated_doc_ids: list[str] = Field(default_factory=list)
    # Total size of the entity cluster (primary + all merged aliases).
    # 1 for unmerged nodes. Lets agents prefer cluster-rich nodes when
    # picking BFS crawl candidates. Populated by the related-entities
    # walker from `entity_aliases` keyed on the primary.
    member_count: int = 1
    # Distinct source_systems across the cluster (from the primary's
    # consolidated `graph_node_provenance` -- Phase 1 merges alias
    # provenance into the primary at merge time). [] for unmerged nodes
    # whose node hasn't been provenance-stamped yet (edge case; normal
    # ingest stamps it). Lets agents see "this person is GitHub +
    # Slack + Linear" without an extra round-trip.
    member_sources: list[str] = Field(default_factory=list)


class MatchProvenance(BaseModel):
    """Per-result trace of which retrieval channel surfaced this node.

    A single QueryResult can have multiple entries -- e.g. a Document
    reached via vector AND graph walks carries two MatchProvenance rows.
    Each entry also carries `intent_idx` identifying which router intent
    surfaced this match (0 for single-intent / pre-fan-out callers).
    """

    channel: Literal[
        "vector", "bm25", "graph", "inferred_edge", "id_lookup", "directed"
    ]
    rank: int
    score: float
    intent_idx: int = 0
    # Populated only when channel == "inferred_edge":
    anchor_doc_id: str | None = None
    edge_type: str | None = None
    confidence: str | None = None
    why: str | None = None  # LLM justification from properties.why


class QueryResultBase(BaseModel):
    """Common shape across all polymorphic QueryResult variants.

    Subclasses set `node_type` to a literal -- Pydantic uses that as the
    discriminator when parsing `list[QueryResult]`.
    """

    canonical_id: str
    score: float
    rank: int  # 1-indexed final rank in QueryResponse.results
    matched_via: list[MatchProvenance] = Field(default_factory=list)


class QueryDocumentResult(QueryResultBase):
    """A Document surfaced by retrieval, with its body chunks nested.

    `chunks` is forward-compatible with the doc-grouped retrieval branch
    (feat/doc-grouped-retrieval): when their PR lands first, their list
    of QueryChunks per Document slots directly into this field.
    """

    node_type: Literal["Document"] = "Document"
    doc_id: str  # equals canonical_id
    doc_version: int
    source_system: SourceSystem
    source_url: str
    title: str | None = None
    author_id: str | None = None
    created_at: datetime
    updated_at: datetime
    chunks: list[QueryChunk] = Field(default_factory=list)
    chunk_count: int = 0
    retriever_scores: dict[str, float] = Field(default_factory=dict)


class QueryEntityResult(QueryResultBase):
    """A non-Document graph node returned as a primary search result.

    Distinct from `RelatedEntity` (which is a post-fusion crawl-candidate
    enrichment): an EntityResult appears in `QueryResponse.results`
    alongside Documents because the user's query asked about it.
    """

    node_type: Literal["Entity"] = "Entity"
    label: str  # NodeLabel value
    display_name: str | None = None
    properties: dict[str, object] = Field(default_factory=dict)
    # Up to 5 doc_ids the entity is attached to, ordered by recency.
    attached_doc_ids: list[str] = Field(default_factory=list)
    # Distinct edge_types observed on the 1-hop neighborhood.
    edge_types: list[str] = Field(default_factory=list)
    # Total 1-hop Document count, NOT capped at len(attached_doc_ids).
    doc_count: int = 0


# Discriminated union: Pydantic v2 routes parsing to the right subclass
# by inspecting `node_type` literally.
QueryResult = Annotated[
    QueryDocumentResult | QueryEntityResult,
    Field(discriminator="node_type"),
]


class IntentAggregation(BaseModel):
    """Per-intent aggregation output (count / group_by).

    Doc-ranking intents produce chunks fused into `results` via RRF.
    Aggregation intents produce shape-incompatible payloads (counts,
    group rows) and are appended here keyed by intent position.
    """

    intent_idx: int
    operation: Literal["count", "group_by"]
    payload: dict[str, Any]


class QueryResponse(BaseModel):
    query: str
    # Polymorphic per-node results -- Document or Entity, discriminated on
    # `node_type`. Documents carry their body chunks nested under `chunks`.
    results: list[QueryResult] = Field(default_factory=list)
    total_candidates: int
    router_hit_cache: bool
    aggregations: list[IntentAggregation] = Field(default_factory=list)
    # Flat aggregate over GraphEvidence entries on every chunk in the
    # response. Keys always present; zeros when no graph hits surfaced.
    confidence_breakdown: dict[str, int] = Field(
        default_factory=lambda: {"EXTRACTED": 0, "INFERRED": 0, "AMBIGUOUS": 0}
    )
    applied_temporal: dict[str, object] | None = None
    applied_sort: dict[str, object] | None = None
    applied_entity_filter: dict[str, object] | None = None
    applied_mode: str | None = None
    applied_doc_types: list[str] | None = None
    applied_min_confidence: str | None = None
    extracted_entities: list[dict[str, object]] = Field(default_factory=list)
    aggregation: dict[str, object] | None = None
    timing_ms: dict[str, float] = Field(default_factory=dict)
    trace_id: str
    # Three-state contract per codex-B4:
    #   None        -> not requested (top_k_related == 0) OR walk failed
    #                 (also see related_entities_error below). Also None
    #                 when list_pipeline returns aggregation rows (no
    #                 result docs to walk from).
    #   []          -> requested, walked, no neighbors found (legitimate empty)
    #   [item, ...] -> requested, walked, found neighbors
    related_entities: list[RelatedEntity] | None = None
    # Populated only on walk failure (set to type(exc).__name__). Lets MCP
    # consumers and ops distinguish "feature broken" from "no neighbors".
    related_entities_error: str | None = None
    # Search-agent (gatherer) self-reported notes. Optional; absent on
    # router/list-only paths and on pre-gatherer responses. Schema is the
    # `GathererNotes` Pydantic shape from
    # `services.retrieval.agent.models`; dumped here as a dict so this
    # module doesn't need to import the gatherer model (which would close
    # a layering loop on startup-time imports). Consumers that don't
    # know about this field (older MCP clients) ignore it under
    # Pydantic's default extra='ignore' semantics.
    gatherer_notes: dict[str, object] | None = None
    # The doc the query is most about — the explicit "root" anchor of
    # the result set. Surfaced so downstream consumers (esp. the
    # dashboard's chain-of-reasoning graph viz) can deterministically
    # pin a root node instead of guessing from `results[0]`. Computed
    # by the adapter from the top-ranked Document in the result set
    # (falling back to the top extracted_entity's canonical_id when no
    # Document was emitted). None when there's nothing meaningful to
    # anchor on (empty result set, entity-only queries with no
    # extracted entities).
    query_root_doc_id: str | None = None


class AnswerRequest(QueryRequest):
    """Same retrieval knobs as QueryRequest, plus synthesis configuration.

    Inherits everything (top_k, temporal, sort, entity_must_match, etc.) so
    callers can use the same body shape and just toggle the endpoint.
    """

    # AnswerResponse has no related_entities field; the walk would run and
    # be discarded, costing one DB round-trip per /query for nothing.
    # Caller can opt back in with top_k_related > 0 if needed for debug.
    top_k_related: int = Field(
        default=0,
        ge=0,
        le=50,
        description=(
            "Override of QueryRequest.top_k_related: defaults to 0 on the "
            "synthesis path because AnswerResponse does not propagate "
            "related_entities. Set explicitly to enable the walk."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "Synthesis model in '<provider>/<model>' form. Defaults to "
            "anthropic/claude-sonnet-4-6. See shared.constants.SYNTHESIS_MODELS "
            "for the full allowed list."
        ),
    )
    max_tokens: int = Field(
        default=600,
        ge=64,
        le=4096,
        description="Cap on synthesis output length.",
    )


class AnswerResponse(BaseModel):
    query: str
    answer: str
    citations: list[dict[str, object]] = Field(default_factory=list)
    insufficient_context: bool = False
    model: str
    # Mirrors QueryResponse.results -- polymorphic Document/Entity.
    # Documents carry their cited chunks nested under `chunks`.
    results: list[QueryResult] = Field(default_factory=list)
    total_candidates: int
    confidence_breakdown: dict[str, int] = Field(
        default_factory=lambda: {"EXTRACTED": 0, "INFERRED": 0, "AMBIGUOUS": 0}
    )
    applied_temporal: dict[str, object] | None = None
    applied_sort: dict[str, object] | None = None
    applied_entity_filter: dict[str, object] | None = None
    applied_mode: str | None = None
    applied_doc_types: list[str] | None = None
    extracted_entities: list[dict[str, object]] = Field(default_factory=list)
    aggregation: dict[str, object] | None = None
    timing_ms: dict[str, float] = Field(default_factory=dict)
    trace_id: str


class SourceResponse(BaseModel):
    """Full source content for a document, reassembled from its chunks.

    Agents drilling down from a retrieved chunk into broader context
    fetch this. The same chunks that powered retrieval get concatenated
    in `chunk_index` order, so the result is the exact text we ingested
    — no live API calls, no rate limits, no stale-vs-live divergence.
    """

    doc_id: str
    doc_version: int
    source_system: SourceSystem
    source_id: str
    source_url: str
    title: str | None
    content: str
    author_id: str | None = None
    chunk_count: int
    body_size_bytes: int
    metadata: dict[str, object] = Field(default_factory=dict)
    entities: list[dict[str, object]] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    ingested_at: datetime
    deleted_at: datetime | None = None


class SourceViewSection(BaseModel):
    chunk_index: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    score: float | None = None


class SourceViewResponse(BaseModel):
    """Bounded source view for agent/MCP callers.

    Unlike SourceResponse, this never returns a full document by default.
    Callers pick a mode (preview/search/grep/range/chunk/tail/full) and the
    service enforces max line/byte ceilings. mode="full" is the explicit
    opt-in for whole-document retrieval, gated only by a high
    OOM-defense cap.
    """

    doc_id: str
    doc_version: int
    source_system: SourceSystem
    source_url: str
    title: str | None
    content: str
    author_id: str | None = None
    mode: Literal["preview", "search", "grep", "range", "chunk", "tail", "full"]
    sections: list[SourceViewSection] = Field(default_factory=list)
    line_start: int | None = None
    line_end: int | None = None
    total_lines: int
    next_cursor: str | None = None
    truncated: bool
    chunk_count: int
    body_size_bytes: int
    max_bytes: int
    limit_lines: int


class BootstrapConfig(BaseModel):
    customer_id: str
    display_name: str
    r2_bucket_name: str
    api_key: str
    oauth_urls: dict[SourceSystem, str]


# ---------------------------------------------------------------------------
# usage_events: dashboard-facing audit trail of retrieval calls.
# Written by services/retrieval/middleware.py, read by /usage/{feed,stats,search}.
# ---------------------------------------------------------------------------


class UsageEventOut(BaseModel):
    """One usage_events row, as serialized to dashboard / SDK callers."""

    event_id: str
    occurred_at: datetime
    caller_kind: str
    caller_subject: str | None = None
    event_type: str
    request_id: str | None = None
    endpoint: str
    summary: str | None = None
    status: str
    error_class: str | None = None
    latency_ms: int | None = None
    result_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UsageFeedResponse(BaseModel):
    """Top-N most recent events within a window. Used by /usage/feed and
    /usage/search (which is feed-shaped + FTS predicate)."""

    events: list[UsageEventOut]
    window: str
    count: int


class UsageStatsResponse(BaseModel):
    """Aggregate counts + latency percentiles over a window.

    Latency percentiles are computed only across status='ok' rows — error
    rows have no meaningful latency (they may have errored before any
    retrieval ran). `error_count` is reported separately so the dashboard
    can show error rate without polluting the latency bars.
    """

    total: int
    by_caller_kind: dict[str, int] = Field(default_factory=dict)
    by_event_type: dict[str, int] = Field(default_factory=dict)
    latency_p50_ms: int | None = None
    latency_p95_ms: int | None = None
    error_count: int
    window: str


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
    and are resolved to node_ids after their GraphNodeSpec peers are upserted.

    `confidence` is the spec's three-tier tag (EXTRACTED | INFERRED | AMBIGUOUS).
    Default 'EXTRACTED' so existing connectors don't have to know about it.
    Code-graph emits 'AMBIGUOUS' for unresolved CALLS targets; PR-B's Graphify
    proposer/promoter emits 'INFERRED'.
    """

    edge_type: EdgeType
    from_label: NodeLabel
    from_canonical_id: str
    to_label: NodeLabel
    to_canonical_id: str
    properties: dict[str, Any] = Field(default_factory=dict)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: str = "EXTRACTED"
    aliased_from_canonical_id: str | None = None
    aliased_to_canonical_id: str | None = None


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


class CodeRepoStateUpdate(BaseModel):
    """One row to upsert into `code_repo_state` (code-graph cache).

    Allows the code-graph connector to record per-file extraction status
    inside the same write txn the normalizer opens for documents/chunks/
    graph_nodes/graph_edges. Other connectors leave the list empty; the
    normalizer's _persist no-ops on them.

    `language` may be the sentinel `_skipped_secrets` when the file matched
    the secrets skip-list — see `services/ingestion/code_graph/secrets.py`.
    """

    repo: str
    file_path: str
    content_hash: str
    language: str
    symbol_count: int = 0
    extractor_version: str


class PreChunkedDocument(BaseModel):
    """A Document the connector has already chunked itself.

    Used by code-graph (Path 2 file-as-Document model) to emit one
    Document per file with N pre-built ChunkPiece entries — one per
    symbol body — bypassing the normalizer's default
    `chunk_text(doc.body)` call. The token-window chunker would
    otherwise split `def foo` from its docstring at arbitrary
    boundaries; that's wrong for code where the symbol is the
    natural chunk unit.

    `document.body` MUST be None for pre-chunked Documents (the
    body-guard in normalizer enforces this). The chunks list IS
    the source of truth.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    document: Document
    chunks: list[ChunkPiece]
    # Optional metadata chunk for the Document — same role as
    # `_metadata_piece` in normalizer (synthetic key:value text the
    # search layer hits for identity queries). Omit when the Document's
    # content chunks already contain identifying text.
    metadata_chunk: ChunkPiece | None = None


class NormalizationResult(BaseModel):
    """Uniform output shape produced by every connector's .normalize().

    The normalizer persists each field into its canonical table. Empty lists
    are fine — a connector that never touches the graph just returns `graph_nodes=[]`.
    """

    documents: list[Document] = Field(default_factory=list)
    # Pre-chunked Documents the normalizer must NOT re-chunk. The
    # connector owns chunking when symbol/structural granularity matters
    # (today: code-graph). Default empty for everyone else.
    documents_with_chunks: list[PreChunkedDocument] = Field(default_factory=list)
    graph_nodes: list[GraphNodeSpec] = Field(default_factory=list)
    graph_edges: list[GraphEdgeSpec] = Field(default_factory=list)
    acl_snapshots: list[ACLSnapshotRow] = Field(default_factory=list)
    # Code-graph only: per-file cache state to upsert into `code_repo_state`.
    # Other connectors leave this empty; normalizer._persist no-ops on it.
    code_repo_state_updates: list[CodeRepoStateUpdate] = Field(default_factory=list)
    # Non-fatal reason this event produced no documents (e.g. "slack edit of deleted msg").
    skipped_reason: str | None = None
    # Set True by handlers when this event should trigger the
    # investigation pipeline (Plan 4 wires this up). Default False
    # keeps every existing connector unchanged.
    requires_investigation: bool = False
    # Set True by handlers (PD/incident.io) when this event is the
    # incident-resolution signal that the post-approval dispatch seam
    # listens for (services/post_approval/dispatch.py:on_resolution_event).
    # Default False keeps every existing connector unchanged.
    requires_resolution_check: bool = False

    @property
    def is_empty(self) -> bool:
        return not (
            self.documents
            or self.documents_with_chunks
            or self.graph_nodes
            or self.graph_edges
            or self.acl_snapshots
            or self.code_repo_state_updates
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
    # Transient: populated during exchange_oauth_code by connectors that
    # capture workspace info from the token-exchange response (e.g. Notion).
    # Pydantic exclude=True keeps it out of model_dump(), so save_token()
    # never persists it. Cleared on every load_token() trip through the DB.
    install_metadata: dict[str, Any] | None = Field(default=None, exclude=True)

    # Device-scoped credentials (Phase 1 use: claude_code per-laptop tokens).
    # Non-device sources leave these as None.
    device_id: str | None = Field(default=None)
    device_metadata: dict[str, Any] | None = Field(default=None)


class ExternalWorkspaceRef(BaseModel):
    """A source-side workspace/team/org identifier linked to a PRBE customer.

    Returned by `Connector.identify_workspaces` after OAuth token exchange.
    Stored in `customer_source_mapping` so incoming webhooks can resolve
    their owning customer from the payload alone (no X-Prbe-Customer header).
    """

    external_id: str
    external_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
