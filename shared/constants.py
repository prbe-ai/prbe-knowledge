"""Phase 0 canonical enums. Every string used as a type/label/edge/status lives here."""

import os
from enum import StrEnum


class SourceSystem(StrEnum):
    SLACK = "slack"
    LINEAR = "linear"
    GITHUB = "github"
    NOTION = "notion"
    SENTRY = "sentry"
    GRANOLA = "granola"
    CLAUDE_CODE = "claude_code"
    # Codex CLI sessions arrive shimmed into Claude-Code shape by the plugin's
    # sanitizer. Doc shape and unit extraction are identical to claude_code;
    # this label exists so dashboard queries can distinguish provenance.
    CODEX = "codex"
    MANUAL_UPLOAD = "manual_upload"
    CUSTOM_INGEST = "custom_ingest"
    # Curated team-knowledge layer (runbooks, decisions, service cards, feature
    # notes). Pages are authored programmatically via /api/wiki/pages/* — no
    # external webhook. doc_class distinguishes human authorship (MANUAL_ENTRY)
    # from agent-compiled summaries (COMPILED_WIKI).
    WIKI = "wiki"
    CODE_GRAPH = "code_graph"
    PAGERDUTY = "pagerduty"
    INCIDENT_IO = "incident_io"


# Canonical display labels for each SourceSystem. Exposed to the
# dashboard via /api/sources (TODO) and mirrored verbatim in
# prbe-dashboard/src/lib/sources.ts. The connector classes also carry
# `display_name: ClassVar[str]` for the same string at handler-instance
# scope; these are kept aligned by code review (a future cleanup can
# derive one from the other through the connector registry).
SOURCE_DISPLAY_NAMES: dict[SourceSystem, str] = {
    SourceSystem.SLACK: "Slack",
    SourceSystem.LINEAR: "Linear",
    SourceSystem.GITHUB: "GitHub",
    SourceSystem.NOTION: "Notion",
    SourceSystem.SENTRY: "Sentry",
    SourceSystem.GRANOLA: "Granola",
    SourceSystem.CLAUDE_CODE: "Claude Code",
    SourceSystem.CODEX: "Codex",
    SourceSystem.MANUAL_UPLOAD: "Manual upload",
    SourceSystem.CUSTOM_INGEST: "Custom Ingest",
    SourceSystem.WIKI: "Wiki",
    SourceSystem.CODE_GRAPH: "Code",
    SourceSystem.PAGERDUTY: "PagerDuty",
    SourceSystem.INCIDENT_IO: "incident.io",
}


class DocClass(StrEnum):
    RAW_SOURCE = "raw_source"
    COMPILED_WIKI = "compiled_wiki"
    MANUAL_ENTRY = "manual_entry"
    AGENT_ARTIFACT = "agent_artifact"


class DocType(StrEnum):
    SLACK_MESSAGE = "slack.message"
    SLACK_THREAD = "slack.thread"
    LINEAR_ISSUE = "linear.issue"
    LINEAR_COMMENT = "linear.comment"
    GITHUB_PULL_REQUEST = "github.pull_request"
    GITHUB_ISSUE = "github.issue"
    GITHUB_COMMIT = "github.commit"
    GITHUB_COMMIT_COMMENT = "github.commit_comment"
    GITHUB_REVIEW = "github.review"
    GITHUB_RELEASE = "github.release"
    GITHUB_CODEOWNERS = "github.codeowners"
    NOTION_PAGE = "notion.page"
    NOTION_DATABASE = "notion.database"
    SENTRY_ISSUE = "sentry.issue"
    SENTRY_EVENT = "sentry.event"
    GRANOLA_MEETING = "granola.meeting"
    CLAUDE_CODE_SESSION = "claude_code.session"
    CLAUDE_CODE_QA = "claude_code.qa"
    CLAUDE_CODE_CODE_CHANGE = "claude_code.code_change"
    CLAUDE_CODE_DECISION = "claude_code.decision"
    CLAUDE_CODE_FILE_REF = "claude_code.file_ref"
    MANUAL_UPLOAD_TEXT = "manual_upload.text"
    MANUAL_UPLOAD_MARKDOWN = "manual_upload.markdown"
    MANUAL_UPLOAD_DOCX = "manual_upload.docx"
    MANUAL_UPLOAD_FILE = "manual_upload.file"
    CUSTOM_DOCUMENT = "custom.document"
    # Wiki pages use a free-form `wiki.<type>` doc_type stamped at write
    # time from the LLM-emitted `wiki_type` slug — no enum, no validation
    # gate. The synthesis agent decides what page kinds are useful for a
    # given customer's corpus (typically `repo`, `runbook`, `person`, but
    # nothing prevents new ones). The auto-generated overview page is
    # written under `wiki.index`. Anywhere we need to filter for wiki
    # pages in SQL: `WHERE doc_type LIKE 'wiki.%'`.
    # LEGACY (PR-A pre-Path-2): one Document per symbol. Migration 0050
    # hard-deletes existing rows of this type when the file-as-Document
    # rewrite (CODE_FILE below) ships. Keep the constant defined so the
    # search pipeline + dashboard renderer can recognize stragglers
    # (e.g. an old chunk that escaped DELETE) and still display them.
    CODE_SYMBOL = "code.symbol"
    # Path 2: one Document per file. Body is None; chunks are pre-emitted
    # by the pipeline (one ChunkPiece per symbol body + one metadata chunk
    # carrying repo+file+symbol-list identifying text). The repo name
    # lives in the embedded metadata chunk so semantic search ranks
    # repo-qualified queries correctly.
    CODE_FILE = "code.file"
    INCIDENT = "incident"
    INCIDENT_INVESTIGATION = "incident.investigation"
    # Standalone Document carrying the LLM-drafted + human-approved "why
    # this PR exists" rationale produced by prbe-apps on PR merge. Persisted
    # alongside the FEATURE GraphNode (see feature_nodes_routes.py) so the
    # rationale text lands in BM25 + vector indexes. Prefix is `github.`
    # so doc_type_resolver's SourceSystem.GITHUB narrowing includes it.
    FEATURE_RATIONALE = "github.feature_rationale"
    # Post-approval wiki artifacts authored by the postmortem / wiki-edit
    # agents after an incident investigation is approved AND resolved.
    # These doc_types share the `wiki.` prefix so existing wiki listings
    # (`doc_type LIKE 'wiki.%'`) include them, while remaining
    # distinguishable from human-authored wiki pages.
    #
    # Visibility (DRAFT vs APPROVED) gates retrieval -- DRAFT artifacts are
    # excluded from search until a reviewer approves them via the
    # wiki_review_queue lifecycle.
    WIKI_POSTMORTEM = "wiki.postmortem"
    WIKI_KNOWLEDGE_PAGE = "wiki.knowledge_page"
    WIKI_CORRECTION = "wiki.correction"
    # TODO(post-approval): wiki-listing queries in services/ingestion/wiki_routes.py,
    #   services/synthesis/wiki_agent.py, and services/synthesis/persistence.py fan
    #   over `doc_type LIKE 'wiki.%'` without filtering by visibility. Once the
    #   writeback route (Component 5) starts persisting these doc types as drafts,
    #   those queries need a `visibility = 'approved'` predicate.


class Visibility(StrEnum):
    """Retrieval-visibility gate on a Document / Chunk.

    DRAFT rows are excluded from search and synthesis until promoted to
    APPROVED via the post-approval review pipeline. Used by the
    post-approval wiki artifacts (postmortems, knowledge pages,
    corrections); existing wiki/source documents default to APPROVED at
    write time, matching pre-existing behavior.
    """

    DRAFT = "draft"
    APPROVED = "approved"


# SQL pattern matching every wiki page doc_type (excludes the singleton
# index page so listings don't show themselves). The schema stamps
# wiki pages as `wiki.<wiki_type>` with no validation; this prefix +
# the explicit `<> 'wiki.index'` exclusion is the canonical filter.
WIKI_DOC_TYPE_PREFIX = "wiki."
WIKI_INDEX_DOC_TYPE = "wiki.index"


class NodeLabel(StrEnum):
    """Graph node labels — four canonical kinds post-migration 0091.

    Sub-type discrimination (Module vs Function for code; PR vs Issue for
    documents) lives in ``properties['kind']`` using the typed enums below
    (CodeSymbolKind, DocumentKind). Emit via the factories in shared.models
    (`make_code_symbol`, `make_document`, `make_person`, `make_feature`)
    rather than constructing GraphNodeSpec directly — the factories enforce
    the label-to-kind relationship.

    Other domain labels (SERVICE, SERVICE_CARD, DECISION, RUNBOOK,
    ERROR_GROUP, AGENT, WORKFLOW, FIX_ARTIFACT, VERIFICATION_RESULT) are
    out of scope for the collapse — they're either unused for acme
    today or carry distinct semantics worth preserving.
    """

    # ---- Canonical labels ----
    PERSON = "Person"
    DOCUMENT = "Document"
    FEATURE = "Feature"
    CODE_SYMBOL = "CodeSymbol"

    # ---- Domain labels (untouched by 0091) ----
    SERVICE = "Service"
    ERROR_GROUP = "ErrorGroup"
    SERVICE_CARD = "ServiceCard"
    DECISION = "Decision"
    RUNBOOK = "Runbook"
    AGENT = "Agent"
    WORKFLOW = "Workflow"
    FIX_ARTIFACT = "FixArtifact"
    VERIFICATION_RESULT = "VerificationResult"


class CodeSymbolKind(StrEnum):
    """Sub-type of a CODE_SYMBOL node, stored in ``properties['kind']``.

    Tree-sitter extractors classify each emitted symbol with one of these.
    Comparison sites (e.g. ``if symbol.kind == CodeSymbolKind.MODULE``)
    use this enum to keep symbol-kind reasoning type-safe.
    """

    MODULE = "Module"
    FUNCTION = "Function"
    CLASS = "Class"
    METHOD = "Method"
    # Generic "Symbol" — tree-sitter emits this for symbols that don't fit
    # the four categories above (interfaces, enums, constants, etc.).
    SYMBOL = "Symbol"


class DocumentKind(StrEnum):
    """Sub-type of a DOCUMENT node, stored in ``properties['kind']``.

    Optional — plain source documents (a slack message, a notion page) leave
    properties['kind'] unset. The five values below are the categories that
    collapsed INTO the Document label during migration 0091, where the
    sub-type is still meaningful enough to query against.
    """

    PR = "PR"
    ISSUE = "Issue"
    TICKET = "Ticket"
    CHANNEL = "Channel"
    REPO = "Repo"


# ---------------------------------------------------------------------------
# Router entity_type -> graph_nodes.label mapping (single source of truth).
#
# The router (services/retrieval/router.py) emits typed entities like
# {"entity_type": "pr", "canonical_id": "175"}. The graph retriever
# (services/retrieval/retrievers/graph.py) needs to know which NodeLabel
# corresponds to each router type to look those entities up. Historically
# this dict lived inside the retriever, drifted from the router's enum
# (services/retrieval/router.py:118-133), and silently dropped any
# router-emitted type the retriever didn't recognise -- producing zero
# graph hits with no error message.
#
# Keeping this in shared/ alongside NodeLabel guarantees the retriever
# can't fall behind when the router adds a new entity type. mypy catches
# any value that isn't a real NodeLabel; missing keys still need a human
# to add them, but the graph retriever now logs + falls back gracefully
# (see services/retrieval/retrievers/graph.py).
# ---------------------------------------------------------------------------
ROUTER_ENTITY_TO_LABEL: dict[str, NodeLabel] = {
    # Router-emitted typed entities (kept in sync with the router prompt's
    # entity_type enum at services/retrieval/router.py).
    #
    # After migration 0091's label collapse, multiple router entity_types
    # share a target NodeLabel — e.g. "pr", "issue", "ticket", "channel",
    # "repo", "file_path", "session" all map to DOCUMENT. The router prompt
    # still emits the fine-grained entity_type; this dict is the single
    # mapping into the (now coarser) graph_nodes.label vocabulary.
    "service":     NodeLabel.SERVICE,
    "person":      NodeLabel.PERSON,
    "feature":     NodeLabel.FEATURE,
    "decision":    NodeLabel.DECISION,
    "error_group": NodeLabel.ERROR_GROUP,

    # All addressable-source-document entities collapse to DOCUMENT post-0091.
    "repo":        NodeLabel.DOCUMENT,
    "ticket":      NodeLabel.DOCUMENT,
    "pr":          NodeLabel.DOCUMENT,
    "issue":       NodeLabel.DOCUMENT,
    "channel":     NodeLabel.DOCUMENT,
    "file_path":   NodeLabel.DOCUMENT,
    "session":     NodeLabel.DOCUMENT,

    # Code-graph entities (extracted by tree-sitter at ingest, not by the
    # router LLM, but the router can still emit these from queries that
    # mention qualified symbol names like `Normalizer.process_queue_row`).
    # All collapse to CODE_SYMBOL post-0091.
    "function":    NodeLabel.CODE_SYMBOL,
    "method":      NodeLabel.CODE_SYMBOL,
    "class":       NodeLabel.CODE_SYMBOL,
    "module":      NodeLabel.CODE_SYMBOL,
    "symbol":      NodeLabel.CODE_SYMBOL,
}


class EdgeType(StrEnum):
    OWNS = "OWNS"
    MENTIONS = "MENTIONS"
    AUTHORED = "AUTHORED"
    BLOCKS = "BLOCKS"
    SUPERSEDES = "SUPERSEDES"
    DUPLICATES = "DUPLICATES"
    TOUCHES = "TOUCHES"
    FIRES_IN = "FIRES_IN"
    MEMBER_OF = "MEMBER_OF"
    LINKED_FROM = "LINKED_FROM"

    CONFLICTS_WITH = "CONFLICTS_WITH"
    VERIFIED_BY = "VERIFIED_BY"
    DERIVED_FROM = "DERIVED_FROM"
    FIXES = "FIXES"
    REGRESSES = "REGRESSES"
    ASSIGNED_TO = "ASSIGNED_TO"
    COMPILED_FROM = "COMPILED_FROM"
    DESCRIBES = "DESCRIBES"

    CALLS = "CALLS"
    IMPORTS = "IMPORTS"
    INHERITS = "INHERITS"
    IMPLEMENTS = "IMPLEMENTS"
    REFERENCES = "REFERENCES"
    DEFINED_IN = "DEFINED_IN"

    # LLM-inferred edge types (Lane B). These are emitted only by the
    # inferred_edges extractor and carry INFERRED or AMBIGUOUS confidence.
    DISCUSSES = "DISCUSSES"
    DOCUMENTS = "DOCUMENTS"
    RESOLVES = "RESOLVES"
    MENTIONS_ENTITY = "MENTIONS_ENTITY"
    RELATES_TO = "RELATES_TO"


class EdgeConfidence(StrEnum):
    """Confidence tier on a graph edge — mirrors the string literals used
    in `graph_edges.confidence` SQL CASEs and `_stronger_confidence`
    (services/ingestion/graph_writer.py). StrEnum so members compare
    equal to the bare string ("EXTRACTED" == EdgeConfidence.EXTRACTED),
    keeping the existing string-based SQL + asyncpg parameter binding
    paths working unchanged.
    """

    EXTRACTED = "EXTRACTED"  # explicit upstream signal (webhook, API field)
    INFERRED = "INFERRED"    # LLM-derived from text
    AMBIGUOUS = "AMBIGUOUS"  # LLM-derived with low certainty


class PrincipalType(StrEnum):
    USER = "user"
    GROUP = "group"
    CHANNEL = "channel"
    WORKSPACE = "workspace"
    SYSTEM = "system"
    AGENT = "agent"


class Permission(StrEnum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class QueueStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    DLQ = "dlq"


class IngestionEventStatus(StrEnum):
    RECEIVED = "received"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    SKIPPED = "skipped"


class IngestionEventType(StrEnum):
    WEBHOOK = "webhook"
    SYNC = "sync"
    BACKFILL = "backfill"
    MANUAL = "manual"
    REPROCESS = "reprocess"


class BackfillStatus(StrEnum):
    IDLE = "idle"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class IntegrationStatus(StrEnum):
    ACTIVE = "active"
    AUTH_FAILED = "auth_failed"
    REVOKED = "revoked"


class CustomerStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class CompileTrigger(StrEnum):
    SCHEDULED = "scheduled"
    SOURCE_UPDATE = "source_update"
    MANUAL = "manual"
    QUERY_FILING = "query_filing"
    NORMALIZER_REPROCESS = "normalizer_reprocess"


class RefType(StrEnum):
    MENTIONS = "mentions"
    LINKS_TO = "links_to"
    EMBEDS = "embeds"
    REPLIES_TO = "replies_to"


class AttachmentKind(StrEnum):
    IMAGE = "image"
    FILE = "file"
    URL = "url"
    CODE_LINK = "code_link"
    BLOCK_REFERENCE = "block_reference"


class EntityType(StrEnum):
    SERVICE = "service"
    REPO = "repo"
    PERSON = "person"
    TICKET = "ticket"
    PR = "pr"
    ERROR_GROUP = "error_group"
    FEATURE = "feature"
    DECISION = "decision"
    FILE_PATH = "file_path"
    CHANNEL = "channel"


# Gemini-2 is the sole embedder (OpenAI text-embedding-3-large was retired
# 2026-05-14, PR #263; the OpenAI embedder + SDK were stripped in a follow-up).
# The v1 column on `chunks` is nullable and unindexed post-0067; reads target
# `embedding_v2`.
#
# Treat the two V2 constants below as the single source of truth for
# "what's the embedder?" — swapping models later means flipping these
# values, not chasing string literals across the codebase.
EMBEDDING_V2_MODEL = "google/gemini-embedding-2"
EMBEDDING_V2_DIM = 3072
# Bare model id as exposed by the LiteLLM proxy's `model_list` (matches the
# `gemini-embedding-*` alias). Use this when routing through the gateway;
# prefixing with `gemini/` falls through to the proxy's `*` catch-all and
# returns "invalid model ID" because the catch-all routes to OpenAI.
# Direct-SDK callers (google-genai) also accept this bare form.
EMBEDDING_V2_PROXY_ALIAS = "gemini-embedding-2"
# Per https://ai.google.dev/gemini-api/docs/embeddings, gemini-embedding-2
# accepts up to 8192 input tokens. The chunker's DEFAULT_CHUNK_TOKENS (512)
# is well under this; this constant is the absolute upper bound the chunker
# is allowed to use so we can't silently truncate Gemini-side input if the
# chunker is ever retuned.
EMBEDDING_V2_MAX_INPUT_TOKENS = 8192
CHUNKER_VERSION = "naive-v1"

# Per-symbol cap for code_graph chunks. Matches DEFAULT_CHUNK_TOKENS so code
# and prose live on the same retrieval scale: a unified retriever ranks
# candidates across sources and assumes chunks are roughly comparable units.
# Pre-cap, individual code symbols could land as 6000+ token chunks, which
# (a) made BM25 fire harder on identifier tokens than a 512-token prose
# chunk and (b) blew Anthropic 25KB tool-result caps on Probe MCP responses
# (one symbol = 30KB). 0.3x demote (commit 7745043c) was a band-aid for the
# ranking side; this constant attacks the size mismatch at the source.
MAX_SYMBOL_CHUNK_TOKENS = 512
NORMALIZER_VERSION = "v1"

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"

# Inferred-edges (Lane B) extractor configuration. Lives here per the
# RRF_K / RRF_BREADTH_ALPHA tuning-knob convention so eval sweeps and
# per-tenant overrides don't require code edits.
#
# Model selection: picked Gemini 3.1 Flash Lite over Claude Haiku 4.5
# after a real-prod eval showed Flash Lite + wider bundle produces
# better edge quality (65% specificity vs 49%, 0% vs 8% hallucination
# rate) at the same cost as Haiku-current. See
# scripts/eval_inferred_edges_widebundle.py.
#
# Provider dispatch is by prefix in services.ingestion.inferred_edges.
# extractor: "claude-*" -> anthropic SDK; "gemini-*" -> google-genai.
INFERRED_EDGES_MODEL = "gemini-3.1-flash-lite"

# Pricing per 1M tokens, as of 2026-05. Used only for the cost_usd
# telemetry gauge; pipeline correctness is unaffected by drift here.
INFERRED_EDGES_MODEL_PRICES: dict[str, tuple[float, float]] = {
    # (input_per_1M, output_per_1M) USD
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}

# Bundle caps. Wider than v1 (60K / 20 / 5 / 10) because Flash Lite's
# 1M context window plus its better specificity at higher candidate
# counts means the LLM picks more specific edge_types when given more
# evidence. Real-prod eval: at v1 caps Flash Lite was 37% specific
# (worse than Haiku); at these caps it's 65% specific (best of all
# tested combos). Wider bundle ate Flash Lite's cost advantage --
# net cost is ~the same as Haiku-v1 -- but quality is higher.
INFERRED_EDGES_BUNDLE_TOKEN_BUDGET = 300_000
INFERRED_EDGES_BUNDLE_MAX_1HOP = 50
INFERRED_EDGES_BUNDLE_MAX_VECTOR_SIMILAR = 20
INFERRED_EDGES_BUNDLE_MAX_TIME_WINDOW = 30

# Models supported by the /query synthesis layer. Keys are the
# "<provider>/<model>" identifier callers pass; values are provider names
# the synthesis dispatcher uses to pick a client.
SYNTHESIS_MODELS: dict[str, str] = {
    "anthropic/claude-haiku-4-5-20251001": "anthropic",
    "anthropic/claude-sonnet-4-6": "anthropic",
    "google/gemini-3-flash-preview": "google",
    "google/gemini-3.1-flash-lite": "google",
}
DEFAULT_SYNTHESIS_MODEL = "anthropic/claude-sonnet-4-6"

MAX_WEBHOOK_ATTEMPTS = 5
QUEUE_HEARTBEAT_INTERVAL_SECONDS = 30
QUEUE_RECLAIM_THRESHOLD_SECONDS = 300

# Per-source-system queue priority at enqueue time. Worker._claim_one
# orders by priority DESC, so higher numbers claim first. Tiers:
#
#   100  — interactive webhooks: github, slack, notion, linear, granola, sentry
#    75  — claude_code: bursty, deprioritized vs interactive (search-indexable,
#          not user-blocking; one chatty CC user shouldn't block other connectors)
#    50  — backfill rows (set in backfill_runner.py); never blocks live work
#
# Sources not in this map fall back to DEFAULT_INGESTION_PRIORITY.
DEFAULT_INGESTION_PRIORITY = 100
SOURCE_INGESTION_PRIORITY: dict[SourceSystem, int] = {
    SourceSystem.CLAUDE_CODE: 75,
    # CODEX is the OpenAI Codex CLI sibling source — same coalescing
    # semantics + doc shape as CLAUDE_CODE, so it gets the same priority
    # tier. Keeps a chatty Codex user from preempting interactive
    # webhooks at the queue claim layer.
    SourceSystem.CODEX: 75,
    SourceSystem.CUSTOM_INGEST: 75,
    # CODE_GRAPH is bursty (initial-backfill batches) and search-indexable,
    # not user-facing latency-critical — sits in the same tier as backfill
    # rows so a large repo onboarding can't block live webhooks.
    SourceSystem.CODE_GRAPH: 50,
    # Incident sources are first-class authored signals (same tier as Slack /
    # Linear / GitHub webhooks). Sentry is unlisted (defaults to 100); these
    # are explicit so the intent is obvious to future readers.
    SourceSystem.PAGERDUTY: 100,
    SourceSystem.INCIDENT_IO: 100,
}

TOP_K_VECTOR = 50
TOP_K_BM25 = 50
TOP_K_GRAPH = 20
TOP_K_DIRECTED = 20
RRF_K = 60
DEDUP_COSINE_THRESHOLD = 0.95

# Graph-explore endpoint (POST /graph/explore + /graph/search) caps. These
# bound the visualization payload that the dashboard renders client-side --
# force-directed layout starts to crawl above a few thousand nodes, and the
# wire payload itself dwarfs everything else above 5k edges. Lives here per
# the RRF_K / RRF_BREADTH_ALPHA tuning-knob convention so tuning doesn't
# require an env-var deploy.
#
# Default mode: top-N nodes by graph_nodes.degree DESC, 1-hop edges among
# the selected set. Anchor mode: tiered BFS centered on a node, hop1 cap +
# hop2 cap (total = hop1 + hop2). Edge cap is enforced regardless of node
# count; if hit, truncated=True flips in the response. WHY_MAX_CHARS caps
# per-edge LLM-generated rationale at serialization time.
GRAPH_EXPLORE_NODE_CAP = 2000
GRAPH_EXPLORE_EDGE_CAP = 5000
GRAPH_EXPLORE_HOP1_CAP = 500
GRAPH_EXPLORE_HOP2_CAP = 1500
GRAPH_EXPLORE_WHY_MAX_CHARS = 200
GRAPH_SEARCH_DEFAULT_LIMIT = 10
GRAPH_SEARCH_MAX_LIMIT = 25

# Directed-vectors feature: doc-level retrieval signal contributed by
# per-document trigger phrases stored in the directed_vectors table.
# Eval-tuned; commits in the same change that bumps it. Set to 0.0 to
# disable contribution without removing the retriever from the fan-out.
DIRECTED_RETRIEVAL_WEIGHT: float = 1.0

# Cap on LLM-generated directed phrases per wiki document. Engineer-pinned
# phrases get their own cap (MAX_HUMAN_DIRECTED_PER_DOC) so a runaway LLM
# can't bury legitimate pins.
MAX_DIRECTED_VECTORS_PER_DOC: int = 16

# Cap on engineer-pinned directed phrases per wiki document. Higher than
# the LLM cap because explicit pins are intentional, but bounded so a
# malicious / typo'd frontmatter can't balloon the table.
MAX_HUMAN_DIRECTED_PER_DOC: int = 32

# Per-phrase character cap. Trigger phrases are short by design (5-12
# tokens per the prompt); 256 chars is generous slack against natural
# English while still rejecting megabyte-long pathological inputs that
# would bloat embedding cost / storage / log noise.
MAX_DIRECTED_PHRASE_CHARS: int = 256

# Cosine distance threshold below which two candidate trigger phrases are
# considered near-duplicates and one is dropped (humans always win on
# collision; LLM duplicates of human pins are suppressed).
DIRECTED_DEDUPE_COSINE_THRESHOLD: float = 0.05

# Doc-grouped fusion: weight applied to the sum of NON-best content-chunk RRF
# scores when collapsing per-doc. doc_score = max(rrfs) + alpha * sum(others) +
# metadata_sum. Prevents long docs from drowning shorter ones; preserves
# best-chunk-wins-ties; rewards docs whose multiple chunks all matched.
RRF_BREADTH_ALPHA = 0.3

# Per-source-system score multiplier applied AFTER RRF fusion. Values < 1.0
# demote a source's docs so they rank below other sources at equal vector
# relevance. Defaults to 1.0 (no change) for any source not listed.
#
# Rationale: claude_code transcripts are high-volume and lower-signal-density
# than authored team artifacts (Slack threads, Linear tickets, PR descriptions),
# so we down-weight them to keep authored content surfacing first.
SOURCE_SCORE_MULTIPLIERS: dict[SourceSystem, float] = {
    SourceSystem.CLAUDE_CODE: 0.5,
    # CODEX docs are the same shape and signal density as CLAUDE_CODE --
    # both are agent transcripts, not authored team artifacts. Apply the
    # same demotion so they rank consistently with each other below
    # Slack/Linear/PR docs at equal vector relevance.
    SourceSystem.CODEX: 0.5,
    # CODE_GRAPH chunks are over-represented in top-K relative to their
    # signal strength: production query_traces (7d, acme) show
    # code_graph at 36.5% of top-5 results despite an avg post-fusion score
    # ~3.4x lower than github and ~4.1x lower than claude_code. BM25 fires
    # on common identifier tokens ("session", "tenant", "customer"),
    # surfacing weak code matches above genuine non-code answers (e.g. a
    # "lindy.ai onboarding pilot" query returning triage_worker.py at rank 1).
    # 0.3 ~= inverse of the avg-score ratio: a code_graph chunk now needs to
    # be ~3x stronger than the code_graph average to compete with an average
    # non-code hit, so genuinely strong vector matches still survive while
    # keyword-noise chunks fall out of top-K. Validated via offline replay
    # at multipliers [1.0, 0.7, 0.5, 0.3, 0.2]: knee at 0.3 (top-5 share
    # 36.5% -> 24.4%); 0.2 only buys 1.3pp more.
    SourceSystem.CODE_GRAPH: 0.3,
}

# Baseline recency half-life (days) applied to every source. Smaller = faster
# decay. Acts as the universal floor so backfilled tenants don't see 8-12 month
# old docs ranked equally with last week's. Per-source overrides below win when
# a source is noisier than baseline and needs faster decay.
#
# At 120d: a 4-month-old doc keeps 50% of its score, 8-month 25%, 12-month 12%.
# Strongly-relevant old docs still win on raw signal; tied semantic matches go
# to the fresher one.
DEFAULT_RECENCY_HALF_LIFE_DAYS = 120.0

# Per-source-system half-life (days) overrides for recency decay applied after
# the multiplier. Smaller = faster decay. Sources not listed fall back to the
# caller-supplied global half_life_days if set, else DEFAULT_RECENCY_HALF_LIFE_DAYS.
#
# Rationale: a CC session is a point-in-time scratchpad — by week two it's
# almost always stale or contradicted by something authored elsewhere. Slack/
# Linear/PR docs stay relevant for months and ride the baseline by design.
SOURCE_HALF_LIFE_DAYS: dict[SourceSystem, float] = {
    SourceSystem.CLAUDE_CODE: 7.0,
    # CODEX transcripts are scratchpads with the same staleness curve as
    # CLAUDE_CODE -- both lose relevance fast as authored docs catch up.
    SourceSystem.CODEX: 7.0,
    # Code symbols decay slower than CC/Codex transcripts (a function still
    # exists weeks later) but faster than authored docs (refactors happen).
    SourceSystem.CODE_GRAPH: 30.0,
    # Incident records remain relevant for post-mortems and pattern-matching
    # for months; 200d keeps them well above the 120d baseline while still
    # allowing slow decay relative to very recent incidents.
    SourceSystem.PAGERDUTY: 200.0,
    SourceSystem.INCIDENT_IO: 200.0,
}

# Inferred-edge retrieval channel tuning. The channel walks INFERRED Doc-Doc
# edges from top-K primary docs and surfaces the linked docs as additional
# results with `matched_via.channel = "inferred_edge"`. Knobs live here so
# eval sweeps + per-tenant overrides can adjust without code edits, per the
# RRF_K / RRF_BREADTH_ALPHA precedent (feedback_prbe_knowledge_tuning_consts).

# Cap on linked docs returned by `inferred_edge_search`. 5 keeps the result
# set predictable; the inferred-edge channel is supplementary, not primary.
INFERRED_EDGE_TOP_K = 5

# Per-hop dampening applied to inferred-edge results. The final score is:
#   dampening * 1/(1 + anchor_rank) * SOURCE_SCORE_MULTIPLIERS[src]
#                                  / (1 + ln(linked_edge_count))
# 0.2 keeps a 2-hop result (anchor at rank 1) at base score 0.10 -- well
# below direct vector hits but still surfacing above weakly-matched primary
# docs. Was 0.5 in v1; lowered after observing inferred-edge results
# outranking the primary doc that surfaced them (the codex session #1
# case where a 2-hop chained inference dominated rank 1 for a query the
# linked doc didn't actually contain).
INFERRED_EDGE_DAMPENING = 0.2

# Max chunks per inferred-edge-derived QueryDocumentResult. Hydrated from
# the chunks table by `chunk_index ASC` (first chunks are usually the most
# identity-bearing -- title metadata + opening body). Without hydration the
# chunks list is empty and the dashboard renders "0 matched"; the
# synthesizer also can't cite the doc -- both regressions of the v1 design.
INFERRED_EDGE_HYDRATION_CHUNKS = 3

# ---- Search agent (gatherer) -------------------------------------------------
# The gatherer is the retrieval pipeline (see
# docs/specs/agentic-search.md). Tunables below are read at agent loop
# construction; changing them requires a redeploy (no hot-reload).

# IMPORTANT: no provider-prefix expansion. When `shared.llm.acompletion`
# forces `custom_llm_provider="openai"` for gateway-routed calls (so the
# upstream honors response_format), LiteLLM forwards this model id
# verbatim in the OpenAI chat-completion request body. The LiteLLM proxy
# matches `model_name: "cerebras/*"` in its modelList (per
# prbe-backend/charts/litellm/values.yaml) and the same model id flows
# through to the upstream call.
#
# Env-overridable so we can A/B-test alternative providers without a code
# change. Set SEARCH_AGENT_INFERENCE_MODEL on the retrieval pod to e.g.
# `claude-sonnet-4-6` (routed via the proxy's claude-* modelList entry)
# and that's what gets called.
#
# Default flipped 2026-05-18 from `accounts/fireworks/models/gpt-oss-120b`
# to `cerebras/gpt-oss-120b`. Same model id, ~10x the output throughput
# on Cerebras's wafer-scale chips — eliminated the 90s gatherer-timeout
# cascade on conceptual queries. The Fireworks route was simultaneously
# dropped from the LiteLLM modelList (prbe-backend PR #342), so a stale
# install relying on the old default now 404s at the proxy with a clean
# error instead of silently routing through the catch-all.
SEARCH_AGENT_INFERENCE_MODEL = os.getenv(
    "SEARCH_AGENT_INFERENCE_MODEL",
    "cerebras/gpt-oss-120b",
)

# Soft budget: total tool calls across all turns. Covers turn-1 mandatory 4
# + ~16 exploration calls across 2-3 follow-up turns. The agent may extend
# by emitting need_deeper{reason}; +10 per extension, max 2 extensions.
SEARCH_AGENT_TOOL_BUDGET = 20
SEARCH_AGENT_EXTENSION_GRANT = 10
SEARCH_AGENT_MAX_EXTENSIONS = 2

# Soft turn cap: once the model has completed this many turns without
# emitting the terminal, the next exploration turn triggers a forcing
# nudge to call `emit_gatherer_output` on the turn after that. Same
# mechanism as the budget-exhausted nudge, but tripped by turn count
# instead of tool-call count.
#
# Set to 1 because the parallel 4-channel fan-out (vector + BM25 +
# graph + inferred edges) already runs PRE-LOOP in run_gatherer and
# its results are baked into the model's turn-1 evidence pack as
# `<channel_results>`. Vector and BM25 already surface anything one
# hop from a real answer, so the only useful in-loop exploration is
# at most one 1-hop follow-up (graph_walk / fetch_doc); beyond that
# is provably noise.
#
# Concretely the loop runs at most:
#   model turn 1 — sees prefanout, may emit terminal OR pick one
#                  exploration tool (cap not yet tripped: turn_count
#                  was 0 on entry).
#   model turn 2 — should emit terminal; if it picks another tool,
#                  the cap (turn_count was 1 on entry, >= 1) fires
#                  on the way out of this iteration.
#   model turn 3 — forced-emit turn after the nudge (only reached
#                  if turns 1 and 2 both explored).
#
# Effective ceiling: SOFT_TURN_CAP + 2 = 3 LLM turns (~3-5s on
# Cerebras), down from the prior 67-90s oscillation that hit the
# 90s SEARCH_AGENT_LOOP_TIMEOUT_SECONDS wall-clock.
SEARCH_AGENT_SOFT_TURN_CAP = 1

# Hard ceiling. Even with extensions the agent never exceeds this.
SEARCH_AGENT_HARD_CAP = SEARCH_AGENT_TOOL_BUDGET + (
    SEARCH_AGENT_EXTENSION_GRANT * SEARCH_AGENT_MAX_EXTENSIONS
)  # 40

# Min curated results the agent should return before falling back to "no
# confident match". If the agent emits fewer than this, harness logs a
# `gatherer.under_min_output` anomaly for trace review.
SEARCH_AGENT_MIN_OUTPUT = 5

# Per-tool top_k defaults. The agent may override at call time. See plan
# section "Per-tool top_k defaults" for the bytes/turn budget reasoning.
SEARCH_AGENT_VECTOR_TOP_K = 15
SEARCH_AGENT_BM25_TOP_K = 15
SEARCH_AGENT_GRAPH_TOP_K = 10
SEARCH_AGENT_INFERRED_EDGE_TOP_K = 10
SEARCH_AGENT_GRAPH_WALK_TOP_K = 20
SEARCH_AGENT_EXPAND_NEIGHBORS_TOP_K = 10
SEARCH_AGENT_FETCH_CHUNKS_MAX = 10

# Per-channel result byte cap. Node properties / chunk content trimmed to
# this when assembled into a tool return. Keeps the per-turn evidence pack
# around 15K tokens.
SEARCH_AGENT_PER_HIT_PROPERTIES_CAP = 2048

# Cerebras prefix-cache discount only fires when consecutive turns —
# AND consecutive queries from the same customer — land on the same
# replica. We set `x-session-affinity` per customer (not per query) so
# the static prefix (system prompt + tool defs) cache-hits across queries,
# and multi-turn cache continuity is preserved because Cerebras's prefix
# cache is content-addressed (turn 1 still hits the warm prefix turn 0
# wrote to the same replica). This is the acceptance gate observed via
# query_traces.cache_hit_rate; if production rate drops below this,
# hard-query cost roughly doubles. See `loop._affinity_key`.
SEARCH_AGENT_CACHE_HIT_RATE_FLOOR = 0.7

# Wall-clock cap on a single agent turn (model + tool execution combined).
# Aborts a stuck turn loudly rather than waiting on the LiteLLM default.
SEARCH_AGENT_TURN_TIMEOUT_SECONDS = 30.0

# Overall agent loop cap. Prevents pathological queries from monopolising
# a worker. p99 should land far below this; trip = log + 503.
SEARCH_AGENT_LOOP_TIMEOUT_SECONDS = 90.0

# Fraction of gatherer runs whose full per-turn transcript gets persisted
# to R2 alongside the query_traces summary row. 1.0 = persist every run.
# Drop via `kubectl set env DEPLOY SEARCH_AGENT_TRACE_SAMPLE_RATE=0.1`
# without a deploy if R2 spend spikes. Sampled-out rows still get the
# summary in `query_traces`; only the full blob is skipped.
SEARCH_AGENT_TRACE_SAMPLE_RATE = float(
    os.getenv("SEARCH_AGENT_TRACE_SAMPLE_RATE", "1.0")
)

# Prefix used in `integration_tokens.scope` to signal the row represents a
# GitHub App installation rather than an OAuth access_token. The installation
# id follows the colon; tokens are minted on demand from the App private key.
GITHUB_INSTALLATION_SCOPE_PREFIX = "installation:"

# Granola: API tier prefix in integration_tokens.scope. Personal keys see only
# the issuing user's notes + shared. Enterprise keys see the whole workspace.
GRANOLA_SCOPE_PERSONAL = "tier:personal"
GRANOLA_SCOPE_ENTERPRISE = "tier:enterprise"

# pg_notify channel the worker LISTENs on for sub-second manual-refresh wake.
# The /admin/.../granola/refresh endpoint NOTIFYs after enqueuing so
# BackfillWorker doesn't wait for its 5s poll cycle.
GRANOLA_REFRESH_CHANNEL = "granola_refresh"

# Steady-state polling cadence: re-enqueue Granola backfills this often once
# the initial sync is complete. Read by services/ingestion/poller.
GRANOLA_POLL_INTERVAL_SECONDS = 300

# Per-customer rate-budget for outbound calls to the Granola API.
# Granola docs: 5 rps / 25 in 5s burst. We sleep this long between calls inside
# the connector's backfill loop, leaving 20% headroom under the documented limit.
GRANOLA_REQUEST_INTERVAL_SECONDS = 0.25

# Manual-refresh debounce. Repeated /refresh hits within this window collapse
# into a single enqueue + notify; the second hit returns 429 with Retry-After.
GRANOLA_REFRESH_DEBOUNCE_SECONDS = 30


# pg_notify channels for the wiki synthesis pipeline.
#
# Pre-redesign: a single `wiki_synthesize` channel — Normalizer._persist
# fired NOTIFY on every webhook, the in-process cron drained immediately,
# resulting in continuous daytime synthesis. That model didn't match the
# wiki's actual scope (slow-moving company knowledge). The redesign:
#
# - Normalizer._persist NO LONGER fires NOTIFY. Queue rows accumulate
#   silently during the day at status='pending'.
# - The wiki-cron fly app fires NOTIFY on `wiki_synthesize_pending`
#   nightly at 02:00 UTC (per opted-in customer with pending rows). The
#   /api/wiki/synthesize/trigger endpoint also fires it for manual wakes
#   from the dashboard "Generate Wiki Now" button.
# - The wiki-worker (triage) app LISTENs on `wiki_synthesize_pending` →
#   drains pending rows through triage → marks them triaged/rejected/
#   verifier_rejected → fires NOTIFY on `wiki_synthesize_triaged` from
#   the same transaction that committed the UPDATE (Postgres delivers
#   NOTIFY only after COMMIT, so listeners never wake on un-visible rows).
# - The wiki-synthesis app LISTENs on `wiki_synthesize_triaged` → drains
#   triaged rows through verifier + synthesize → writes wiki pages →
#   regenerates the index.
WIKI_PENDING_CHANNEL = "wiki_synthesize_pending"
WIKI_TRIAGED_CHANNEL = "wiki_synthesize_triaged"

# Backfill pipeline wake channel — fired by the
# /api/wiki/backfill/trigger route after it inserts pending rows. Empty
# payload; the BackfillWorker treats this as a "drain pending rows now"
# wake hint and claims rows via FOR UPDATE SKIP LOCKED. Distinct from
# WIKI_PENDING_CHANNEL because the daily-replay path operates on the v4
# queue, while backfill reads from source APIs.
WIKI_BACKFILL_CHANNEL = "wiki_backfill_pending"

# Backfill cancel channel — fired by the trigger route's force-cancel
# path. Payload is a JSON object ``{customer_id, run_ids: [int]}``;
# every BackfillWorker LISTENing on this channel cancels in-flight
# tasks whose run_id matches. Coarse 10s drain window — see
# BACKFILL_CANCEL_DRAIN_TIMEOUT_SECONDS.
WIKI_BACKFILL_CANCEL_CHANNEL = "wiki_backfill_cancel"

# Cooperative drain window the trigger route waits after firing the
# cancel NOTIFY before proceeding to wipe + insert new pending rows.
# Sized larger than the worker's per-tick cadence but small enough that
# admin-initiated force-restart still feels interactive in the dashboard.
BACKFILL_CANCEL_DRAIN_TIMEOUT_SECONDS = 10.0

# Per-machine cap on concurrent backfill crawler agents. Read at boot
# from the BACKFILL_PARALLELISM env var by ``BackfillWorker``; the
# constant here is the default. Sized at 6 against the 4 GB / 2 vCPU
# fly machine envelope (idle ~150 MB, ~150-250 MB per active crawler ->
# ~1.5 GB peak crawler load + headroom). Tune via env, not code.
BACKFILL_PARALLELISM = 6

# How many wiki_synthesis_queue rows the cron claims per drain tick. Triage is
# token-budget batched on top of this; this is just the upper bound on rows
# pulled into memory at once.
WIKI_SYNTHESIS_CLAIM_BATCH = 200

# Token budget per Haiku triage call, expressed in *estimated Anthropic
# tokens* (post-multiplier — see `services.synthesis.triage`). Rows are
# packed greedily until this ceiling is hit, then the batch fires. The
# packer adds prompt + tool-schema + per-event framing overhead on top
# of body tokens before comparing to this budget, so it represents the
# user-content slice of the wire request, not the full request size.
#
# Headroom: Anthropic Haiku's hard context limit is 200K tokens. We
# budget 150K for content; the remaining 50K is left as margin for
# (1) prompt + tool-schema + envelope (~2K), (2) tokenizer drift between
# our cl100k estimate and Anthropic's true tokenizer, and (3) the model's
# own response. Production drains were DLQ'ing entire batches at the
# previous 120K budget because the packer counted only raw body text in
# cl100k and Anthropic's tokenizer + request envelope pushed the wire
# count past 200K (e.g. batch_size=66 produced 208K Anthropic tokens).
WIKI_TRIAGE_TOKEN_BUDGET = 150_000

# Output-side budget for the triage Anthropic call.
#
# Haiku 4.5's `max_tokens` ceiling is 8192. We set 8000 to leave a 192-
# token cushion against SDK-version drift / per-conversation token
# bookkeeping. The original 4096 was way too low: a batch of ~28+ events
# would produce more verdicts than fit in 4096 output tokens, so Haiku
# would stop at max_tokens with NO tool_use block — causing a Pydantic
# crash on the missing `verdicts` field and DLQ-ing the whole batch.
WIKI_TRIAGE_MAX_OUTPUT_TOKENS = 8000

# Per-verdict size estimate, in Anthropic tokens. TriageVerdict is
# {important: bool, score: float, reason: str ≤ ~100 tokens}; with the
# JSON envelope `"queue_id": {...}` and pretty-printing, a verdict lands
# around 80-120 Anthropic tokens. 150 is the conservative cap.
WIKI_TRIAGE_VERDICT_TOKENS = 150

# Output-side cap on events per batch:
#   floor(WIKI_TRIAGE_MAX_OUTPUT_TOKENS / WIKI_TRIAGE_VERDICT_TOKENS)
#   = 8000 / 150 = 53 → round down to 50 for envelope + drift margin.
# The packer enforces MIN(input-token-budget, this-event-cap) so the
# limiting factor is whichever binds first for a given batch.
WIKI_TRIAGE_MAX_EVENTS_PER_BATCH = 50

# Importance threshold for triage to keep an event. Below this score the row
# is marked 'rejected' and never reaches synthesis. Raised from 5.0 → 7.0
# to align triage with the wiki's actual scope (slow-moving company
# knowledge, not a per-event log). Step down stepwise (7.0 → 6.0 → 5.0)
# if the wiki is under-populated; step up if it gets spammy.
WIKI_TRIAGE_SCORE_THRESHOLD = 7.0

# Per-row attempt cap before a queue row is parked in 'failed'.
WIKI_SYNTHESIS_MAX_ATTEMPTS = 3

# Defensive periodic wake interval. The cron also wakes on every NOTIFY; this
# is a safety net if a notify is missed during a connection drop.
WIKI_SYNTHESIS_PERIODIC_WAKE_SECONDS = 1800  # 30 min

# Provider knob for the triage stage. v4 uses the wiki agent (Gemini
# Pro) for synthesis, so the synthesis + verifier provider knobs are
# gone. Triage is provider-pluggable: flip the value and redeploy.
# Recognized values:
#   "haiku" | "claude-haiku"            -> Anthropic Haiku 4.5
#   "gemini-flash-lite" | "gemini-3.1-flash-lite" -> Gemini 3.1 Flash Lite
#   "gemini-3.5-flash"                  -> Gemini 3.5 Flash (default; 2026-05-19)
# No env-var override path — the prior `getattr(settings, ...)` plumbing
# referenced fields that didn't exist on Settings, so the env var was
# silently inert. Constants-only is honest.
#
# Default flipped 2026-05-19 from "haiku" to "gemini-3.5-flash" after the
# A/B sweep in scripts/eval_3_5_flash_sweep.py (report:
# ~/.gstack/projects/prbe-knowledge/eval-3-5-flash-sweep-20260520T025718Z.md).
# 20 fixtures x 2 trials per model. Label accuracy: both 100%. Opus-judged
# quality: 9.3 (haiku) vs 9.4 (3.5-flash) — statistical tie. p50 latency:
# 1913ms → 1614ms (~16% faster). Cost per call: $0.00225 → $0.00060
# (~3.75x cheaper). Net: equal quality at <30% of the wire cost on a
# high-volume hot path.
WIKI_TRIAGE_MODEL = "gemini-3.5-flash"

# Directed-phrase generation runs once per wiki page during synthesis to
# emit 5-10 trigger phrases that boost retrieval ranking when an engineer's
# symptom-style query semantically matches them. The 2026-05-09 model
# shootout (scripts/eval_directed_phrases.py, judged by Opus 4.7) picked
# Gemini 3 Flash: specificity 8.6/retrieval-fit 8.2 vs Haiku 7.8/7.8, at
# ~1/4 the cost ($0.0005 vs $0.0022 per call). Flip to "haiku" or
# "gemini-3.1-flash-lite" via this constant + redeploy.
DIRECTED_PHRASES_MODEL = "gemini-3-flash-preview"

# Concurrency caps. The wiki-worker fans out customers, then triage
# batches per customer. (The v4 wiki agent uses
# WIKI_AGENT_GLOBAL_CONCURRENCY for synthesis-side fan-out plus a
# per-customer advisory lock; it doesn't cluster events anymore.)
WIKI_SYNTHESIS_CUSTOMER_CONCURRENCY = 4
WIKI_TRIAGE_BATCH_CONCURRENCY = 8

# Manual trigger rate limit (advisory-lock + lookback in the BFF). The
# /api/wiki/synthesize/trigger endpoint here surfaces the same value so
# the dashboard can render an accurate "try again in Xs" toast.
WIKI_TRIGGER_RATE_LIMIT_SECONDS = 300

# Hour-of-day (UTC) the wiki-cron fly machine fires its nightly NOTIFY.
# 02:00 UTC = 18:00 PT / 21:00 ET — picked so the drain finishes before
# the team's morning standup but doesn't compete with the rest of the
# nightly pipeline (Granola steady-poll cycles, etc.).
WIKI_NIGHTLY_HOUR_UTC = 2


# ---------------------------------------------------------------------------
# Wiki agent loop (v4: Gemini 3.1 Pro driving the synthesis stage)
# ---------------------------------------------------------------------------

# Hard cap on agent turns per drain. Picked at 200 to leave headroom for
# pebble's ~3000-event drains; smaller customers typically finish in
# 10-50 turns. Exceeding this cap halts the drain and DLQs the in-flight
# rows; admin reset is the recovery path.
WIKI_AGENT_TURN_CAP = 200

# Hard cap on staged page updates per drain. The wiki is supposed to
# move slowly — 30 page edits per night is generous. Exceeding this cap
# means the agent is hallucinating page mass and we'd rather DLQ than
# write 100 brand-new pages.
WIKI_AGENT_UPDATE_CAP = 30

# Stall threshold. If the agent makes no consequential tool call (no
# page update / create / skip) for this many consecutive turns, halt.
#
# Bumped from 3 to 15 after acme' run 105 stalled with 200
# events DLQ'd despite the agent making real progress on reads. The
# old "one read-page, one read-event, one think" math was wrong — a
# realistic decision flow on a chunk of 200 events looks like:
#   next_events -> list_wiki_pages -> read_page x3 -> get_event_body x2
#   -> update_page (CONSEQUENTIAL)
# That's 7 read-style turns before the first consequential one.
# Three would have halted in the middle of normal exploration. 15
# leaves margin for the agent to read 5 pages and 3 event bodies
# before deciding, with extra room for a re-read or thought-only
# turn. Stuck loops still trip eventually.
WIKI_AGENT_STALL_TURNS = 15

# Auto-compaction trigger. When estimated input tokens cross this
# fraction of Gemini 3.1 Pro's 2M context window, summarize the
# conversation history (preserving structured runtime state) before
# the next turn.
WIKI_AGENT_COMPACT_THRESHOLD = 0.60

# Number of triaged events per next_events() page. Gemini reads the
# day in batches; the agent re-calls next_events() until drain_complete.
WIKI_AGENT_BATCH_SIZE = 200

# Maximum number of customer drains running simultaneously per
# wiki-synthesis fly machine. Higher than per-customer concurrency (1
# under advisory lock) so two small customers can drain in parallel
# while pebble holds its own machine.
WIKI_AGENT_GLOBAL_CONCURRENCY = 2

# Gemini model used by the wiki agent loop. Triage stays Flash Lite;
# only the agent uses Pro because per-cluster reasoning + cross-event
# pattern recognition need the bigger model.
WIKI_AGENT_MODEL = "gemini-3.1-pro-preview"

# Compactor model. Cheaper Flash variant since it only summarizes the
# conversation; preserves the structured runtime state untouched.
WIKI_AGENT_COMPACTOR_MODEL = "gemini-3.1-flash-lite"

# Per-source backfill crawler models. Default to the same Pro model the
# daily-replay agent uses; per-source knobs let us swap a cheaper /
# bigger model for one source without redeploying the rest. Mentioned
# under "Per-source models" in docs/wiki-backfill-plan.md.
WIKI_BACKFILL_MODEL_GITHUB = "gemini-3.1-pro-preview"

# Stop-walking heuristic for backfill crawlers. After this many
# consecutive source items that don't cause the agent to call
# update_page / create_page, the crawler treats the repo as drained and
# moves on. Picked at 50 to match the system prompt's stopping rule.
WIKI_BACKFILL_QUIET_STREAK = 50

# Time horizon (days) for GitHub PR + issue ingestion. Commits walk
# all-time per the locked plan so old structural commits ("first added
# auth middleware") still surface even when ticket history is bounded.
WIKI_BACKFILL_GITHUB_PRS_DAYS = 365


# Cap on Phase 2 fan-out per (customer, source). After Phase 1 completes,
# the orchestrator queries the source's discoverer for targets (e.g.,
# repos for GitHub) and inserts one Phase 2 row per target up to this
# cap. Above the cap, take the top-N by recent activity. At ~$0.30-0.60
# per Phase 2 agent run (Gemini Pro), 30 caps worst-case spend at
# ~$15/backfill on the largest customers we have today.
BACKFILL_MAX_TARGETS_PER_SOURCE = int(os.environ.get("BACKFILL_MAX_TARGETS_PER_SOURCE", "30"))

# Agent's CachedContent TTL. Re-create on miss; alert if hit_rate < 80%.
WIKI_AGENT_CACHE_TTL = "3600s"


# --- DB pool init backoff ---------------------------------------------------
# Connect-with-backoff knobs for shared.db.init_pool, kept here per the
# RRF_K / RRF_BREADTH_ALPHA tuning-knob convention (feedback_prbe_knowledge_
# tuning_consts) so the ceiling is one explicit number, not buried in db.py.
#
# Sizing: in the k8s data-plane, app pods only start after the migrate
# sentinel exists (prbe-data-plane-image / chart change), so Postgres is up
# and migrated by the time init_pool runs. The remaining retries cover only
# transient boot blips -- NetworkPolicy settling, DNS, pool limits, a
# credential race -- which clear in a second or two, not a minute. So the
# old 6-attempt / base-1s / x2 schedule (1+2+4+8+16 = 31s of pure sleep,
# ~90s worst case with connect timeouts) is far longer than needed.
#
# New ceiling: 4 attempts, base 0.5s, x2, capped per-attempt at 5s ->
# backoffs of 0.5 + 1 + 2 = 3.5s of sleep across 3 retries; worst case
# (connect timeout fully consumed each attempt) ~3.5s + 4 * connect_timeout.
# A real transient blip recovers in single-digit seconds; genuine outages
# still surface a readable DatabaseUnavailable rather than a silent hang.
DB_INIT_RETRY_ATTEMPTS = 4
DB_INIT_RETRY_BASE_SECONDS = 0.5
DB_INIT_RETRY_BACKOFF_CAP_SECONDS = 5.0
