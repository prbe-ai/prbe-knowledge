"""Phase 0 canonical enums. Every string used as a type/label/edge/status lives here."""

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
    GITHUB_REVIEW = "github.review"
    GITHUB_CODEOWNERS = "github.codeowners"
    NOTION_PAGE = "notion.page"
    NOTION_DATABASE = "notion.database"
    SENTRY_ISSUE = "sentry.issue"
    SENTRY_EVENT = "sentry.event"
    GRANOLA_MEETING = "granola.meeting"
    CLAUDE_CODE_SESSION     = "claude_code.session"
    CLAUDE_CODE_QA          = "claude_code.qa"
    CLAUDE_CODE_CODE_CHANGE = "claude_code.code_change"
    CLAUDE_CODE_DECISION    = "claude_code.decision"
    CLAUDE_CODE_FILE_REF    = "claude_code.file_ref"
    MANUAL_UPLOAD_TEXT      = "manual_upload.text"
    MANUAL_UPLOAD_MARKDOWN  = "manual_upload.markdown"
    MANUAL_UPLOAD_DOCX      = "manual_upload.docx"
    MANUAL_UPLOAD_FILE      = "manual_upload.file"
    WIKI_SERVICE_CARD = "wiki.service_card"
    WIKI_DECISION = "wiki.decision"
    WIKI_FEATURE = "wiki.feature"
    WIKI_RUNBOOK = "wiki.runbook"


class NodeLabel(StrEnum):
    SERVICE = "Service"
    REPO = "Repo"
    PERSON = "Person"
    CHANNEL = "Channel"
    TICKET = "Ticket"
    PR = "PR"
    ISSUE = "Issue"
    DOCUMENT = "Document"
    ERROR_GROUP = "ErrorGroup"

    SERVICE_CARD = "ServiceCard"
    DECISION = "Decision"
    FEATURE = "Feature"
    RUNBOOK = "Runbook"
    WIKI_PERSON = "WikiPerson"

    AGENT = "Agent"
    WORKFLOW = "Workflow"
    FIX_ARTIFACT = "FixArtifact"
    VERIFICATION_RESULT = "VerificationResult"


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


EMBEDDING_MODEL = "openai/text-embedding-3-large"
EMBEDDING_DIM = 3072
CHUNKER_VERSION = "naive-v1"
NORMALIZER_VERSION = "v1"

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"

# Models supported by the /query synthesis layer. Keys are the
# "<provider>/<model>" identifier callers pass; values are provider names
# the synthesis dispatcher uses to pick a client.
SYNTHESIS_MODELS: dict[str, str] = {
    "anthropic/claude-haiku-4-5-20251001": "anthropic",
    "anthropic/claude-sonnet-4-6":          "anthropic",
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
}

TOP_K_VECTOR = 50
TOP_K_BM25 = 50
TOP_K_GRAPH = 20
RRF_K = 60
DEDUP_COSINE_THRESHOLD = 0.95

# Per-source-system score multiplier applied AFTER RRF fusion. Values < 1.0
# demote a source's docs so they rank below other sources at equal vector
# relevance. Defaults to 1.0 (no change) for any source not listed.
#
# Rationale: claude_code transcripts are high-volume and lower-signal-density
# than authored team artifacts (Slack threads, Linear tickets, PR descriptions),
# so we down-weight them to keep authored content surfacing first.
SOURCE_SCORE_MULTIPLIERS: dict[SourceSystem, float] = {
    SourceSystem.CLAUDE_CODE: 0.5,
    # CODEX docs are the same shape and signal density as CLAUDE_CODE —
    # both are agent transcripts, not authored team artifacts. Apply the
    # same demotion so they rank consistently with each other below
    # Slack/Linear/PR docs at equal vector relevance.
    SourceSystem.CODEX: 0.5,
}

# Per-source-system half-life (days) for recency decay applied after the
# multiplier. Smaller = faster decay. Sources not listed fall back to the
# caller-supplied global half_life_days, or no decay if that's None.
#
# Rationale: a CC session is a point-in-time scratchpad — by week two it's
# almost always stale or contradicted by something authored elsewhere.
# Slack/Linear/PR docs stay relevant for months by design.
SOURCE_HALF_LIFE_DAYS: dict[SourceSystem, float] = {
    SourceSystem.CLAUDE_CODE: 7.0,
    # CODEX transcripts are scratchpads with the same staleness curve as
    # CLAUDE_CODE — both lose relevance fast as authored docs catch up.
    SourceSystem.CODEX: 7.0,
}

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
