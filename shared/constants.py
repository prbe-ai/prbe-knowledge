"""Phase 0 canonical enums. Every string used as a type/label/edge/status lives here."""

from enum import StrEnum


class SourceSystem(StrEnum):
    SLACK = "slack"
    LINEAR = "linear"
    GITHUB = "github"
    NOTION = "notion"
    SENTRY = "sentry"
    GRANOLA = "granola"


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

MAX_WEBHOOK_ATTEMPTS = 5
QUEUE_HEARTBEAT_INTERVAL_SECONDS = 30
QUEUE_RECLAIM_THRESHOLD_SECONDS = 300

TOP_K_VECTOR = 50
TOP_K_BM25 = 50
TOP_K_GRAPH = 20
RRF_K = 60
DEDUP_COSINE_THRESHOLD = 0.95

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
