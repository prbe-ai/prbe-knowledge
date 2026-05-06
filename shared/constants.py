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
    CUSTOM_INGEST = "custom_ingest"
    # Curated team-knowledge layer (runbooks, decisions, service cards, feature
    # notes). Pages are authored programmatically via /api/wiki/pages/* — no
    # external webhook. doc_class distinguishes human authorship (MANUAL_ENTRY)
    # from agent-compiled summaries (COMPILED_WIKI).
    WIKI = "wiki"


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
    WIKI_SERVICE_CARD = "wiki.service_card"
    WIKI_DECISION = "wiki.decision"
    WIKI_FEATURE = "wiki.feature"
    WIKI_RUNBOOK = "wiki.runbook"
    # Auto-generated table of contents. Exactly one per customer; regenerated
    # at the end of each synthesis run from the live set of wiki pages.
    WIKI_INDEX = "wiki.index"


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
    "anthropic/claude-sonnet-4-6": "anthropic",
    "google/gemini-3-flash-preview": "google",
    "google/gemini-3.1-flash-lite-preview": "google",
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

# Bootstrap pipeline channel — fired by the /api/wiki/bootstrap/trigger
# route and the OAuth-callback per-source hook. The bootstrap fly app's
# listener wakes per NOTIFY, parses the payload (json: customer_id +
# optional sources + wipe_first + reason), and calls the orchestrator.
# Distinct from WIKI_PENDING_CHANNEL because the daily-replay path
# operates on the v4 queue, while bootstrap reads from source APIs.
WIKI_BOOTSTRAP_CHANNEL = "wiki_bootstrap_pending"

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
# gone. Triage is provider-pluggable: flip the value and redeploy to
# switch from Anthropic Haiku -> Gemini Flash Lite (or back). Defaults
# to Anthropic.
# Recognized values:
#   "haiku" | "claude-haiku"           -> Anthropic Haiku 4.5
#   "gemini-flash-lite" | "gemini-flash-lite-preview" -> Gemini Flash Lite
# No env-var override path — the prior `getattr(settings, ...)` plumbing
# referenced fields that didn't exist on Settings, so the env var was
# silently inert. Constants-only is honest.
WIKI_TRIAGE_MODEL = "haiku"

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
# Bumped from 3 to 15 after probe-founders' run 105 stalled with 200
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
WIKI_AGENT_COMPACTOR_MODEL = "gemini-flash-lite-preview"

# Agent's CachedContent TTL. Re-create on miss; alert if hit_rate < 80%.
WIKI_AGENT_CACHE_TTL = "3600s"
