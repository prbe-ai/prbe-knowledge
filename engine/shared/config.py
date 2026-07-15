"""Process config. All runtime settings load from env (`.env.local` in dev).

One Settings object per process. Access via `get_settings()` (cached).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from engine.shared.constants import (
    DB_INIT_RETRY_ATTEMPTS,
    DB_INIT_RETRY_BACKOFF_CAP_SECONDS,
    DB_INIT_RETRY_BASE_SECONDS,
)

# "self-host" is what the community Helm chart ships as its default
# ENVIRONMENT (deploy/helm/values.yaml `config.environment`). It behaves
# like any other non-local environment (signature dev-bypasses stay off,
# boot-secret validation applies).
Environment = Literal["local", "dev", "staging", "main", "self-host"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env.local", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Environment = "local"
    log_level: str = "INFO"
    service_name: str = "prbe-knowledge"

    # --- Postgres ------------------------------------------------------------
    # Local dev default connects as the ``prbe`` superuser for migration
    # convenience. In every managed-shared deployment (staging / prod)
    # the operator-set DATABASE_URL secret MUST point at the
    # non-privileged ``probe_app`` role instead — superuser bypasses FORCE
    # RLS and would silently disable tenant isolation. The boot path
    # (``shared.db.init_pool``) emits ``db.superuser_in_managed_env`` if it
    # lands on a superuser anywhere ``environment != "local"``; see
    # ``docs/database-url-cutover.md`` for the operator switch.
    database_url: str = "postgresql://prbe:prbe@localhost:5432/prbe_knowledge"
    # Direct (non-pooler) DSN used ONLY by LISTEN/NOTIFY consumers. Neon's
    # pooler endpoint runs pgbouncer in transaction mode, which resets
    # session state ("UNLISTEN *; RESET ALL") between every transaction —
    # so a LISTEN registered on a pooler conn never receives any NOTIFY.
    # The fix: listeners hold a dedicated asyncpg conn against the direct
    # endpoint (same Neon project, hostname without the `-pooler` suffix).
    # Other queries continue to use database_url (pooled) for connection
    # efficiency. If unset, listeners fall back to database_url, preserving
    # local-dev + non-pooler deploys without a config burden.
    database_url_unpooled: str | None = None
    # RLS role guard (bug #46 follow-up). ``shared.db.init_pool`` always
    # WARN-logs in any non-local environment if DATABASE_URL connects as a
    # superuser OR a BYPASSRLS role — either one silently disables FORCE RLS
    # tenant isolation. When this flag is True it additionally FAILS CLOSED
    # (refuses to start) on such a role.
    #
    # Default is False (warn-only) deliberately: on 2026-07-10 the live
    # managed cluster showed the data-plane pools on the non-privileged
    # ``probe_app`` role (36 conns, rolsuper=f/rolbypassrls=f) BUT also 21
    # live connections as the ``probe`` superuser (idle health/auth probes —
    # `SELECT 1`, pgbouncer rolvaliduntil auth_query). Until every process
    # that calls init_pool (incl. the token-health CronJob) is confirmed to
    # use probe_app, a fail-closed default could crash-loop a pooled service.
    # Flip to enforcement per-env with REQUIRE_NON_SUPERUSER_DB=true once
    # that trace is done — the warn line is the signal to watch first.
    require_non_superuser_db: bool = False
    db_pool_min_size: int = 2
    # >= machine_count * WORKER_MAX_CONCURRENT so claim loops never queue on
    # the pool. 18 * 6 = 108 slots; 30 covers the steady-state working set
    # (claims hold a conn only briefly per heartbeat/commit).
    db_pool_max_size: int = 30
    # 5 min, not 30s. We're a write-heavy multi-tenant queue worker: a single
    # batched UPSERT can wait on row locks held by sibling workers operating
    # on the same hot graph nodes. 30s caused pre-batched-writes contention
    # to surface as TimeoutError → DLQ instead of just slow throughput. The
    # batched-writes shape (PR #40 + this change) means a single statement
    # genuinely shouldn't take this long; treat it as a backstop, not a SLO.
    # Stuck workers are still caught by cron_stuck_queue_reclaim.py via the
    # heartbeat (30s).
    db_statement_timeout_ms: int = 300_000
    # init_pool connect-with-backoff. Defaults tracked in shared.constants
    # (DB_INIT_RETRY_*) so the boot-retry ceiling is one explicit knob;
    # tightened for the post-migrate-sentinel world where retries only cover
    # transient blips. Override per-env via DB_INIT_RETRY_* env vars.
    db_init_retry_attempts: int = DB_INIT_RETRY_ATTEMPTS
    db_init_retry_base_seconds: float = DB_INIT_RETRY_BASE_SECONDS
    db_init_retry_backoff_cap_seconds: float = DB_INIT_RETRY_BACKOFF_CAP_SECONDS
    db_connect_timeout_seconds: float = 10.0
    # Heartbeat is liveness, NOT progress. The runner spawns a background task
    # that pings heartbeat_at every N seconds regardless of whether events are
    # being enqueued, so a healthy-but-paused runner (Slack rate limit, large
    # channel mid-paginate) can't be reclaimed while it's still alive. The
    # stale threshold is 6x the interval to absorb GC pauses + network blips.
    backfill_heartbeat_interval_seconds: int = 30
    backfill_stale_heartbeat_seconds: int = 180

    # --- Object storage (R2 in prod, MinIO locally) -------------------------
    r2_endpoint_url: str = "http://localhost:9000"
    r2_access_key_id: str = "minioadmin"
    r2_secret_access_key: SecretStr = SecretStr("minioadmin")
    r2_region: str = "auto"
    r2_bucket_prefix: str = "prbe-knowledge"

    # --- External model providers -------------------------------------------
    openai_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = Field(
        default=SecretStr("dev-only"),
        description="Used for claude_code unit extraction.",
    )
    google_api_key: SecretStr = SecretStr("")
    claude_code_extraction_model: str = Field(default="claude-sonnet-4-6")

    # --- LLM gateway (managed-shared / self-host: route LLM + embedding
    #     calls through a central LiteLLM proxy instead of direct provider
    #     SDKs — plan D1, `shared/llm.py`) ------------------------------------
    # When `llm_gateway_url` is set, callers that go through `shared.llm`
    # forward to it (LiteLLM's `api_base`) with `llm_gateway_key` as the
    # bearer (`api_key`). Empty `llm_gateway_url` => direct provider calls
    # using the `*_api_key` fields above (the self-host-with-own-keys path,
    # and dev). A key without a URL is ignored. (`shared.llm` reads these
    # from the environment, not this Settings object — these fields exist so
    # the values appear in one inventory + so non-`shared.llm` code can read
    # them too; the env var names are `LLM_GATEWAY_URL` / `LLM_GATEWAY_KEY`.)
    llm_gateway_url: str = ""
    llm_gateway_key: SecretStr = SecretStr("")

    # --- Token encryption (Fernet key, 32 url-safe base64 bytes) -----------
    token_encryption_key: SecretStr = SecretStr("")

    # --- Internal-knowledge API key -----------------------------------------
    # Shared secret for service-to-service trust within the prbe-knowledge
    # platform. Sent as `X-Internal-Knowledge-Key`. Used by the dashboard to
    # gate /admin/* routes (503 when unset) and by orchestrator/MCP to call
    # retrieval + /internal/ingest. Compared with hmac.compare_digest.
    internal_knowledge_api_key: SecretStr | None = None

    # --- prbe-backend (upstream identity service) ---------------------------
    # Knowledge calls backend's /internal/* endpoints for things only backend
    # should hold credentials for — most importantly, minting GitHub App
    # installation tokens (the App private key lives in backend, not here).
    # `backend_base_url` is typically `http://prbe-backend.internal:8080` over
    # Fly's 6PN private networking; `internal_backend_api_key` is shared with
    # backend and sent as `X-Internal-Backend-Key`.
    backend_base_url: str = ""
    internal_backend_api_key: SecretStr = SecretStr("")

    # --- Single-tenant community mode (self-host) ---------------------------
    # When the control plane is absent, the engine runs as a single tenant.
    # `default_customer_id` (e.g. "default") becomes the tenant for every
    # request: `with_tenant()` falls back to it (RLS stays ON, trivially
    # satisfied) and a `customers` row is seeded on boot. `knowledge_api_token`
    # is the static bearer for /query + /retrieve — its sha256 is seeded as the
    # default customer's `api_key_hash`, so the EXISTING bearer auth path
    # resolves it to `default_customer_id` with no special-casing. Both unset
    # => hosted multi-tenant behavior is unchanged (per-request tenant +
    # dashboard-minted bearer tokens + gateway trust).
    default_customer_id: str = ""
    knowledge_api_token: SecretStr = SecretStr("")

    # --- Per-source OAuth + webhook secrets ----
    # After the gateway migration, ACTIVE code in this service only uses
    # client_secrets (token exchange in /api/oauth/{source}/exchange). The
    # webhook signing secrets and OAuth client_ids no longer have an active
    # caller — they're owned by the gateway (prbe-backend) which verifies
    # signatures before forwarding and builds authorize URLs.
    #
    # The remaining fields below are kept as None-defaults because the
    # connector classes still expose `verify_signature` and `oauth_install_url`
    # methods that reference them. Those methods are dead in production
    # but still exercised by unit tests in tests/handlers/. Removing the
    # fields would force a surgery across all connectors; keeping them
    # None-default costs nothing.
    slack_client_id: str | None = None
    slack_client_secret: SecretStr | None = None
    slack_signing_secret: SecretStr | None = None  # unused in prod (gateway verifies)

    # GitHub App credentials. In HOSTED mode the App private key lives in
    # prbe-backend and knowledge fetches installation tokens over HTTP via
    # `shared.backend_client.fetch_github_installation_token`. In STANDALONE
    # community mode (no BACKEND_BASE_URL) the self-hoster registers their own
    # GitHub App and the engine mints installation tokens locally via
    # `shared.github_app` using `github_app_id` + `github_app_private_key`.
    # Both paths share the same call site; the branch is in backend_client.
    github_app_slug: str | None = None  # unused in prod (gateway builds install URL)
    github_webhook_secret: SecretStr | None = None  # in-process webhook verify (standalone)
    github_app_id: str = ""  # standalone: numeric GitHub App id for local token minting
    github_app_private_key: SecretStr = SecretStr("")  # standalone: App private key (PEM)

    linear_client_id: str | None = None
    linear_client_secret: SecretStr | None = None
    linear_webhook_secret: SecretStr | None = None  # unused in prod

    notion_client_id: str | None = None
    notion_client_secret: SecretStr | None = None
    # Per-subscription token Notion sends in the verification handshake (paste
    # it back into Notion's UI to verify, then set this secret). Once set,
    # every Notion webhook is HMAC-SHA256 signed with this token in the
    # `X-Notion-Signature` header. Unset => connector accepts unsigned in
    # local dev only and rejects in prod (see verify_signature).
    notion_webhook_verification_token: SecretStr | None = None

    sentry_client_id: str | None = None
    sentry_client_secret: SecretStr | None = None
    sentry_webhook_secret: SecretStr | None = None  # unused in prod

    # --- Worker tuning ------------------------------------------------------
    worker_poll_interval_seconds: float = 1.0
    # Parallel claim loops per worker process. Embedding is I/O-bound, so
    # one vCPU can fan out many concurrent OpenAI calls. Set in
    # fly.worker.toml to size the fleet against the OpenAI tier's TPM cap:
    # Tier 1 (1M TPM) → 2; Tier 5 (10M TPM) → 6 today (OpenAI permits 8+;
    # 3gb VM memory is the actual ceiling — resize before pushing higher).
    worker_max_concurrent: int = 2
    # Effectively retry-forever for transient errors (TimeoutError, lock
    # waits, network blips). The queue is the buffer; rows that hit the
    # ceiling here are data we silently dropped on the customer's behalf,
    # which is the worst possible failure mode. Permanent-DLQ-on-first-try
    # is still possible for deterministic errors via PrbeError(transient=
    # False) — that path is unchanged.
    worker_max_attempts: int = 50
    # Soft per-customer cap on simultaneously processing rows. Original
    # value (10) was conservative against the per-row-loop contention model
    # in graph_writer/normalizer that PR #41 retired (batched writes +
    # sorted lock order + 5min timeout + 50 retries). With those layers in
    # place, 30 is comfortable: 30 contending txs on a hot node serialize
    # at ~5ms per acquisition, well under the 5min ceiling. Sized to keep
    # at least 2-3 customers' worth of headroom against the 108-slot
    # fleet (18 machines * 6 concurrency) when several burst at once.
    # Snapshot count, not a hard lock — slight over-spill under racing
    # claims is fine.
    worker_per_customer_max_inflight: int = 30

    # GitHub backfill: how many repos walk in parallel via the GraphQL v4
    # engine. Each walker holds one in-flight POST /graphql at a time; sized
    # to stay well under the 5000-points/hour cap at ~3 points/page.
    github_backfill_repo_concurrency: int = 4

    # --- Backfill batching --------------------------------------------------
    # Backfill consumer batches events: one R2 gather (16-wide) + one asyncpg
    # executemany per batch (pipelined prepared inserts, ~5-10x fewer
    # round-trips end-to-end). 100 keeps batches small enough that a single
    # failure rolls back at most ~100 events. Upper bound 500 caps in-memory
    # payload buffer at ~500MB worst-case (one heavy PR body per event), well
    # under the 1Gi ingestion-pod request.
    backfill_batch_size: int = Field(default=100, ge=1, le=500)

    # --- HTTP / outbound ----------------------------------------------------
    http_timeout_seconds: float = 30.0

    # --- Embeddings batching ------------------------------------------------
    embedding_batch_size: int = Field(default=256, ge=1, le=2048)

    # --- Claude Code session completer --------------------------------------
    claude_code_session_idle_minutes: int = Field(default=5, ge=1)

    # --- Custom ingest -------------------------------------------------------
    custom_ingest_max_request_bytes: int = Field(default=5_000_000, ge=1)
    custom_ingest_max_body_bytes: int = Field(default=1_000_000, ge=1)
    custom_ingest_max_metadata_bytes: int = Field(default=64_000, ge=1)

    # --- Device source reconciliation
    # When True, verify_device_token escalates an integration_tokens row's
    # source_system from the default "claude_code" to a non-default source
    # if the webhook route says otherwise. Escalate-only — never demotes.
    # Toggle off only during incident triage; mislabeled rows then require
    # manual SQL or re-pair to correct.
    auto_reconcile_device_source: bool = Field(
        default=True,
        description=(
            "Promote mislabeled device rows on first webhook hit. "
            "Toggle off only during incident triage; mislabeled rows then "
            "require manual SQL or re-pair to correct."
        ),
    )

    @property
    def is_local(self) -> bool:
        return self.environment == "local"

    def bucket_for(self, customer_id: str) -> str:
        """Per-tenant bucket naming. Kept in one place so bootstrap + runtime agree."""
        return f"{self.r2_bucket_prefix}-{customer_id}"


# Placeholder secret values shipped in deploy/helm/values.yaml. Empty/unset
# secrets are allowed (they disable the corresponding feature); a placeholder
# silently *enables* the feature with a publicly-known credential, which is
# strictly worse — refuse to boot instead. Deliberately NOT included:
# docker-compose defaults ("minioadmin", "local-internal-key", ...) — the
# compose stack pins ENVIRONMENT=local, where this check never runs.
_PLACEHOLDER_SECRET_VALUES = frozenset({"CHANGEME", "changeme"})


def validate_boot_secrets(settings: Settings) -> None:
    """Refuse to boot outside ``local`` when a security-critical secret is
    still a known shipped placeholder (helm values.yaml "CHANGEME").

    Called from ``shared.db.init_pool`` so every service entrypoint gets it
    without per-lifespan wiring. Managed-shared / hosted deployments set
    real values via operator-managed Secrets (never these placeholders),
    so this cannot fire there.
    """
    if settings.environment == "local":
        return
    checked: dict[str, str] = {
        "KNOWLEDGE_API_TOKEN": settings.knowledge_api_token.get_secret_value(),
        "TOKEN_ENCRYPTION_KEY": settings.token_encryption_key.get_secret_value(),
        "INTERNAL_KNOWLEDGE_API_KEY": (
            settings.internal_knowledge_api_key.get_secret_value()
            if settings.internal_knowledge_api_key is not None
            else ""
        ),
        "R2_ACCESS_KEY_ID": settings.r2_access_key_id,
        "R2_SECRET_ACCESS_KEY": settings.r2_secret_access_key.get_secret_value(),
        "GOOGLE_API_KEY": settings.google_api_key.get_secret_value(),
    }
    offending = sorted(
        name
        for name, value in checked.items()
        if value.strip() in _PLACEHOLDER_SECRET_VALUES
    )
    if offending:
        raise RuntimeError(
            "refusing to start: placeholder value(s) still set for "
            f"security-critical secret(s) {', '.join(offending)} in "
            f"environment '{settings.environment}'. Replace the CHANGEME "
            "defaults from deploy/helm/values.yaml with real secrets "
            "(see .env.example for how to generate each), or set "
            "ENVIRONMENT=local for local development."
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def repo_root() -> Path:
    # engine/shared/config.py -> engine/shared -> engine -> repo root
    return Path(__file__).resolve().parent.parent.parent
