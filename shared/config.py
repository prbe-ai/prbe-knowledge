"""Process config. All runtime settings load from env (`.env.local` in dev).

One Settings object per process. Access via `get_settings()` (cached).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "dev", "staging", "main"]


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
    database_url: str = "postgresql://prbe:prbe@localhost:5432/prbe_knowledge"
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
    db_init_retry_attempts: int = 6
    db_init_retry_base_seconds: float = 1.0

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

    # --- Token encryption (Fernet key, 32 url-safe base64 bytes) -----------
    token_encryption_key: SecretStr = SecretStr("")

    # --- Internal-knowledge API key -----------------------------------------
    # Shared secret for service-to-service trust within the prbe-knowledge
    # platform. Sent as `X-Internal-Knowledge-Key`. Used by the dashboard to
    # gate /admin/* routes (503 when unset) and by orchestrator/MCP to call
    # retrieval + /internal/ingest. Compared with hmac.compare_digest.
    internal_knowledge_api_key: SecretStr | None = None

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

    github_app_id: str | None = None
    github_app_private_key: SecretStr | None = None
    github_app_slug: str | None = None  # unused in prod (gateway builds install URL)
    github_webhook_secret: SecretStr | None = None  # unused in prod

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
    # Soft per-customer cap on simultaneously processing rows. Sized so one
    # workspace's install-time burst can't monopolize the fleet; spare slots
    # go to other customers. With 18 machines * 4 concurrency = 72 slots,
    # 10 per customer means at least 7 customers can have headroom in
    # parallel. Snapshot count, not a hard lock — slight over-spill under
    # racing claims is fine.
    worker_per_customer_max_inflight: int = 10

    # --- HTTP / outbound ----------------------------------------------------
    http_timeout_seconds: float = 30.0

    # --- Embeddings batching ------------------------------------------------
    embedding_batch_size: int = Field(default=256, ge=1, le=2048)

    # --- Claude Code session completer --------------------------------------
    claude_code_session_idle_minutes: int = Field(default=5, ge=1)

    @property
    def is_local(self) -> bool:
        return self.environment == "local"

    def bucket_for(self, customer_id: str) -> str:
        """Per-tenant bucket naming. Kept in one place so bootstrap + runtime agree."""
        return f"{self.r2_bucket_prefix}-{customer_id}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent
