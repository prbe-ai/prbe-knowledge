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
    db_pool_max_size: int = 10
    db_statement_timeout_ms: int = 30_000

    # --- Object storage (R2 in prod, MinIO locally) -------------------------
    r2_endpoint_url: str = "http://localhost:9000"
    r2_access_key_id: str = "minioadmin"
    r2_secret_access_key: SecretStr = SecretStr("minioadmin")
    r2_region: str = "auto"
    r2_bucket_prefix: str = "prbe-knowledge"

    # --- External model providers -------------------------------------------
    openai_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")
    google_api_key: SecretStr = SecretStr("")

    # --- Token encryption (Fernet key, 32 url-safe base64 bytes) -----------
    token_encryption_key: SecretStr = SecretStr("")

    # --- Admin API (gates /admin/* routes on the ingestion service) ---------
    # Unset → /admin/* returns 503. Set to a high-entropy token; compared with
    # hmac.compare_digest to avoid timing-based token recovery.
    admin_api_key: SecretStr | None = None

    # --- Dashboard redirect (optional) --------------------------------------
    # When set, OAuth callbacks 302 to <url>/oauth-landed?source=...&customer_id=...
    # instead of rendering the backend's own HTML success page, so users land
    # back on the dashboard after connecting an integration.
    dashboard_base_url: str | None = None

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

    sentry_client_id: str | None = None
    sentry_client_secret: SecretStr | None = None
    sentry_webhook_secret: SecretStr | None = None  # unused in prod

    # --- Worker tuning ------------------------------------------------------
    worker_poll_interval_seconds: float = 1.0
    worker_max_concurrent: int = 4
    worker_max_attempts: int = 5

    # --- HTTP / outbound ----------------------------------------------------
    http_timeout_seconds: float = 30.0

    # --- Embeddings batching ------------------------------------------------
    embedding_batch_size: int = Field(default=96, ge=1, le=2048)

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
