from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "dev"
    log_level: str = "INFO"
    service_name: str = "prbe-knowledge-mcp"

    # ---- Auth mode switch (hosted vs community) ----------------------------
    # `oauth`   — hosted/managed: validate OAuth 2.1 JWTs against the issuer's
    #             JWKS (the Phase-G fields below). Existing behavior; the
    #             default whenever OAuth env is present.
    # `static`  — community self-host: a single shared bearer (MCP_API_TOKEN)
    #             scoped to DEFAULT_CUSTOMER_ID. No issuer, no JWKS, no
    #             control-plane callbacks.
    # Empty = auto: `static` when MCP_API_TOKEN is set AND no JWKS URL is
    # configured; otherwise `oauth`. Set MCP_AUTH_MODE to override.
    mcp_auth_mode: str = Field(
        default="",
        description="'oauth' | 'static' | '' (auto — see resolved_auth_mode)",
    )
    # Community-mode shared bearer (Authorization: Bearer <token>), compared
    # constant-time. Ignored in oauth mode.
    mcp_api_token: str = Field(
        default="",
        description="Static bearer for community (static) auth mode",
    )
    # Tenant id requests resolve to in static single-tenant mode. Mirrors the
    # engine's DEFAULT_CUSTOMER_ID single-tenant default (Spec A Part 4).
    default_customer_id: str = Field(
        default="default",
        description="customer_id used in static single-tenant mode",
    )

    # ---- Inbound auth (Phase D fallback) -----------------------------------
    # Internal-key auth for prbe-backend → MCP. Phase G accepts OAuth
    # JWTs primarily; this stays as a dev/test escape hatch.
    internal_backend_api_key: str = Field(
        default="",
        description="Shared secret prbe-backend presents on X-Internal-Backend-Key",
    )

    # ---- Inbound auth (Phase G primary) ------------------------------------
    # Customer AI agents present an OAuth 2.1 access token issued by api.knowledge.prbe.ai.
    # We validate signature against the issuer's JWKS and check iss/aud.
    mcp_oauth_jwks_url: str = Field(
        default="",
        description="JWKS endpoint of the OAuth issuer (e.g. https://api.knowledge.prbe.ai/oauth/jwks)",
    )
    mcp_oauth_issuer: str = Field(
        default="https://api.knowledge.prbe.ai",
        description="Required `iss` claim on incoming JWTs",
    )
    mcp_oauth_audience: str = Field(
        default="https://mcp.knowledge.prbe.ai",
        description="Required `aud` claim on incoming JWTs",
    )
    mcp_oauth_jwks_ttl_s: int = Field(
        default=600,
        description="Cache TTL for the issuer's JWKS",
    )

    # ---- Outbound (to prbe-knowledge) -------------------------------------
    knowledge_query_url: str = Field(
        default="",
        description="prbe-knowledge retrieval base URL (serves /retrieve, /query, /sources)",
    )
    knowledge_timeout_s: float = Field(
        # The upstream envelope can spend 30s extracting entities, 90s in
        # the gatherer loop (including each 60s provider-chain client call),
        # and another 30s synthesizing /query responses. Leave headroom for
        # grounding, fan-out, and response serialization.
        default=180.0,
        description="HTTP read timeout for retrieval calls; exceeds upstream LLM budgets",
    )
    knowledge_connect_timeout_s: float = Field(
        default=10.0,
        description="HTTP connect timeout for retrieval calls",
    )
    knowledge_write_timeout_s: float = Field(
        default=30.0,
        description="HTTP request-body write timeout for retrieval calls",
    )
    knowledge_pool_timeout_s: float = Field(
        default=10.0,
        description="HTTP connection-pool wait timeout for retrieval calls",
    )
    internal_knowledge_api_key: str = Field(
        default="",
        description="X-Internal-Knowledge-Key sent to prbe-knowledge",
    )

    # ---- Revocation check (oauth mode) ------------------------------------
    # Access tokens are non-expiring, so the issuer can't claw them back. We
    # enforce revocation here: on each request, ask prbe-backend whether the
    # token's `sid` session is still alive. Reuses the BACKEND_BASE_URL the MCP
    # service is already given (same host it fetches JWKS from).
    #
    # Gated behind an explicit flag (default OFF) so the check is dormant until
    # deliberately enabled — decoupled from BACKEND_BASE_URL (which is already
    # populated in managed). This lets the code roll out with zero behavior
    # change, then enforcement is switched on via config once the issuer is
    # confirmed deployed and tokens carry `sid`. Managed-shared sets it "true"
    # in the chart configmap; community/static self-host leaves it off.
    mcp_revocation_check_enabled: bool = Field(
        default=False,
        description="Enable the per-request session-liveness (revocation) check. Off by default.",
    )
    backend_base_url: str = Field(
        default="",
        description="prbe-backend in-cluster base URL for /oauth/introspect (BACKEND_BASE_URL). Empty also disables the revocation check.",
    )
    revocation_check_timeout_s: float = Field(
        default=3.0,
        description="HTTP timeout for the per-request session-liveness check; fail-closed (reject) on timeout/error",
    )

    @property
    def resolved_auth_mode(self) -> str:
        """The effective auth mode after applying the auto rule.

        Explicit MCP_AUTH_MODE wins. Otherwise: community `static` only when a
        static token is configured and no JWKS issuer is — so a hosted deploy
        (JWKS set, no static token) always resolves to `oauth` and is never
        affected by this switch.
        """
        mode = (self.mcp_auth_mode or "").strip().lower()
        if mode in ("oauth", "static"):
            return mode
        if self.mcp_api_token and not self.mcp_oauth_jwks_url:
            return "static"
        return "oauth"


@lru_cache
def get_settings() -> Settings:
    return Settings()
