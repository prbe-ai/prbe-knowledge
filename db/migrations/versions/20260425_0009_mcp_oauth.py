"""mcp_oauth: client/code/refresh-token tables for the MCP OAuth provider

Revision ID: 0009_mcp_oauth
Revises: 0008_enrichment_runs
Create Date: 2026-04-25

prbe-backend acts as the OAuth 2.1 issuer for prbe-knowledge-mcp.
Customer AI agents (Claude Desktop, Cursor, etc.) register dynamically
via RFC 7591, walk an authorization-code-with-PKCE flow on
api.prbe.ai/oauth/{authorize,token}, and present the issued JWT to
mcp.prbe.ai.

Three small tables. All live in the shared Neon DB so the backend
(issuer) and the MCP server (validator) can both read them when needed
— though the JWT itself is self-contained, so MCP doesn't need any of
these on the hot path. They exist for issuance, refresh, and audit.

Prefix is `mcp_oauth_*` to keep this surface separate from the
per-source integration OAuth flow (Linear/Slack/etc), which doesn't
have its own tables — those tokens land in `integration_tokens`.
"""

from __future__ import annotations

from alembic import op

revision = "0009_mcp_oauth"
down_revision = "0008_enrichment_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        -- Registered MCP clients (auto-approved RFC 7591 dynamic registration).
        CREATE TABLE mcp_oauth_clients (
            client_id                  TEXT PRIMARY KEY,
            client_name                TEXT NOT NULL,
            redirect_uris              TEXT[] NOT NULL,
            grant_types                TEXT[] NOT NULL
                                       DEFAULT ARRAY['authorization_code','refresh_token'],
            response_types             TEXT[] NOT NULL DEFAULT ARRAY['code'],
            token_endpoint_auth_method TEXT NOT NULL DEFAULT 'none',  -- public PKCE clients
            software_id                TEXT,
            software_version           TEXT,
            scope                      TEXT NOT NULL DEFAULT 'mcp:read',
            created_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        -- Short-lived (5 min) authorization codes. Single-use; PKCE required.
        CREATE TABLE mcp_oauth_codes (
            code                  TEXT PRIMARY KEY,
            client_id             TEXT NOT NULL
                                  REFERENCES mcp_oauth_clients(client_id) ON DELETE CASCADE,
            user_id               TEXT NOT NULL,         -- Neon Auth user id (for audit)
            customer_id           TEXT NOT NULL
                                  REFERENCES customers(customer_id) ON DELETE CASCADE,
            redirect_uri          TEXT NOT NULL,
            code_challenge        TEXT NOT NULL,
            code_challenge_method TEXT NOT NULL,         -- 'S256' (we reject 'plain')
            scope                 TEXT NOT NULL,
            issued_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at            TIMESTAMPTZ NOT NULL,
            used_at               TIMESTAMPTZ            -- single-use; non-null after exchange
        );
        CREATE INDEX mcp_oauth_codes_expires_at ON mcp_oauth_codes(expires_at);

        -- Long-lived (default 30 days) refresh tokens. Stored as sha256(token).
        CREATE TABLE mcp_oauth_refresh_tokens (
            token_id     TEXT PRIMARY KEY,               -- sha256(refresh_token), hex
            client_id    TEXT NOT NULL
                         REFERENCES mcp_oauth_clients(client_id) ON DELETE CASCADE,
            user_id      TEXT NOT NULL,
            customer_id  TEXT NOT NULL
                         REFERENCES customers(customer_id) ON DELETE CASCADE,
            scope        TEXT NOT NULL,
            issued_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at   TIMESTAMPTZ NOT NULL,
            revoked_at   TIMESTAMPTZ                     -- non-null after rotation/revoke
        );
        CREATE INDEX mcp_oauth_refresh_tokens_user
            ON mcp_oauth_refresh_tokens(user_id, customer_id)
            WHERE revoked_at IS NULL;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS mcp_oauth_refresh_tokens CASCADE;
        DROP TABLE IF EXISTS mcp_oauth_codes CASCADE;
        DROP TABLE IF EXISTS mcp_oauth_clients CASCADE;
        """
    )
