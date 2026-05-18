"""customer_incident_mcp_servers

Per-customer external MCP server config used by the incident
investigation agent (orchestrator side). Each row carries the URL +
Fernet-encrypted auth token for an MCP server (Sentry, Datadog,
Cloudflare, k8s, or a per-customer Knowledge override). The agent's
`incident_mcp_config.get_enabled_for_customer` reads this table at run
time and composes a multi-MCP toolset.

The schema lives in prbe-knowledge's alembic chain even though the
readers live in prbe-orchestrator, because the two services share one
Postgres database (see app/db/runs.py comment in prbe-orchestrator).

Revision ID: 0081_incident_mcp_servers
Revises: 0080_incident_investigations
Create Date: 2026-05-17
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision = "0081_incident_mcp_servers"
down_revision = "0080_incident_investigations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "customer_incident_mcp_servers",
        sa.Column("customer_id", sa.Text(), nullable=False),
        sa.Column("mcp_kind", sa.Text(), nullable=False),
        sa.Column("mcp_url", sa.Text(), nullable=False),
        sa.Column("secret_ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column(
            "enabled", sa.Boolean(), nullable=False, server_default=sa.text("true"),
        ),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True),
            nullable=False, server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at", sa.TIMESTAMP(timezone=True),
            nullable=False, server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("customer_id", "mcp_kind"),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.customer_id"], ondelete="CASCADE",
            name="customer_incident_mcp_servers_customer_fkey",
        ),
        sa.CheckConstraint(
            "mcp_kind IN ('sentry','datadog','cloudflare','k8s','knowledge')",
            name="customer_incident_mcp_servers_kind_chk",
        ),
    )
    op.execute(
        "ALTER TABLE customer_incident_mcp_servers ENABLE ROW LEVEL SECURITY"
    )
    op.execute(
        "ALTER TABLE customer_incident_mcp_servers FORCE ROW LEVEL SECURITY"
    )
    op.execute(
        "CREATE POLICY tenant_isolation ON customer_incident_mcp_servers "
        "USING (customer_id = current_setting('app.current_customer_id', true)) "
        "WITH CHECK (customer_id = current_setting('app.current_customer_id', true))"
    )


def downgrade() -> None:
    op.drop_table("customer_incident_mcp_servers")
