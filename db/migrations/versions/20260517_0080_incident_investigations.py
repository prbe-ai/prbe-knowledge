"""incident_investigations

Per-incident review lifecycle state. The report content lives in
documents; this table is the small relational state side (current
state, version history JSONB, reviewer outcome).

`incident_doc_id` and `current_report_doc_id` are intentionally
unconstrained soft references — `documents` is SCD2-versioned and has
no single-row PK on `doc_id` alone. Cleanup on customer deprovision
runs through the `customer_id` CASCADE.

Revision ID: 0080_incident_investigations
Revises: 0079_search_agent_trace_blob
Create Date: 2026-05-17
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "0080_incident_investigations"
down_revision = "0079_search_agent_trace_blob"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "incident_investigations",
        sa.Column("customer_id", sa.Text(), nullable=False),
        sa.Column("incident_doc_id", sa.Text(), nullable=False),
        sa.Column("current_report_doc_id", sa.Text(), nullable=True),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column(
            "versions",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("reviewer_id", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("customer_id", "incident_doc_id"),
        sa.CheckConstraint(
            "state IN ('pending_dispatch','running','pending_review',"
            "'approved','rejected','failed_pending_review')",
            name="incident_investigations_state_chk",
        ),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.customer_id"], ondelete="CASCADE",
            name="incident_investigations_customer_fkey",
        ),
    )
    op.execute("ALTER TABLE incident_investigations ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE incident_investigations FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY tenant_isolation ON incident_investigations "
        "USING (customer_id = current_setting('app.current_customer_id', true)) "
        "WITH CHECK (customer_id = current_setting('app.current_customer_id', true))"
    )
    op.create_index(
        "incident_investigations_state_idx",
        "incident_investigations",
        ["customer_id", "state"],
    )


def downgrade() -> None:
    op.drop_index("incident_investigations_state_idx", table_name="incident_investigations")
    op.drop_table("incident_investigations")
