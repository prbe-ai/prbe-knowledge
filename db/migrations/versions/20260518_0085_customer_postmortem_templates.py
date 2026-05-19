"""customer_postmortem_templates

Per-customer postmortem template override. Two modes:

- ``inline`` — ``body_markdown`` holds the template directly.
- ``doc_ref`` — ``ref_doc_id`` points at an existing prbe-knowledge
  document whose body is fetched at postmortem-render time (lets ops
  keep the template alongside their other wiki content).

The default template lives as a code constant in
``shared/templates/postmortem.py`` (added in a later task); this table
holds OVERRIDES only — missing row means "use the default template".

The mode-consistency check enforces the two modes are mutually exclusive:
each row has exactly one of ``body_markdown`` or ``ref_doc_id`` populated.

RLS: tenant_isolation matching every other tenant table.

Revision ID: 0085_postmortem_templates
Revises: 0084_wiki_review_queue
Create Date: 2026-05-18
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision = "0085_postmortem_templates"
down_revision = "0084_wiki_review_queue"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "customer_postmortem_templates",
        sa.Column("customer_id", sa.Text(), nullable=False),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column("body_markdown", sa.Text(), nullable=True),
        sa.Column("ref_doc_id", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("customer_id"),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.customer_id"], ondelete="CASCADE",
            name="customer_postmortem_templates_customer_fkey",
        ),
        sa.CheckConstraint(
            "mode IN ('inline','doc_ref')",
            name="customer_postmortem_templates_mode_chk",
        ),
        sa.CheckConstraint(
            "(mode = 'inline' AND body_markdown IS NOT NULL AND ref_doc_id IS NULL) "
            "OR (mode = 'doc_ref' AND ref_doc_id IS NOT NULL AND body_markdown IS NULL)",
            name="customer_postmortem_templates_mode_consistency_chk",
        ),
    )
    op.execute(
        "ALTER TABLE customer_postmortem_templates ENABLE ROW LEVEL SECURITY"
    )
    op.execute(
        "ALTER TABLE customer_postmortem_templates FORCE ROW LEVEL SECURITY"
    )
    op.execute(
        "CREATE POLICY tenant_isolation ON customer_postmortem_templates "
        "USING (customer_id = current_setting('app.current_customer_id', true)) "
        "WITH CHECK (customer_id = current_setting('app.current_customer_id', true))"
    )


def downgrade() -> None:
    op.drop_table("customer_postmortem_templates")
