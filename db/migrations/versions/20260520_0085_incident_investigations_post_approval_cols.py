"""incident_investigations_post_approval_cols

Extends ``incident_investigations`` (created in 0080) with the columns
the post-approval downstream-actions pipeline needs:

- ``approved_at`` — set by the review approve path. The "investigation
  is approved AND incident has resolved" trigger reads this alongside
  ``resolved_at`` to decide whether to dispatch Pass 2.
- ``resolved_at`` — set by the incident.io/PD resolution-event handler.
- ``post_approval_dispatched_at`` — one-shot guard. Set inside the
  dispatch txn so concurrent triggers can't double-fire Pass 2.
- ``evidence_pack`` (jsonb) — cache of the Pass 1 EvidencePack. Pass 2
  consumes it directly; future re-runs reuse the same pack so postmortem
  authoring is deterministic w.r.t. the gathered evidence.

All four columns are nullable: existing rows pre-date the post-approval
pipeline and there is no sensible backfill value.

Revision ID: 0085_inv_post_approval
Revises: 0082_visibility_columns
Create Date: 2026-05-18
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers
revision = "0085_inv_post_approval"
down_revision = "0084_visibility_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "incident_investigations",
        sa.Column("approved_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "incident_investigations",
        sa.Column("resolved_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "incident_investigations",
        sa.Column(
            "post_approval_dispatched_at",
            sa.TIMESTAMP(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "incident_investigations",
        sa.Column("evidence_pack", JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("incident_investigations", "evidence_pack")
    op.drop_column("incident_investigations", "post_approval_dispatched_at")
    op.drop_column("incident_investigations", "resolved_at")
    op.drop_column("incident_investigations", "approved_at")
