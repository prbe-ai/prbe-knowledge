"""incident_investigations_metadata

Adds a ``metadata`` JSONB column to ``incident_investigations``. The
post-approval dispatch seam (``services/post_approval/dispatch.py``)
records non-fatal lifecycle flags here — most importantly
``post_approval_dispatch_failed=true`` when the orchestrator POST hit
retry exhaustion, so the dashboard's "Re-trigger" recovery flow can
surface failed dispatches without scanning logs.

Defaulted to ``'{}'::jsonb`` so existing rows + new inserts both
satisfy NOT NULL. The default makes the COALESCE in the dispatcher's
metadata-mutation SQL technically redundant, but the COALESCE stays for
defense-in-depth (in case a future row is inserted without the default
honored — e.g. raw COPY paths).

Revision ID: 0086_inv_metadata
Revises: 0085_postmortem_templates
Create Date: 2026-05-18
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers
revision = "0086_inv_metadata"
down_revision = "0085_postmortem_templates"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "incident_investigations",
        sa.Column(
            "metadata",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("incident_investigations", "metadata")
