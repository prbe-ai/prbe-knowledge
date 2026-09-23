"""`ingestion_queue.extraction_outcome`: how a session's last mining pass went.

A coding-agent session has ended when an end signal is the newest key on its
queue row (engine/shared/session_signals.py), and the sweep leaves an ended
session alone. That is right after a complete mine and wrong after a partial
one: a pass that lost a segment, hit the cap, had the tool declined, or ran
with extraction switched off leaves the session "ended" with fewer units than
it has, and nothing would ever revisit it until a new batch arrived.

The worker records each complete pass here; the sweep re-queues the ones that
were not authoritative, a bounded number of times. Nullable and without a
default: NULL means "no pass recorded since this column existed", which the
sweep treats as nothing to retry. Adding a nullable column with no default is
a catalog-only change.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0138_queue_extraction_outcome"
down_revision = "0137_queue_first_enqueued_at"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ingestion_queue",
        sa.Column("extraction_outcome", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("ingestion_queue", "extraction_outcome")
