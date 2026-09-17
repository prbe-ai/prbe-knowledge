"""`ingestion_queue.first_enqueued_at`: when the work ARRIVED, not when it was
last touched.

`enqueued_at` is bumped by every transcript batch (the session UPSERT does it
so `session_completer` can read MAX(enqueued_at) as an idle signal). That makes
it useless as a backlog-age signal: an active session whose batches keep
arriving looks permanently young, so a queue that is not draining reads as a
queue with nothing old in it. The 33-minute median wait on 2026-09-16 was found
by hand for exactly this reason -- nothing could have alerted on it.

Set once, on insert, and never updated. Existing rows backfill to `enqueued_at`,
which is the best available estimate and is exactly right for every row that
was never re-enqueued.

Also drops the `priority` column default from 100 to 75. Under the tier table
0136 installed, 100 now means PRIORITY_RESEARCH_CONTENT -- so a column default
of 100 would silently give the top tier to any insert that forgot to name one.
Every insert in the codebase names one; this is about the next one.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0137_queue_first_enqueued_at"
down_revision = "0136_retier_pending_queue_rows"
branch_labels = None
depends_on = None

_INDEX = "ingestion_queue_pending_first_enqueued_idx"


def upgrade() -> None:
    op.add_column(
        "ingestion_queue",
        sa.Column(
            "first_enqueued_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    # The server_default stamped every existing row with the migration's own
    # clock, which would report a fleet-wide backlog age of zero at exactly the
    # moment someone starts trusting the number.
    op.execute("UPDATE ingestion_queue SET first_enqueued_at = enqueued_at")
    # 100 used to mean "a live webhook", the safe default. It now means
    # "research content", the TOP tier -- see 0136.
    op.execute("ALTER TABLE ingestion_queue ALTER COLUMN priority SET DEFAULT 75")
    with op.get_context().autocommit_block():
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_INDEX} "
            "ON ingestion_queue (first_enqueued_at) "
            "WHERE status IN ('pending', 'processing')"
        )


def downgrade() -> None:
    op.execute("ALTER TABLE ingestion_queue ALTER COLUMN priority SET DEFAULT 100")
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_INDEX}")
    op.drop_column("ingestion_queue", "first_enqueued_at")
