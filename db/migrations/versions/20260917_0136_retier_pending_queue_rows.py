"""Move already-queued rows onto the new tier table.

The tiers changed (shared.constants.PRIORITY_*): research content 100, live
integrations 75, agent captures 60, background 50. `priority` is stamped at
enqueue and read at claim, so a deploy alone re-tiers nothing already in the
queue -- and a transcript session row, which is created once and resumed for
the life of the session, would keep the tier it was born with indefinitely.

Only `pending` rows. A `processing` row is mid-flight and its tier no longer
decides anything; a `done` row is history and rewriting it would falsify what
the queue actually did.

Deliberately NOT data-loss shaped: worst case a row keeps its old number and
claims slightly out of order until it drains.
"""

from __future__ import annotations

from alembic import op

revision = "0136_retier_pending_queue_rows"
down_revision = "0135_fk_support_indexes"
branch_labels = None
depends_on = None

# Written out rather than imported from shared.constants on purpose: a
# migration records what was true when it ran. If the tier table moves again,
# this migration must keep applying the numbers it was written for, and the
# NEXT migration carries the next set.
_TIERS = {
    100: ("custom_ingest",),
    75: (
        "slack", "github", "linear", "notion", "granola", "sentry",
        "pagerduty", "incident_io", "manual_uploads",
    ),
    60: ("claude_code", "codex", "pi"),
    50: ("code_graph",),
}


def upgrade() -> None:
    for priority, sources in _TIERS.items():
        listed = ", ".join(f"'{s}'" for s in sources)
        op.execute(
            f"UPDATE ingestion_queue SET priority = {priority} "
            f"WHERE status = 'pending' AND priority <> {priority} "
            f"AND source_system = ANY(ARRAY[{listed}])"
        )


def downgrade() -> None:
    # The pre-change table: everything agent-shaped at 75, everything else at
    # its old default of 100, backfill-tier at 50.
    op.execute(
        "UPDATE ingestion_queue SET priority = 75 "
        "WHERE status = 'pending' "
        "AND source_system = ANY(ARRAY['custom_ingest', 'claude_code', 'codex', 'pi'])"
    )
    op.execute(
        "UPDATE ingestion_queue SET priority = 100 "
        "WHERE status = 'pending' "
        "AND source_system = ANY(ARRAY['slack', 'github', 'linear', 'notion', "
        "'granola', 'sentry', 'pagerduty', 'incident_io', 'manual_uploads'])"
    )
