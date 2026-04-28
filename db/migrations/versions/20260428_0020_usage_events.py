"""usage_events: per-tenant retrieval audit trail (feed/stats/search)

Revision ID: 0020_usage_events
Revises: 0019_graph_nodes_loose_match
Create Date: 2026-04-28

Every call to /retrieve, /query, and /sources writes one row here from a
post-response BackgroundTask. The dashboard's /query/usage page reads these
rows back via three endpoints — feed (top-N within a window), stats
(percentile latency + counts), and search (FTS over the summary text).

Why a dedicated table (instead of audit_log):
  * audit_log is general-purpose JSON-blob; usage events have a fixed
    columnar shape (caller_kind, event_type, latency_ms, status) that
    drives stats/percentile queries. Giving it its own table keeps the
    /usage/stats SQL simple and indexable.
  * Retention policy will diverge: audit_log is forever; usage_events
    will eventually grow a 180d retention sweep (see TODOS.md P3).

RLS:
  * `app.current_customer_id` GUC (matches graph_nodes / graph_edges).
  * Set by `with_tenant()` at the start of every read AND write. The
    write path runs in a post-response BackgroundTask that calls
    `with_tenant(event.customer_id)` so failed auth never reaches this
    table.

Indexes:
  * (customer_id, occurred_at DESC) — primary feed access pattern
  * (customer_id, event_type, occurred_at DESC) — feed filtered by type
  * GIN (to_tsvector('simple', summary)) — FTS search

The 'simple' tsvector dictionary (vs 'english') keeps query strings
verbatim — no stemming, no stopword removal — because users searching
for "klavis" or "auth-refactor-pr-42" want exact tokens, not English
linguistic transforms. Matches plainto_tsquery('simple', $q) at read
time so callers can drop arbitrary text without tsquery-injection risk.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0020_usage_events"
down_revision = "0019_graph_nodes_loose_match"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "usage_events",
        sa.Column(
            "event_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "customer_id",
            sa.Text(),
            sa.ForeignKey("customers.customer_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "occurred_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("caller_kind", sa.Text(), nullable=False),
        sa.Column("caller_subject", sa.Text(), nullable=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("endpoint", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error_class", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("result_count", sa.Integer(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )

    op.create_index(
        "idx_usage_events_customer_time",
        "usage_events",
        ["customer_id", sa.text("occurred_at DESC")],
    )
    op.create_index(
        "idx_usage_events_customer_type_time",
        "usage_events",
        ["customer_id", "event_type", sa.text("occurred_at DESC")],
    )
    op.execute(
        """
        CREATE INDEX idx_usage_events_search
            ON usage_events
            USING gin (to_tsvector('simple', summary))
        """
    )

    op.execute("ALTER TABLE usage_events ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE usage_events FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY usage_events_tenant_isolation ON usage_events
            USING (customer_id = current_setting('app.current_customer_id', true))
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS usage_events_tenant_isolation ON usage_events")
    op.execute("DROP INDEX IF EXISTS idx_usage_events_search")
    op.drop_index("idx_usage_events_customer_type_time", table_name="usage_events")
    op.drop_index("idx_usage_events_customer_time", table_name="usage_events")
    op.drop_table("usage_events")
