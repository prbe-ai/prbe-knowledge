"""query_traces: full request/response payload log per retrieval call

Revision ID: 0030_query_traces
Revises: 0029_agent_kind_subject_kind
Create Date: 2026-05-02

Sister table to usage_events. Where usage_events stores thin metrics
(latency, status, caller_kind, summary text), query_traces stores the
full parsed request body and response body so we can evaluate retrieval
effectiveness over time — zero-result rate, retrieve->get_source
click-through, score distributions, query repeats, latency-vs-quality
drift.

Why split from usage_events:
  * Rows here are ~50x fatter (full JSONB request + response). Mixing
    them into usage_events.metadata bloats the index heap pages for the
    /usage/feed and /usage/stats hot paths.
  * Retention windows may diverge: usage_events at 180d (TODOS.md), but
    query_traces could justify shorter windows or selective per-event
    retention later.
  * The two tables share `request_id` and a 1:1 row relationship under
    normal operation, so joining is cheap. UNIQUE on request_id was
    considered and rejected: clients may supply X-Request-Id and a
    misbehaving retry would silently lose the second trace under a
    UNIQUE constraint.

Schema choices:
  * `response_truncated boolean` (vs a JSONB sentinel like
    {"_truncated": true}) so consumers can distinguish a stub row from
    a real response that happens to contain a `_truncated` key.
  * `schema_version smallint` so future shape changes can ALTER without
    rewriting old rows.
  * Same RLS pattern as usage_events: ENABLE + FORCE + tenant_isolation
    policy on `app.current_customer_id`.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0030_query_traces"
down_revision = "0029_agent_kind_subject_kind"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "query_traces",
        sa.Column(
            "trace_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "request_id",
            postgresql.UUID(as_uuid=True),
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
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column(
            "schema_version",
            sa.SmallInteger(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column("request", postgresql.JSONB(), nullable=False),
        sa.Column("response", postgresql.JSONB(), nullable=False),
        sa.Column("response_size_bytes", sa.Integer(), nullable=False),
        sa.Column(
            "response_truncated",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )

    op.create_index(
        "idx_query_traces_customer_time",
        "query_traces",
        ["customer_id", sa.text("occurred_at DESC")],
    )
    # Plain BTREE (NOT UNIQUE) — request_id can be supplied by clients via
    # X-Request-Id; a buggy double-fire under UNIQUE would silently drop
    # the second trace, hiding exactly the retry pattern we'd want to study.
    op.create_index(
        "idx_query_traces_request_id",
        "query_traces",
        ["request_id"],
    )

    op.execute("ALTER TABLE query_traces ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE query_traces FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY query_traces_tenant_isolation ON query_traces
            USING (customer_id = current_setting('app.current_customer_id', true))
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS query_traces_tenant_isolation ON query_traces")
    op.drop_index("idx_query_traces_request_id", table_name="query_traces")
    op.drop_index("idx_query_traces_customer_time", table_name="query_traces")
    op.drop_table("query_traces")
