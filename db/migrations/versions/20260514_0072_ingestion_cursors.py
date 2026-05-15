"""ingestion_cursors — per-customer-per-source-per-resource cursor state.

For self-host customers (and any customer with `INGESTION_MODE=poll`),
ingestion is driven by a polling loop in the worker pod, NOT inbound
webhooks. Each source / resource combination needs a cursor so the
poller can ask the upstream API "give me everything since X" and avoid
re-ingesting the world on every tick.

Why a dedicated table (vs reusing integration_tokens.metadata JSON):
  * Cursor values are written every poll tick (potentially hundreds of
    rows per customer per minute as the worker drains the resource
    list). A JSONB UPDATE on integration_tokens would churn the same
    hot row every tick and contend with the OAuth-refresh / token-rotate
    paths that read it.
  * Per-resource granularity. GitHub polls many repos under one OAuth
    install; Slack polls many channels under one bot token; Notion
    polls many pages under one integration. Each needs its own cursor.
    integration_tokens is one row per source — too coarse.
  * Cursor *shape* varies by source (an ISO timestamp for Linear, an
    ETag string for GitHub, a Slack `ts`, a Notion `last_edited_time`,
    a Sentry `statsPeriod` anchor). A free-form TEXT column accepts
    them all; the per-source poller knows how to parse its own.

RLS:
  * Tenant-scoped via `app.current_customer_id` (matches every other
    per-tenant table in this codebase). The poller calls
    `with_tenant(customer_id)` before reading or writing.
  * `customers` FK with ON DELETE CASCADE — when a customer is
    deprovisioned the cursors go with them.
  * NO FORCE RLS — the poll_scheduler walks all customers' cursors in
    one tick (claim-style work distribution), so the worker needs to
    read rows for every tenant without setting the GUC per row. This
    matches the inferred_edges_queue pattern (see migration 0068).

Indexes:
  * PRIMARY KEY (customer_id, source, resource_id) — one cursor per
    resource per source per tenant.
  * (source, polled_at DESC) — operator queries "which resources of
    source X have been polled most recently".

Revision ID: 0072_ingestion_cursors
Revises: 0071_chunks_embed_v1_nullable
Create Date: 2026-05-14
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0072_ingestion_cursors"
down_revision = "0071_chunks_embed_v1_nullable"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ingestion_cursors",
        sa.Column(
            "customer_id",
            sa.Text(),
            sa.ForeignKey("customers.customer_id", ondelete="CASCADE"),
            nullable=False,
        ),
        # The SourceSystem enum value (e.g. "github", "slack"). Plain
        # TEXT — no CHECK constraint — so adding a new source in
        # shared/constants.py doesn't require a follow-up migration.
        sa.Column("source", sa.Text(), nullable=False),
        # Source-specific resource id. Shape varies:
        #   github: "<owner>/<repo>"
        #   slack:  "<channel_id>"
        #   linear: "<team_id>" (or "*" for the customer-wide cursor)
        #   notion: "<page_or_database_id>"
        #   sentry: "<project_slug>"
        sa.Column("resource_id", sa.Text(), nullable=False),
        # The cursor value the poller passes back to the upstream API on
        # the next tick. Free-form — each poller parses its own. NULL
        # means "first poll", which the poller interprets as
        # source-specific (often "since beginning of time" or "last 7d").
        sa.Column("cursor_value", sa.Text(), nullable=True),
        sa.Column(
            "polled_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        # When the cursor was first created. Useful for "how long has
        # this resource been polling" diagnostics.
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        # Last-error tracking. Cleared on every successful poll. Useful
        # for surfacing "this resource hasn't successfully polled in N
        # ticks" to the dashboard.
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "last_error_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.PrimaryKeyConstraint(
            "customer_id", "source", "resource_id", name="ingestion_cursors_pkey"
        ),
    )
    op.create_index(
        "ix_ingestion_cursors_source_polled_at",
        "ingestion_cursors",
        ["source", sa.text("polled_at DESC")],
    )
    op.execute("ALTER TABLE ingestion_cursors ENABLE ROW LEVEL SECURITY")
    # NO FORCE RLS — see module docstring. The scheduler reads
    # cross-tenant for work distribution; per-tenant writes set the GUC
    # via with_tenant() so the policy still gates writes.
    op.execute(
        """
        CREATE POLICY ingestion_cursors_tenant_isolation ON ingestion_cursors
            USING (customer_id = current_setting('app.current_customer_id', true))
            WITH CHECK (customer_id = current_setting('app.current_customer_id', true))
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS ingestion_cursors_tenant_isolation ON ingestion_cursors")
    op.drop_index("ix_ingestion_cursors_source_polled_at", table_name="ingestion_cursors")
    op.drop_table("ingestion_cursors")
