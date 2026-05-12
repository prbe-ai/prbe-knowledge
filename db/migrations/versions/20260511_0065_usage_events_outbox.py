"""usage_events: outbox columns for the data-plane telemetry uploader (option A).

Revision ID: 0065_usage_events_outbox
Revises: 0064_merge_three_heads
Create Date: 2026-05-11

Background — schema conflict resolution (option A: ONE table):
  prbe-knowledge already owns `usage_events` (created in 0020) — it's the
  PRODUCER: the retrieval middleware writes one row per /retrieve, /query,
  /sources, /source-view call, and three /usage/* read endpoints feed the
  dashboard live-feed.

  Separately, prbe-backend's data-plane (PR #130, migration
  `0002_usage_events`) defined a minimal outbox-shaped `usage_events`:
  `id text PK`, `customer_id text`, `event_type text`,
  `occurred_at timestamptz default now()`, `counters jsonb default '{}'`,
  `uploaded_at timestamptz nullable`, with a partial index
  `ix_usage_events_pending` on `(customer_id, occurred_at) WHERE
  uploaded_at IS NULL`. The telemetry uploader
  (apps/data_plane/services/control_plane/telemetry.py) drains rows
  `WHERE uploaded_at IS NULL`, batches them, POSTs to the control plane's
  /telemetry/usage, then sets `uploaded_at = now()`.

  Option A keeps ONE table: prbe-knowledge's `usage_events` is the
  producer; this migration adds the outbox columns the uploader needs so
  it can drain the same table directly. NON-DESTRUCTIVE — adds columns +
  one partial index, renames nothing (the dashboard read endpoints depend
  on the existing column names).

What the existing table already has that the outbox shape wants:
  * `customer_id`  -> already TEXT (NOT a UUID — no type conversion
                      needed; the uploader's TEXT assumption holds).
  * `occurred_at`  -> already `timestamptz NOT NULL DEFAULT NOW()`. The
                      partial index goes on this existing column; we do
                      NOT add a duplicate.
  * `event_type`   -> already `TEXT NOT NULL`. The uploader treats it as
                      an opaque label; prbe-knowledge's values
                      (knowledge.retrieve|query|get_source|unknown) are
                      fine.
  * `id` vs `event_id` -> prbe-knowledge uses `event_id uuid` PK; the
                      outbox migration named it `id text`. We do NOT
                      rename. Follow-up for whoever owns the uploader:
                      SELECT `event_id` (cast `::text` if it wants a
                      string id) instead of `id`. Cheap one-liner.

What this migration ADDS:
  * `uploaded_at timestamptz NULL`            — NULL = "needs flushing".
  * `counters    jsonb NOT NULL DEFAULT '{}'` — token/usage counters; the
                      retrieval write path leaves this `{}` for now (a
                      separate task threads real token counts in).
  * partial index `ix_usage_events_pending` on `(customer_id, occurred_at)
                  WHERE uploaded_at IS NULL` — the uploader's drain query.

Postgres-safety: no CHECK-on-bool, no native enum. `counters` server
default is `'{}'::jsonb`. The partial index uses `postgresql_where=` so
SQLite (if the migration CI ever runs there) silently drops the dialect
kwarg — but this repo's test DB is real Postgres (scripts/neon-migrate.sh
local), so the partial index materializes correctly under test.

This migration adds columns + an index only; it does not UPDATE/DELETE
any tenant rows, so the NO FORCE / FORCE RLS toggle pattern does not
apply. Existing rows get `uploaded_at = NULL` and `counters = '{}'` from
the column defaults — meaning the historical backlog is immediately
visible to the uploader's drain query. That's intentional: the control
plane dedupes on event_id, and a one-time backlog flush is harmless.

Lessons reminder: revision string MUST be <=32 chars (alembic_version
column is varchar(32)); '0065_usage_events_outbox' is 24 chars — fine.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0065_usage_events_outbox"
down_revision = "0064_merge_three_heads"
branch_labels = None
depends_on = None


PENDING_INDEX_NAME = "ix_usage_events_pending"


def upgrade() -> None:
    op.add_column(
        "usage_events",
        sa.Column(
            "uploaded_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "usage_events",
        sa.Column(
            "counters",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.create_index(
        PENDING_INDEX_NAME,
        "usage_events",
        ["customer_id", "occurred_at"],
        unique=False,
        postgresql_where=sa.text("uploaded_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(PENDING_INDEX_NAME, table_name="usage_events")
    op.drop_column("usage_events", "counters")
    op.drop_column("usage_events", "uploaded_at")
