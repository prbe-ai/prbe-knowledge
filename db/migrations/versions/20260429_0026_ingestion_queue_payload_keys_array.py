"""ingestion_queue: payload_s3_keys[] + version (coalescing support)

Revision ID: 0026_queue_payload_keys
Revises: 0025_system_settings
Create Date: 2026-04-29

Adds two columns to support same-session enqueue coalescing for claude_code:

- `payload_s3_keys text[]` — array of R2 paths for the queue row's payloads.
  For non-CC connectors this is always a 1-element array (one payload per row).
  For claude_code, multiple batches under the same session_id append to this
  array via UPSERT on (customer_id, source_system, source_event_id).
- `version int` — monotonic counter bumped on every UPSERT into the row.
  Worker captures it on claim, runs Phase A (no-write embed) outside any
  transaction, then commits Phase B with an atomic CAS UPDATE:
      WHERE queue_id = $1 AND version = $captured
  If a new batch landed mid-Phase-A, version advanced, CAS matches 0 rows,
  and the row stays 'pending' for re-claim with the extended array. This
  closes the race window the same-session NOT EXISTS clause from PR #33
  was approximating.

Backfill: every existing row gets payload_s3_keys = ARRAY[payload_s3_key].
The legacy `payload_s3_key` column is intentionally NOT dropped in this
migration — that would race with mid-deploy workers still on the old code.
A follow-up PR drops it once this deploy stabilizes.

Why this matters: live claude_code batches were landing in date-partitioned
R2 keys but `fetch_supplementary` was listing the per-session prefix to
merge. The prefixes never lined up for live traffic, so each batch only
processed its own events and the chunk-diff loop expired prior batches'
chunks. Net effect: only the latest 30s of any CC session was searchable.
Coalescing into one queue row per session, with payload_s3_keys carrying
every batch's R2 key, fixes the silent data loss as a side effect.
"""

from __future__ import annotations

from alembic import op

revision = "0026_queue_payload_keys"
down_revision = "0025_system_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add the array column, default to empty so the NOT NULL constraint
    # passes for any concurrent INSERT racing with the migration.
    op.execute(
        """
        ALTER TABLE ingestion_queue
        ADD COLUMN IF NOT EXISTS payload_s3_keys text[] NOT NULL DEFAULT '{}'
        """
    )

    # Backfill: existing rows had payload_s3_key (single string, NOT NULL).
    # Every row becomes a 1-element array of itself. After the deploy the
    # worker only reads payload_s3_keys; payload_s3_key lingers as dead
    # data until a follow-up cleanup PR drops it.
    op.execute(
        """
        UPDATE ingestion_queue
        SET payload_s3_keys = ARRAY[payload_s3_key]
        WHERE payload_s3_keys = '{}'
          AND payload_s3_key IS NOT NULL
        """
    )

    # Monotonic version counter. Bumped on every UPSERT into the row;
    # the worker uses it for compare-and-swap on commit.
    op.execute(
        """
        ALTER TABLE ingestion_queue
        ADD COLUMN IF NOT EXISTS version int NOT NULL DEFAULT 0
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE ingestion_queue DROP COLUMN IF EXISTS version")
    op.execute("ALTER TABLE ingestion_queue DROP COLUMN IF EXISTS payload_s3_keys")
