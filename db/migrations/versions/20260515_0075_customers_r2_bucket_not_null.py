"""customers.r2_bucket — backfill stragglers and lock NOT NULL.

PR 3 of the R2 bucket rename trilogy. Migration 0073 added the column
nullable and backfilled every existing row from the
``<R2_BUCKET_PREFIX>-<customer_id>`` formula. Migration 0024 on the CP
side started writing ``prbe-<slug>`` via the mirror so new tenants get
a non-NULL value at INSERT. With both deployed, the runtime's legacy-
prefix fallback is dead code — every row has ``r2_bucket`` populated.

This migration:
  * Defensively backfills any rows whose ``r2_bucket`` is still NULL
    (race between 0073 deploy and the CP mirror starting to send the
    column). Uses the same formula as 0073.
  * ``ALTER COLUMN r2_bucket SET NOT NULL`` — schema-level guarantee.

The runtime's ``_load_bucket`` fallback in ``shared.storage`` is
removed in the same PR — they go together.

Revision ID: 0075_r2_bucket_not_null
Revises: 0074_merge_r2_cascade
Create Date: 2026-05-15
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op

revision = "0075_r2_bucket_not_null"
down_revision = "0074_merge_r2_cascade"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Defensive backfill — same formula 0073 used. Idempotent (only acts
    # on NULL rows). The window in which a row could still be NULL: a
    # tenant created between PR 1's column-add deploy and PR 2's CP
    # mirror starting to send r2_bucket. In prod that window was a few
    # minutes, but we sweep here in case the deploys interleaved.
    prefix = os.environ.get("R2_BUCKET_PREFIX") or "prbe-knowledge"
    op.execute(
        sa.text(
            "UPDATE customers SET r2_bucket = :prefix || '-' || customer_id "
            "WHERE r2_bucket IS NULL"
        ).bindparams(prefix=prefix)
    )

    # BEFORE INSERT trigger to fill r2_bucket from customer_id when NULL.
    # Postgres DEFAULTs can't reference other columns, so a trigger is the
    # only way to derive r2_bucket from customer_id at insert time. The
    # CP→DP mirror (production INSERT path) always supplies r2_bucket
    # explicitly, so the trigger only fires for tests / self-host's
    # seed-customer Job that forgets to set it. On the DP, ``customer_id``
    # IS the slug (see prbe-backend's apps/control_plane/routers/me/
    # provision.py:432-439), so ``prbe-<customer_id>`` == ``prbe-<slug>``
    # — the trigger produces the canonical new-policy name.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION customers_fill_r2_bucket() RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.r2_bucket IS NULL THEN
                NEW.r2_bucket := 'prbe-' || NEW.customer_id;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER customers_fill_r2_bucket_trg
            BEFORE INSERT ON customers
            FOR EACH ROW
            EXECUTE FUNCTION customers_fill_r2_bucket();
        """
    )
    op.alter_column("customers", "r2_bucket", nullable=False)


def downgrade() -> None:
    op.alter_column("customers", "r2_bucket", nullable=True)
    op.execute("DROP TRIGGER IF EXISTS customers_fill_r2_bucket_trg ON customers")
    op.execute("DROP FUNCTION IF EXISTS customers_fill_r2_bucket()")
