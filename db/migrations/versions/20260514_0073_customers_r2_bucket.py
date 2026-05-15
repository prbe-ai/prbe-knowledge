"""customers.r2_bucket — per-tenant R2 bucket name, persisted on the row.

Why this exists
---------------
Until now the data plane formed the per-tenant bucket name at runtime as
``f"{R2_BUCKET_PREFIX}-{customer_id}"`` (see ``shared/config.py``). That
formula is fine when the suffix is always ``customer_id``, but the control
plane is moving new tenants to ``prbe-<slug>`` (human-friendly, no UUID
in the URL/path). Two suffix shapes can't both come out of one prefix +
``customer_id`` formula, so we stop computing the name and start storing
it.

This migration is additive and backwards-compatible:
  * Adds ``customers.r2_bucket TEXT`` (NULLABLE for now).
  * Backfills every existing row with the CURRENT computed name
    (``<R2_BUCKET_PREFIX>-<customer_id>``) so the live R2 bucket each
    tenant is already using gets recorded on the row — nothing in prod
    moves. ``R2_BUCKET_PREFIX`` is read from env at migration time; falls
    back to ``"prbe-knowledge"`` (the dataclass default in
    ``shared/config.py``) if unset (dev / SQLite-CI).
  * ``r2_bucket`` stays NULLABLE here. The runtime falls back to the old
    prefix-formula when it's NULL, so the migration and the CP-side
    writer can land in any order. A follow-up migration (after CP starts
    populating r2_bucket on every mirror call) will ALTER ... SET NOT NULL
    and the runtime will drop the fallback.

No RLS policy change: ``customers`` already has the right policy (see
the table's original definition); we're just adding a column.

Revision ID: 0073_customers_r2_bucket
Revises: 0072_ingestion_cursors
Create Date: 2026-05-14
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op

revision = "0073_customers_r2_bucket"
down_revision = "0072_ingestion_cursors"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "customers",
        sa.Column("r2_bucket", sa.Text(), nullable=True),
    )
    # Backfill every existing row so the bucket name a tenant is ALREADY
    # using in R2 gets recorded — this migration must not silently move
    # any tenant's live data to a new bucket. We read the prefix from env
    # the same way ``shared.config.Settings`` does, so the backfill picks
    # up whatever the chart wired in (``prbe-customer`` in managed-shared
    # prod today). Fallback matches the dataclass default for dev/CI.
    prefix = os.environ.get("R2_BUCKET_PREFIX") or "prbe-knowledge"
    op.execute(
        sa.text(
            "UPDATE customers SET r2_bucket = :prefix || '-' || customer_id "
            "WHERE r2_bucket IS NULL"
        ).bindparams(prefix=prefix)
    )


def downgrade() -> None:
    op.drop_column("customers", "r2_bucket")
