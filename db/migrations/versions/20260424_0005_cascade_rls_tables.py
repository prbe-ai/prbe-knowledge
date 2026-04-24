"""cascade FK on audit_log + query_cache

Revision ID: 0005_cascade_rls_tables
Revises: 0004_backfill_progress
Create Date: 2026-04-24

audit_log and query_cache carried a customer_id column but no FK to
customers — relying on RLS only. That made customer delete awkward
(had to issue extra DELETEs before DELETE FROM customers succeeded).
Adding the FK with ON DELETE CASCADE means a single DELETE FROM
customers now cascades to every child row in the DB.
"""

from __future__ import annotations

from alembic import op

revision = "0005_cascade_rls_tables"
down_revision = "0004_backfill_progress"
branch_labels = None
depends_on = None


_TABLES = ("audit_log", "query_cache")


def upgrade() -> None:
    for table in _TABLES:
        op.execute(
            f"""
            ALTER TABLE {table}
            ADD CONSTRAINT {table}_customer_id_fkey
            FOREIGN KEY (customer_id)
            REFERENCES customers(customer_id)
            ON DELETE CASCADE
            """
        )


def downgrade() -> None:
    for table in _TABLES:
        op.execute(
            f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {table}_customer_id_fkey"
        )
