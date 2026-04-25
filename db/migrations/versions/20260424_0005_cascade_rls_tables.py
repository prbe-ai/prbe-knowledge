"""backfill missing foreign keys

Revision ID: 0005_cascade_rls_tables
Revises: 0004_backfill_progress
Create Date: 2026-04-24

Closes three FK gaps:
  - audit_log.customer_id and query_cache.customer_id were RLS-only,
    no FK. DELETE FROM customers had to issue extra DELETEs first.
  - documents.ingestion_event_id was declared but never constrained,
    so there was no way to assert docs point at real events.

Cascades on customer_id are ON DELETE CASCADE (tenant delete nukes
everything). documents → ingestion_events is ON DELETE SET NULL so
periodic retention sweeps on the event log don't take docs with them.
"""

from __future__ import annotations

from alembic import op

revision = "0005_cascade_rls_tables"
down_revision = "0004_backfill_progress"
branch_labels = None
depends_on = None


_CUSTOMER_FK_TABLES = ("audit_log", "query_cache")


def upgrade() -> None:
    # Idempotent. Fresh installs run schema.sql (via 0001) which already
    # creates these constraints with the same names, so 0005 is a no-op
    # there. Upgrading an existing DB from 0004 actually adds them.
    # query_cache may already be dropped (via migration 0006 on a fresh
    # install where schema.sql doesn't include it) — skip if missing.
    for table in _CUSTOMER_FK_TABLES:
        op.execute(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_class
                    WHERE relname = '{table}' AND relkind = 'r'
                ) THEN
                    RETURN;
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = '{table}_customer_id_fkey'
                ) THEN
                    ALTER TABLE {table}
                    ADD CONSTRAINT {table}_customer_id_fkey
                    FOREIGN KEY (customer_id)
                    REFERENCES customers(customer_id)
                    ON DELETE CASCADE;
                END IF;
            END $$;
            """
        )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'documents_ingestion_event_id_fkey'
            ) THEN
                ALTER TABLE documents
                ADD CONSTRAINT documents_ingestion_event_id_fkey
                FOREIGN KEY (ingestion_event_id)
                REFERENCES ingestion_events(event_id)
                ON DELETE SET NULL;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_ingestion_event_id_fkey"
    )
    for table in _CUSTOMER_FK_TABLES:
        op.execute(
            f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {table}_customer_id_fkey"
        )
