"""drop query_cache

Revision ID: 0006_drop_query_cache
Revises: 0005_cascade_rls_tables
Create Date: 2026-04-24

The router-output cache was a 1h TTL on Haiku responses. We're removing it:
single-tenant for now, Haiku call is ~50ms, the cache adds operational
weight (sweep cron, query_cache table, RLS) without protecting any meaningful
volume of traffic. Every /query now calls Haiku fresh.
"""

from __future__ import annotations

from alembic import op

revision = "0006_drop_query_cache"
down_revision = "0005_cascade_rls_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_query_cache_expiry")
    op.execute("DROP INDEX IF EXISTS idx_query_cache_customer")
    op.execute("DROP TABLE IF EXISTS query_cache")


def downgrade() -> None:
    op.execute(
        """
        CREATE TABLE query_cache (
            cache_key TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            query_text_hash TEXT NOT NULL,
            entities JSONB NOT NULL,
            expansions JSONB NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute("CREATE INDEX idx_query_cache_expiry ON query_cache (expires_at)")
    op.execute(
        "CREATE INDEX idx_query_cache_customer ON query_cache (customer_id, query_text_hash)"
    )
