"""drop query_cache

Revision ID: 0006_drop_query_cache
Revises: 0005_cascade_rls_tables
Create Date: 2026-04-24

The router cache wasn't paying back at our scale (low query volume,
~10% hit rate in practice) and made temporal-aware retrieval awkward
— relative phrases like "last month" had to resolve fresh per request,
not from cached results. Drop the table; Haiku is on the path for
every query now. Re-introduce an in-memory LRU at the retrieval
service level if/when query volume justifies it.
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
            cache_key            TEXT PRIMARY KEY,
            customer_id          TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            query_text_hash      TEXT NOT NULL,
            entities             JSONB NOT NULL,
            expansions           JSONB NOT NULL,
            expires_at           TIMESTAMPTZ NOT NULL,
            created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute("CREATE INDEX idx_query_cache_expiry ON query_cache (expires_at)")
    op.execute(
        "CREATE INDEX idx_query_cache_customer ON query_cache (customer_id, query_text_hash)"
    )
