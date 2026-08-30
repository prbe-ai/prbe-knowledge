"""pg_prewarm, so the guardian can warm the retrieval hot set after a promotion.

Every failover ships one guaranteed engine_timeout today: the promoted
instance serves its first searches from a cold cache, and the first
storm-shaped query measured 66.8s against the caller's 30s budget (observed
again 2026-08-30: first post-switchover query 30s engine_timeout). The
guardian's post-promotion hook (`prewarm_indexes`) reads the hot indexes
into the OS page cache; this migration is only the extension it calls.

TOLERANT of absence and privilege, and that is the load-bearing part:
pg_prewarm is an UNTRUSTED extension -- unlike vector/pg_trgm/btree_gin it
needs superuser (or an explicit grant) to create -- and alembic runs every
migration on every install, so a bare CREATE EXTENSION here would fail the
whole upgrade on exactly the self-host whose operator was never told to
preinstall it (docs list only the three trusted ones). The guardian degrades
gracefully when the extension is absent (`guardian.prewarm_skipped`), so
nothing hard-depends on it and this migration must not either.
"""

from alembic import op

revision = "0123_pg_prewarm_extension"
down_revision = "0122_documents_source_key_expr"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_prewarm') THEN
                BEGIN
                    CREATE EXTENSION IF NOT EXISTS pg_prewarm;
                EXCEPTION WHEN insufficient_privilege THEN
                    RAISE NOTICE 'pg_prewarm not created (insufficient privilege); guardian prewarm will skip';
                END;
            END IF;
        END $$
        """
    )


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS pg_prewarm")
