"""pg_prewarm, so the guardian can warm the retrieval hot set after a promotion.

Every failover ships one guaranteed engine_timeout today: the promoted
instance serves its first searches from a cold cache, and the first
storm-shaped query measured 66.8s against the caller's 30s budget (observed
again 2026-08-30: first post-switchover query 30s engine_timeout). The
guardian's post-promotion hook (`prewarm_indexes`) reads the HNSW and BM25
indexes into the OS page cache; this migration is only the extension it
calls.

`IF NOT EXISTS` keeps this idempotent against a database where an operator
installed it by hand first. The guardian itself degrades gracefully when the
extension is absent (a self-host that never runs this migration just logs
`guardian.prewarm_skipped`), so nothing hard-depends on it.
"""

from alembic import op

revision = "0123_pg_prewarm_extension"
down_revision = "0122_documents_source_key_expr"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS pg_prewarm")
