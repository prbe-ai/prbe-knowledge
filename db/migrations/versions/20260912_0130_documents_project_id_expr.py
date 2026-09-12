"""expression index on documents project_id -- planner statistics for scope

Revision ID: 0130_documents_project_id_expr
Revises: 0129_retrieve_pages
Create Date: 2026-09-12

WHY
---
`QueryRequest.scope.project_id` adds
    d.metadata->>'project_id' = $n
to every retriever predicate on a scoped request. A JSONB extraction has NO
column statistics, so the planner falls back to default selectivity (0.005
for equality); with an `ORDER BY <distance> LIMIT` on the ANN path a
200x-underestimated pass rate makes the estimated index walk look worse than
a table scan -- the exact failure 0122 fixed for `source_key` (measured 37-52s
vs ~300ms on the research primary). The index's VALUE is the expression
statistics ANALYZE collects for it; serving equality lookups is a bonus.

CONCURRENTLY (plain btree -- the pg_search CIC hazard does not apply), with
the autocommit block alembic requires for it, mirroring 0122's shape.
"""

from __future__ import annotations

from alembic import op

revision = "0130_documents_project_id_expr"
down_revision = "0129_retrieve_pages"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_project_id_expr "
            "ON documents ((metadata->>'project_id'))"
        )
    op.execute("ANALYZE documents")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_documents_project_id_expr")
