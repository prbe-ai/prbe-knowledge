"""expression index on documents source_key -- the planner cannot see JSONB

Revision ID: 0122_documents_source_key_expr
Revises: 0121_guardian_known_absent
Create Date: 2026-08-26

WHY
---
Unified search sends `source_keys` + `source_keys_include_keyless` on every
request, so every retriever predicate carries

    (d.metadata->>'source_key' = ANY($n) OR d.metadata->>'source_key' IS NULL)

A JSONB extraction has NO column statistics, so the planner falls back to
default selectivity guesses -- and with an `ORDER BY <distance> LIMIT` on the
ANN path, an underestimated pass-rate makes the estimated index walk look
worse than a table scan. Measured on the research primary 2026-08-26: the
identical pool query planned as a Parallel Seq Scan + top-N heapsort with the
predicate present, and as an HNSW index scan (~300ms warm) once this index
existed. The real pass rate is high (~77% of a tenant's documents are
keyless), which is exactly what the planner could not know.

The index's VALUE here is the expression statistics ANALYZE collects for it
(null_frac, ndistinct on the extracted key); serving equality lookups is a
bonus. It was built by hand on the research primary during the incident; this
migration makes it schema, so a fresh database and the drift checker both
know it.

CONCURRENTLY (plain btree -- the pg_search CIC hazard does not apply), with
the autocommit block alembic requires for it, mirroring 0102's shape.
"""

from __future__ import annotations

from alembic import op

revision = "0122_documents_source_key_expr"
down_revision = "0121_guardian_known_absent"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_source_key_expr "
            "ON documents ((metadata->>'source_key'))"
        )
    op.execute("ANALYZE documents")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_documents_source_key_expr")
