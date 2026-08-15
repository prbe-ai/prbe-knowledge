"""documents: index the parent lookup that child retirement depends on

Revision ID: 0105_documents_parent_live_index
Revises: 0104_wiki_live_page_size
Create Date: 2026-08-15

`documents.parent_doc_id` has never had an index. Nothing needed one, because
nothing queried by it -- children were only ever reached by walking down from a
doc already in hand.

Retiring a session's superseded units queries the other direction:

    UPDATE documents SET valid_to = NOW()
     WHERE customer_id = $1 AND parent_doc_id = ANY($2) AND valid_to IS NULL
       AND NOT (doc_id = ANY($3))

The best existing index is `idx_documents_live (customer_id, doc_id) WHERE
valid_to IS NULL`, whose leading column after customer_id is doc_id -- it cannot
satisfy a parent_doc_id predicate. So this plans as a scan over every live
document for the tenant, taking row locks as it goes, on the largest table in
the system.

That would be tolerable once. It is not: the path fires on every session
finalize, again on every batch of a session that resumes after finalizing (the
completion marker is sticky), and in bulk on the nightly sweep. With
db_statement_timeout at 300s it would not fail loudly either -- it would just
get slower and start contending with retrieval, which is the same shape as the
2026-04-29 lock incident.

PARTIAL, matching the predicate: `WHERE valid_to IS NULL` keeps the index to
live rows only, which is the minority of a table that accumulates every prior
version forever. CONCURRENTLY because this table is hot and an ACCESS EXCLUSIVE
lock on it during a deploy is not acceptable -- which is also why this migration
runs outside a transaction.
"""
from __future__ import annotations

from alembic import op

revision = "0105_documents_parent_live_index"
down_revision = "0104_wiki_live_page_size"
branch_labels = None
depends_on = None

# CREATE INDEX CONCURRENTLY cannot run inside a transaction block, and alembic
# wraps migrations in one by default.
def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_parent_live
                ON documents (customer_id, parent_doc_id)
             WHERE valid_to IS NULL
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_documents_parent_live")
