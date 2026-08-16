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
satisfy a parent_doc_id predicate. Without this the retire scans every live
document for the tenant, on a path that fires per finalize and in bulk nightly.

WHY NOT CONCURRENTLY -- learned the hard way, 2026-08-16
-------------------------------------------------------
The first version of this migration used CREATE INDEX CONCURRENTLY, reasoning
that `documents` is the hottest table in the system and an ACCESS EXCLUSIVE lock
during a deploy was unacceptable. That reasoning was sound in the abstract and
wrong here, and it took prod deploys down for two hours.

CONCURRENTLY builds the index and then WAITS for every transaction older than
the build to finish before marking it valid. It waits indefinitely. One
long-lived transaction anywhere on the database is enough. That is exactly what
happened: the index finished building at 152 kB and sat in its final wait until
helm's 15m post-upgrade timeout killed the release -- twice -- leaving
`indisvalid = false` behind. And because the retry used `IF NOT EXISTS`, it
would have skipped the invalid leftover forever, so the index would never have
existed while every deploy kept failing.

The table is 22,543 rows and 93 MB. A plain build takes well under a second and
the momentary lock is genuinely fine. The cost of CONCURRENTLY here was not a
tradeoff, it was a self-inflicted outage in exchange for nothing.

The lesson generalises: CONCURRENTLY belongs in an out-of-band ops step, never
in a deploy hook with a timeout, and never before checking whether the table is
actually big enough to need it.
"""
from __future__ import annotations

from alembic import op

revision = "0105_documents_parent_live_index"
down_revision = "0104_wiki_live_page_size"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Clear the INVALID leftover from the CONCURRENTLY attempt before creating.
    # An invalid index is unusable by the planner AND satisfies IF NOT EXISTS,
    # so leaving it would silently mean this index never exists. Guarded on
    # indisvalid so a healthy index on a re-run is left alone.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                  FROM pg_class c
                  JOIN pg_index i ON i.indexrelid = c.oid
                 WHERE c.relname = 'idx_documents_parent_live'
                   AND NOT i.indisvalid
            ) THEN
                DROP INDEX idx_documents_parent_live;
            END IF;
        END $$;
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_parent_live
            ON documents (customer_id, parent_doc_id)
         WHERE valid_to IS NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_documents_parent_live")
