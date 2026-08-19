"""ingestion stats: cover the two aggregates behind the /knowledge header

Revision ID: 0111_ingestion_stats_indexes
Revises: 0110_drop_wiki
Create Date: 2026-08-19

`GET /api/stats/ingestion` backs the DOCUMENTS / CHUNKS readout at the top of
the dashboard's /knowledge tab. Measured on the research plane, tenant `probe`
(19,573 live docs / 101,745 live chunks):

    query      cold      warm    buffer accesses
    docs     5187 ms    64 ms             10,533
    chunks   2936 ms   450 ms            148,808

Neither aggregate had an index it could use.

DOCS -- the expensive part is the heap, not the rows
----------------------------------------------------
    SELECT source_system, COUNT(*), MAX(ingested_at)
      FROM documents
     WHERE customer_id = $1 AND valid_to IS NULL AND deleted_at IS NULL
     GROUP BY source_system

The best index available was `idx_documents_parent_live (customer_id,
parent_doc_id) WHERE valid_to IS NULL`, used purely as a customer_id prefix.
Everything the query actually selects -- source_system, ingested_at -- and one
of its filters -- deleted_at -- live only in the heap, so the plan bitmap-scans
**10,494 heap pages to answer for 19,573 rows**. `documents` averages ~1.5 rows
per page (407 MB heap + 248 MB TOAST over 229k rows), so a row count in the tens
of thousands means a page count in the same order.

CHUNKS -- 77% of the cost is the join, not the count
-----------------------------------------------------
    SELECT d.source_system, COUNT(*)
      FROM chunks c
      JOIN documents d ON d.customer_id = c.customer_id AND d.doc_id = c.doc_id
       AND d.valid_to IS NULL AND d.deleted_at IS NULL
     WHERE c.customer_id = $1 AND c.valid_to IS NULL
     GROUP BY d.source_system

115,245 of the 148,808 buffer accesses were the Memoize'd nested loop into
`documents`, resolving source_system one chunk at a time across 104,598 loops.
The chunks side was no better: with no `(customer_id, ...) WHERE valid_to IS
NULL` index the planner BitmapAnds two indexes that are each wrong on one axis
-- `idx_chunks_doc_live` (partial on valid_to but no customer_id, 133,030 rows)
against `idx_chunks_customer` (customer_id but every version, 392,443 rows) --
to arrive at 104,598.

WHY THE JOIN STAYS
------------------
It is tempting to read the join as pure source_system lookup and delete it by
denormalizing that column onto `chunks` (the shape migration 0100 used for
`title`). The join is doing a second job: a chunk can be live while its document
is not. Measured here:

    tenant             live chunks   joined   difference
    probe                  104,835  101,745        3,090
    anthrogen               22,075   20,992        1,083
    new-workspace            5,725    5,348          377

~3% of live chunks hang off a superseded or soft-deleted document. Dropping the
join overcounts by exactly that. So both sides get an index instead, and the
join becomes a hash join between two index-only scans -- no heap on either side.
`(customer_id, doc_id)` is unique among live non-deleted documents (verified:
zero doc_ids with more than one live version, all tenants), so the join is 1:1
and COUNT(*) cannot be inflated by fan-out.

ONE INDEX ON `documents`, NOT TWO
---------------------------------
The two queries want different column orders -- the semi-join wants doc_id
first, the aggregate wants source_system first. A single index ordered
`(customer_id, doc_id, source_system, ingested_at)` serves both: the join by
prefix, and the aggregate by scanning the whole partial index (19.5k tuples for
probe) and hash-aggregating. Sorting the aggregate's way would save nothing --
it has to read every live document either way -- and `documents` is the hottest
table in the system, so one index is worth more than a marginally better plan.

WHY NOT CONCURRENTLY
--------------------
Migration 0105 took prod deploys down for two hours doing exactly that:
CONCURRENTLY waits indefinitely for older transactions, and a helm post-upgrade
hook has a timeout. Both indexes here are partial and small (live rows only:
~133k chunks, ~20k documents across all tenants), a plain build is seconds, and
the momentary lock is fine. Do not "improve" this to CONCURRENTLY.

THESE INDEXES NEED A VACUUM TO PAY OFF
--------------------------------------
An index-only scan reads the visibility map, and only VACUUM populates it. At
the time of writing neither table had EVER been vacuumed on the research plane
-- autovacuum_count = 0, last_autovacuum = NULL, 397k dead tuples on chunks --
while every neighbouring table vacuumed normally. `documents` has since been
vacuumed by hand (25.6s) and its side of this now goes index-only with 15 heap
fetches. `chunks` has not, and cannot: see below.

AND ON `chunks` THIS INDEX COSTS SOMETHING
------------------------------------------
The reason `chunks` never vacuums is a CORRUPT `idx_chunks_bm25_v2` -- VACUUM
opens its segment file, finds it short, and dies, on every run. So every chunk
insert or version-supersede now also writes an entry into `idx_chunks_stats_live`
that nothing will ever reclaim, on a table already at 397k dead against 575k
live. It is a far smaller constant than the `source_system` denormalization this
work rejected for the same reason (a 3.4 MB index against a 575k-row rewrite),
and it buys a better plan even with no visibility map -- 148,808 buffers down to
72,182. But it is the same kind of debt, and the honest sequencing is to
REINDEX INDEX CONCURRENTLY idx_chunks_bm25_v2 FIRST and let `chunks` vacuum,
rather than filing that repair as follow-up. See tasks/knowledge-stats-latency.md.
"""

from __future__ import annotations

from alembic import op

revision = "0111_ingestion_stats_indexes"
down_revision = "0110_drop_wiki"
branch_labels = None
depends_on = None

_CHUNKS_INDEX = "idx_chunks_stats_live"
_DOCUMENTS_INDEX = "idx_documents_stats_live"


def _drop_if_invalid(index_name: str) -> None:
    """Clear an INVALID leftover before creating.

    Straight from 0105: a failed CONCURRENTLY build leaves `indisvalid = false`,
    which the planner will not use and which nonetheless satisfies IF NOT
    EXISTS. Leaving one behind means the index silently never exists. Guarded on
    indisvalid so a healthy index on a re-run is left alone.
    """
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                  FROM pg_class c
                  JOIN pg_index i ON i.indexrelid = c.oid
                 WHERE c.relname = '{index_name}'
                   AND NOT i.indisvalid
            ) THEN
                EXECUTE 'DROP INDEX {index_name}';
            END IF;
        END
        $$;
        """
    )


def upgrade() -> None:
    # Bound the lock wait, same as 0094/0095/0097 do for these two tables.
    # CREATE INDEX needs SHARE for the whole build (16.8s on chunks, measured),
    # and an unbounded wait behind a long retrieval scan queues every writer
    # after it -- inside a helm post-upgrade hook that has a timeout. That is
    # 0105's outage arriving through a different door. Both statements below are
    # IF NOT EXISTS, so failing fast and retrying is a no-op on whichever index
    # already got built.
    op.execute("SET lock_timeout = '5s'")

    _drop_if_invalid(_CHUNKS_INDEX)
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {_CHUNKS_INDEX}
            ON chunks (customer_id, doc_id)
            WHERE valid_to IS NULL
        """
    )

    _drop_if_invalid(_DOCUMENTS_INDEX)
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {_DOCUMENTS_INDEX}
            ON documents (customer_id, doc_id, source_system, ingested_at)
            WHERE valid_to IS NULL AND deleted_at IS NULL
        """
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {_DOCUMENTS_INDEX}")
    op.execute(f"DROP INDEX IF EXISTS {_CHUNKS_INDEX}")
