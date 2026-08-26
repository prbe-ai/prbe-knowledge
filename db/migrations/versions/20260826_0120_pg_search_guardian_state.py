"""pg_search guardian: remember the timeline so a promotion is detectable

Revision ID: 0120_pg_search_guardian_state
Revises: 0119_wfmem_misc_backfill_rls
Create Date: 2026-08-26

WHY THIS TABLE EXISTS
---------------------
On 2026-08-25 a CNPG failover promoted a standby whose `idx_chunks_bm25_v2`
was a 0-byte file. pg_search Community does not replicate its index storage to
physical standbys, so the standby had carried an empty copy since the index was
built five days earlier. `indisvalid` still said true, so the planner kept
choosing it and every query that PLANNED against `chunks` failed with XX001 --
search channels and the ingestion worker alike, ~10 errors/sec for forty
minutes until a human dropped the index by hand.

The guardian (`scripts/cron_pg_search_guardian.py`) exists so that window is
~1 minute and unattended instead. This table is the one piece of state it
cannot derive from the catalog.

WHY NOT JUST LOOK FOR A 0-BYTE INDEX
------------------------------------
Because that is only the LOUD half of the failure. A standby cloned from a
healthy primary via pg_basebackup copies the index files as they were at clone
time, and then pg_search's writes never replicate -- so after promotion the
index is NONZERO and silently stale, missing every document ingested since the
clone. Size-based detection cannot see that at all: the index looks fine, plans
fine, and quietly returns incomplete results. The only reliable signal that
"this instance's pg_search indexes are now suspect" is that a promotion
happened, and the only durable record of a promotion is the timeline ID
(`pg_control_checkpoint().timeline_id`), which increments on each one.

So the guardian reads the timeline every tick and compares it against the last
value it saw. A change means: re-ANALYZE (a promoted standby carries no planner
statistics of its own -- on 2026-08-25 that alone cost 19.3s of grounding
latency per query until someone ran ANALYZE) and flag every pg_search index for
rebuild regardless of its size.

WHY NO RLS
----------
There is no `customer_id` here and there cannot be: this is one row about one
Postgres instance, not tenant data. `node_post_write_queue` set the precedent
for an operational table that carries no tenant column and therefore no policy.
A CHECK-constrained singleton id keeps it one row forever -- the guardian
UPSERTs, so a second row would mean two disagreeing memories of the same fact.

FIRST TICK AFTER THIS MIGRATION
-------------------------------
The table starts EMPTY, deliberately. The guardian treats "no row" as "record
the current timeline, take no action" rather than as a change, so installing it
does not fire a spurious promotion alert on a cluster that never failed over.
The cost is that a promotion happening between this migration and the first
tick is missed by the timeline path -- the size-0 path still catches the loud
half of that case, which is the one that takes the database down.
"""

from __future__ import annotations

from alembic import op

revision = "0120_pg_search_guardian_state"
down_revision = "0119_wfmem_misc_backfill_rls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS pg_search_guardian_state (
            id                SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            last_timeline_id  BIGINT NOT NULL,
            observed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS pg_search_guardian_state")
