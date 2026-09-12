"""backfill_chunks_project_id: redo 0131's backfill, this time seeing the rows

Revision ID: 0132_backfill_chunks_project_id
Revises: 0131_chunks_project_id_bm25_v3
Create Date: 2026-09-12

WHAT WENT WRONG IN 0131
-----------------------
0131 added `chunks.project_id`, its triggers and its backfill. The column and
the triggers landed. The backfill touched ZERO rows and reported success.

`chunks` and `documents` are under FORCE ROW LEVEL SECURITY, and the policy is
`customer_id = current_setting('app.current_customer_id', true)`. With no
tenant GUC set that reduces to `customer_id = NULL`, which matches nothing --
so the UPDATE ran, affected nothing, and committed cleanly. Measured on the
research plane immediately afterwards: 31 of 15,431 target chunks carried a
project_id, and those 31 were the INSERT trigger firing on new rows, not the
backfill.

FORCE is the part that matters. Ordinary RLS exempts a table's owner; FORCE
removes that exemption. Migrations connect as `app`, which owns both tables
and has `rolbypassrls = false`, so it is subject to the policy like anyone
else.

WHY `NO FORCE` AND NOT A PER-TENANT LOOP
----------------------------------------
Both work. `NO FORCE` is what migration 0028 already established in this repo
for exactly this situation, and it is the better fit here: the backfill is one
set-based statement whose whole point is that it spans every tenant, and
looping it per tenant would run the same 25 GB join once per customer to get
the same answer.

Alembic wraps a migration in ONE transaction, so a failure rolls the
`NO FORCE` back along with the UPDATE. The RLS posture is identical before and
after this migration, and there is no window in which a request-path
connection sees a weakened policy -- `NO FORCE` affects the OWNER's exemption,
not the policy that constrains `app`'s request-path sessions... which are the
same role. That is the one caveat worth stating plainly: for the duration of
this transaction, an `app` session IS exempt from the chunks/documents
policies. It is a migration window measured in minutes on a plane where
migrations already hold DDL locks, and it is the trade 0028 made.

IDEMPOTENT
----------
`project_id IS DISTINCT FROM` means re-running this costs a scan and writes
nothing. Safe to run on a plane where 0131's backfill somehow did work.
"""

from __future__ import annotations

from alembic import op

revision = "0132_backfill_chunks_project_id"
down_revision = "0131_chunks_project_id_bm25_v3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE chunks NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE documents NO FORCE ROW LEVEL SECURITY")
    try:
        # Newest document version in each chunk's version RANGE wins. A chunk
        # spans a range and a document can be re-projected inside it, so a
        # chunk can have two candidate project_ids; `DISTINCT ON` pinned to
        # `version DESC` takes the current one, which is what a searcher means.
        #
        # `customer_id` is in the UPDATE's join even though `chunk_id` is
        # effectively unique: it keeps the statement correct by construction
        # rather than by a property of the id format, and it lets the planner
        # use chunks_pkey.
        op.execute(
            """
            UPDATE chunks c
               SET project_id = src.project_id
              FROM (
                    SELECT DISTINCT ON (c2.customer_id, c2.chunk_id)
                           c2.customer_id,
                           c2.chunk_id,
                           d.metadata->>'project_id' AS project_id
                      FROM chunks c2
                      JOIN documents d
                        ON d.doc_id = c2.doc_id
                       AND d.customer_id = c2.customer_id
                       AND d.version BETWEEN c2.first_seen_version
                                         AND c2.last_seen_version
                     WHERE d.metadata->>'project_id' IS NOT NULL
                     ORDER BY c2.customer_id, c2.chunk_id, d.version DESC
                   ) src
             WHERE c.customer_id = src.customer_id
               AND c.chunk_id = src.chunk_id
               AND c.project_id IS DISTINCT FROM src.project_id
            """
        )
    finally:
        # Restored inside the same transaction. If the UPDATE raised, alembic
        # rolls both the statement and these back together; the `finally` is
        # belt to that braces, so a future refactor that stops wrapping
        # migrations in one transaction cannot leave FORCE off.
        op.execute("ALTER TABLE documents FORCE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE chunks FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    # Deliberately not a revert. Clearing project_id would undo the INSERT
    # trigger's correct work alongside this backfill's, and an empty column is
    # not a safer state than a filled one -- it is the state that made a
    # scoped search silently return nothing.
    pass
