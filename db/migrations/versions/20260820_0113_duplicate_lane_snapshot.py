"""entity_merge_edge_snapshot: permit the duplicate-lane operation it already writes

Revision ID: 0113_duplicate_lane_snapshot
Revises: 0112_repair_bootstrap_drift
Create Date: 2026-08-20

`entity_clusters_routes.py:363` inserts `operation = 'deleted_duplicate_lane'`.
The column's CHECK has only ever permitted `'deleted_self_loop'`. Those two facts
coexisted without incident because the code that writes the row COULD NOT RUN:

    merge_cluster's body is inside with_tenant()'s explicit transaction. The
    UniqueViolation that triggers the duplicate-lane branch aborted that
    transaction the instant it was raised, so the branch's INSERT died with
    InFailedSQLTransactionError long before Postgres ever evaluated this CHECK.
    The merge then rolled back whole. Production signature on the managed plane:
    88 duplicate-key errors in 24h paired 1:1 with 88 "current transaction is
    aborted", and not one completed merge.

Adding the SAVEPOINT that lets the branch run is what surfaced this. It is the
second of three layers, and worth stating plainly because it is good evidence
the path was dead rather than merely rare: had it EVER executed, the very first
attempt would have raised CheckViolationError.

    layer 1  the handler could not run          -> SAVEPOINT (routes)
    layer 2  its INSERT violates this CHECK     -> this migration
    layer 3  unmerge restored only self-loops   -> unmerge step 5 (routes)

WHY WIDEN THE CHECK RATHER THAN REUSE 'deleted_self_loop'
---------------------------------------------------------
They are different deletions and unmerge has to tell them apart. A self-loop is
restored as `($1, $1)` — both endpoints collapsed onto the restored alias node.
A duplicate-lane edge kept its original two distinct endpoints and has to be
restored by resolving `pre_from_canonical_id` / `pre_to_canonical_id` back to
node ids. Overloading one value would make the restore ambiguous and silently
turn a normal edge into a self-loop.

Idempotent: the constraint is dropped IF EXISTS before being recreated, so a
re-run and a database that already carries the widened form both no-op.
"""

from __future__ import annotations

from alembic import op

revision = "0113_duplicate_lane_snapshot"
down_revision = "0112_repair_bootstrap_drift"
branch_labels = None
depends_on = None

_CONSTRAINT = "entity_merge_edge_snapshot_operation_check"
_TABLE = "entity_merge_edge_snapshot"


def upgrade() -> None:
    # Bound the wait: this is a brief ACCESS EXCLUSIVE on a table the merge
    # endpoint writes. Same reasoning as 0094/0095/0097.
    op.execute("SET lock_timeout = '5s'")
    op.execute(f"ALTER TABLE {_TABLE} DROP CONSTRAINT IF EXISTS {_CONSTRAINT}")
    op.execute(
        f"""
        ALTER TABLE {_TABLE}
            ADD CONSTRAINT {_CONSTRAINT}
            CHECK (operation IN ('deleted_self_loop', 'deleted_duplicate_lane'))
        """
    )


def downgrade() -> None:
    # Narrowing the CHECK would fail against any row this release wrote, so
    # delete those rows first. That is lossy and it is the correct trade: a
    # snapshot row whose operation the constraint forbids is unrestorable by
    # definition, and leaving it would make the downgrade fail instead of the
    # data be smaller.
    op.execute("SET lock_timeout = '5s'")
    op.execute(f"DELETE FROM {_TABLE} WHERE operation = 'deleted_duplicate_lane'")
    op.execute(f"ALTER TABLE {_TABLE} DROP CONSTRAINT IF EXISTS {_CONSTRAINT}")
    op.execute(
        f"""
        ALTER TABLE {_TABLE}
            ADD CONSTRAINT {_CONSTRAINT}
            CHECK (operation IN ('deleted_self_loop'))
        """
    )
