"""guardian: remember which indexes it already reported absent

Revision ID: 0121_guardian_known_absent
Revises: 0120_pg_search_guardian_state
Create Date: 2026-08-26

WHY
---
The guardian detects a BROKEN pg_search index (valid in the catalog, 0 bytes
on disk) and repairs it by dropping. But its own repair produces a state it
could not see: after the drop, the index is simply ABSENT, the table works,
BM25 quietly returns zero hits -- and the guardian's tick reads
`broken_count: 0`, a false all-clear. That is exactly what happened after the
2026-08-26 04:13 switchover: the guardian dropped the corrupted index at
04:14 (correctly), and from then on reported healthy while lexical search was
dead. Nothing prompted the rebuild.

So the guardian also has to say "an index that SHOULD exist is missing". An
absence, unlike breakage, persists for hours by design (rebuilds are
attended), so alerting on it every one-minute tick would be an alert storm --
the same failure mode the timeline marker's record-LAST discipline exists to
avoid. This column remembers which absences have already been reported;
the guardian alerts on the TRANSITION (newly absent, or newly restored), not
the state.

Same table, same no-RLS rationale as 0120: one row about one Postgres
instance, not tenant data.
"""

from __future__ import annotations

from alembic import op

revision = "0121_guardian_known_absent"
down_revision = "0120_pg_search_guardian_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE pg_search_guardian_state
            ADD COLUMN IF NOT EXISTS known_absent TEXT[] NOT NULL DEFAULT '{}'
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE pg_search_guardian_state DROP COLUMN IF EXISTS known_absent")
