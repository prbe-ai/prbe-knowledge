"""record who published a clause without waiting for a second human

Revision ID: 0112_wfmem_clause_publication
Revises: 0111_wfmem_capability_prefs
Create Date: 2026-08-19

THE PROBLEM THIS FIXES. A clause is visible to the team only once TWO distinct
humans' evidence backs it (`engine.shared.wfmem.visibility`). That guard is
right for mined and inferred rules -- one person's working note is not the
team's practice -- but it makes the most important case impossible: a lead sits
down and declares twenty existing team rules, and every one of them is invisible
to everybody, because the lead is one person. Nothing errors. The feature simply
appears dead, which is exactly the state the external tenant's onboarding
session would have produced on day one.

So publication becomes a separate axis from corroboration. `shared_by` NULL
means "visible under the ordinary two-human rule"; non-NULL means a named person
published it on their own authority.

WHY TWO COLUMNS RATHER THAN A BOOLEAN. `shared` would say a clause was
force-published and lose who did it. Unilateral publication is an act of
authority over what a team is told to do, and an unattributable one is worse
than none -- there is nobody to ask when the rule turns out to be wrong, and
nothing to show a person who says "I never agreed to this". `shared_at` is
separate from `created_at` for the same reason: publishing an existing private
clause later is a distinct event from writing it.

NOT A STATUS VALUE. The `status` ladder describes how strong the EVIDENCE is;
this describes whether the clause has been PUBLISHED. Those are orthogonal --
a force-published rule is still `declared`, with exactly one human behind it,
and folding publication into the ladder would destroy the distinction the
ladder exists to make. Serving surfaces are expected to LABEL a force-published
single-author clause rather than hide it: the house rule is label, never gate.

NO BACKFILL. Every existing clause keeps NULL, which preserves current
behaviour exactly -- nothing becomes newly visible when this runs.

RAW SQL VIA `op.execute`, matching 0110 rather than using alembic's `add_column`
/ `create_index` helpers. Not a style preference: the schema-drift guard in
tests/test_workflow_memory_isolation.py replays these migrations by swapping
`op` for a recorder that collects `execute` calls, and an op helper would be
invisible to it -- the guard would compare schema.sql against a migration it
only half read, and pass. The guard is the only thing keeping schema.sql honest,
because CI never runs the migration chain at all.
"""

from __future__ import annotations

from alembic import op

revision = "0112_wfmem_clause_publication"
down_revision = "0111_wfmem_capability_prefs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE clauses
            ADD COLUMN shared_by TEXT,
            ADD COLUMN shared_at TIMESTAMPTZ
        """
    )
    # Both or neither. A `shared_at` with no `shared_by` is an unattributable
    # publication, and a `shared_by` with no timestamp cannot be ordered against
    # the evidence that may later corroborate it -- both are the "who decided
    # this, and when" question the columns exist to answer, half-answered.
    op.execute(
        """
        ALTER TABLE clauses
            ADD CONSTRAINT ck_clauses_publication_is_attributed
            CHECK ((shared_by IS NULL) = (shared_at IS NULL))
        """
    )
    # Partial: published clauses are the minority and the only ones this index
    # serves. The visibility predicate tests `shared_by IS NOT NULL`, which a
    # full index over a mostly-NULL column would answer no faster.
    op.execute(
        """
        CREATE INDEX clauses_published_idx ON clauses (customer_id, shared_at)
            WHERE shared_by IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS clauses_published_idx")
    op.execute(
        "ALTER TABLE clauses DROP CONSTRAINT IF EXISTS ck_clauses_publication_is_attributed"
    )
    op.execute("ALTER TABLE clauses DROP COLUMN IF EXISTS shared_at")
    op.execute("ALTER TABLE clauses DROP COLUMN IF EXISTS shared_by")
