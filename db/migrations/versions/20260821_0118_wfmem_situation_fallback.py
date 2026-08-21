"""wfmem: a fallback situation, so an unclassifiable rule is not unreachable

Revision ID: 0118_wfmem_situation_fallback
Revises: 0117_wfmem_clause_embedding
Create Date: 2026-08-21

THE BUG THIS CLOSES HAS NO ERROR IN IT. `declare` attaches a clause to a
situation only when it HAS one, and the classifier honestly answers `unknown`
for prose it cannot place. Those two correct behaviours compose into a silent
loss: the clause is written, gets a real id, returns 200, appears in an
unfiltered `probe rule list` -- and is invisible to every situation-scoped read,
which is the only read the store exists to serve. Observed in production on the
first rule anyone declared: it was plainly about opening a pull request, and
asking for `open-pr` rules returned nothing.

So there is now somewhere for those to land. Two columns of thought:

WHY A COLUMN AND NOT A RESERVED SLUG. `situations` is per-tenant and editable --
the seeder deliberately leaves an existing slug's label and description alone
because a tenant may have rewritten them. A rule keyed on the literal string
'misc' therefore breaks the moment somebody renames it, and it would have to be
repeated in the classifier, the declare path and the serving path, which is
three places to forget. `classifiable` is a property of the row, survives a
rename, and gives Phase 1's mining somewhere to put future non-label buckets.

WHY THE DEFAULT IS TRUE. Every situation that exists today is a classifier
label; the fallback is the exception. A default of false would silently empty
the label space of any tenant seeded before this migration -- and an empty label
space classifies everything as `unknown`, which is the failure mode this whole
change exists to fix, arrived at by the fix for it.

Backfill is deliberately scoped to tenants that ALREADY have a vocabulary. A
tenant with zero situations has never had the capability enabled, and giving
them a lone `misc` row would put them in a state no code path produces: enabled-
looking, with a vocabulary that cannot classify anything. `enabled_tenants_
missing_situations` is what names those tenants, and it must keep working.
"""

from __future__ import annotations

from alembic import op

revision = "0118_wfmem_situation_fallback"
down_revision = "0117_wfmem_clause_embedding"
branch_labels = None
depends_on = None


#: Kept in sync with `engine.shared.wfmem.situations.FALLBACK_SITUATION`; a test
#: compares the two. Duplicated rather than imported on purpose -- a migration
#: that imports app code runs whatever that code says TODAY, not what it said
#: when the migration was written, so a later edit silently rewrites history.
_FALLBACK_SLUG = "misc"
_FALLBACK_LABEL = "Anything else"
_FALLBACK_DESCRIPTION = (
    "A rule that does not belong to any of the situations above. This is a "
    "holding bucket, not a label: nothing is ever classified INTO it, and rules "
    "land here only because no situation fit them."
)


def upgrade() -> None:
    op.execute("ALTER TABLE situations ADD COLUMN classifiable BOOLEAN NOT NULL DEFAULT true")

    # One fallback row per tenant that already has a vocabulary. ON CONFLICT so a
    # re-run is a no-op and a tenant who somehow already owns the slug keeps
    # whatever they wrote -- except `classifiable`, which is not theirs to choose:
    # a fallback in the label space is the pollution this column exists to stop.
    op.execute(
        f"""
        INSERT INTO situations (customer_id, slug, label, description, classifiable)
        SELECT DISTINCT s.customer_id, '{_FALLBACK_SLUG}', '{_FALLBACK_LABEL}',
               '{_FALLBACK_DESCRIPTION}', false
          FROM situations s
        ON CONFLICT (customer_id, slug)
        DO UPDATE SET classifiable = false
        """
    )

    # Adopt the clauses that were already stranded. Without this the fix only
    # helps rules declared from now on, and the rules most likely to be orphaned
    # are the ones written before anybody noticed the problem.
    op.execute(
        f"""
        INSERT INTO clause_situation_edges (customer_id, clause_id, situation_id, classification)
        SELECT c.customer_id, c.id, s.id, '{{"method": "fallback_backfill_0118"}}'::jsonb
          FROM clauses c
          JOIN situations s
            ON s.customer_id = c.customer_id AND s.slug = '{_FALLBACK_SLUG}'
         WHERE NOT EXISTS (
                   SELECT 1 FROM clause_situation_edges e WHERE e.clause_id = c.id
               )
        ON CONFLICT (clause_id, situation_id) DO NOTHING
        """
    )


def downgrade() -> None:
    # The edges go first: dropping the column would leave the fallback rows
    # indistinguishable from real labels, and deleting the rows without the
    # edges would cascade the edges away anyway. Explicit beats incidental.
    op.execute(
        f"""
        DELETE FROM clause_situation_edges e
         USING situations s
         WHERE e.situation_id = s.id AND s.slug = '{_FALLBACK_SLUG}'
        """
    )
    op.execute(f"DELETE FROM situations WHERE slug = '{_FALLBACK_SLUG}'")
    op.execute("ALTER TABLE situations DROP COLUMN classifiable")
