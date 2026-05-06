"""wiki_synthesis_runs.kind CHECK: allow both 'bootstrap' and 'backfill'

Revision ID: 0048_extend_kind_for_backfill
Revises: 0047_wiki_bootstrap_states
Create Date: 2026-05-06

The wiki bootstrap pipeline is being renamed to "backfill" across code,
routes, and structlog events. The DB still stores kind='bootstrap' on
existing rows; this migration only widens the CHECK constraint so a
future PR can migrate row data + drop the old value.

Mirrors the swap pattern from 0044/0047: drop the named constraint,
recreate it with the extended set. Existing rows are unaffected.
Downgrade narrows the set back; any rows already at kind='backfill' at
that point are remapped to 'bootstrap' so the prior CHECK passes.
"""

from __future__ import annotations

from alembic import op

revision = "0048_extend_kind_for_backfill"
down_revision = "0047_wiki_bootstrap_states"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE wiki_synthesis_runs DROP CONSTRAINT IF EXISTS ck_wsr_kind")
    op.execute(
        """
        ALTER TABLE wiki_synthesis_runs ADD CONSTRAINT ck_wsr_kind CHECK (
            kind IN ('onboarding','wake','scheduled','bootstrap','backfill')
        )
        """
    )


def downgrade() -> None:
    op.execute("UPDATE wiki_synthesis_runs SET kind = 'bootstrap' WHERE kind = 'backfill'")
    op.execute("ALTER TABLE wiki_synthesis_runs DROP CONSTRAINT IF EXISTS ck_wsr_kind")
    op.execute(
        """
        ALTER TABLE wiki_synthesis_runs ADD CONSTRAINT ck_wsr_kind CHECK (
            kind IN ('onboarding','wake','scheduled','bootstrap')
        )
        """
    )
