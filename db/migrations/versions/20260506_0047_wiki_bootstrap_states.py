"""wiki_synthesis_runs status: add 'pending' and 'cancelled'

Revision ID: 0047_wiki_bootstrap_states
Revises: 0046_custom_ingest_tokens
Create Date: 2026-05-06

The wiki-bootstrap fly app moves from inline-NOTIFY-dispatch to a
queue+claim worker model. Two new ``wiki_synthesis_runs.status`` values
fall out:

  * ``pending``    — trigger route inserts rows in this state; the
                     BootstrapWorker on each machine claims them via
                     ``FOR UPDATE SKIP LOCKED`` and flips them to
                     ``running``. Reclaim flips stale ``running`` rows
                     back to ``pending`` so a self-healing retry happens
                     on machine death.
  * ``cancelled``  — admin-initiated ``?force=true`` re-trigger marks
                     in-flight rows ``cancelled`` and notifies workers
                     to ``task.cancel()`` the live crawler. Distinct
                     from ``failed`` so audit history can tell apart a
                     real crawler error from an operator override.

Existing rows are unaffected by both directions of the swap. v4
daily-replay rows (kind != 'bootstrap') don't use the new states.

The CHECK constraint name is ``ck_wsr_status`` (set by 0033). Drop and
re-add with the extended set; reverse the swap on downgrade. Any rows
in the new states at downgrade time are remapped: ``pending`` ->
``running`` (lets reclaim sweep them) and ``cancelled`` -> ``failed``
(closest v3-shaped terminal).
"""

from __future__ import annotations

from alembic import op

revision = "0047_wiki_bootstrap_states"
down_revision = "0046_custom_ingest_tokens"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE wiki_synthesis_runs DROP CONSTRAINT IF EXISTS ck_wsr_status")
    op.execute(
        """
        ALTER TABLE wiki_synthesis_runs ADD CONSTRAINT ck_wsr_status CHECK (
            status IN ('pending','running','complete','failed','partial','cancelled')
        )
        """
    )


def downgrade() -> None:
    # Remap rows in the new states so the v3 CHECK passes.
    op.execute(
        "UPDATE wiki_synthesis_runs SET status = 'running' WHERE status = 'pending'"
    )
    op.execute(
        "UPDATE wiki_synthesis_runs SET status = 'failed' WHERE status = 'cancelled'"
    )
    op.execute("ALTER TABLE wiki_synthesis_runs DROP CONSTRAINT IF EXISTS ck_wsr_status")
    op.execute(
        """
        ALTER TABLE wiki_synthesis_runs ADD CONSTRAINT ck_wsr_status CHECK (
            status IN ('running','complete','failed','partial')
        )
        """
    )
