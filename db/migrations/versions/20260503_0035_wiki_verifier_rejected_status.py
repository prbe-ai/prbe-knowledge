"""Add 'verifier_rejected' status to wiki_synthesis_queue

Revision ID: 0035_wiki_verifier_rejected
Revises: 0034_wiki_synthesis_no_rls
Create Date: 2026-05-03

The upcoming triage redesign adds a verifier stage between Haiku/Gemini
triage and the synthesize call. The verifier may decide a triaged cluster
of events isn't actually wiki-worthy after a closer look — e.g. they
restate content already on the page. Previously those rows would land in
status='done' with synthesis_error set, conflating verifier-reject with
successful-synthesize. Adding a distinct terminal state so dashboard /
audit queries can tell the difference: 'done' = wiki page written;
'verifier_rejected' = verifier said no.

Single ALTER with DROP CONSTRAINT + ADD CONSTRAINT in one statement is
atomic — no window where the table is unconstrained.
"""

from __future__ import annotations

from alembic import op

revision = "0035_wiki_verifier_rejected"
down_revision = "0034_wiki_synthesis_no_rls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE wiki_synthesis_queue
            DROP CONSTRAINT ck_wsq_status,
            ADD CONSTRAINT ck_wsq_status CHECK (
                status IN ('pending','triaging','triaged','rejected',
                           'synthesizing','done','failed','verifier_rejected')
            )
        """
    )


def downgrade() -> None:
    # Map any verifier_rejected rows to 'done' before re-tightening the
    # constraint, otherwise ADD CONSTRAINT will fail validation.
    op.execute(
        """
        UPDATE wiki_synthesis_queue
        SET status = 'done',
            synthesis_error = COALESCE(synthesis_error, '')
                              || ' (downgraded from verifier_rejected)'
        WHERE status = 'verifier_rejected'
        """
    )
    op.execute(
        """
        ALTER TABLE wiki_synthesis_queue
            DROP CONSTRAINT ck_wsq_status,
            ADD CONSTRAINT ck_wsq_status CHECK (
                status IN ('pending','triaging','triaged','rejected',
                           'synthesizing','done','failed')
            )
        """
    )
