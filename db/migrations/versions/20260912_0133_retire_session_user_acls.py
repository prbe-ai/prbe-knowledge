"""Retire the per-user ACL assertions on captured coding sessions.

Session documents used to carry one ACL principal: the author, as a USER. No
part of the system ever enforced it — the transcript route scopes by
customer_id alone and retrieval has no ACL filter — so every teammate could
already read every session. The row asserted a protection that did not exist,
which is worse than asserting none: it is the shape of guarantee someone reads
and then relies on. It was found while tracing a live AWS key that sat readable
by its author's whole team for ten days behind an ACL naming only them.

The connector now writes a WORKSPACE principal, which is what actually holds.
That fixes new sessions and leaves every already-ingested one asserting the old
thing — including exactly the historical sessions the 2026-08-30 investigation
was about. Without this migration the code change means less than its docstring
claims.

CLOSES OUT, DOES NOT DELETE. `acl_snapshots` is temporal: `valid_to` is how an
assertion stops being true, and the history of what was asserted, when, is the
point of the table. Deleting the rows would erase the evidence that we ever
made the claim.

Scope is deliberately narrow — USER principals on documents from the three
coding-agent connectors, and only rows still open. Other connectors' USER ACLs
(Slack, Linear, Notion) describe real upstream permissions and are untouched.
"""

from __future__ import annotations

from alembic import op

revision = "0133_retire_session_user_acls"
down_revision = "0132_backfill_chunks_project_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE acl_snapshots
           SET valid_to = NOW()
         WHERE valid_to IS NULL
           AND principal_type = 'user'
           AND resource_type = 'document'
           AND source_system IN ('claude_code', 'codex', 'pi')
        """
    )


def downgrade() -> None:
    # Re-opening every row this closed is not expressible: `valid_to` was NULL
    # before and NOW() after, and NOW() is also a legitimate value for an
    # assertion that genuinely ended at that moment. Reopening on the timestamp
    # would revive rows this migration never touched. The forward direction is
    # the honest one; there is nothing to restore that anything reads.
    pass
