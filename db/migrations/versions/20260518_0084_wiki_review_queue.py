"""wiki_review_queue

Per-artifact review-state row for wiki artifacts produced by the
post-approval orchestrator pipeline (postmortem, knowledge_page,
correction). Each artifact has exactly one row keyed by
``(customer_id, artifact_doc_id)``; re-runs create a NEW row whose
``parent_artifact_doc_id`` points at the prior version (versions are
tracked via the parent-link chain, not as a jsonb history blob, so
queries can join the chain without parsing nested arrays).

Columns:
- ``state`` — lifecycle: ``pending_writeback`` (orchestrator writing
  the document body), ``pending_review`` (waiting on a reviewer),
  ``approved`` / ``rejected`` (terminal reviewer outcomes),
  ``failed_pending_review`` (orchestrator hit a non-retryable error
  but produced a partial artifact that still needs reviewer attention).
- ``artifact_kind`` — one of postmortem / knowledge_page / correction.
- ``target_doc_id`` — the document the artifact corrects. ONLY
  populated for ``correction`` artifacts; postmortems and knowledge
  pages are new docs, not corrections. Enforced via check constraint.
- ``metadata`` jsonb — orchestrator-side hints (model used, token
  budget, citations) so the review UI can surface "why this draft".

RLS: tenant_isolation bound to ``current_setting('app.current_customer_id')``
matching every other tenant table. ``FORCE ROW LEVEL SECURITY`` so
even the table owner role obeys the policy.

Revision ID: 0084_wiki_review_queue
Revises: 0083_inv_post_approval
Create Date: 2026-05-18
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers
revision = "0084_wiki_review_queue"
down_revision = "0083_inv_post_approval"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "wiki_review_queue",
        sa.Column("customer_id", sa.Text(), nullable=False),
        sa.Column("artifact_doc_id", sa.Text(), nullable=False),
        sa.Column("incident_doc_id", sa.Text(), nullable=False),
        sa.Column("artifact_kind", sa.Text(), nullable=False),
        sa.Column("target_doc_id", sa.Text(), nullable=True),
        sa.Column("parent_artifact_doc_id", sa.Text(), nullable=True),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("reviewer_id", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "metadata",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("customer_id", "artifact_doc_id"),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.customer_id"], ondelete="CASCADE",
            name="wiki_review_queue_customer_fkey",
        ),
        sa.CheckConstraint(
            "artifact_kind IN ('postmortem','knowledge_page','correction')",
            name="wiki_review_queue_kind_chk",
        ),
        sa.CheckConstraint(
            "state IN ('pending_writeback','pending_review','approved',"
            "'rejected','failed_pending_review')",
            name="wiki_review_queue_state_chk",
        ),
        sa.CheckConstraint(
            "(artifact_kind = 'correction' AND target_doc_id IS NOT NULL) "
            "OR (artifact_kind <> 'correction' AND target_doc_id IS NULL)",
            name="wiki_review_queue_target_consistency_chk",
        ),
    )
    op.execute("ALTER TABLE wiki_review_queue ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE wiki_review_queue FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY tenant_isolation ON wiki_review_queue "
        "USING (customer_id = current_setting('app.current_customer_id', true)) "
        "WITH CHECK (customer_id = current_setting('app.current_customer_id', true))"
    )
    op.create_index(
        "wiki_review_queue_state_idx",
        "wiki_review_queue",
        ["customer_id", "state", sa.text("updated_at DESC")],
    )
    op.create_index(
        "wiki_review_queue_incident_idx",
        "wiki_review_queue",
        ["customer_id", "incident_doc_id"],
    )


def downgrade() -> None:
    op.drop_index("wiki_review_queue_incident_idx", table_name="wiki_review_queue")
    op.drop_index("wiki_review_queue_state_idx", table_name="wiki_review_queue")
    op.drop_table("wiki_review_queue")
