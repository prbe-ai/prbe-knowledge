"""merge two 0032 heads

Revision ID: 0035_merge_heads
Revises: 0032_manual_uploads, 0033_wiki_synthesis_no_rls
Create Date: 2026-05-03

Two parallel migrations both branched from 0031_codex_device_source_bkfl:
  * 0032_manual_uploads
  * 0032_wiki_synthesis (renumbered to 0033) -> 0033_wiki_synthesis_no_rls

`alembic upgrade head` refused to run while both were unmerged heads, which
broke deploy-ingestion on every push since PR #75 landed (06:27 UTC),
which in turn skipped deploy-retrieval / deploy-worker / deploy-poller.
This is a no-op merge that collapses the DAG so future deploys can upgrade
cleanly. No DDL — both parent revisions already carry their own.
"""

from __future__ import annotations

revision = "0035_merge_heads"
down_revision = ("0032_manual_uploads", "0033_wiki_synthesis_no_rls")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
