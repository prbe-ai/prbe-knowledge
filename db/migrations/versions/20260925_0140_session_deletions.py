"""`session_deletions`: deleted coding-agent sessions stay deleted.

A customer can ask for specific captured sessions, or every session one person
authored, to be erased (kb/session_deletion.py). Removing the rows and raw
objects is not enough on its own: the capture client retries and re-uploads,
and the worker rebuilds a session from its raw batches on every pass. This
table is the record every writer checks under the session's lock before it
writes (engine/shared/session_suppression.py), and the journal a deletion run
resumes from.

New table only; no existing row is touched. The DDL lives in
kb/session_deletions_schema.sql, which db/schema.sql mirrors (CI builds from
schema.sql and stamps head, it never replays this chain).
"""

from __future__ import annotations

from pathlib import Path

from alembic import op

revision = "0140_session_deletions"
down_revision = "0138_queue_extraction_outcome"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        (Path(__file__).resolve().parents[3] / "kb/session_deletions_schema.sql").read_text()
    )


def downgrade() -> None:
    raise RuntimeError(
        "session_deletions keeps deleted sessions from being re-uploaded; dropping it "
        "would let a retrying client restore data a customer asked us to erase"
    )
