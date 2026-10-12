"""`session_media`: the pasted images of captured sessions.

A capture client asks for an upload URL per pasted image (kb/session_media.py,
`POST /api/sessions/{source}/{session_id}/media`); the bytes go to a staging
key and the worker promotes them into the session's folder once a batch that
names the image is accepted. This table records each URL handed out (the
customer's daily budget reads it) and when its image was promoted.

New table only; no existing row is touched, so there is nothing to backfill
and FORCE RLS cannot silence anything here. The DDL lives in
kb/session_media_schema.sql, which db/schema.sql mirrors (CI builds from
schema.sql and stamps head, it never replays this chain).
"""

from __future__ import annotations

from pathlib import Path

from alembic import op

revision = "0149_session_media"
down_revision = "0148_session_streams_protocol_3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        (Path(__file__).resolve().parents[3] / "kb/session_media_schema.sql").read_text()
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS session_media")
