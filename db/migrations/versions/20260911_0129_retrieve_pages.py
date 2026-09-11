"""retrieve_pages: the surplus of a retrieval, so a caller can page it.

THE GRANT IS THE HALF THAT IS EASY TO FORGET. The app role differs by
deployment -- `app` on research, `probe_app` on managed, possibly neither on a
self-host install -- so a CREATE TABLE alone leaves the engine unable to write
here. That failure is quiet by design on the write path (a page that cannot be
stored costs a cursor, never an answer), which means it would have looked
exactly like "pagination shipped and nobody used it". The DO block grants to
whichever known app role actually exists and no-ops otherwise, following
0112's pattern for the same reason it was written.
"""

from pathlib import Path

from alembic import op

revision = "0129_retrieve_pages"
down_revision = "0128_session_receipts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # One source: the same file `db/schema.sql` embeds, so a migrated database
    # and a freshly bootstrapped one cannot differ -- including the grant.
    op.execute((Path(__file__).resolve().parents[3] / "kb/retrieve_pages_schema.sql").read_text())


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS retrieve_pages")
