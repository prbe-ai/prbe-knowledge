"""retrieve_pages: the surplus of a retrieval, so a caller can page it."""

from pathlib import Path

from alembic import op

revision = "0129_retrieve_pages"
down_revision = "0128_session_receipts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute((Path(__file__).resolve().parents[3] / "kb/retrieve_pages_schema.sql").read_text())


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS retrieve_pages")
