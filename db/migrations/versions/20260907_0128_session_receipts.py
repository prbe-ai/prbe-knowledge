"""Immutable protocol-2 transcript acceptance receipts."""

from pathlib import Path

from alembic import op

revision = "0128_session_receipts"
down_revision = "0127_github_installation_control"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute((Path(__file__).resolve().parents[3] / "kb/session_receipts_schema.sql").read_text())


def downgrade() -> None:
    raise RuntimeError(
        "Session receipts fence legacy writers and must survive application rollback"
    )
