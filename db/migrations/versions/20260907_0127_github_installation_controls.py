"""Installation-scoped GitHub live controls and durable v2 history jobs.

Additive cutover: new queue status values are intentionally invisible to old
workers. Legacy token/backfill rows are not rewritten or removed. Controls only
apply after a caller explicitly adopts protocol 2. Worker fleet enumeration reads
customers, then enters tenant RLS before touching either control table.
"""

from pathlib import Path

from alembic import op

revision = "0127_github_installation_control"
down_revision = "0126_drop_full_hnsw_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = Path(__file__).resolve().parents[3] / "kb" / "github_control_schema.sql"
    # Migration and bootstrap share the exact additive definitions.
    op.execute(schema.read_text())


def downgrade() -> None:
    raise RuntimeError("GitHub v2 jobs must be drained before a coordinated rollback")
