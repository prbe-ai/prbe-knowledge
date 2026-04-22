"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-04-21

Executes db/schema.sql verbatim. The SQL file is canonical; this migration
does not redefine it here to avoid drift.
"""

from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCHEMA_SQL_PATH = Path(__file__).resolve().parents[3] / "db" / "schema.sql"


def upgrade() -> None:
    sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    op.execute(sql)


def downgrade() -> None:
    # Order matters: drop children before parents.
    op.execute("""
        DROP TABLE IF EXISTS query_cache CASCADE;
        DROP TABLE IF EXISTS graph_edges CASCADE;
        DROP TABLE IF EXISTS graph_nodes CASCADE;
        DROP TABLE IF EXISTS audit_log CASCADE;
        DROP TABLE IF EXISTS ingestion_events CASCADE;
        DROP TABLE IF EXISTS failed_chunks CASCADE;
        DROP TABLE IF EXISTS integration_tokens CASCADE;
        DROP TABLE IF EXISTS backfill_state CASCADE;
        DROP TABLE IF EXISTS ingestion_queue CASCADE;
        DROP TABLE IF EXISTS acl_snapshots CASCADE;
        DROP TABLE IF EXISTS chunks CASCADE;
        DROP TABLE IF EXISTS documents CASCADE;
        DROP TABLE IF EXISTS customers CASCADE;
    """)
