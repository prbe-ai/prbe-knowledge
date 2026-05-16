"""query_traces: add router-intelligence telemetry columns.

Revision ID: 0077_router_intel_telemetry
Revises: 0076_entity_clusters
Create Date: 2026-05-15

Adds seven nullable/defaulted columns so the v1 router emits enough
signal to answer "is Haiku good enough" without unnesting JSONB on
every aggregate query. See
docs/superpowers/specs/2026-05-14-router-intelligence-design.md §Telemetry.

(Rebase note: originally landed as 0071_router_intel_telemetry in the
feature branch; renumbered to 0077 when rebasing onto main, which had
advanced through 0071_chunks_embed_v1_nullable .. 0076_entity_clusters
in the interim.)
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0077_router_intel_telemetry"
down_revision = "0076_entity_clusters"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("query_traces", sa.Column("grounding_bundle", sa.dialects.postgresql.JSONB(), nullable=True))
    op.add_column("query_traces", sa.Column("router_raw", sa.dialects.postgresql.JSONB(), nullable=True))
    op.add_column("query_traces", sa.Column("intents_count", sa.Integer(), nullable=True))
    op.add_column("query_traces", sa.Column("intent_dispatch", sa.dialects.postgresql.JSONB(), nullable=True))
    op.add_column("query_traces", sa.Column("cache_tokens", sa.dialects.postgresql.JSONB(), nullable=True))
    op.add_column("query_traces", sa.Column("router_model", sa.Text(), nullable=True))
    op.add_column(
        "query_traces",
        sa.Column("failure_recovered", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    for col in (
        "failure_recovered", "router_model", "cache_tokens", "intent_dispatch",
        "intents_count", "router_raw", "grounding_bundle",
    ):
        op.drop_column("query_traces", col)
