"""query_traces: add search-agent (gatherer) telemetry columns.

Revision ID: 0078_search_agent_telemetry
Revises: 0077_router_intel_telemetry
Create Date: 2026-05-16

Extends PR #282's 0077 telemetry with agent-loop fields so production
traces answer "how is the gatherer behaving" without an unnest on every
aggregate query.

Columns:
- gatherer_status (text)       — one of "ok", "passthrough_harness_fallback",
                                 "loop_timeout", "schema_violation". Null on
                                 router-only rows (pre-cutover or schema-skip).
- tool_calls_count (int)       — total tool calls fired across all turns.
- need_deeper_extensions (int) — 0/1/2; how many soft-budget extensions the
                                 agent asked for.
- confidence (text)            — "high"/"medium"/"low" self-reported by the
                                 agent on its final emission.
- dropped_count (int)          — entries in GathererOutput.gatherer_notes.dropped.
                                 Full list lives in R2 keyed by trace_id.
- cache_hit_rate (numeric 4,3) — averaged across turns:
                                 (cache_read_tokens / total_input_tokens).
                                 Acceptance gate: stay >= 0.70.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0078_search_agent_telemetry"
down_revision = "0077_router_intel_telemetry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("query_traces", sa.Column("gatherer_status", sa.Text(), nullable=True))
    op.add_column("query_traces", sa.Column("tool_calls_count", sa.Integer(), nullable=True))
    op.add_column("query_traces", sa.Column("need_deeper_extensions", sa.Integer(), nullable=True))
    op.add_column("query_traces", sa.Column("confidence", sa.Text(), nullable=True))
    op.add_column("query_traces", sa.Column("dropped_count", sa.Integer(), nullable=True))
    op.add_column("query_traces", sa.Column("cache_hit_rate", sa.Numeric(4, 3), nullable=True))


def downgrade() -> None:
    for col in (
        "cache_hit_rate",
        "dropped_count",
        "confidence",
        "need_deeper_extensions",
        "tool_calls_count",
        "gatherer_status",
    ):
        op.drop_column("query_traces", col)
