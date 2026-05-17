"""query_traces: add trace_blob_key pointer to the R2 transcript.

Revision ID: 0079_search_agent_trace_blob
Revises: 0078_search_agent_telemetry
Create Date: 2026-05-17

0078 added the summary columns (gatherer_status, tool_calls_count, etc).
This migration adds the pointer into R2 where the full per-turn agent
transcript lives. The blob carries state.messages, per-turn cache hit
rates, the final GathererOutput, status, timing, and the raw query for
nightly trace-analyzer to read.

Column:
- trace_blob_key (text, nullable) — R2 object key inside the per-tenant
                                    bucket, e.g.
                                    `search-traces/2026-05-17/<request_id>.json.gz`.
                                    NULL on sampled-out rows and on rows
                                    where the R2 PUT failed. Lookup by
                                    `request_id` via the existing
                                    `idx_query_traces_request_id` index;
                                    no new index needed.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0079_search_agent_trace_blob"
down_revision = "0078_search_agent_telemetry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("query_traces", sa.Column("trace_blob_key", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("query_traces", "trace_blob_key")
