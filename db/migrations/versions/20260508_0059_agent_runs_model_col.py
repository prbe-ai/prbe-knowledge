"""agent_runs: add model column for spend attribution

Revision ID: 0059_agent_runs_model_col
Revises: 0058_rename_to_agent_runs
Create Date: 2026-05-08

The orchestrator already tracks token_usage_input / token_usage_output
per run, but not which model produced those tokens. Without the model
id, downstream spend queries can't price the tokens correctly --
gemini-3-flash-preview, sonnet-4.x, and opus-4.x have different rates,
and which one ran is currently invisible.

Add a single nullable TEXT column. The orchestrator's mark_done writer
will populate it on success; rows from failed / pending runs leave it
NULL. Purely additive -- no rewrite, no constraint, no risk to readers
that don't know about the column yet.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0059_agent_runs_model_col"
down_revision = "0058_rename_to_agent_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column("model", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "model")
