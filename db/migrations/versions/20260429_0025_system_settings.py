"""system_settings table + ingestion_killswitch row

Revision ID: 0025_system_settings
Revises: 0024_queue_priority
Create Date: 2026-04-29

Adds a small key/value store for global config knobs read at request time.
First user is the global ingestion killswitch — a single row whose JSONB
value gates every plugin webhook. Default state is enabled (open).

Schema choice — JSONB instead of typed columns — so future settings
(per-customer overrides, per-source toggles, feature flags) can land
without a migration. Per-key cache lives in services/system_settings/store.py.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0025_system_settings"
down_revision = "0024_queue_priority"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "system_settings",
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("updated_by", sa.Text(), nullable=True),
    )

    # Seed the killswitch row in the enabled (open) state. Idempotent —
    # ON CONFLICT DO NOTHING means rerunning the migration on a DB that
    # already has the row leaves it alone.
    op.execute(
        sa.text(
            """
            INSERT INTO system_settings (key, value, description)
            VALUES (
                'ingestion_killswitch',
                '{"enabled": true, "reason": null}'::jsonb,
                'Master switch for all plugin ingestion. Set value->>enabled to false to halt webhooks globally.'
            )
            ON CONFLICT (key) DO NOTHING
            """
        )
    )


def downgrade() -> None:
    op.drop_table("system_settings")
