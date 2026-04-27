"""per-tenant primary keys on documents and chunks

Revision ID: 0012_per_tenant_pk
Revises: 0011_graph_source_system
Create Date: 2026-04-26

doc_id and chunk_id are computed deterministically from source-side data
(`source_system:channel:ts` and `doc_id + content_hash` respectively). When
two tenants ingest the same source content (e.g. the same Slack workspace
during testing or as part of a workspace handoff), the second tenant's
INSERT collides with the first tenant's PK and `ON CONFLICT DO NOTHING`
silently drops the write — leaving customer B with zero docs even though
their queue ran clean.

Fix: include customer_id in both PKs so the same source identity can coexist
across tenants. customer_id is already NOT NULL on both tables, so the
upgrade is data-safe; existing rows can't collide on the wider key.
"""

from __future__ import annotations

from alembic import op

revision = "0012_per_tenant_pk"
down_revision = "0011_graph_source_system"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE documents DROP CONSTRAINT documents_pkey;
        ALTER TABLE documents ADD PRIMARY KEY (customer_id, doc_id, version);

        ALTER TABLE chunks DROP CONSTRAINT chunks_pkey;
        ALTER TABLE chunks ADD PRIMARY KEY (customer_id, chunk_id);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE chunks DROP CONSTRAINT chunks_pkey;
        ALTER TABLE chunks ADD PRIMARY KEY (chunk_id);

        ALTER TABLE documents DROP CONSTRAINT documents_pkey;
        ALTER TABLE documents ADD PRIMARY KEY (doc_id, version);
        """
    )
