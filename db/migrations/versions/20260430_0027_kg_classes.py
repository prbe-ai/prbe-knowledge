"""kg_classes: debugging knowledge graph class table

Revision ID: 0027_kg_classes
Revises: 0026_queue_payload_keys
Create Date: 2026-04-30

First migration in the Phase 1 foundation of the debugging knowledge
graph (see docs/superpowers/specs/2026-04-29-debugging-knowledge-graph-design.md
§5.1). Adds the core `kg_classes` table that holds one row per debugging
class per tenant.

Shape:
  * `frontmatter jsonb` — all structured fields (signature, related,
    context_sources, evidence). Stored as JSONB end-to-end; no YAML
    serialization round-trip.
  * `body text` — opaque markdown playbook prose with [[wiki-links]];
    DEFAULT '' so a class can be created with frontmatter only.
  * `signature_embedding vector(1536)` — pgvector embedding of the
    class signature seed. NULLABLE because the embed pass runs after
    write; classes start unembedded and get filled in by the
    maintenance worker / classifier (Phase 2).
  * `(customer_id, class_id)` PRIMARY KEY — class_id is the slug
    (e.g. "auth-401-jwt-refresh"); the customer_id prefix gives
    tenant isolation at the PK level on top of the RLS policy.

Out of scope for this migration (separate Phase 1 tasks):
  * GIN index on `frontmatter->'related'` for edge lookups (Task 4).
  * ivfflat index on `signature_embedding` for similarity search
    (Task 4).
  * RLS enable + tenant_isolation policy (Task 5).

The `vector` extension is already created in db/schema.sql; no
CREATE EXTENSION needed here.

customer_id is TEXT to match the existing repo convention
(customers, usage_events, graph_nodes all use TEXT customer_id), and
to allow the RLS policy in Task 5 to compare directly against the
`app.current_customer_id` GUC, which is set as TEXT. References
customers(customer_id) with ON DELETE CASCADE so a tenant offboard
cleans up its KG with the rest of its data.

Why raw SQL via op.execute rather than op.create_table: SQLAlchemy core
has no first-class pgvector type, and the existing migrations in this
repo already use op.execute for any DDL that doesn't map cleanly to
sa.Column (see 0024_queue_priority and 0026_queue_payload_keys, both
of which use raw SQL throughout). Keeping the pattern consistent.
"""

from __future__ import annotations

from alembic import op

revision = "0027_kg_classes"
down_revision = "0026_queue_payload_keys"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE kg_classes (
            customer_id          TEXT         NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            class_id             TEXT         NOT NULL,
            frontmatter          JSONB        NOT NULL,
            body                 TEXT         NOT NULL DEFAULT '',
            signature_embedding  vector(1536),
            created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            updated_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            PRIMARY KEY (customer_id, class_id)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS kg_classes")
