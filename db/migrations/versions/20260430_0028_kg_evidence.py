"""kg_evidence: episodic learning trail for debugging knowledge graph

Revision ID: 0028_kg_evidence
Revises: 0027_kg_classes
Create Date: 2026-04-30

Second migration in the Phase 1 foundation of the debugging knowledge
graph (see docs/superpowers/specs/2026-04-29-debugging-knowledge-graph-design.md
§5.1, §7). Adds the `kg_evidence` table — the episodic learning trail
referenced by the maintenance agent.

Distinct from the `frontmatter.evidence` summary on `kg_classes` (which
holds aggregate counts/exemplars rendered into the playbook): this
table records one row per refinement observation tied to a specific
ticket. The maintenance agent reads it on each run to decide what to
adjust on the class. A single ticket can produce multiple refinement
rows over time, so `observed_at` is part of the primary key.

Shape:
  * `(customer_id, class_id, ticket_id, observed_at)` PRIMARY KEY —
    composite to allow multiple refinements per (customer, class,
    ticket) without collisions.
  * `customer_id` TEXT — matches the existing repo convention
    (customers, usage_events, graph_nodes, kg_classes all use TEXT
    customer_id) and lines up with the `app.current_customer_id` GUC
    used by RLS.
  * `(customer_id, class_id)` composite FK to
    `kg_classes(customer_id, class_id)` with ON DELETE CASCADE so
    deleting a class drops its full evidence trail. This requires the
    existing PRIMARY KEY on kg_classes to back the reference, which
    it does (Task 1 / 0027).
  * `customer_id` also FKs to `customers(customer_id)` ON DELETE
    CASCADE so a tenant offboard cleans up evidence even if the
    parent class row was already gone.
  * `refinement` TEXT — opaque prose describing what the maintenance
    agent decided to change based on this ticket. No length limit
    enforced at the DB layer; long-form notes are expected.
  * `observed_at TIMESTAMPTZ DEFAULT NOW()` — when the refinement was
    recorded.

Out of scope for this migration (separate Phase 1 tasks):
  * RLS enable + tenant_isolation policy (Task 5).
  * Any indexes beyond the implicit PK index (none required at
    foundation; the access pattern is "all evidence for a given
    (customer_id, class_id)" which the leading PK columns already
    cover).

Why raw SQL via op.execute rather than op.create_table: keeping the
pattern consistent with 0027_kg_classes and the other recent DDL
(0024_queue_priority, 0026_queue_payload_keys), and keeping the
composite FK readable inline.
"""

from __future__ import annotations

from alembic import op

revision = "0028_kg_evidence"
down_revision = "0027_kg_classes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE kg_evidence (
            customer_id  TEXT         NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            class_id     TEXT         NOT NULL,
            ticket_id    TEXT         NOT NULL,
            refinement   TEXT         NOT NULL,
            observed_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            PRIMARY KEY (customer_id, class_id, ticket_id, observed_at),
            FOREIGN KEY (customer_id, class_id)
                REFERENCES kg_classes(customer_id, class_id)
                ON DELETE CASCADE
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS kg_evidence")
