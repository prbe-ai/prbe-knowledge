"""kg_candidates: debugging-agent observation queue with two-layer dedup

Revision ID: 0032_kg_candidates
Revises: 0031_kg_evidence
Create Date: 2026-04-30

Third migration in the Phase 1 foundation of the debugging knowledge
graph (see docs/superpowers/specs/2026-04-29-debugging-knowledge-graph-design.md
§5.1, §7.3). Adds the `kg_candidates` table — the queue the debugging
agent writes to when something is off (incident matched poorly,
traversal went outside the playbook, agent disagreed with playbook
ordering, etc.). The maintenance agent (Phase 2) drains it.

Two-layer dedup design (see spec §7.3 — read before changing this
schema):

  * `payload_hash` is a sha256 over STRUCTURED fields only (`type`,
    `top_class_match.id`, `score_bucket`, `agent_action`,
    `incident_signature_keys`). Prose `notes` are NEVER hashed; they
    live in `payload.notes` (JSONB) for the maintenance agent to read.
  * `payload_hash` is **NOT UNIQUE**. Two genuinely-different
    observations can share structured fields (e.g., a 401 from JWT
    clock-skew vs a 401 from tenant-config propagation lag, both
    classified as the same class with the same score bucket). The
    hash is a *clustering key* used to find layer-1 candidates for
    dedup, not a definitive dedup key. Treating hash collision as
    automatic dedup would silently lose distinct signal.
  * `notes_embedding vector(1536)` is the layer-2 signal. On insert,
    if any pending row in the same tenant shares `payload_hash`, the
    agent compares notes embeddings; only when cosine > 0.85 against
    a matching row is dedup confirmed (in which case `repeat_count`
    is incremented on the matching row, no new row written).
    Otherwise a new row is inserted with the same `payload_hash`.
  * `repeat_count` therefore counts CONFIRMED duplicates only.

Shape:
  * `(customer_id, candidate_id)` PRIMARY KEY — `candidate_id` is a
    UUID generated server-side via `gen_random_uuid()` (pgcrypto;
    already used elsewhere in the schema, e.g., `oauth_tokens`,
    `ingestion_events` — no `CREATE EXTENSION` needed here).
  * `customer_id TEXT REFERENCES customers(customer_id) ON DELETE
    CASCADE` — matches the existing repo convention (customers,
    usage_events, graph_nodes, kg_classes, kg_evidence all use TEXT
    customer_id) and lines up with the `app.current_customer_id` GUC
    used by RLS. No FK to `kg_classes`: a candidate may name a class
    via `payload->'top_class_match'->>'id'` but is intentionally not
    tightly coupled — candidates can predate class creation, and
    candidates referencing a deleted class should remain in the
    queue for the maintenance agent to triage.
  * `status` four-state machine enforced by CHECK constraint:
    `pending` → `accepted` (incorporated into a class), `pending`
    → `merged` (matched an existing class), `pending` → `rejected`
    (no action taken). See spec §7.2 step 7 and §7.3 closed-loop
    section.
  * `notes_embedding` is nullable — populated by the agent on insert
    (layer-2 lookup), but a fresh row written without a prior
    hash-collision skips the embedding step until a future collision
    forces it.
  * `resolved_at` is nullable — set when status leaves `pending`.

Indexes in this migration:
  * `kg_candidates_dedup` on `(customer_id, payload_hash, status,
    created_at)` — supports the layer-1 lookup "pending candidates
    with this hash in the last 24h for this tenant" (spec §7.3 step
    2). Leading `customer_id` keeps the index tenant-scoped under
    RLS; trailing `created_at` lets the planner range-scan the time
    window.

Out of scope for this migration (separate Phase 1 tasks):
  * ivfflat index on `notes_embedding` lands in Task 4 alongside the
    other vector indexes.
  * RLS enable + tenant_isolation policy (Task 5).

Why raw SQL via op.execute rather than op.create_table: keeping the
pattern consistent with 0030_kg_classes, 0031_kg_evidence, and the
other recent DDL (0024_queue_priority, 0026_queue_payload_keys), and
avoiding SQLAlchemy core's lack of a first-class pgvector type.
"""

from __future__ import annotations

from alembic import op

revision = "0032_kg_candidates"
down_revision = "0031_kg_evidence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE kg_candidates (
            customer_id      TEXT         NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            candidate_id     UUID         NOT NULL DEFAULT gen_random_uuid(),
            payload_hash     TEXT         NOT NULL,
            payload          JSONB        NOT NULL,
            notes_embedding  vector(1536),
            repeat_count     INTEGER      NOT NULL DEFAULT 1,
            status           TEXT         NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'accepted', 'rejected', 'merged')),
            created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            resolved_at      TIMESTAMPTZ,
            PRIMARY KEY (customer_id, candidate_id)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX kg_candidates_dedup
            ON kg_candidates (customer_id, payload_hash, status, created_at)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS kg_candidates_dedup")
    op.execute("DROP TABLE IF EXISTS kg_candidates")
