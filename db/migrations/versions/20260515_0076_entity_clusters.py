"""Entity clusters: tables + graph_edges changes for manual identity merging (B-promote).

Revision ID: 0076_entity_clusters
Revises: 0075_r2_bucket_not_null
Create Date: 2026-05-15

Five new RLS-isolated tables backing manual entity merging (design doc:
``docs/superpowers/specs/2026-05-13-entity-clusters-design.md``):

  * ``entity_merge_audit`` -- one row per merge action.
  * ``entity_merge_node_snapshot`` -- full pre-state of each hard-deleted
    alias node (incl. inlined provenance JSONB).
  * ``entity_merge_edge_snapshot`` -- pre-state of each edge deleted as a
    self-loop during merge (the only deletion type under composite UNIQUE).
  * ``entity_aliases`` -- forward routing (alias -> primary), consulted by
    ``graph_writer`` at ingest and by retrieval at anchor lookup (Phase 2).
  * ``entity_cluster_metadata`` -- sparse display-name override; created
    empty, populated only by the Phase 3 dashboard.

Plus two ``graph_edges`` schema changes:

  * New nullable columns ``aliased_from_canonical_id``,
    ``aliased_to_canonical_id`` -- provenance for alias-resolved or
    merge-rewritten edges. NULL for all existing rows.
  * The existing single-column UNIQUE
    ``(customer_id, edge_type, from_node_id, to_node_id)`` is replaced by
    a composite UNIQUE INDEX that COALESCEs the two new alias columns to ''.
    This lets different alias lanes coexist as distinct rows (post-merge),
    while still dedupping the common-case "both aliased_from cols NULL"
    inserts (the existing graph_writer upsert path).

We use a ``CREATE UNIQUE INDEX`` (not a UNIQUE constraint) because the
COALESCE expressions are only legal inside an expression index in PG 16.
``ON CONFLICT`` can reference the index by column list or by name, so
this is functionally equivalent for the graph_writer upsert path.

Migration is purely additive: all new tables are empty at creation, and
the column ADDs on graph_edges are instant (ADD COLUMN on a nullable TEXT
is O(1) in PG 11+).

Verification
------------
::

    \\d entity_merge_audit
    \\d entity_aliases
    \\d entity_cluster_metadata

    SELECT relname, relrowsecurity AS rls, relforcerowsecurity AS forced
    FROM   pg_class
    WHERE  relname IN ('entity_merge_audit', 'entity_aliases',
                       'entity_cluster_metadata',
                       'entity_merge_node_snapshot',
                       'entity_merge_edge_snapshot');

    \\d graph_edges  -- verify the new index and columns
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0076_entity_clusters"
down_revision: str | Sequence[str] | None = "0075_r2_bucket_not_null"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_POLICY_EXPR = "customer_id = current_setting('app.current_customer_id', true)"


def upgrade() -> None:
    # ----- entity_merge_audit (created first; entity_aliases FKs it) -----
    op.execute(
        """
        CREATE TABLE entity_merge_audit (
            merge_id                    UUID PRIMARY KEY,
            customer_id                 TEXT NOT NULL
                                        REFERENCES customers(customer_id)
                                        ON DELETE CASCADE,
            label                       TEXT NOT NULL,
            primary_canonical_id        TEXT NOT NULL,
            merged_alias_canonical_ids  TEXT[] NOT NULL,
            performed_by_user_id        UUID NOT NULL,
            performed_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            reason                      TEXT NULL,
            status                      TEXT NOT NULL DEFAULT 'active'
                                        CHECK (status IN ('active', 'reversed'))
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_entity_merge_audit_primary "
        "ON entity_merge_audit (customer_id, label, primary_canonical_id)"
    )

    # ----- entity_merge_node_snapshot -----
    op.execute(
        """
        CREATE TABLE entity_merge_node_snapshot (
            merge_id      UUID NOT NULL REFERENCES entity_merge_audit(merge_id),
            customer_id   TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            label         TEXT NOT NULL,
            canonical_id  TEXT NOT NULL,
            properties    JSONB NOT NULL,
            degree        INT  NOT NULL,
            community_id  INT  NULL,
            created_at    TIMESTAMPTZ NOT NULL,
            provenance    JSONB NOT NULL,
            PRIMARY KEY (merge_id, label, canonical_id)
        )
        """
    )

    # ----- entity_merge_edge_snapshot -----
    op.execute(
        """
        CREATE TABLE entity_merge_edge_snapshot (
            merge_id                       UUID NOT NULL REFERENCES entity_merge_audit(merge_id),
            snapshot_seq                   INT  NOT NULL,
            customer_id                    TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            operation                      TEXT NOT NULL
                                           CHECK (operation IN ('deleted_self_loop')),
            pre_edge_type                  TEXT NOT NULL,
            pre_from_canonical_id          TEXT NOT NULL,
            pre_from_label                 TEXT NOT NULL,
            pre_to_canonical_id            TEXT NOT NULL,
            pre_to_label                   TEXT NOT NULL,
            pre_properties                 JSONB NOT NULL,
            pre_confidence                 TEXT NOT NULL,
            pre_valid_from                 TIMESTAMPTZ NOT NULL,
            pre_valid_to                   TIMESTAMPTZ NULL,
            pre_source_system              TEXT NULL,
            pre_extractor_id               TEXT NULL,
            pre_extracted_at               TIMESTAMPTZ NULL,
            pre_aliased_from_canonical_id  TEXT NULL,
            pre_aliased_to_canonical_id    TEXT NULL,
            PRIMARY KEY (merge_id, snapshot_seq)
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_entity_merge_edge_snapshot_merge "
        "ON entity_merge_edge_snapshot (merge_id)"
    )

    # ----- entity_aliases -----
    op.execute(
        """
        CREATE TABLE entity_aliases (
            customer_id           TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            label                 TEXT NOT NULL,
            alias_canonical_id    TEXT NOT NULL,
            primary_canonical_id  TEXT NOT NULL,
            merge_id              UUID NOT NULL REFERENCES entity_merge_audit(merge_id),
            added_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (customer_id, label, alias_canonical_id),
            CONSTRAINT entity_aliases_not_self
                CHECK (alias_canonical_id <> primary_canonical_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_entity_aliases_primary "
        "ON entity_aliases (customer_id, label, primary_canonical_id)"
    )
    op.execute(
        "CREATE INDEX idx_entity_aliases_merge "
        "ON entity_aliases (merge_id)"
    )

    # ----- entity_cluster_metadata -----
    op.execute(
        """
        CREATE TABLE entity_cluster_metadata (
            customer_id                  TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            label                        TEXT NOT NULL,
            primary_canonical_id         TEXT NOT NULL,
            display_name                 TEXT NOT NULL,
            display_name_last_edited_by  UUID NULL,
            display_name_last_edited_at  TIMESTAMPTZ NULL,
            PRIMARY KEY (customer_id, label, primary_canonical_id)
        )
        """
    )

    # ----- RLS: ENABLE + FORCE + USING/WITH CHECK on all 5 new tables -----
    for table in (
        "entity_merge_audit",
        "entity_merge_node_snapshot",
        "entity_merge_edge_snapshot",
        "entity_aliases",
        "entity_cluster_metadata",
    ):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY tenant_isolation ON {table}
                USING ({_POLICY_EXPR})
                WITH CHECK ({_POLICY_EXPR})
            """
        )

    # ----- graph_edges: drop old UNIQUE, add columns, create new UNIQUE INDEX -----
    # Order is critical: drop the old constraint BEFORE adding the new columns
    # and creating the new index, or both UNIQUE definitions would coexist.
    op.execute(
        "ALTER TABLE graph_edges "
        "DROP CONSTRAINT graph_edges_customer_id_edge_type_from_node_id_to_node_id_key"
    )
    op.execute(
        """
        ALTER TABLE graph_edges
            ADD COLUMN aliased_from_canonical_id TEXT NULL,
            ADD COLUMN aliased_to_canonical_id   TEXT NULL
        """
    )
    # UNIQUE INDEX (not UNIQUE CONSTRAINT) -- only expression indexes accept
    # COALESCE(...) in their key list in PG 16. Equivalent for ON CONFLICT.
    op.execute(
        """
        CREATE UNIQUE INDEX graph_edges_unique_lane ON graph_edges (
            customer_id, edge_type, from_node_id, to_node_id,
            COALESCE(aliased_from_canonical_id, ''),
            COALESCE(aliased_to_canonical_id, '')
        )
        """
    )


def downgrade() -> None:
    # graph_edges first -- restore original UNIQUE constraint.
    op.execute("DROP INDEX IF EXISTS graph_edges_unique_lane")
    op.execute("ALTER TABLE graph_edges DROP COLUMN IF EXISTS aliased_to_canonical_id")
    op.execute("ALTER TABLE graph_edges DROP COLUMN IF EXISTS aliased_from_canonical_id")
    op.execute(
        """
        ALTER TABLE graph_edges
            ADD CONSTRAINT graph_edges_customer_id_edge_type_from_node_id_to_node_id_key
                UNIQUE (customer_id, edge_type, from_node_id, to_node_id)
        """
    )

    # Drop tables in reverse FK order (aliases / snapshots -> audit).
    for table in (
        "entity_cluster_metadata",
        "entity_aliases",
        "entity_merge_edge_snapshot",
        "entity_merge_node_snapshot",
        "entity_merge_audit",
    ):
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"DROP TABLE IF EXISTS {table}")
