"""collapse graph_nodes labels: 13 → 4 (Document | Person | Feature | CodeSymbol)

Revision ID: 0091_collapse_node_labels
Revises: 0090_agent_runs_if_missing
Create Date: 2026-05-20

Context: see plan ~/.claude/plans/unified-conjuring-peacock.md

Label remap:
  PR | Issue | Ticket | Channel | Repo            → Document
  WikiPerson                                       → Person
  Function | Class | Method | Module | Symbol     → CodeSymbol
  Feature, Person, Document, and other domain labels untouched.

Phase 0 — handle WikiPerson ↔ Person canonical_id collisions
  Unique constraint graph_nodes_customer_id_label_canonical_id_key permits the
  same canonical_id under two labels today (e.g. acme has
  `AshwaryeYadav` as both Person and WikiPerson). After the Phase 1 relabel
  both rows would key to (customer_id, 'Person', 'AshwaryeYadav') and violate
  the constraint. Phase 0 resolves these by promoting the WikiPerson row's
  edges + provenance onto the surviving Person row, then deleting the
  WikiPerson row.

  graph_edges has ON DELETE CASCADE from graph_nodes, so we MUST re-point
  edges before the DELETE — or the WikiPerson row's edges go with it.

  graph_edges_unique_lane is keyed by
    (customer_id, edge_type, from_node_id, to_node_id,
     COALESCE(aliased_from_canonical_id, ''),
     COALESCE(aliased_to_canonical_id, ''))
  so the re-point UPDATE filters out cases where the target Person row already
  has an edge with the same (edge_type, from, to, aliased_*). The leftover
  WikiPerson edge is dropped by the cascading DELETE in Phase 0 step 4.

Phase 1 — bulk relabel via CASE UPDATE.

Phase 2 — partial functional indexes on Person.properties->>{employee_id,
  login, email}. Targets the new resolve_to_person_canonical_ids retrieval
  path. CREATE INDEX CONCURRENTLY uses alembic's autocommit_block per the
  pattern in migration 0015_documents_listing_index.

Down-migration is LOSSY. Re-maps CodeSymbol → Function (no way to recover the
pre-collapse Module/Class/Method distinction) and leaves Document rows that
were PR/Issue/Ticket/Channel/Repo alone (no way to recover their kind from
the row alone — properties.kind survives the upgrade for entity-shape
Document rows but isn't restored to the label on downgrade). Provided only
so local dev can roundtrip; never run against managed-shared. See downgrade()
docstring.

NOTE: this PR removes the deprecated NodeLabel enum members in the SAME
deploy as the migration (single-shot, no observation window). The rolling
deploy means old pods running the previous binary still emit the deprecated
labels during their last few seconds before termination — those rows
become orphans that the auto-merge backfill cleans up post-deploy.

NOTE: revision id "0091_collapse_node_labels" is 27 chars (<=32) per
alembic_version.version_num cap (feedback_alembic_version_32char_cap).
"""

from __future__ import annotations

from alembic import op

revision = "0091_collapse_node_labels"
down_revision = "0090_agent_runs_if_missing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- Phase 0: WikiPerson ↔ Person canonical_id collision cleanup ----
    #
    # Capture (keep_id, drop_id) pairs into a temp table once. Temp tables
    # are session-scoped so they survive across op.execute calls within
    # this migration's transaction.
    op.execute(
        """
        CREATE TEMP TABLE _wikiperson_dups (
            keep_id BIGINT NOT NULL,
            drop_id BIGINT NOT NULL
        ) ON COMMIT DROP;

        INSERT INTO _wikiperson_dups (keep_id, drop_id)
        SELECT a.node_id, b.node_id
        FROM graph_nodes a
        JOIN graph_nodes b
          ON a.customer_id  = b.customer_id
         AND a.canonical_id = b.canonical_id
         AND a.label = 'Person'
         AND b.label = 'WikiPerson';
        """
    )

    # Re-point graph_edges.from_node_id from the dup row to the kept row.
    # NOT EXISTS guard preserves graph_edges_unique_lane uniqueness — if the
    # kept row already has an equivalent edge, we drop the dup edge (which
    # dies via CASCADE in Phase 0 step 4).
    op.execute(
        """
        UPDATE graph_edges e
        SET from_node_id = d.keep_id
        FROM _wikiperson_dups d
        WHERE e.from_node_id = d.drop_id
          AND NOT EXISTS (
            SELECT 1 FROM graph_edges e2
            WHERE e2.customer_id  = e.customer_id
              AND e2.edge_type    = e.edge_type
              AND e2.from_node_id = d.keep_id
              AND e2.to_node_id   = e.to_node_id
              AND COALESCE(e2.aliased_from_canonical_id, '')
                  = COALESCE(e.aliased_from_canonical_id, '')
              AND COALESCE(e2.aliased_to_canonical_id, '')
                  = COALESCE(e.aliased_to_canonical_id, '')
          );
        """
    )

    # Mirror for to_node_id.
    op.execute(
        """
        UPDATE graph_edges e
        SET to_node_id = d.keep_id
        FROM _wikiperson_dups d
        WHERE e.to_node_id = d.drop_id
          AND NOT EXISTS (
            SELECT 1 FROM graph_edges e2
            WHERE e2.customer_id  = e.customer_id
              AND e2.edge_type    = e.edge_type
              AND e2.from_node_id = e.from_node_id
              AND e2.to_node_id   = d.keep_id
              AND COALESCE(e2.aliased_from_canonical_id, '')
                  = COALESCE(e.aliased_from_canonical_id, '')
              AND COALESCE(e2.aliased_to_canonical_id, '')
                  = COALESCE(e.aliased_to_canonical_id, '')
          );
        """
    )

    # Promote provenance rows from the dup row to the kept row. Conflict on
    # PRIMARY KEY (node_id, source_system) leaves the existing kept-row
    # provenance alone. graph_node_provenance columns verified against
    # db/schema.sql:677-684.
    op.execute(
        """
        INSERT INTO graph_node_provenance (
            node_id, customer_id, source_system, first_seen_at, last_seen_at
        )
        SELECT d.keep_id, p.customer_id, p.source_system,
               p.first_seen_at, p.last_seen_at
        FROM graph_node_provenance p
        JOIN _wikiperson_dups d ON p.node_id = d.drop_id
        ON CONFLICT (node_id, source_system) DO NOTHING;
        """
    )

    # Delete the WikiPerson dup row. CASCADE cleans up any residual edges +
    # provenance keyed to drop_id.
    op.execute(
        """
        DELETE FROM graph_nodes
        WHERE node_id IN (SELECT drop_id FROM _wikiperson_dups);
        """
    )

    # ---- Phase 1: bulk relabel ----
    op.execute(
        """
        UPDATE graph_nodes SET label = CASE
            WHEN label IN ('PR','Issue','Ticket','Channel','Repo') THEN 'Document'
            WHEN label = 'WikiPerson' THEN 'Person'
            WHEN label IN ('Function','Class','Method','Module','Symbol') THEN 'CodeSymbol'
            ELSE label
        END
        WHERE label IN (
            'PR','Issue','Ticket','Channel','Repo',
            'WikiPerson',
            'Function','Class','Method','Module','Symbol'
        );
        """
    )

    # entity_aliases.label mirrors graph_nodes.label for the cluster key.
    op.execute(
        """
        UPDATE entity_aliases SET label = 'Person'
        WHERE label = 'WikiPerson';
        """
    )

    # ---- Phase 2: partial functional indexes for Person property lookups ----
    # CREATE INDEX CONCURRENTLY cannot run inside a transaction. Use alembic's
    # autocommit_block per migration 0015_documents_listing_index pattern.
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS
                idx_graph_nodes_person_employee_id
            ON graph_nodes ((properties->>'employee_id'))
            WHERE label = 'Person' AND properties->>'employee_id' IS NOT NULL;
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS
                idx_graph_nodes_person_login
            ON graph_nodes ((properties->>'login'))
            WHERE label = 'Person' AND properties->>'login' IS NOT NULL;
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS
                idx_graph_nodes_person_email
            ON graph_nodes ((properties->>'email'))
            WHERE label = 'Person' AND properties->>'email' IS NOT NULL;
            """
        )


def downgrade() -> None:
    """LOSSY rollback — for local dev only.

    Information lost on collapse cannot be recovered: a Document node
    that was originally a PR vs. Issue vs. Repo is indistinguishable
    post-collapse. This downgrade re-maps each collapsed group to a
    single representative old label so the alembic chain is reversible
    in name only.

    Never run against managed-shared CNPG. The two-phase enum cleanup
    plan keeps the old NodeLabel members defined in code during the
    observation window, so a deploy-time downgrade is unnecessary.
    """
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX IF EXISTS idx_graph_nodes_person_email;")
        op.execute("DROP INDEX IF EXISTS idx_graph_nodes_person_login;")
        op.execute("DROP INDEX IF EXISTS idx_graph_nodes_person_employee_id;")

    # Best-effort label remap; cannot reconstruct PR/Issue/Ticket/Channel/Repo
    # distinction from a Document row.
    op.execute(
        """
        UPDATE graph_nodes SET label = CASE
            WHEN label = 'CodeSymbol' THEN 'Function'
            -- 'Document' stays 'Document'; we cannot recover PR/Issue/etc.
            ELSE label
        END
        WHERE label = 'CodeSymbol';
        """
    )
