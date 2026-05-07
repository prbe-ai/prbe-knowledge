"""CodeGraph Path 2: hard-delete legacy code.symbol data + cache reset

Revision ID: 0052_codegraph_file_as_document
Revises: 0051_wipe_wiki_freeform_types
Create Date: 2026-05-07

The Path 2 refactor replaces per-symbol Documents (`doc_type='code.symbol'`)
with per-file Documents (`doc_type='code.file'`) that carry pre-chunked
symbol bodies + a synthetic metadata chunk (repo + file + symbol-list)
so semantic search ranks repo-qualified queries correctly.

Two data shapes can't coexist in production: search would return both
the old per-symbol chunks AND the new per-file metadata chunks for the
same underlying symbol, polluting fusion and confusing users. This
migration wipes the old shape and the per-file content_hash cache so the
new pipeline re-extracts cleanly on the next push.

Order matters:

  1. Toggle FORCE RLS off on graph tables — `neondb_owner` has BYPASSRLS=true
     so this is mostly belt-and-suspenders, but the documented pitfall in
     `feedback_graph_nodes_rls_force` is real for older client paths.
  2. DELETE chunks belonging to code.symbol Documents (FK-style cascade)
  3. DELETE the code.symbol Documents themselves
  4. DELETE code_repo_state entirely — content_hash cache is invalid under
     the new pipeline; clearing it forces re-extract under code.file shape
  5. DELETE graph_node_provenance rows scoped to source_system='code_graph'
  6. DELETE code-graph-specific edges (CALLS/IMPORTS/INHERITS/IMPLEMENTS/
     REFERENCES/DEFINED_IN globally, plus COMPILED_FROM whose target is
     a code-graph node label)
  7. DELETE code-graph node labels (Function/Method/Class/Module/Symbol)
  8. Restore FORCE RLS

The downgrade is intentionally a no-op — once the data is gone we can't
restore it; the next backfill rebuilds under whatever shape main has.

Deploy order (enforced operationally, not by Alembic):
  1. Pipeline + normalizer code lands first via the deploy workflow
  2. This migration runs as part of the same deploy (alembic upgrade)
  3. Operator re-runs `scripts/code_graph_backfill_existing.py`
"""

from __future__ import annotations

from alembic import op

revision = "0052_codegraph_file_as_document"
down_revision = "0051_wipe_wiki_freeform_types"
branch_labels = None
depends_on = None


_CODE_GRAPH_NODE_LABELS = ("Function", "Method", "Class", "Module", "Symbol")
_CODE_GRAPH_EDGE_TYPES = (
    "CALLS",
    "IMPORTS",
    "INHERITS",
    "IMPLEMENTS",
    "REFERENCES",
    "DEFINED_IN",
)


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE graph_nodes NO FORCE ROW LEVEL SECURITY;
        ALTER TABLE graph_edges NO FORCE ROW LEVEL SECURITY;
        ALTER TABLE graph_node_provenance NO FORCE ROW LEVEL SECURITY;
        """
    )

    # Chunks belonging to code.symbol Documents go before the Documents
    # themselves to keep the chunk → document FK consistent during the
    # delete window.
    op.execute(
        """
        DELETE FROM chunks
        WHERE doc_id IN (
            SELECT doc_id FROM documents
            WHERE source_system = 'code_graph' AND doc_type = 'code.symbol'
        )
        """
    )

    op.execute(
        """
        DELETE FROM documents
        WHERE source_system = 'code_graph' AND doc_type = 'code.symbol'
        """
    )

    # Per-file extraction cache is invalid under the new shape — old
    # content_hashes correspond to the old per-symbol Document layout.
    # Wipe so the new pipeline re-extracts every file on the next push.
    op.execute("DELETE FROM code_repo_state")

    # graph_node_provenance: drop the code_graph source rows. Whatever
    # nodes were ONLY visible via this source become candidates for
    # deletion in the next step.
    op.execute(
        "DELETE FROM graph_node_provenance WHERE source_system = 'code_graph'"
    )

    # Code-graph-specific edge types (no other source emits these).
    edge_type_list = ", ".join(f"'{t}'" for t in _CODE_GRAPH_EDGE_TYPES)
    op.execute(
        f"""
        DELETE FROM graph_edges
        WHERE edge_type IN ({edge_type_list})
        """
    )

    # COMPILED_FROM is shared with other doc_types (e.g. wiki Documents
    # COMPILED_FROM their source clusters), so scope by the target node's
    # label — only delete COMPILED_FROM whose target is a code-graph
    # node label.
    label_list = ", ".join(f"'{lbl}'" for lbl in _CODE_GRAPH_NODE_LABELS)
    op.execute(
        f"""
        DELETE FROM graph_edges
        WHERE edge_type = 'COMPILED_FROM'
          AND to_node_id IN (
              SELECT node_id FROM graph_nodes
              WHERE label IN ({label_list})
          )
        """
    )

    # Nodes last — edges referencing them are gone.
    op.execute(
        f"""
        DELETE FROM graph_nodes
        WHERE label IN ({label_list})
        """
    )

    op.execute(
        """
        ALTER TABLE graph_nodes FORCE ROW LEVEL SECURITY;
        ALTER TABLE graph_edges FORCE ROW LEVEL SECURITY;
        ALTER TABLE graph_node_provenance FORCE ROW LEVEL SECURITY;
        """
    )


def downgrade() -> None:
    # Forward-only: deleted symbol-shape data cannot be reconstructed.
    # Re-running the backfill script under whatever pipeline shape main
    # carries is the recovery path.
    pass
