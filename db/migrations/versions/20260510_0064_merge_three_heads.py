"""merge two heads (codegraph_rechunk_inval + graph_nodes_degree_idx chain)

Revision ID: 0064_merge_three_heads
Revises: 0063_codegraph_rechunk_inval, 0064_graph_nodes_degree_idx
Create Date: 2026-05-10

History (this got tangled):

  PR #212 + PR #229 landed minutes apart and both stamped
  `down_revision = "0062_chunks_content_tsv"`, putting main into a
  3-head state along with this branch's 0063_codegraph_rechunk_inval.
  The original version of THIS file merged all three.

  PR #231 then independently re-chained graph_nodes_degree_idx
  AFTER embedding_v2_hnsw, renaming it to 0064_graph_nodes_degree_idx
  with `down_revision = "0063_embedding_v2_hnsw"`. So when PR #230
  finally merged, the old `0063_graph_nodes_degree_idx` reference
  was dangling — the tests + deploy on merge commit 914c5fe failed
  with "Can't find revision 0063_graph_nodes_degree_idx" and main
  went red again.

  This hotfix updates the `down_revision` tuple to point at the
  current heads:

    - 0063_codegraph_rechunk_inval    (PR #230's content_hash invalidation)
    - 0064_graph_nodes_degree_idx     (PR #231's re-chained version)

  0063_embedding_v2_hnsw is now an ancestor of 0064_graph_nodes_degree_idx
  (per PR #231) so it doesn't need to be a direct parent here — it gets
  reached transitively through the second tuple entry.

The merge stays a no-op: no schema change, just collapses two
parallel chains into a single head.

Lessons reminder: revision id ≤ 32 chars
('0064_merge_three_heads' is 22 — fine; keeping the original
revision id since this hotfix amends the same logical migration
rather than introducing a new one).
"""

from __future__ import annotations

revision = "0064_merge_three_heads"
down_revision = (
    "0063_codegraph_rechunk_inval",
    "0064_graph_nodes_degree_idx",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    # No-op merge: collapses two sibling heads into one. Schema state
    # is identical pre- and post-migration; only the alembic_version
    # row advances to '0064_merge_three_heads'.
    pass


def downgrade() -> None:
    # No-op: rolling back this row only puts alembic_version back into
    # the multi-head state, which is what the merge fixed. Use the
    # constituent migrations' downgrades to revert actual schema work.
    pass
