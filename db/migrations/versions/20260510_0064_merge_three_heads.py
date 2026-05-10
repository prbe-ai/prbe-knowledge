"""merge two remaining heads after #231 hotfix.

Revision ID: 0064_merge_three_heads
Revises: 0063_codegraph_rechunk_inval, 0064_graph_nodes_degree_idx
Create Date: 2026-05-10

History:
  PR #229 landed `0063_graph_nodes_degree_idx` (down_revision=0062),
  PR #212 had landed `0063_embedding_v2_hnsw` (down_revision=0062),
  PR #230 added `0063_codegraph_rechunk_inval` (down_revision=0062).
  Three heads, all stamped 0062.

  Hotfix #231 then renamed `0063_graph_nodes_degree_idx` to
  `0064_graph_nodes_degree_idx` chained after `0063_embedding_v2_hnsw`.
  That removed one head, so this merge migration's original tuple of
  three 0063 entries became stale -- the third element references a
  deleted revision.

  Updated tuple merges the two remaining heads:
    - 0063_codegraph_rechunk_inval (PR #230's own 0063)
    - 0064_graph_nodes_degree_idx  (transitively chains
                                    0063_embedding_v2_hnsw -> 0062)

  Result: single head `0064_merge_three_heads`. File name kept for
  git-history continuity even though it now merges two heads, not three.

Lessons reminder: revision id <= 32 chars
('0064_merge_three_heads' is 22 -- fine).
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
    # No-op merge: collapses three sibling heads into one. Schema state
    # is identical pre- and post-migration; only the alembic_version
    # row advances to '0064_merge_three_heads'.
    pass


def downgrade() -> None:
    # No-op: rolling back this row only puts alembic_version back into
    # the multi-head state, which is what the merge fixed. Use the
    # constituent migrations' downgrades to revert actual schema work.
    pass
