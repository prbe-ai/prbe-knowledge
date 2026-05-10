"""merge three 0063 heads (rechunk_inval + embedding_v2_hnsw + graph_nodes_degree_idx)

Revision ID: 0064_merge_three_heads
Revises: 0063_codegraph_rechunk_inval, 0063_embedding_v2_hnsw, 0063_graph_nodes_degree_idx
Create Date: 2026-05-10

Three migrations all chain off `0062_chunks_content_tsv`:

  - 0063_codegraph_rechunk_inval     (this branch — content_hash invalidation)
  - 0063_embedding_v2_hnsw           (PR #212 — Stage 3 of Gemini migration)
  - 0063_graph_nodes_degree_idx      (PR #229 — graph-viz endpoints)

PR #212 and PR #229 landed minutes apart on main and both stamped
`down_revision = "0062_chunks_content_tsv"` independently — main has
been red since because `alembic upgrade head` fails with "Multiple
heads are present; please specify a single target revision". This
branch's 0063 makes it three heads.

Resolution: a merge migration with a tuple `down_revision` that
collapses all three into a single head (`0064_merge_three_heads`).
No schema change — just chains the alembic graph back into a line.

Lessons reminder: revision id ≤ 32 chars
('0064_merge_three_heads' is 22 — fine).
"""

from __future__ import annotations

revision = "0064_merge_three_heads"
down_revision = (
    "0063_codegraph_rechunk_inval",
    "0063_embedding_v2_hnsw",
    "0063_graph_nodes_degree_idx",
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
