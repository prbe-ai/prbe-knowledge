"""Drop the full HNSW index: 7.7GB serving a query shape nobody sends.

The live-only partial twin (0124, `WHERE valid_to IS NULL`) serves every
default-mode ANN query -- TemporalMode.LATEST's chunk predicate implies its
WHERE clause, and the planner has been choosing it since it shipped. The
full index's remaining consumers were audited 2026-08-31 and none exist:

  * temporal `AS_OF` / `ALL` requests: ZERO in 1,059 production requests
    over 14 days (every one was `latest`). If one ever arrives, its vector
    channel degrades to a bounded seq-scan + sort (~3.4s measured on this
    corpus) instead of an index walk -- a priced trade on a mode with no
    measured users, not a regression on a hot path.
  * the inferred-edges bundle: orders by `1 - (embedding <=> $x)`, an
    expression form NO ANN index can serve (pgvector requires the bare
    distance as the sort key), so it never used this index at all -- and it
    already filters `valid_to IS NULL` besides.
  * sort_by="recency": cannot use an ANN index by construction.

What the drop buys: 7.7GB of disk and cache pressure gone (the index grew
~580MB/day and could never shrink -- HNSW pages are not reclaimed), per-write
index maintenance on a ~1M-row table halved, and one fewer giant index in
every failover/rebuild story. The IndexContract that bound the ANN ORDER BY
shape moves to the live index in the same change, so the guard survives.

Plain DROP INDEX: it takes a brief AccessExclusive lock on chunks -- an
instant on an index nothing selects from.
"""

from alembic import op

revision = "0126_drop_full_hnsw_index"
down_revision = "0125_cap_removed_chunk_versions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_v2_hnsw")


def downgrade() -> None:
    # Rebuildable, not restorable: recreating a 7.7GB HNSW index is a
    # multi-minute attended build, so downgrade recreates the definition
    # plainly and the operator owns the timing.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_v2_hnsw "
        "ON chunks USING hnsw (embedding_v2 halfvec_cosine_ops)"
    )
