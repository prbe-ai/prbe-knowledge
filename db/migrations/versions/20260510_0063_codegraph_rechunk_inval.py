"""invalidate code_repo_state for code_graph rechunk

Revision ID: 0063_codegraph_rechunk_inval
Revises: 0062_chunks_content_tsv
Create Date: 2026-05-10

Forces re-extraction of every code_graph file on the next reindex pass.

Why
---
The companion code change in services/ingestion/code_graph/chunking.py +
pipeline.py caps emitted chunks at MAX_SYMBOL_CHUNK_TOKENS (== 512, same
as DEFAULT_CHUNK_TOKENS). Pre-cap, individual symbols landed as 7-30KB
single chunks. Production trace: one Probe MCP search_knowledge call
returned 196KB / 12 docs, with 7 code_graph files contributing ~152KB
across 7 single-chunk symbols (linear.py, slack.py, normalizer.py,
claude_code.py, agent.py, fusion.py, main.py). That blew the 25KB
tool-result envelope and triggered disk-spill fallback in agent runtimes.

The new chunker only fires on freshly-ingested files. Already-extracted
files short-circuit at the file-level cache check in
services/ingestion/code_graph/pipeline.py:300:

    if cached_state.get(rel) == ch:
        # cache hit -> skip extraction, no rechunking happens

Setting `code_repo_state.content_hash = ''` invalidates every cached
row. Next time a customer's repos get re-pushed (organic) or a manual
reindex runs (services/ingestion/code_graph/reindex.py), the cache miss
forces re-extraction with the new chunker.

Old chunks stay in the chunks table with valid_to set by SCD2 logic in
`_apply_chunk_plan`. Search filters on `valid_to IS NULL` so stale
30KB chunks don't pollute results once their replacements land.

The MCP response byte budget (separate PR in prbe-knowledge-mcp)
catches any 30KB chunk that survives the rollout window via
truncation + cursor.

RLS posture
-----------
code_repo_state has RLS ENABLED but not FORCED (migration 0049):

    op.execute("ALTER TABLE code_repo_state ENABLE ROW LEVEL SECURITY")

So the alembic migration role bypasses the tenant_isolation policy as
table owner. No NO FORCE / FORCE toggle is needed (unlike graph_nodes
and graph_edges, which ARE forced and require the toggle per
feedback_graph_nodes_rls_force memory).

Lessons reminder: revision string MUST be <=32 chars
(alembic_version.version_num is varchar(32)).
'0063_codegraph_rechunk_inval' is 28 chars - fine.
"""

from __future__ import annotations

from alembic import op

revision = "0063_codegraph_rechunk_inval"
down_revision = "0062_chunks_content_tsv"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Empty content_hash forces a cache miss on every code_graph file at
    # the next reindex. Idempotent: re-running this migration is a no-op
    # because content_hash is repopulated from the file's actual SHA-256
    # on each successful re-extraction.
    op.execute("UPDATE code_repo_state SET content_hash = ''")


def downgrade() -> None:
    # No-op: content_hash is recomputed on next ingestion. The migration
    # only triggers a one-time cache miss; rolling back doesn't restore
    # the prior cache state (which is fine, cache misses are non-
    # destructive). Avoids needing a backup column or pg_dump snapshot.
    pass
