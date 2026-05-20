"""Entity auto-merge: LLM-judged dedup for graph_nodes.

For each candidate (label, canonical_id), the analyzer:
  1. Finds nearby entities via pg_trgm on LOWER(canonical_id) + LOWER(properties->>'name')
     and HNSW on graph_nodes.embedding (when populated)
  2. Filters candidates with conflicting stable properties (mismatched email, etc.)
  3. Sends survivors to Cerebras gpt-oss-120b with a `response_format=AutoMergeVerdict`
     Pydantic schema
  4. On confidence='high': posts to the existing entity-clusters merge txn
  5. On medium/low: inserts a row into entity_merge_suggestions for dashboard review

Backfill via `scripts/run_auto_merge_backfill.py` over existing graph_nodes.
Real-time integration into graph_writer.upsert_nodes is a follow-up PR.

See: entity auto-merge plan, /plan-eng-review 2026-05-19.
"""

from services.ingestion.auto_merge.analyzer import (
    AutoMergeAnalyzer,
    AutoMergeResult,
)
from services.ingestion.auto_merge.models import AutoMergeVerdict

__all__ = ["AutoMergeAnalyzer", "AutoMergeResult", "AutoMergeVerdict"]
