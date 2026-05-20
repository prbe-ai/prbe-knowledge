"""Run the AutoMergeAnalyzer across all of a customer's existing graph_nodes.

Iterates `graph_nodes` for the target customer, runs the analyzer on each
non-path-canonical row, and either fires merges (when --execute) or writes
suggestions for the dashboard. Idempotent: rows that were already merged
into another cluster get skipped by the analyzer's pre-LLM filter
(entity_aliases NOT EXISTS check), and the suggestions table has a unique
constraint that prevents duplicate pending rows.

Pair with `scripts/backfill_graph_node_embeddings.py` so the vector path
contributes candidates. Without embeddings, candidate generation falls
back to trigram-only — still works, just lower recall on names that
diverged across sources.

Usage::

    # Suggestion-only (safe, write to entity_merge_suggestions)
    .venv/bin/python -m scripts.run_auto_merge_backfill --customer probe-founders

    # Execute high-confidence merges directly
    .venv/bin/python -m scripts.run_auto_merge_backfill --customer probe-founders --execute

    # Limit scope for testing
    .venv/bin/python -m scripts.run_auto_merge_backfill --customer probe-founders --limit 50
    .venv/bin/python -m scripts.run_auto_merge_backfill --customer probe-founders --label Person

Each row gets one LLM call. At ~$0.001 per gpt-oss-120b call on Cerebras and
~6000 nodes per customer at probe-founders scale, full backfill is < $10.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter

from services.ingestion.auto_merge import AutoMergeAnalyzer
from shared.config import get_settings
from shared.db import close_pool, init_pool, with_tenant
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)


async def run(
    customer_id: str,
    *,
    execute: bool,
    label: str | None,
    limit: int | None,
) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    analyzer = AutoMergeAnalyzer(execute_high_confidence=execute)
    action_counts: Counter = Counter()
    candidate_counts: list[int] = []
    processed = 0

    try:
        async with with_tenant(customer_id) as conn:
            # Snapshot the node_id list first — running the analyzer can mutate
            # the table (deletions on merge), so we don't want to iterate a live
            # cursor.
            if label:
                rows = await conn.fetch(
                    """
                    SELECT node_id FROM graph_nodes
                    WHERE label = $1
                    ORDER BY node_id
                    """,
                    label,
                )
            else:
                rows = await conn.fetch(
                    "SELECT node_id FROM graph_nodes ORDER BY node_id"
                )

            node_ids = [r["node_id"] for r in rows]
            if limit:
                node_ids = node_ids[:limit]
            log.info(
                "auto_merge_backfill.starting",
                customer=customer_id,
                total=len(node_ids),
                execute=execute,
                label_filter=label,
            )

            for node_id in node_ids:
                # Skip rows that may have been deleted by an earlier merge in this pass
                # (the analyzer's _load_node returns None and we'd record 'skipped').
                result = await analyzer.analyze(conn, customer_id, node_id)
                action_counts[result.action] += 1
                candidate_counts.append(result.candidate_count)
                processed += 1

                if processed % 50 == 0:
                    log.info(
                        "auto_merge_backfill.progress",
                        customer=customer_id,
                        processed=processed,
                        total=len(node_ids),
                        actions=dict(action_counts),
                    )

        avg_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0
        log.info(
            "auto_merge_backfill.complete",
            customer=customer_id,
            processed=processed,
            actions=dict(action_counts),
            avg_candidate_count=round(avg_candidates, 2),
        )
        print(f"\nProcessed {processed} nodes for customer={customer_id}")
        print(f"  Actions: {dict(action_counts)}")
        print(f"  Avg candidates surfaced per node: {avg_candidates:.2f}")
    finally:
        await close_pool()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--customer", required=True, help="customer_id to backfill")
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Fire high-confidence merges; otherwise suggestion-only",
    )
    ap.add_argument("--label", default=None, help="Restrict to one NodeLabel")
    ap.add_argument("--limit", type=int, default=None, help="Cap nodes processed (testing)")
    args = ap.parse_args()
    try:
        asyncio.run(
            run(
                args.customer,
                execute=args.execute,
                label=args.label,
                limit=args.limit,
            )
        )
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
