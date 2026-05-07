"""One-shot inferred-edges backfill for existing documents.

Enqueues every existing (customer_id, doc_id) pair into inferred_edges_queue
so the side-queue worker can build bundles and extract cross-source LLM edges
for content that was ingested before this pipeline was deployed.

The queue worker picks up rows from inferred_edges_queue and processes them
asynchronously -- this script only writes queue rows, it does NOT run the
LLM extraction itself.

Idempotent end-to-end:
  - Uses ON CONFLICT DO NOTHING on (customer_id, anchor_doc_id) so re-running
    doesn't double-enqueue the same doc.
  - Rows that have already been processed (done_at IS NOT NULL) are skipped
    by the drain worker automatically.

Usage:
    # One tenant (recommended for first runs)
    .venv/bin/python -m scripts.inferred_edges_backfill_existing --customer-id cust-prbe-founders

    # All tenants (gated behind --yes)
    .venv/bin/python -m scripts.inferred_edges_backfill_existing --all-customers --yes

    # See what would happen without writing
    .venv/bin/python -m scripts.inferred_edges_backfill_existing --customer-id cust-prbe-founders --dry-run

    # Limit to recent docs (useful for a quick test run)
    .venv/bin/python -m scripts.inferred_edges_backfill_existing --customer-id cust-prbe-founders --days 7

Environment (same as the worker -- point at prod via env, secrets from fly):
    DATABASE_URL, TOKEN_ENCRYPTION_KEY.

DO NOT RUN: This script is provided for ops use post-deployment.
Run from a fly ssh console:
    flyctl ssh console -a prbe-knowledge-worker
    uv run python -m scripts.inferred_edges_backfill_existing --customer-id ...

Per memory feedback_ci_autodeploys_on_push.md:
    prbe-knowledge deploys automatically on push to main. Sequence:
    1. Merge Lane B.
    2. Wait for migration to apply (inferred_edges_queue table created).
    3. Run this backfill script.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from services.ingestion.inferred_edges.prompts.v1 import PROMPT_VERSION
from shared.config import get_settings
from shared.db import close_pool, init_pool, raw_conn
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)


async def _list_customers(customer_id: str | None) -> list[str]:
    """Return customer_ids to backfill."""
    where = ["status = 'active'"]
    params: list[object] = []
    if customer_id is not None:
        params.append(customer_id)
        where.append(f"customer_id = ${len(params)}")
    sql = f"SELECT customer_id FROM customers WHERE {' AND '.join(where)} ORDER BY customer_id"
    async with raw_conn() as conn:
        rows = await conn.fetch(sql, *params)
    return [r["customer_id"] for r in rows]


async def _count_docs(customer_id: str, days: int | None) -> int:
    """Count live documents for a customer, optionally limited to recent days."""
    where = ["customer_id = $1", "valid_to IS NULL"]
    params: list[object] = [customer_id]
    if days is not None:
        params.append(days)
        where.append(f"updated_at > NOW() - make_interval(days => ${len(params)})")
    sql = f"SELECT count(*) FROM documents WHERE {' AND '.join(where)}"
    async with raw_conn() as conn:
        return await conn.fetchval(sql, *params)


async def _enqueue_customer(
    customer_id: str,
    *,
    dry_run: bool,
    days: int | None,
    batch_size: int = 500,
) -> int:
    """Enqueue all live docs for a customer into inferred_edges_queue.

    Returns the number of rows inserted.
    """
    where = ["customer_id = $1", "valid_to IS NULL"]
    params: list[object] = [customer_id]
    if days is not None:
        params.append(days)
        where.append(f"updated_at > NOW() - make_interval(days => ${len(params)})")
    sql = f"SELECT doc_id FROM documents WHERE {' AND '.join(where)} ORDER BY doc_id"

    async with raw_conn() as conn:
        doc_rows = await conn.fetch(sql, *params)

    doc_ids = [r["doc_id"] for r in doc_rows]
    if not doc_ids:
        return 0

    if dry_run:
        print(f"  [dry-run] would enqueue {len(doc_ids)} docs for {customer_id}")
        return len(doc_ids)

    inserted = 0
    async with raw_conn() as conn:
        # Batch inserts to avoid a single massive statement.
        for i in range(0, len(doc_ids), batch_size):
            batch = doc_ids[i : i + batch_size]
            await conn.executemany(
                """
                INSERT INTO inferred_edges_queue
                    (customer_id, anchor_doc_id, extractor_id)
                VALUES ($1, $2, $3)
                ON CONFLICT DO NOTHING
                """,
                [(customer_id, doc_id, PROMPT_VERSION) for doc_id in batch],
            )
            inserted += len(batch)
            log.info(
                "backfill.enqueued_batch",
                customer=customer_id,
                batch_start=i,
                batch_end=i + len(batch),
                total_so_far=inserted,
            )

    return inserted


async def _main(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    try:
        customers = await _list_customers(args.customer_id)
        if not customers:
            print("No matching active customers found.", file=sys.stderr)
            return 1
        if args.all_customers and not args.yes and len(customers) > 1:
            print(
                f"--all-customers matched {len(customers)} customers; pass --yes to "
                "confirm. Use --customer-id <id> to target one.",
                file=sys.stderr,
            )
            return 1

        total_enqueued = 0
        for customer_id in customers:
            doc_count = await _count_docs(customer_id, args.days)
            print(
                f"\n== {customer_id} ({doc_count} live docs"
                + (f" in last {args.days}d" if args.days else "")
                + ") =="
            )
            enqueued = await _enqueue_customer(
                customer_id,
                dry_run=args.dry_run,
                days=args.days,
            )
            total_enqueued += enqueued
            print(f"  {'would-enqueue' if args.dry_run else 'enqueued'}: {enqueued}")

        print(
            f"\nDone. customers={len(customers)} "
            f"{'would-enqueue' if args.dry_run else 'enqueued'}={total_enqueued}"
        )
        return 0
    finally:
        await close_pool()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--customer-id", help="Single customer (cust-...)")
    g.add_argument(
        "--all-customers",
        action="store_true",
        help="Every active customer",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List docs that would be enqueued; don't write queue rows",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Required with --all-customers when more than one customer matches",
    )
    p.add_argument(
        "--days",
        type=int,
        default=None,
        help="Limit to docs updated in the last N days (useful for test runs)",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(_parse_args())))
