"""One-time catch-up: seed wiki_synthesis_queue from existing documents.

Use case: customers onboarded BEFORE Phase 2 shipped have populated
`documents` rows but no queue rows (the Normalizer hook didn't exist yet).
This script enqueues every live, non-deleted, non-wiki document for a
customer in one INSERT, then NOTIFYs the synthesis cron.

Idempotent on (customer_id, doc_id, doc_version) — running it twice is a
no-op for already-queued rows.

Not part of any automated path; runs once per existing customer on Phase 2
rollout day.

Usage:
    .venv/bin/python -m scripts.wiki_synthesis_catchup <customer_id> [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from shared.config import get_settings
from shared.constants import WIKI_PENDING_CHANNEL, SourceSystem
from shared.db import close_pool, init_pool, with_tenant
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)


async def seed(customer_id: str, *, dry_run: bool) -> int:
    """Insert one queue row per live, non-wiki document for `customer_id`.

    Returns the number of rows inserted (or that would be inserted, if dry_run).
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    try:
        async with with_tenant(customer_id) as conn:
            count = await conn.fetchval(
                """
                SELECT count(*)
                FROM documents
                WHERE customer_id = $1
                  AND valid_to IS NULL
                  AND deleted_at IS NULL
                  AND source_system <> $2
                """,
                customer_id,
                SourceSystem.WIKI.value,
            )
            count = int(count or 0)
            print(f"customer={customer_id} eligible_documents={count}")
            if dry_run:
                print("(dry run — no rows enqueued)")
                return count
            inserted = await conn.fetchval(
                """
                WITH inserted AS (
                    INSERT INTO wiki_synthesis_queue
                        (customer_id, doc_id, doc_version, source_system,
                         doc_type, status, enqueued_at)
                    SELECT customer_id, doc_id, version, source_system,
                           doc_type, 'pending', NOW()
                    FROM documents
                    WHERE customer_id = $1
                      AND valid_to IS NULL
                      AND deleted_at IS NULL
                      AND source_system <> $2
                    ON CONFLICT (customer_id, doc_id, doc_version) DO NOTHING
                    RETURNING 1
                )
                SELECT count(*) FROM inserted
                """,
                customer_id,
                SourceSystem.WIKI.value,
            )
            inserted = int(inserted or 0)

            # Open a wiki_synthesis_runs row with kind='onboarding'. The
            # cron itself opens its own per-tick 'wake'/'scheduled' rows
            # while draining; this row is the durable marker that an
            # onboarding-style mass enqueue happened, and is what the
            # dashboard reads to surface "Wiki being generated, X events
            # left." Status stays 'running' until something explicitly
            # closes it (a future finalize step that fires when no
            # queue rows for this customer remain pending).
            run_id = await conn.fetchval(
                """
                INSERT INTO wiki_synthesis_runs
                    (customer_id, kind, events_total, status)
                VALUES ($1, 'onboarding', $2, 'running')
                RETURNING run_id
                """,
                customer_id,
                inserted,
            )

            await conn.execute(
                "SELECT pg_notify($1, $2)",
                WIKI_PENDING_CHANNEL,
                customer_id,
            )
            print(
                f"customer={customer_id} enqueued={inserted} "
                f"already_queued={count - inserted} run_id={run_id} "
                f"notified={WIKI_PENDING_CHANNEL}"
            )
            return inserted
    finally:
        await close_pool()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("customer_id")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows that would be inserted without writing.",
    )
    args = parser.parse_args(argv)
    asyncio.run(seed(args.customer_id, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
