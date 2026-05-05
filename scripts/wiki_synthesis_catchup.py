"""Catchup / backfill: seed wiki_synthesis_queue from existing documents.

Two use cases:

1. **Onboarding catchup** (the original purpose). A customer onboarded
   BEFORE Phase 2 shipped — their `documents` rows exist but no queue
   rows (the Normalizer hook didn't exist yet). Run with no flags:

       .venv/bin/python -m scripts.wiki_synthesis_catchup <customer_id>

   This INSERTs queue rows for every live non-wiki document. Idempotent
   on (customer_id, doc_id, doc_version) — re-running is a no-op for
   already-queued rows.

2. **Full backfill** (use after a triage prompt change, threshold tune,
   or model swap). Re-evaluate every historical event under the current
   triage by also resetting any terminal queue rows back to 'pending':

       .venv/bin/python -m scripts.wiki_synthesis_catchup <customer_id> \\
           --reset-terminal

   Terminal states reset: done, rejected, failed, synthesis_skipped, dlq.
   In-flight states (triaging, synthesizing, triaged) are left alone —
   they're either being processed right now or will be picked up by the
   next claim.

Other flags:
- `--dry-run` prints what would change without writing.
- `--no-notify` skips firing pg_notify after the writes (workers will
  pick up the new pending rows on their next periodic wake or the next
  nightly cron tick).
- `--all-enabled` iterates every customer with
  preferences.wiki_generation_enabled=true. Requires `--yes` to confirm
  the multi-tenant scope. Useful for a fleet-wide reset after a triage
  change.

Operator notes:
- Does NOT bypass the customer opt-in flag. If a customer's
  preferences.wiki_generation_enabled is false, queue rows still go in
  but the worker skips the customer until the flag is flipped on.
- For a true backfill of v3-era queue rows after the v4 redesign:
  pair this script with the dashboard's "Generate Wiki Now" button
  (or a fresh pg_notify) to wake the wiki-worker immediately rather
  than waiting for the periodic 30-min wake cycle.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from shared.config import get_settings
from shared.constants import WIKI_PENDING_CHANNEL, SourceSystem
from shared.db import close_pool, init_pool, raw_conn, with_tenant
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)


# Terminal queue statuses we reset to 'pending' under --reset-terminal.
# 'triaging' and 'synthesizing' are deliberately excluded — they're
# in-flight; resetting them mid-claim would produce duplicate work
# when the heartbeat-reclaim loop also retries them. 'triaged' is
# excluded because it's already past triage and will drain via
# wiki-synthesis on the next NOTIFY.
_TERMINAL_STATUSES = (
    "done",
    "rejected",
    "failed",
    "synthesis_skipped",
    "dlq",
)


async def _enqueue_missing(conn, customer_id: str) -> tuple[int, int]:
    """Insert queue rows for live non-wiki docs that aren't queued yet.

    Returns (eligible_total, inserted). eligible_total counts every doc
    that *could* be queued; inserted counts only the new rows ON CONFLICT
    skipped.
    """
    eligible = await conn.fetchval(
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
    eligible = int(eligible or 0)
    inserted = await conn.fetchval(
        """
        WITH inserted AS (
            INSERT INTO wiki_synthesis_queue
                (customer_id, doc_id, doc_version, source_system,
                 doc_type, source_ts, status, enqueued_at)
            SELECT customer_id, doc_id, version, source_system,
                   doc_type, created_at, 'pending', NOW()
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
    return eligible, int(inserted or 0)


async def _reset_terminal(conn, customer_id: str) -> int:
    """Reset terminal queue rows back to 'pending' so v4 triage re-evaluates.

    Wipes per-row triage decisions, attempts, and DLQ markers but leaves
    the doc_id / doc_version / enqueued_at intact (these are identity
    columns, not state). The agent's source_ts ordering is also
    preserved.
    """
    reset = await conn.fetchval(
        """
        WITH reset AS (
            UPDATE wiki_synthesis_queue
            SET status = 'pending',
                triage_score = NULL,
                triage_error = NULL,
                triage_completed_at = NULL,
                synthesis_run_id = NULL,
                synthesis_completed_at = NULL,
                synthesis_error = NULL,
                dlq_reason = NULL,
                dlq_at = NULL,
                attempts = 0,
                heartbeat_at = NULL
            WHERE customer_id = $1
              AND status = ANY($2::text[])
            RETURNING 1
        )
        SELECT count(*) FROM reset
        """,
        customer_id,
        list(_TERMINAL_STATUSES),
    )
    return int(reset or 0)


async def seed(
    customer_id: str,
    *,
    dry_run: bool,
    reset_terminal: bool,
    notify: bool,
) -> dict[str, int | None]:
    """Run the catchup / backfill for one customer.

    Returns a stats dict — useful when called from `--all-enabled`.
    """
    async with with_tenant(customer_id) as conn:
        eligible, inserted = (
            (
                await conn.fetchval(
                    """
                    SELECT count(*) FROM documents
                    WHERE customer_id = $1
                      AND valid_to IS NULL
                      AND deleted_at IS NULL
                      AND source_system <> $2
                    """,
                    customer_id,
                    SourceSystem.WIKI.value,
                ),
                0,
            )
            if dry_run
            else await _enqueue_missing(conn, customer_id)
        )
        eligible = int(eligible or 0)

        terminal_pending = (
            await conn.fetchval(
                """
                SELECT count(*) FROM wiki_synthesis_queue
                WHERE customer_id = $1
                  AND status = ANY($2::text[])
                """,
                customer_id,
                list(_TERMINAL_STATUSES),
            )
            if reset_terminal
            else 0
        )
        terminal_pending = int(terminal_pending or 0)

        reset = 0
        if reset_terminal and not dry_run:
            reset = await _reset_terminal(conn, customer_id)

        run_id: int | None = None
        # Mark the onboarding-style mass enqueue. The dashboard reads
        # this to surface "Wiki being generated, X events left."
        # Backfill runs (reset_terminal) don't open a separate run
        # row — the regular wake/scheduled run row the worker opens
        # on next claim is enough audit.
        if not dry_run and inserted > 0 and not reset_terminal:
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

        if not dry_run and notify and (inserted > 0 or reset > 0):
            await conn.execute(
                "SELECT pg_notify($1, $2)",
                WIKI_PENDING_CHANNEL,
                customer_id,
            )

        stats = {
            "customer_id": customer_id,
            "eligible_documents": eligible,
            "inserted": inserted,
            "already_queued": eligible - inserted,
            "terminal_to_reset": terminal_pending if reset_terminal else None,
            "reset": reset,
            "run_id": run_id,
            "notified": (
                not dry_run
                and notify
                and (inserted > 0 or reset > 0)
            ),
            "dry_run": dry_run,
        }
        _print_stats(stats)
        return stats


def _print_stats(s: dict[str, int | None]) -> None:
    suffix = " (dry run — no rows changed)" if s["dry_run"] else ""
    parts = [
        f"customer={s['customer_id']}",
        f"eligible={s['eligible_documents']}",
        f"inserted={s['inserted']}",
        f"already_queued={s['already_queued']}",
    ]
    if s["terminal_to_reset"] is not None:
        parts.append(f"terminal={s['terminal_to_reset']}")
        parts.append(f"reset={s['reset']}")
    if s["run_id"] is not None:
        parts.append(f"run_id={s['run_id']}")
    if s["notified"]:
        parts.append(f"notified={WIKI_PENDING_CHANNEL}")
    print(" ".join(parts) + suffix)


async def _list_enabled_customers() -> list[str]:
    """Customers with preferences.wiki_generation_enabled = true.

    Reads the global view (no with_tenant) — pairs with raw_conn rather
    than with_tenant since we need cross-customer visibility.
    """
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT customer_id
            FROM customers
            WHERE status = 'active'
              AND preferences->>'wiki_generation_enabled' = 'true'
            ORDER BY customer_id
            """
        )
    return [r["customer_id"] for r in rows]


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    try:
        if args.all_enabled:
            customers = await _list_enabled_customers()
            if not customers:
                print("No customers have wiki_generation_enabled=true.")
                return 0
            print(
                f"Will run on {len(customers)} customer(s): "
                f"{', '.join(customers)}"
            )
            if not args.yes:
                print(
                    "Refusing to proceed without --yes. Re-run with "
                    "--yes to confirm the multi-tenant scope."
                )
                return 2
            for cust in customers:
                await seed(
                    cust,
                    dry_run=args.dry_run,
                    reset_terminal=args.reset_terminal,
                    notify=args.notify,
                )
            return 0

        await seed(
            args.customer_id,
            dry_run=args.dry_run,
            reset_terminal=args.reset_terminal,
            notify=args.notify,
        )
        return 0
    finally:
        await close_pool()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument(
        "customer_id",
        nargs="?",
        help="Customer to backfill (omit when using --all-enabled).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows that would change without writing.",
    )
    parser.add_argument(
        "--reset-terminal",
        action="store_true",
        help=(
            "Also reset terminal queue rows (done, rejected, failed, "
            "synthesis_skipped, dlq) back to pending so v4 triage "
            "re-evaluates them. Default: off (onboarding-only INSERT)."
        ),
    )
    parser.add_argument(
        "--no-notify",
        dest="notify",
        action="store_false",
        default=True,
        help=(
            "Skip pg_notify after writes. Workers will pick up new "
            "pending rows on the next periodic wake (~30 min) or the "
            "next nightly cron tick instead."
        ),
    )
    parser.add_argument(
        "--all-enabled",
        action="store_true",
        help=(
            "Iterate every customer with "
            "preferences.wiki_generation_enabled=true. Requires --yes."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm --all-enabled multi-tenant scope.",
    )
    args = parser.parse_args(argv)

    if not args.all_enabled and not args.customer_id:
        parser.error("customer_id is required unless --all-enabled is set")

    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
