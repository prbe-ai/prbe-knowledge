"""Catchup / backfill: seed wiki_synthesis_queue from existing documents.

Thin CLI over ``kb.synthesis.persistence`` — the queue SQL lives THERE
(single SQL boundary), this file only parses flags, wires a connection,
and prints stats. Two use cases:

1. **Onboarding catchup** (the original purpose). A customer's
   `documents` rows exist but carry no queue rows (ingested before the
   Normalizer hook, while the opt-in flag was off, or dropped by the
   best-effort enqueue). Run with no flags:

       .venv/bin/python -m scripts.wiki_synthesis_catchup <customer_id>

   Idempotent on (customer_id, doc_id, doc_version) — re-running is a
   no-op for already-queued rows. The nightly trigger's reconcile step
   performs this same seed automatically for enabled tenants; the CLI
   remains for immediate, targeted runs.

2. **Full backfill** (after a triage prompt change, threshold tune, or
   model swap). Re-evaluate history under the current triage by also
   resetting terminal queue rows back to 'pending':

       .venv/bin/python -m scripts.wiki_synthesis_catchup <customer_id> \\
           --reset-terminal

   Terminal states reset: done, rejected, failed, synthesis_skipped.
   Only rows whose (doc_id, doc_version) still matches a live eligible
   document are reset — superseded versions, deleted docs, and excluded
   sources stay terminal. `dlq` rows are NOT reset unless you also pass
   `--include-dlq`: dead-letter reasons deserve a human read before a
   redrive. In-flight states (triaging, synthesizing, triaged) are
   always left alone.

Other flags:
- `--dry-run` prints the REAL would-insert / would-reset counts (an
  anti-join against the queue) without writing.
- `--no-notify` skips firing pg_notify after the writes (workers pick
  the rows up on their next periodic wake or the nightly cron tick).
- `--all-enabled` iterates every customer with
  preferences.wiki_generation_enabled=true. Requires `--yes` to confirm
  the multi-tenant scope.

Operator notes:
- Does NOT bypass the customer opt-in flag. If a customer's flag is
  false, queue rows still go in but the worker skips the customer until
  the flag is flipped on.
- Pair with the dashboard's "Generate Wiki Now" button (or keep the
  default notify) to wake the wiki-worker immediately rather than
  waiting for the periodic 30-min wake cycle.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from engine.shared.config import get_settings
from engine.shared.constants import WIKI_PENDING_CHANNEL
from engine.shared.customer_prefs import wiki_enabled_sql
from engine.shared.db import close_pool, init_pool, raw_conn, with_tenant
from engine.shared.logging import configure_logging, get_logger
from kb.synthesis import persistence

log = get_logger(__name__)


async def seed(
    customer_id: str,
    *,
    dry_run: bool,
    reset_terminal: bool,
    notify: bool,
    include_dlq: bool = False,
) -> dict[str, int | bool | str | None]:
    """Run the catchup / backfill for one customer.

    Returns a stats dict — useful when called from `--all-enabled`.
    """
    async with with_tenant(customer_id) as conn:
        if dry_run:
            eligible, would_insert = await persistence.count_seedable_docs(
                conn, customer_id
            )
            inserted = 0
            reset = 0
            would_reset = (
                await persistence.count_resettable_rows(
                    conn, customer_id, include_dlq=include_dlq
                )
                if reset_terminal
                else None
            )
        else:
            eligible, inserted = await persistence.seed_missing_docs(
                conn, customer_id
            )
            would_insert = inserted
            reset = (
                await persistence.reset_terminal_rows(
                    conn, customer_id, include_dlq=include_dlq
                )
                if reset_terminal
                else 0
            )
            would_reset = reset if reset_terminal else None

        run_id: int | None = None
        # Mark the onboarding-style mass enqueue. The dashboard reads
        # this to surface "Wiki being generated, X events left."
        # Backfill runs (reset_terminal) don't open a separate run
        # row — the regular wake/scheduled run row the worker opens
        # on next claim is enough audit.
        if not dry_run and inserted > 0 and not reset_terminal:
            run_id = await persistence.open_onboarding_run(
                conn, customer_id, events_total=inserted
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
            "would_insert": would_insert,
            "already_queued": eligible - would_insert,
            "terminal_to_reset": would_reset,
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


def _print_stats(s: dict[str, int | bool | str | None]) -> None:
    suffix = " (dry run — no rows changed)" if s["dry_run"] else ""
    parts = [
        f"customer={s['customer_id']}",
        f"eligible={s['eligible_documents']}",
        f"already_queued={s['already_queued']}",
    ]
    if s["dry_run"]:
        parts.append(f"would_insert={s['would_insert']}")
    else:
        parts.append(f"inserted={s['inserted']}")
    if s["terminal_to_reset"] is not None:
        parts.append(f"terminal={s['terminal_to_reset']}")
        if not s["dry_run"]:
            parts.append(f"reset={s['reset']}")
    if s["run_id"] is not None:
        parts.append(f"run_id={s['run_id']}")
    if s["notified"]:
        parts.append(f"notified={WIKI_PENDING_CHANNEL}")
    print(" ".join(str(p) for p in parts) + suffix)


async def _list_enabled_customers() -> list[str]:
    """Customers with preferences.wiki_generation_enabled = true.

    Reads the global view (no with_tenant) — pairs with raw_conn rather
    than with_tenant since we need cross-customer visibility.
    """
    async with raw_conn() as conn:
        rows = await conn.fetch(
            f"""
            SELECT customer_id
            FROM customers
            WHERE status = 'active'
              AND {wiki_enabled_sql()}
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
                    include_dlq=args.include_dlq,
                )
            return 0

        await seed(
            args.customer_id,
            dry_run=args.dry_run,
            reset_terminal=args.reset_terminal,
            notify=args.notify,
            include_dlq=args.include_dlq,
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
        help="Report the real would-insert/would-reset counts, write nothing.",
    )
    parser.add_argument(
        "--reset-terminal",
        action="store_true",
        help=(
            "Also reset terminal queue rows (done, rejected, failed, "
            "synthesis_skipped) back to pending so v4 triage "
            "re-evaluates them. Only rows still matching a live "
            "eligible document are touched. Default: off "
            "(onboarding-only INSERT)."
        ),
    )
    parser.add_argument(
        "--include-dlq",
        action="store_true",
        help=(
            "With --reset-terminal: also redrive dlq rows. Off by "
            "default — read the dlq_reason before resetting poison rows."
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
    if args.include_dlq and not args.reset_terminal:
        parser.error("--include-dlq requires --reset-terminal")

    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
