"""One-shot Leiden community detection for ops use.

Runs the same per-tenant Leiden logic as the nightly cron but can be
targeted at a single customer or a subset. Useful for:
  - Back-filling community_id after first deploy of the migration
  - Debugging Leiden output for a specific tenant
  - Manually re-running after a large ingestion batch

Usage:
    # One tenant
    .venv/bin/python -m scripts.leiden_one_shot --customer-id cust-prbe-founders

    # All tenants (same as running the cron manually)
    .venv/bin/python -m scripts.leiden_one_shot --all-customers --yes

    # Dry-run: shows edge/node counts without writing community_id
    .venv/bin/python -m scripts.leiden_one_shot --customer-id cust-prbe-founders --dry-run

Environment: same as the worker -- set DATABASE_URL to point at prod.
All other vars are read from the standard Settings object.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from engine.community.leiden import MIN_EDGES_FOR_LEIDEN, run_leiden_for_tenant
from engine.shared.config import get_settings
from engine.shared.db import close_pool, init_pool, with_tenant
from engine.shared.logging import configure_logging, get_logger

log = get_logger(__name__)


async def _run(
    customer_ids: list[str],
    *,
    dry_run: bool = False,
) -> None:
    settings = get_settings()
    await init_pool(settings)

    try:
        for customer_id in customer_ids:
            log.info("leiden_one_shot.starting", customer_id=customer_id)
            if dry_run:
                # Read counts inside with_tenant so FORCE-RLS on graph_edges
                # and graph_nodes passes on the per-customer GUC under
                # probe_app. Belt-and-suspenders with the explicit
                # customer_id WHERE filter.
                async with with_tenant(customer_id) as conn:
                    edge_count = await conn.fetchval(
                        "SELECT COUNT(*) FROM graph_edges WHERE customer_id = $1",
                        customer_id,
                    )
                    node_count = await conn.fetchval(
                        "SELECT COUNT(*) FROM graph_nodes WHERE customer_id = $1",
                        customer_id,
                    )
                log.info(
                    "leiden_one_shot.dry_run",
                    customer_id=customer_id,
                    edge_count=edge_count,
                    node_count=node_count,
                    would_skip=edge_count < MIN_EDGES_FOR_LEIDEN,
                )
            else:
                # run_leiden_for_tenant expects a tenant-scoped conn so
                # the FORCE-RLS UPDATE on graph_nodes passes on the GUC.
                async with with_tenant(customer_id) as conn:
                    stats = await run_leiden_for_tenant(conn, customer_id)
                log.info("leiden_one_shot.done", **stats)
    finally:
        await close_pool()


def main() -> int:
    configure_logging()

    parser = argparse.ArgumentParser(description="One-shot Leiden community detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--customer-id", help="Run for a single customer")
    group.add_argument("--all-customers", action="store_true", help="Run for all customers")
    parser.add_argument(
        "--yes", action="store_true", help="Required when using --all-customers"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show edge/node counts without writing community_id",
    )
    args = parser.parse_args()

    if args.all_customers and not args.yes:
        print("ERROR: --all-customers requires --yes (safety gate)", file=sys.stderr)
        return 1

    if args.customer_id:
        customer_ids = [args.customer_id]
    else:
        # Load all customer IDs synchronously before kicking off async
        from engine.shared.config import get_settings as _gs

        async def _list() -> list[str]:
            from engine.shared.db import close_pool as _cp
            from engine.shared.db import init_pool as _ip
            from engine.shared.db import raw_conn as _rc

            await _ip(_gs())
            try:
                async with _rc() as c:
                    rows = await c.fetch("SELECT customer_id FROM customers ORDER BY customer_id")
                    return [r["customer_id"] for r in rows]
            finally:
                await _cp()

        customer_ids = asyncio.run(_list())
        if not customer_ids:
            log.info("leiden_one_shot.no_customers")
            return 0

    asyncio.run(_run(customer_ids, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
