"""Nightly trigger — wakes the wiki-worker once per opted-in customer.

Runs as a fly machine schedule (the fly.wiki-cron.toml `[processes].cron`
entry runs at 02:00 UTC daily). One-shot:

  1. Connect to Postgres.
  2. SELECT customer_ids with at least one pending wiki_synthesis_queue
     row AND `preferences->>'wiki_generation_enabled' = 'true'`.
  3. Per customer: pg_notify('wiki_synthesize_pending', customer_id).
  4. Exit.

The wiki-worker app's NotifyListener wakes immediately, drains the
queue. We can't `await` completion here — the worker decouples on
purpose so a long pebble-scale drain doesn't hold the cron machine
alive.

Also exposed as a function so tests can drive it without spinning up a
process.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import asyncpg

from shared.config import get_settings
from shared.constants import WIKI_PENDING_CHANNEL
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)


async def trigger_nightly_synthesis(dsn: str) -> int:
    """Fire pg_notify on `wiki_synthesize_pending` for every opted-in
    customer with pending rows. Returns the count of customers notified.
    """
    started_at = datetime.now(UTC)
    customer_ids: list[str] = []
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT DISTINCT q.customer_id
            FROM wiki_synthesis_queue q
            JOIN customers c ON c.customer_id = q.customer_id
            WHERE q.status = 'pending'
              AND c.preferences->>'wiki_generation_enabled' = 'true'
            """
        )
        customer_ids = [row["customer_id"] for row in rows]
        for customer_id in customer_ids:
            await conn.execute(
                "SELECT pg_notify($1, $2)",
                WIKI_PENDING_CHANNEL,
                customer_id,
            )
    finally:
        await conn.close()
    log.info(
        "nightly_trigger.fired",
        customers=len(customer_ids),
        elapsed_seconds=(datetime.now(UTC) - started_at).total_seconds(),
        started_at=started_at.isoformat(),
    )
    return len(customer_ids)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("nightly_trigger.start", environment=settings.environment)
    notified = await trigger_nightly_synthesis(settings.database_url)
    log.info("nightly_trigger.done", customers=notified)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
