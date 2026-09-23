"""How coding-agent sessions end, and how often the client's own signal is missed.

A session ends when an end signal is the newest key on its queue row
(engine/shared/session_signals.py): the client's finalize, or the idle sweep's
marker after 24h of silence. The sweep is the backstop, so its share is the
miss rate of the client-side signals -- but only after allowing for a LATE
client finalize (a laptop that slept, then woke and said goodbye), which lands
on top of the marker and is a late signal, not a missing one.

For protocol-2 sessions, whose client finalize is recorded in
`session_streams.finalized`:

    client_finalized   no marker on the row, stream finalized
    late_client        a marker on the row, then the client's finalize on top
    sweep_only         the marker is on top and the stream never finalized
    open               neither (still live, or idle less than the sweep window)

Counted over sessions that STARTED in a window ending `--grace-days` ago, so
each one has had that long to be finalized by its client before it counts as
a miss. Also summarises `extraction_outcome` (how the last pass went) across
all rows.

    python -m scripts.session_end_report --days 7 --grace-days 7
"""

from __future__ import annotations

import argparse
import asyncio
import json

from engine.shared.db import get_pool, init_pool
from engine.shared.session_signals import CRON_MARKER_SUFFIX, V2_KEY_SEGMENT
from kb.session_completer import AGENT_SOURCES, MAX_EXTRACTION_RETRIES

_V2_ENDINGS_SQL = f"""
WITH v2 AS (
    SELECT q.payload_s3_keys AS k, s.finalized AS finalized
      FROM ingestion_queue q
      LEFT JOIN session_streams s
        ON s.customer_id = q.customer_id
       AND s.source_system = q.source_system
       AND s.session_id = q.source_event_id
     WHERE q.source_system = $1
       AND q.customer_id = $2
       AND q.first_enqueued_at >= NOW() - make_interval(days => $3::int + $4::int)
       AND q.first_enqueued_at <  NOW() - make_interval(days => $4::int)
       AND EXISTS (SELECT 1 FROM unnest(q.payload_s3_keys) _k WHERE strpos(_k, '{V2_KEY_SEGMENT}') > 0)
)
SELECT
  count(*) FILTER (WHERE COALESCE(finalized, false)
                     AND NOT EXISTS (SELECT 1 FROM unnest(k) _k WHERE right(_k, 16) = '{CRON_MARKER_SUFFIX}'))
    AS client_finalized,
  count(*) FILTER (WHERE COALESCE(finalized, false)
                     AND right(k[cardinality(k)], 16) <> '{CRON_MARKER_SUFFIX}'
                     AND EXISTS (SELECT 1 FROM unnest(k) _k WHERE right(_k, 16) = '{CRON_MARKER_SUFFIX}'))
    AS late_client,
  count(*) FILTER (WHERE NOT COALESCE(finalized, false)
                     AND right(k[cardinality(k)], 16) = '{CRON_MARKER_SUFFIX}')
    AS sweep_only,
  count(*) FILTER (WHERE NOT COALESCE(finalized, false)
                     AND right(k[cardinality(k)], 16) <> '{CRON_MARKER_SUFFIX}')
    AS open
  FROM v2
"""

_OUTCOMES_SQL = """
SELECT COALESCE(extraction_outcome ->> 'reason', '<none recorded>') AS reason,
       COALESCE(extraction_outcome ->> 'authoritative', '') AS authoritative,
       count(*) AS rows,
       count(*) FILTER (WHERE COALESCE((extraction_outcome ->> 'retries')::int, 0) >= $2
                          AND extraction_outcome ->> 'reason' <> 'disabled') AS retries_exhausted
  FROM ingestion_queue
 WHERE source_system = $1
 GROUP BY 1, 2
 ORDER BY 3 DESC
"""


async def report(*, days: int, grace_days: int) -> dict:
    out: dict = {"window_days": days, "grace_days": grace_days, "v2_endings": {}, "outcomes": {}}
    async with get_pool().acquire() as conn:
        tenants = [r["customer_id"] for r in await conn.fetch("SELECT customer_id FROM customers")]
        has_outcome = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'ingestion_queue' AND column_name = 'extraction_outcome'"
        )
        for source in AGENT_SOURCES:
            totals = {"client_finalized": 0, "late_client": 0, "sweep_only": 0, "open": 0}
            for tenant in tenants:
                # session_streams is FORCE RLS: read it as that tenant.
                async with conn.transaction():
                    await conn.execute("SELECT set_config('app.current_customer_id', $1, true)", tenant)
                    row = await conn.fetchrow(_V2_ENDINGS_SQL, source.value, tenant, days, grace_days)
                for key in totals:
                    totals[key] += row[key]
            ended = totals["client_finalized"] + totals["late_client"] + totals["sweep_only"]
            totals["client_miss_rate"] = round(totals["sweep_only"] / ended, 4) if ended else None
            out["v2_endings"][source.value] = totals
            if has_outcome:
                out["outcomes"][source.value] = [
                    dict(r) for r in await conn.fetch(_OUTCOMES_SQL, source.value, MAX_EXTRACTION_RETRIES)
                ]
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=7, help="Sessions started in this many days...")
    parser.add_argument("--grace-days", type=int, default=7, help="...ending this many days ago.")
    args = parser.parse_args(argv)
    if args.days < 1 or args.grace_days < 1:
        parser.error("--days and --grace-days must be >= 1")
    return args


async def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    await init_pool()
    try:
        result = await report(days=args.days, grace_days=args.grace_days)
    finally:
        from engine.shared.db import close_pool

        await close_pool()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(_main())
