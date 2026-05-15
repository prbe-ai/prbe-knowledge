"""Load 100+ synthetic documents for integration testing.

Generates Slack-shaped fixture payloads and pushes them through the normalizer
directly (bypassing the webhook fast path — we don't need signature exercise here).

Usage:
    .venv/bin/python -m scripts.seed_synthetic --customer cust-smoke --count 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from datetime import UTC, datetime

from services.ingestion.handlers.base import make_default_context
from services.ingestion.normalizer import Normalizer
from shared.config import get_settings
from shared.constants import QueueStatus, SourceSystem
from shared.db import close_pool, init_pool, raw_conn
from shared.logging import configure_logging, get_logger
from shared.storage import get_store

log = get_logger(__name__)


CHANNELS = ["C_PAYMENTS", "C_INFRA", "C_GROWTH", "C_SUPPORT", "C_DEPLOYS"]
USERS = ["U_ALICE", "U_BOB", "U_CARL", "U_DANA", "U_ERIN"]
TOPICS = [
    "the payments service is returning 500s after the last deploy",
    "memory spike on growth-api pod after merge of #3421",
    "new onboarding flow shipping behind feature flag next Tuesday",
    "investigating flaky integration test in billing-service",
    "customer lost access to workspace after SSO rotation",
    "p95 latency regression on /checkout endpoint",
    "roll-forward strategy for the payments migration",
    "docs updated for the new auth middleware",
    "rotating prod credentials end of quarter",
    "CODEOWNERS cleanup across infra repos",
]


def _make_slack_fixture(i: int, base_ts: float) -> tuple[str, bytes, dict]:
    ts = f"{base_ts + i:.6f}"
    channel = random.choice(CHANNELS)
    user = random.choice(USERS)
    text = f"{random.choice(TOPICS)} (seed #{i})"
    payload = {
        "team_id": "T_SEED",
        "type": "event_callback",
        "event": {
            "type": "message",
            "channel": channel,
            "user": user,
            "text": text,
            "ts": ts,
        },
    }
    envelope = {
        "_headers": {},
        "payload": payload,
        "received_at": datetime.now(UTC).isoformat(),
        "trace_id": f"seed-{i}",
    }
    source_event_id = f"{channel}:{ts}"
    return source_event_id, json.dumps(envelope).encode(), payload


async def seed(customer_id: str, count: int) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    store = get_store()
    bucket = await store.bucket_for(customer_id)
    await store.ensure_bucket(bucket)

    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'seed-' || $1, 'seed')
            ON CONFLICT DO NOTHING
            """,
            customer_id,
        )

    ctx = make_default_context()
    normalizer = Normalizer(ctx)
    base_ts = time.time() - (count * 60)

    for i in range(count):
        event_id, envelope_bytes, _payload = _make_slack_fixture(i, base_ts)
        key = f"raw/slack/{customer_id}/seed/{event_id}.json"
        await store.put(bucket, key, envelope_bytes)

        async with raw_conn() as conn:
            await conn.execute(
                """
                INSERT INTO ingestion_queue
                    (customer_id, source_system, source_event_id, payload_s3_key, status)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT DO NOTHING
                """,
                customer_id,
                SourceSystem.SLACK.value,
                event_id,
                key,
                QueueStatus.PENDING.value,
            )

        # Drive normalizer inline — bypasses worker for speed in dev seeding.
        async with raw_conn() as conn:
            row = await conn.fetchrow(
                "SELECT queue_id FROM ingestion_queue WHERE source_event_id=$1 AND customer_id=$2",
                event_id,
                customer_id,
            )
        if row is None:
            continue
        try:
            await normalizer.process_queue_row(
                queue_id=row["queue_id"],
                customer_id=customer_id,
                source_system=SourceSystem.SLACK,
                source_event_id=event_id,
                payload_s3_key=key,
            )
        except Exception as exc:
            log.warning("seed.normalize_failed", i=i, error=str(exc))

    await ctx.http.aclose()
    await close_pool()
    log.info("seed.done", customer=customer_id, count=count)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer", required=True)
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    asyncio.run(seed(args.customer, args.count))


if __name__ == "__main__":
    main()
