"""Historical backfill per (customer, source).

Iterates a connector's `backfill()` generator, writes each event to R2,
and enqueues queue rows. Resumable via `backfill_state.last_cursor`.

Usage:
    .venv/bin/python -m scripts.backfill --customer cust-smoke --source slack
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime

from engine.ingest.handlers.base import make_default_context
from engine.ingest.handlers.registry import build_connector
from engine.shared.config import get_settings
from engine.shared.constants import BackfillStatus, QueueStatus, SourceSystem
from engine.shared.db import close_pool, init_pool, raw_conn
from engine.shared.encryption import decrypt_token
from engine.shared.exceptions import NotSupportedByConnector
from engine.shared.logging import configure_logging, get_logger
from engine.shared.models import IntegrationToken
from engine.shared.storage import get_store

log = get_logger(__name__)


async def backfill(customer_id: str, source_value: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    source = SourceSystem(source_value)

    store = get_store()
    bucket = await store.bucket_for(customer_id)
    await store.ensure_bucket(bucket)

    # Load token + cursor
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT access_token_encrypted, refresh_token_encrypted, expires_at, scope,
                   webhook_secret
            FROM integration_tokens
            WHERE customer_id=$1 AND source_system=$2 AND status='active'
            """,
            customer_id,
            source.value,
        )
        cursor_row = await conn.fetchrow(
            """
            SELECT last_cursor FROM backfill_state
            WHERE customer_id=$1 AND source_system=$2
            """,
            customer_id,
            source.value,
        )

    if row is None:
        raise SystemExit(f"no active integration_tokens row for {customer_id}/{source.value}")

    token = IntegrationToken(
        customer_id=customer_id,
        source_system=source,
        access_token=decrypt_token(row["access_token_encrypted"]),
        refresh_token=(
            decrypt_token(row["refresh_token_encrypted"])
            if row["refresh_token_encrypted"]
            else None
        ),
        expires_at=row["expires_at"],
        scope=row["scope"],
        webhook_secret=row["webhook_secret"],
    )
    cursor = cursor_row["last_cursor"] if cursor_row else None

    ctx = make_default_context()
    connector = build_connector(source, ctx)

    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO backfill_state (customer_id, source_system, status, started_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (customer_id, source_system)
            DO UPDATE SET status = EXCLUDED.status, started_at = NOW()
            """,
            customer_id,
            source.value,
            BackfillStatus.RUNNING.value,
        )

    count = 0
    try:
        try:
            events = connector.backfill(customer_id, token, cursor)
        except NotSupportedByConnector as exc:
            log.warning("backfill.not_supported", source=source.value, error=str(exc))
            return

        async for event in events:
            envelope = {
                "_headers": {},
                "payload": event.raw_payload,
                "received_at": datetime.now(UTC).isoformat(),
                "trace_id": f"backfill-{count}",
            }
            key = f"raw/{source.value}/{customer_id}/backfill/{event.source_event_id}.json"
            await store.put(bucket, key, json.dumps(envelope).encode())

            async with raw_conn() as conn:
                await conn.execute(
                    """
                    INSERT INTO ingestion_queue
                        (customer_id, source_system, source_event_id, payload_s3_key, status)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT DO NOTHING
                    """,
                    customer_id,
                    source.value,
                    event.source_event_id,
                    key,
                    QueueStatus.PENDING.value,
                )
            count += 1
            if count % 100 == 0:
                log.info("backfill.progress", source=source.value, count=count)

        async with raw_conn() as conn:
            await conn.execute(
                """
                UPDATE backfill_state
                SET status=$1, completed_at=NOW(), last_progress_at=NOW()
                WHERE customer_id=$2 AND source_system=$3
                """,
                BackfillStatus.COMPLETE.value,
                customer_id,
                source.value,
            )
    finally:
        await ctx.http.aclose()
        await close_pool()

    log.info("backfill.done", customer=customer_id, source=source.value, count=count)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer", required=True)
    ap.add_argument("--source", required=True, choices=[s.value for s in SourceSystem])
    args = ap.parse_args()
    asyncio.run(backfill(args.customer, args.source))


if __name__ == "__main__":
    main()
