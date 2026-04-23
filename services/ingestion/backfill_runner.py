"""Reusable backfill runner.

One function, used by two callers:
  - `BackfillWorker` (services/ingestion/worker.py) — runs in the worker process,
    drains `backfill_state` rows with status='pending'.
  - `scripts/backfill.py` CLI — operator-triggered ad-hoc runs.

Responsibilities:
  - Resolve the integration token for the (customer, source)
  - Call `connector.backfill(customer_id, token, cursor)`
  - For each yielded WebhookEvent: put raw envelope to R2, insert into
    ingestion_queue. The regular ingestion worker picks it up like any webhook.
  - Update progress (`last_cursor`, `events_enqueued`, `last_progress_at`,
    `heartbeat_at`) every N events.
  - Mark done / failed at the end.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.registry import build_connector
from shared.constants import BackfillStatus, QueueStatus, SourceSystem
from shared.db import get_pool, raw_conn
from shared.encryption import decrypt_token
from shared.exceptions import NotSupportedByConnector
from shared.logging import get_logger
from shared.metrics import counter
from shared.models import IntegrationToken
from shared.storage import get_store

log = get_logger(__name__)

PROGRESS_EVERY_N_EVENTS = 25


async def run_backfill(
    ctx: ConnectorContext,
    customer_id: str,
    source: SourceSystem,
) -> int:
    """Execute a backfill for (customer, source). Returns events enqueued.

    Assumes backfill_state row already exists (status='pending' or 'running').
    Responsible for marking it complete/failed.
    """
    token = await _load_token(customer_id, source)
    if token is None:
        await _mark_failed(customer_id, source, "no active integration_tokens row")
        return 0

    cursor = await _load_cursor(customer_id, source)
    store = get_store()
    bucket = store.bucket_for(customer_id)
    await store.ensure_bucket(bucket)

    connector = build_connector(source, ctx)

    await _mark_running(customer_id, source)
    log.info(
        "backfill.start",
        customer=customer_id,
        source=source.value,
        resume_cursor=bool(cursor),
    )

    enqueued = 0
    try:
        try:
            events = connector.backfill(customer_id, token, cursor)
        except NotSupportedByConnector as exc:
            await _mark_failed(customer_id, source, f"not supported: {exc}")
            return 0

        latest_cursor = cursor
        async for event in events:
            envelope = json.dumps(
                {
                    "_headers": event.headers,
                    "payload": event.raw_payload,
                    "received_at": event.received_at.isoformat(),
                    "trace_id": f"backfill-{customer_id}-{source.value}-{enqueued}",
                    "_backfill": True,
                }
            ).encode()
            key = (
                f"raw/{source.value}/{customer_id}/backfill/"
                f"{event.source_event_id.replace('/', '_')}.json"
            )
            await store.put(bucket, key, envelope)

            async with get_pool().acquire() as conn:
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

            # The runner can discover a new cursor via the event's parse_hint
            # or via the connector yielding a tuple — keep it simple for now:
            # most connectors set a `parse_hint["cursor"]` on their synthesized
            # events specifically so the runner can persist it.
            possible_cursor = (event.raw_payload or {}).get("_cursor")
            if possible_cursor is not None:
                latest_cursor = str(possible_cursor)

            enqueued += 1
            if enqueued % PROGRESS_EVERY_N_EVENTS == 0:
                await _update_progress(customer_id, source, latest_cursor, enqueued)

        await _mark_done(customer_id, source, enqueued, latest_cursor)
        counter(
            "backfill.completed",
            1,
            source=source.value,
            events=enqueued,
        )
        log.info(
            "backfill.done", customer=customer_id, source=source.value, events=enqueued
        )
    except Exception as exc:
        await _mark_failed(customer_id, source, str(exc))
        counter("backfill.failed", 1, source=source.value)
        log.exception("backfill.failed", customer=customer_id, source=source.value)
        raise

    return enqueued


# ---- backfill_state writes -----------------------------------------------


async def enqueue_backfill(customer_id: str, source: SourceSystem) -> None:
    """Insert a `pending` backfill_state row so a worker picks it up."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO backfill_state
                (customer_id, source_system, status, started_at, events_enqueued)
            VALUES ($1, $2, $3, NULL, 0)
            ON CONFLICT (customer_id, source_system)
            DO UPDATE SET status          = EXCLUDED.status,
                          last_cursor     = NULL,
                          last_error      = NULL,
                          events_enqueued = 0,
                          started_at      = NULL,
                          heartbeat_at    = NULL,
                          completed_at    = NULL
            """,
            customer_id,
            source.value,
            BackfillStatus.PENDING.value,
        )


async def _mark_running(customer_id: str, source: SourceSystem) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            UPDATE backfill_state
            SET status = $1, started_at = COALESCE(started_at, NOW()), heartbeat_at = NOW()
            WHERE customer_id = $2 AND source_system = $3
            """,
            BackfillStatus.RUNNING.value,
            customer_id,
            source.value,
        )


async def _update_progress(
    customer_id: str,
    source: SourceSystem,
    cursor: str | None,
    enqueued: int,
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            UPDATE backfill_state
            SET last_cursor      = $1,
                events_enqueued  = $2,
                last_progress_at = NOW(),
                heartbeat_at     = NOW()
            WHERE customer_id = $3 AND source_system = $4
            """,
            cursor,
            enqueued,
            customer_id,
            source.value,
        )


async def _mark_done(
    customer_id: str,
    source: SourceSystem,
    enqueued: int,
    cursor: str | None,
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            UPDATE backfill_state
            SET status           = $1,
                last_cursor      = $2,
                events_enqueued  = $3,
                last_progress_at = NOW(),
                heartbeat_at     = NOW(),
                completed_at     = NOW()
            WHERE customer_id = $4 AND source_system = $5
            """,
            BackfillStatus.COMPLETE.value,
            cursor,
            enqueued,
            customer_id,
            source.value,
        )


async def _mark_failed(
    customer_id: str, source: SourceSystem, error: str
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            UPDATE backfill_state
            SET status       = $1,
                last_error   = $2,
                heartbeat_at = NOW()
            WHERE customer_id = $3 AND source_system = $4
            """,
            BackfillStatus.FAILED.value,
            error[:1000],
            customer_id,
            source.value,
        )


async def _load_cursor(customer_id: str, source: SourceSystem) -> str | None:
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT last_cursor FROM backfill_state WHERE customer_id=$1 AND source_system=$2",
            customer_id,
            source.value,
        )
    return row["last_cursor"] if row else None


async def _load_token(
    customer_id: str, source: SourceSystem
) -> IntegrationToken | None:
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
    if row is None:
        return None
    return IntegrationToken(
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


async def claim_pending_backfill() -> tuple[str, SourceSystem] | None:
    """Pick up one pending backfill_state row atomically. SKIP LOCKED for concurrency."""
    async with get_pool().acquire() as conn, conn.transaction():
        row = await conn.fetchrow(
            """
            SELECT customer_id, source_system
            FROM backfill_state
            WHERE status = 'pending'
            ORDER BY customer_id, source_system
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """
        )
        if row is None:
            return None
        # Claim it by setting status=running immediately (inside the same tx).
        await conn.execute(
            """
            UPDATE backfill_state
            SET status = $1, started_at = NOW(), heartbeat_at = NOW()
            WHERE customer_id = $2 AND source_system = $3
            """,
            BackfillStatus.RUNNING.value,
            row["customer_id"],
            row["source_system"],
        )
    return row["customer_id"], SourceSystem(row["source_system"])


# Date/time helpers used in tests.
def utcnow() -> datetime:
    return datetime.now(UTC)
