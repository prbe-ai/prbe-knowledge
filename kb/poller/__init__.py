"""Integration poller — periodic re-enqueue for poll-only connectors.

Replaces the per-source GranolaScheduler that ran as the retired
prbe-knowledge-poller Fly app. Now lives in-process inside the worker
deployment alongside ReclaimLoop, GranolaNotifyListener, etc.

A connector opts in by setting `poll_config: ClassVar[PollConfig | None]`
on its Connector subclass (see services/ingestion/handlers/base.py). The
poller discovers participating connectors by walking the handler registry.

Flow per tick:
    1. SELECT all customers with an active token for the source whose
       backfill_state is in cfg.eligible_statuses and last_progress_at
       is older than cfg.interval_seconds.
    2. For each: call re_enqueue_for_polling (preserves last_cursor).
    3. NOTIFY cfg.notify_channel so the worker's listener wakes
       BackfillWorker immediately instead of waiting for its poll cycle.

Backfill EXECUTION still happens in the worker process. The poller only
flips backfill_state rows.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Mapping
from datetime import UTC, datetime

from engine.ingest.handlers.base import PollConfig
from engine.ingest.handlers.registry import (
    get_connector_class,
    list_registered,
)
from engine.shared.constants import IntegrationStatus, SourceSystem
from engine.shared.db import get_pool, raw_conn
from engine.shared.logging import get_logger
from kb.backfill_runner import re_enqueue_for_polling

log = get_logger(__name__)


class IntegrationPoller:
    """Tick loop driving every connector with a non-None poll_config."""

    def __init__(
        self,
        *,
        configs: Mapping[SourceSystem, PollConfig] | None = None,
    ) -> None:
        """If `configs` is omitted, discover from the connector registry at
        run() time. Tests pass an explicit dict to bypass registry state."""
        self._explicit_configs = configs
        self._shutdown = asyncio.Event()

    def shutdown(self) -> None:
        self._shutdown.set()

    @staticmethod
    def _discover() -> dict[SourceSystem, PollConfig]:
        """Walk the connector registry, pick up every class whose poll_config is set."""
        out: dict[SourceSystem, PollConfig] = {}
        for source in list_registered():
            cfg = get_connector_class(source).poll_config
            if cfg is not None:
                out[source] = cfg
        return out

    async def run(self) -> None:
        configs: Mapping[SourceSystem, PollConfig] = (
            self._explicit_configs if self._explicit_configs is not None else self._discover()
        )

        if not configs:
            log.info("integration_poller.empty")
            return

        log.info(
            "integration_poller.start",
            sources=[s.value for s in configs],
        )

        # Boot tick so a freshly-deployed worker doesn't wait the full
        # interval before its first sweep.
        await self._tick_all(configs)

        # Per-source last-tick timestamps gate work. Outer sleep granularity
        # is the min interval; sources with longer intervals just skip ticks
        # until their cfg.interval_seconds elapses.
        min_interval = min(c.interval_seconds for c in configs.values())
        last_tick: dict[SourceSystem, float] = {s: time.monotonic() for s in configs}

        while not self._shutdown.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._shutdown.wait(), timeout=min_interval)
            if self._shutdown.is_set():
                break
            now = time.monotonic()
            for source, cfg in configs.items():
                if now - last_tick[source] >= cfg.interval_seconds:
                    await self._tick_source(source, cfg)
                    last_tick[source] = now

        log.info("integration_poller.stop")

    async def _tick_all(self, configs: Mapping[SourceSystem, PollConfig]) -> None:
        for source, cfg in configs.items():
            await self._tick_source(source, cfg)

    async def _tick_source(self, source: SourceSystem, cfg: PollConfig) -> None:
        try:
            customers = await self._fetch_due_customers(source, cfg)
        except Exception:
            log.exception("integration_poller.fetch_failed", source=source.value)
            return

        if not customers:
            log.debug("integration_poller.tick_idle", source=source.value)
            return

        for customer_id in customers:
            try:
                triggered = await re_enqueue_for_polling(customer_id, source)
            except Exception:
                log.exception(
                    "integration_poller.enqueue_failed",
                    customer=customer_id,
                    source=source.value,
                )
                continue

            if not triggered:
                # Row was already pending or running — skip the notify.
                continue

            try:
                async with get_pool().acquire() as conn:
                    await conn.execute(
                        "SELECT pg_notify($1, $2)",
                        cfg.notify_channel,
                        customer_id,
                    )
            except Exception:
                log.exception(
                    "integration_poller.notify_failed",
                    customer=customer_id,
                    source=source.value,
                )

            log.info(
                "integration_poller.re_enqueued",
                customer=customer_id,
                source=source.value,
                tick_at=datetime.now(UTC).isoformat(),
            )

    async def _fetch_due_customers(self, source: SourceSystem, cfg: PollConfig) -> list[str]:
        """Customers whose backfill for `source` is in an eligible status and stale.

        Skips PENDING/RUNNING by virtue of eligible_statuses — re-enqueueing
        an in-flight row would be a no-op anyway, but filtering in SQL avoids
        the round-trip. NULL last_progress_at means the initial backfill
        never recorded progress — include it so it gets re-attempted.
        """
        status_values = [s.value for s in cfg.eligible_statuses]
        async with raw_conn() as conn:
            rows = await conn.fetch(
                """
                SELECT t.customer_id
                FROM integration_tokens t
                JOIN backfill_state b
                  ON b.customer_id = t.customer_id
                 AND b.source_system = t.source_system
                WHERE t.source_system = $1
                  AND t.status = $2
                  AND b.status = ANY($3::text[])
                  AND (
                    b.last_progress_at IS NULL
                    OR b.last_progress_at < NOW() - make_interval(secs => $4)
                  )
                ORDER BY b.last_progress_at NULLS FIRST
                """,
                source.value,
                IntegrationStatus.ACTIVE.value,
                status_values,
                cfg.interval_seconds,
            )
        return [r["customer_id"] for r in rows]


__all__ = ["IntegrationPoller"]
