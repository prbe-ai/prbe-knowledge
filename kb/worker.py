"""Composed ingestion-worker process: engine queue drain + kb integrations.

`python -m services.ingestion.worker` (compose / helm / hosted charts) lands
here via the thin services wrapper. The generic queue-drain Worker and
ReclaimLoop live in engine.ingest.worker; this module adds the
integration-coupled pieces (source backfills, Granola notify listener,
poll-mode scheduler, connector registration) and wires them together.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from datetime import UTC, datetime

import asyncpg

from engine.ingest.handlers.base import ConnectorContext, make_default_context
from engine.ingest.worker import ReclaimLoop, Worker, _build_health_app
from engine.shared.constants import GRANOLA_REFRESH_CHANNEL
from engine.shared.db import apply_connection_setup, init_pool
from engine.shared.logging import bind_trace, get_logger
from kb.poller import IntegrationPoller
from kb.polling.scheduler import PollScheduler
from kb.polling.sink import PollDocumentSink

log = get_logger(__name__)


class BackfillWorker:
    """Drains backfill_state rows with status='pending'.

    Runs alongside the ingestion Worker in the same process. Each claimed row
    is handed to run_backfill which paginates the source and enqueues events
    into ingestion_queue — where the ingestion Worker picks them up like any
    other webhook. The two workers are independent; backfill progress does
    not block webhook processing.

    Wake semantics:
      - Normal cycle: sleep `poll_interval` between claim attempts.
      - On `wake_event` set (via pg_notify from /admin/.../granola/refresh
        or from IntegrationPoller's tick), break sleep early and re-poll.
      - The wake_event is informational — the row was already enqueued by
        the caller; we just want to start work sub-second instead of waiting.
    """

    def __init__(
        self,
        ctx: ConnectorContext,
        poll_interval: float = 5.0,
        wake_event: asyncio.Event | None = None,
        heartbeat_interval_seconds: float = 30.0,
    ) -> None:
        self._ctx = ctx
        self._poll_interval = poll_interval
        self._shutdown = asyncio.Event()
        self._wake = wake_event or asyncio.Event()
        self._heartbeat_interval_seconds = heartbeat_interval_seconds

    async def run(self) -> None:
        from kb.backfill_runner import (
            claim_pending_backfill,
            run_backfill,
        )

        log.info("backfill_worker.start")
        while not self._shutdown.is_set():
            claimed = await claim_pending_backfill()
            if claimed is None:
                await self._sleep_or_wake()
                continue

            customer_id, source = claimed
            bind_trace(f"backfill-{customer_id}-{source.value}")
            try:
                await run_backfill(
                    self._ctx,
                    customer_id,
                    source,
                    heartbeat_interval_seconds=self._heartbeat_interval_seconds,
                )
            except Exception:
                log.exception(
                    "backfill_worker.run_failed",
                    customer=customer_id,
                    source=source.value,
                )
        log.info("backfill_worker.stop")

    async def _sleep_or_wake(self) -> None:
        """Wait until shutdown, wake, or poll_interval timeout — whichever first."""
        shutdown_task = asyncio.create_task(self._shutdown.wait())
        wake_task = asyncio.create_task(self._wake.wait())
        try:
            _done, pending = await asyncio.wait(
                {shutdown_task, wake_task},
                timeout=self._poll_interval,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
        finally:
            if self._wake.is_set():
                self._wake.clear()

    def shutdown(self) -> None:
        self._shutdown.set()


class GranolaNotifyListener:
    """LISTEN granola_refresh on a dedicated asyncpg connection.

    Sets `wake_event` whenever a NOTIFY arrives, which the BackfillWorker uses
    to break its poll-interval sleep early. Reconnects on connection drop with
    exponential backoff (1s → 60s).

    Belt-and-suspenders: the BackfillWorker still polls every `poll_interval`
    seconds even if no notify ever arrives. A missed notify (during reconnect)
    just means up to one poll_interval extra latency.
    """

    def __init__(self, dsn: str, wake_event: asyncio.Event) -> None:
        self._dsn = dsn
        self._wake = wake_event
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        log.info("granola_listener.start", channel=GRANOLA_REFRESH_CHANNEL)
        backoff = 1.0
        while not self._shutdown.is_set():
            try:
                conn = await asyncpg.connect(self._dsn)
                # Direct asyncpg.connect bypasses the pool's on_connect hook
                # (which pins search_path so AGE Cypher resolves ag_catalog.*).
                # Apply the same setup to keep LISTEN-channel callbacks consistent.
                await apply_connection_setup(conn)
            except (asyncpg.PostgresError, OSError) as exc:
                log.warning(
                    "granola_listener.connect_failed",
                    error=str(exc),
                    backoff_seconds=backoff,
                )
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._shutdown.wait(), timeout=backoff)
                backoff = min(backoff * 2, 60.0)
                continue

            backoff = 1.0
            try:

                def _on_notify(_conn, _pid, _channel, payload) -> None:
                    log.info(
                        "granola_listener.notified",
                        payload=payload,
                    )
                    self._wake.set()

                await conn.add_listener(GRANOLA_REFRESH_CHANNEL, _on_notify)
                log.info("granola_listener.ready")
                # Keep the connection alive. Periodic SELECT 1 detects half-open
                # connections (e.g., after a network partition) faster than waiting
                # for a packet to time out.
                while not self._shutdown.is_set():
                    try:
                        await asyncio.wait_for(self._shutdown.wait(), timeout=30.0)
                    except TimeoutError:
                        try:
                            await conn.fetchval("SELECT 1")
                        except (asyncpg.PostgresError, OSError) as exc:
                            log.warning("granola_listener.lost", error=str(exc))
                            break
            finally:
                with contextlib.suppress(Exception):
                    await conn.close()

        log.info("granola_listener.stop")

    def shutdown(self) -> None:
        self._shutdown.set()



async def run_worker_forever() -> None:
    """Entry point for `python -m engine.ingest.worker`.

    Runs the ingestion drain, the backfill drain, and a tiny health
    HTTP server concurrently. They share the same asyncpg pool.
    """
    import os

    import uvicorn

    from engine.shared.config import get_settings
    from engine.shared.logging import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    # Import handlers package so @register_connector decorators run.
    # Side-effect import — handlers' @register_connector decorators run on
    # import. The module name is used by the registry, so ruff doesn't flag
    # it as unused.
    import kb.handlers

    _ = kb.handlers  # defensive against future ruff strictness

    # INGESTION_MODE=poll is set by the self-host chart only — public
    # webhook URLs aren't reachable from inside a customer's cluster, so
    # ingestion has to be outbound-poll. When set, import the per-source
    # poller modules so their module-level register_poller() calls run
    # before PollScheduler.run_forever() scans the registry; instantiate
    # the real document sink that pipes polled docs through the same
    # R2 + ingestion_queue path the inbound-webhook handlers use.
    # Managed-shared leaves this unset and keeps the existing webhook
    # ingestion path untouched.
    ingestion_mode = os.environ.get("INGESTION_MODE", "webhook").strip().lower()
    poll_scheduler: PollScheduler | None = None
    if ingestion_mode == "poll":
        # Side-effect imports — each module's @register_poller decorator
        # runs at import time, registering the per-source BasePoller
        # subclass under its SourceSystem. The scheduler reads from that
        # registry on the first tick. The names ARE used (by the
        # registry), so ruff doesn't flag them as unused — no noqa.
        import kb.polling.github
        import kb.polling.linear
        import kb.polling.notion
        import kb.polling.sentry
        import kb.polling.slack

        # Reference the modules so ruff sees them as "used" — defensive,
        # in case a future ruff version flags side-effect-only imports.
        _ = (
            kb.polling.github,
            kb.polling.linear,
            kb.polling.notion,
            kb.polling.sentry,
            kb.polling.slack,
        )

        poll_scheduler = PollScheduler(sink=PollDocumentSink())
        log.info("worker.boot.polling_enabled", mode=ingestion_mode)

    ctx = make_default_context()
    # Shared wake event: NotifyListener sets it on pg_notify, BackfillWorker
    # reads it to break its poll sleep early. Single asyncio.Event because
    # both live in the same process.
    wake_event = asyncio.Event()
    ingestion_worker = Worker(
        ctx,
        max_attempts=settings.worker_max_attempts,
        concurrency=settings.worker_max_concurrent,
        per_customer_max_inflight=settings.worker_per_customer_max_inflight,
    )
    backfill_worker = BackfillWorker(
        ctx,
        wake_event=wake_event,
        heartbeat_interval_seconds=settings.backfill_heartbeat_interval_seconds,
    )
    granola_listener = GranolaNotifyListener(settings.database_url, wake_event)
    reclaim_loop = ReclaimLoop(
        backfill_threshold_seconds=settings.backfill_stale_heartbeat_seconds,
    )
    # IntegrationPoller replaces the retired prbe-knowledge-poller Fly app.
    # Discovers participating connectors via their poll_config ClassVar.
    integration_poller = IntegrationPoller()
    # Wiki triage + synthesis run in their own dedicated fly apps
    # (prbe-knowledge-wiki-worker / prbe-knowledge-wiki-synthesis), driven
    # by pg_notify channels and a nightly cron. Removed from this app's
    # gather as part of the wiki triage redesign — see services/synthesis/
    # triage_app.py and synthesis_app.py for the new entry points.

    health_port = int(os.environ.get("WORKER_HEALTH_PORT", "8082"))
    health_config = uvicorn.Config(
        _build_health_app(),
        host="0.0.0.0",
        port=health_port,
        log_config=None,
        lifespan="off",
        access_log=False,
    )
    health_server = uvicorn.Server(health_config)

    log.info(
        "worker.boot",
        environment=settings.environment,
        health_port=health_port,
        timestamp=datetime.now(UTC).isoformat(),
    )

    # Signal-driven graceful shutdown. On SIGTERM (fly stop / rolling deploy)
    # or SIGINT, we:
    #   1. Set _shutdown on each loop so they stop claiming new work.
    #   2. Tell uvicorn to drain in-flight HTTP requests.
    #   3. Cancel the gather. CancelledError propagates into BackfillWorker.run's
    #      `await run_backfill(...)`; run_backfill's CancelledError handler
    #      releases its backfill_state claim (status -> pending, last_cursor
    #      preserved) before re-raising. Without this, deploys leave rows
    #      in 'running' until the 5-min reclaim cron sweeps them.
    loop = asyncio.get_running_loop()
    gather_future: asyncio.Future | None = None  # type: ignore[type-arg]
    shutdown_started = False

    def handle_signal(signame: str) -> None:
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True
        log.info("worker.shutdown_signal", signal=signame)
        ingestion_worker.shutdown()
        backfill_worker.shutdown()
        granola_listener.shutdown()
        reclaim_loop.shutdown()
        integration_poller.shutdown()
        if poll_scheduler is not None:
            poll_scheduler.stop()
        health_server.should_exit = True
        if gather_future is not None and not gather_future.done():
            gather_future.cancel()

    for signame in ("SIGTERM", "SIGINT"):
        with contextlib.suppress(NotImplementedError):
            # Some platforms (Windows, sandboxed environments) don't support
            # add_signal_handler; on those, the default Python signal behavior
            # (KeyboardInterrupt for SIGINT, terminate for SIGTERM) applies.
            loop.add_signal_handler(getattr(signal, signame), handle_signal, signame)

    coroutines = [
        ingestion_worker.run(poll_interval=settings.worker_poll_interval_seconds),
        backfill_worker.run(),
        granola_listener.run(),
        reclaim_loop.run(),
        integration_poller.run(),
        health_server.serve(),
    ]
    if poll_scheduler is not None:
        # Polling scheduler runs alongside the existing drains. Its sink
        # writes into the same ingestion_queue the webhook handlers do, so
        # the ingestion drain above picks the rows up unchanged.
        coroutines.append(poll_scheduler.run_forever())

    try:
        gather_future = asyncio.gather(*coroutines)
        try:
            await gather_future
        except asyncio.CancelledError:
            log.info("worker.shutdown_complete")
    finally:
        # Drain in-flight release tasks before asyncio.run's task-cancel sweep
        # interrupts their asyncpg UPDATE mid-roundtrip. See PR #210.
        from kb.backfill_runner import drain_pending_release_tasks

        await drain_pending_release_tasks()
        await ctx.http.aclose()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_worker_forever())
