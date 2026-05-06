"""Entry point for the prbe-knowledge-wiki-bootstrap fly app.

Per-machine ``BootstrapWorker`` that LISTENs on ``WIKI_BOOTSTRAP_CHANNEL``
(wake hint) and ``WIKI_BOOTSTRAP_CANCEL_CHANNEL`` (force-cancel), claims
``wiki_synthesis_runs`` rows where ``kind='bootstrap' AND status='pending'``
via ``FOR UPDATE SKIP LOCKED``, runs each (customer, source) crawl under
a per-(customer, source) advisory lock + per-machine semaphore, and
writes the terminal status. Mirrors ``services/synthesis/synthesis_worker.py``
in shape — both use NotifyListener + asyncpg queue claim.

Concurrent tasks running in this process:

  - ``BootstrapWorker.run`` — pending-row drain loop.
  - ``NotifyListener`` for ``WIKI_BOOTSTRAP_CHANNEL`` (wake hint).
  - Cancel listener (custom asyncpg add_listener that delivers payloads
    to the worker's cancel queue, since the generic NotifyListener
    discards payload bytes).
  - ``BootstrapReclaimLoop`` — periodic stale-run sweep that flips
    long-running rows back to ``pending``.
  - tiny health server (Fly probe on port 8082).
  - signal handler that drains in-flight crawl tasks on SIGTERM.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import asyncpg
import httpx
import orjson
import uvicorn

from services.synthesis.bootstrap_reclaim import BootstrapReclaimLoop
from services.synthesis.crawlers import REGISTRY
from services.synthesis.crawlers.base import (
    BearerResolver,
    BootstrapAgent,
    BootstrapAgentResult,
    empty_result,
)
from services.synthesis.listeners import NotifyListener
from shared.config import Settings, get_settings
from shared.constants import (
    BOOTSTRAP_PARALLELISM,
    WIKI_BOOTSTRAP_CANCEL_CHANNEL,
    WIKI_BOOTSTRAP_CHANNEL,
)
from shared.db import init_pool, raw_conn
from shared.locks import advisory_lock_key
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lock key helpers (thin wrappers around ``shared.locks.advisory_lock_key``
# so call sites stay readable; the wrappers are also the import points
# tests use to assert key derivation).
# ---------------------------------------------------------------------------


def _bootstrap_run_lock_key(customer_id: str, source: str) -> int:
    """Per-(customer, source) lock key for the BootstrapWorker.

    Defense-in-depth on top of ``FOR UPDATE SKIP LOCKED``: the queue
    claim already prevents two workers from grabbing the same row, but
    reclaim's ``running -> pending`` flip can in principle requeue a row
    whose original worker is still alive but stuck. The advisory lock
    catches that overlap before the new worker invokes the agent.
    """
    return advisory_lock_key("bootstrap-run", customer_id, source)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class _Claim:
    """One claimed pending row. Plain class (not dataclass) so the test
    surface stays minimal and asyncpg row -> Claim conversion is explicit.
    """

    __slots__ = ("customer_id", "run_id", "source")

    def __init__(self, *, customer_id: str, source: str, run_id: int) -> None:
        self.customer_id = customer_id
        self.source = source
        self.run_id = run_id


# Type alias mirroring the orchestrator's previous ``CrawlerFactory``.
# Tests pass a factory; production reads ``REGISTRY[source]`` directly.
_CrawlerFactory = Any


def _no_bearer_factory(_customer_id: str, _source: str) -> BearerResolver:
    """Bearer factory used when the caller didn't pass one. Real
    crawlers (Lane D's GitHub, ...) inject one that calls
    ``shared.backend_client.fetch_<source>_token``.
    """

    async def _resolver() -> str | None:
        return None

    return _resolver  # type: ignore[return-value]


def _make_default_bearer_factory(http: httpx.AsyncClient) -> Any:
    """Production bearer factory: dispatches per-source to the
    matching ``shared.backend_client`` mint helper.

    Each source's resolver is a closure that calls the appropriate
    fetch_<source>_token helper. Helpers we don't have yet (slack,
    linear, ...) return None — their crawler will halt with
    ``auth.missing`` until a resolver lands.
    """
    from shared.backend_client import fetch_github_installation_token

    def factory(customer_id: str, source: str) -> BearerResolver:
        if source == "github":

            async def _gh_resolver() -> str | None:
                try:
                    bearer, _expires = await fetch_github_installation_token(
                        http, customer_id=customer_id
                    )
                    return bearer
                except Exception as exc:
                    log.warning(
                        "bootstrap.bearer_resolver_failed",
                        customer=customer_id,
                        source=source,
                        error=str(exc),
                    )
                    return None

            return _gh_resolver  # type: ignore[return-value]

        async def _no_bearer() -> str | None:
            return None

        return _no_bearer  # type: ignore[return-value]

    return factory


class BootstrapWorker:
    """Drain pending bootstrap rows through per-source crawler agents.

    One instance per machine. Wakes on either NOTIFY channel + a 30s
    safety-net timeout. Per-machine cap on concurrent crawls via
    ``Semaphore(BOOTSTRAP_PARALLELISM)``. Cooperative cancel via
    ``WIKI_BOOTSTRAP_CANCEL_CHANNEL``: the trigger route inserts new
    pending rows after firing the cancel + sleeping 10s; in-flight tasks
    receive ``CancelledError`` and (idempotently) mark their row
    ``cancelled``.
    """

    def __init__(
        self,
        *,
        dsn: str,
        http: httpx.AsyncClient,
        settings: Settings,
        parallelism: int | None = None,
        bearer_resolver_factory: Any | None = None,
        crawler_factories: dict[str, _CrawlerFactory] | None = None,
        periodic_wake_seconds: float = 30.0,
    ) -> None:
        self._dsn = dsn
        self._http = http
        self._settings = settings
        self._parallelism = int(
            parallelism
            if parallelism is not None
            else os.environ.get("BOOTSTRAP_PARALLELISM", BOOTSTRAP_PARALLELISM)
        )
        self._bearer_resolver_factory = bearer_resolver_factory or _no_bearer_factory
        # ``REGISTRY`` is the production source of truth; tests pass a
        # crawler_factories dict that overrides it on a per-source basis.
        self._crawler_factories = crawler_factories
        self._periodic = periodic_wake_seconds

        self._sem = asyncio.Semaphore(self._parallelism)
        # Wake events for the two listeners.
        self._wake_pending = asyncio.Event()
        self._wake_cancel = asyncio.Event()
        # Buffer of cancel payloads parsed off the cancel listener.
        self._cancel_payloads: asyncio.Queue[str | None] = asyncio.Queue()

        self._shutdown = asyncio.Event()
        # In-flight crawl tasks keyed by run_id so the cancel listener
        # can ``task.cancel()`` a specific run when a NOTIFY arrives.
        self._in_flight: dict[int, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        log.info(
            "bootstrap_worker.start",
            parallelism=self._parallelism,
            channel=WIKI_BOOTSTRAP_CHANNEL,
            cancel_channel=WIKI_BOOTSTRAP_CANCEL_CHANNEL,
        )

        # Wake-hint listener: generic NotifyListener (payload-less wake).
        wake_listener = NotifyListener(
            self._dsn,
            WIKI_BOOTSTRAP_CHANNEL,
            self._wake_pending,
            log_prefix="bootstrap_worker.wake_listener",
        )
        wake_task = asyncio.create_task(wake_listener.run(), name="bootstrap.wake_listener")
        cancel_listener_task = asyncio.create_task(
            self._cancel_listener_loop(), name="bootstrap.cancel_listener"
        )
        cancel_processor_task = asyncio.create_task(
            self._process_cancel_payloads(), name="bootstrap.cancel_processor"
        )

        # Boot drain: pending rows that already exist at startup must
        # be picked up without waiting for the next NOTIFY. Real cases:
        # machine reboot while pending rows exist; reclaim flipped
        # running -> pending without a NOTIFY; tests that seed rows
        # before starting the worker. Pre-setting the wake event makes
        # the first iteration of the loop drain immediately; subsequent
        # iterations wait normally.
        self._wake_pending.set()
        try:
            while not self._shutdown.is_set():
                await self._wait_for_wake()
                if self._shutdown.is_set():
                    break
                # Drain pending rows in a tight loop until empty. Each
                # claim spawns a crawl task; the semaphore enforces the
                # per-machine cap.
                while not self._shutdown.is_set():
                    await self._sem.acquire()
                    try:
                        claim = await self._claim_one()
                    except Exception:
                        # Surface but don't kill the loop on a transient
                        # DB blip; release the semaphore and bail this tick.
                        log.exception("bootstrap_worker.claim_failed")
                        self._sem.release()
                        break
                    if claim is None:
                        self._sem.release()
                        break
                    task = asyncio.create_task(
                        self._run_one(claim),
                        name=f"bootstrap.run_one[{claim.run_id}]",
                    )
                    self._in_flight[claim.run_id] = task
                    task.add_done_callback(self._on_task_done)
        finally:
            wake_listener.shutdown()
            cancel_listener_task.cancel()
            cancel_processor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await wake_task
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_listener_task
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_processor_task

            # Drain any in-flight tasks before declaring shutdown
            # complete. Bounded wait so a hung crawler can't keep the
            # process alive past Fly's kill_timeout.
            if self._in_flight:
                log.info(
                    "bootstrap_worker.draining",
                    in_flight=len(self._in_flight),
                )
                tasks = list(self._in_flight.values())
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=30.0,
                    )
            log.info("bootstrap_worker.stop")

    def shutdown(self) -> None:
        self._shutdown.set()
        # Nudge the wait so the run loop notices.
        self._wake_pending.set()
        self._wake_cancel.set()

    # ------------------------------------------------------------------
    # Wait / claim
    # ------------------------------------------------------------------

    async def _wait_for_wake(self) -> None:
        """Wait on the pending wake event OR a periodic timeout.

        Cancel-NOTIFY processing happens in its own task; we don't need
        to multiplex it into the same wait. Periodic timeout is a
        safety net for a missed NOTIFY during a connection drop.
        """
        shutdown_task = asyncio.create_task(self._shutdown.wait())
        wake_task = asyncio.create_task(self._wake_pending.wait())
        try:
            _done, pending = await asyncio.wait(
                {shutdown_task, wake_task},
                timeout=self._periodic,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
        finally:
            if self._wake_pending.is_set():
                self._wake_pending.clear()

    async def _claim_one(self) -> _Claim | None:
        """Atomic claim: SELECT pending, lock with FOR UPDATE SKIP LOCKED,
        UPDATE to running in the same txn, return claim. ``None`` when
        no pending row is available.
        """
        async with raw_conn() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT customer_id, source, run_id
                FROM wiki_synthesis_runs
                WHERE kind = 'bootstrap' AND status = 'pending'
                ORDER BY started_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
            )
            if row is None:
                return None
            await conn.execute(
                "UPDATE wiki_synthesis_runs SET status = 'running' WHERE run_id = $1",
                row["run_id"],
            )
            return _Claim(
                customer_id=row["customer_id"],
                source=row["source"],
                run_id=int(row["run_id"]),
            )

    # ------------------------------------------------------------------
    # Per-claim run
    # ------------------------------------------------------------------

    async def _run_one(self, claim: _Claim) -> None:
        """Run a single (customer, source) crawl under per-pair advisory
        lock. Idempotent on the row state machine — if the row was
        cancelled mid-flight, ``_close_run`` skips writing.
        """
        lock_key = _bootstrap_run_lock_key(claim.customer_id, claim.source)
        try:
            async with raw_conn() as lock_conn, lock_conn.transaction():
                acquired = await lock_conn.fetchval(
                    "SELECT pg_try_advisory_xact_lock($1)", lock_key
                )
                if not acquired:
                    # Another worker holds the lock. Don't flip the row
                    # back; reclaim handles the orphan if the holder
                    # genuinely dies.
                    log.info(
                        "bootstrap_worker.skip_concurrent_run",
                        customer=claim.customer_id,
                        source=claim.source,
                        run_id=claim.run_id,
                        lock_key=lock_key,
                    )
                    return

                result = await self._invoke_agent(claim)
                # Apply zero-page-halt -> error synthesis (mirrors PR #126).
                if (
                    result.error is None
                    and result.halt_reason is not None
                    and (result.pages_created + result.pages_updated) == 0
                ):
                    result = result.model_copy(update={"error": f"halt:{result.halt_reason}"})
                await self._close_run(claim, result)
        except asyncio.CancelledError:
            # Hard-cancel path: trigger route fired the cancel NOTIFY
            # and we cancelled this task. Mark the row 'cancelled'
            # (idempotent — the trigger route already updated to
            # 'cancelled' before NOTIFY, this is defense-in-depth) and
            # re-raise so the asyncio.Task ends in cancelled state.
            log.info(
                "bootstrap_worker.cancelled",
                customer=claim.customer_id,
                source=claim.source,
                run_id=claim.run_id,
            )
            with contextlib.suppress(Exception):
                await self._mark_cancelled(claim, error="cancelled by force-trigger")
            raise
        except Exception as exc:
            log.exception(
                "bootstrap_worker.run_failed",
                customer=claim.customer_id,
                source=claim.source,
                run_id=claim.run_id,
            )
            with contextlib.suppress(Exception):
                await self._close_run_with_error(claim, exc)

    async def _invoke_agent(self, claim: _Claim) -> BootstrapAgentResult:
        """Resolve the factory + bearer, build the crawler, run it.

        Constructor / pre-run failures get translated into an
        ``empty_result`` with ``error`` set so ``_close_run`` can do its
        normal three-way status branch instead of needing a special path.
        """
        factory_map: dict[str, _CrawlerFactory] = (
            self._crawler_factories if self._crawler_factories is not None else REGISTRY  # type: ignore[assignment]
        )

        factory = factory_map.get(claim.source)
        if factory is None:
            err = (
                f"unknown crawler source {claim.source!r}; registered: {sorted(factory_map.keys())}"
            )
            log.warning(
                "bootstrap_worker.unknown_source",
                customer=claim.customer_id,
                source=claim.source,
                run_id=claim.run_id,
                detail=err,
            )
            return empty_result(
                source=claim.source,
                customer_id=claim.customer_id,
                run_id=claim.run_id,
                error=err,
            )

        try:
            agent: BootstrapAgent = factory(
                customer_id=claim.customer_id,
                run_id=claim.run_id,
                bearer_resolver=self._bearer_resolver_factory(claim.customer_id, claim.source),
                http=self._http,
                settings=self._settings,
            )
        except Exception as exc:
            log.warning(
                "bootstrap_worker.construct_failed",
                customer=claim.customer_id,
                source=claim.source,
                run_id=claim.run_id,
                error=str(exc),
                error_class=type(exc).__name__,
            )
            return empty_result(
                source=claim.source,
                customer_id=claim.customer_id,
                run_id=claim.run_id,
                error=f"{type(exc).__name__}: {exc}",
            )

        # Crawler exceptions land in ``_run_one``'s except-Exception so
        # CancelledError still propagates correctly through the agent
        # call chain. Don't wrap with try/except here.
        return await agent.run()

    # ------------------------------------------------------------------
    # Run-row close paths
    # ------------------------------------------------------------------

    async def _close_run(self, claim: _Claim, result: BootstrapAgentResult) -> None:
        """Three-way status branch (matches the CHECK constraint):
          - error set                 -> 'failed' (includes zero-page
                                          halts; their error field was
                                          synthesized in _run_one)
          - halted but produced pages -> 'partial' (stalled but productive)
          - clean return              -> 'complete'

        Safety net: read the row's current status first; if 'cancelled',
        do nothing — the cancel NOTIFY happened mid-flight and the
        trigger route's UPDATE already marked the row.
        """
        if result.error is not None:
            status = "failed"
        elif result.halt_reason is not None and (result.pages_created + result.pages_updated) > 0:
            status = "partial"
        else:
            status = "complete"

        async with raw_conn() as conn:
            current = await conn.fetchval(
                "SELECT status FROM wiki_synthesis_runs WHERE run_id = $1",
                claim.run_id,
            )
            if current == "cancelled":
                log.info(
                    "bootstrap_worker.close_skipped_cancelled",
                    customer=claim.customer_id,
                    source=claim.source,
                    run_id=claim.run_id,
                )
                return
            await conn.execute(
                """
                UPDATE wiki_synthesis_runs
                   SET finished_at = NOW(),
                       status = $2,
                       pages_updated = $3,
                       pages_created = $4,
                       error = $5
                 WHERE run_id = $1
                """,
                claim.run_id,
                status,
                result.pages_updated,
                result.pages_created,
                result.error,
            )
        log.info(
            "bootstrap_worker.run_closed",
            customer=claim.customer_id,
            source=claim.source,
            run_id=claim.run_id,
            status=status,
            pages_updated=result.pages_updated,
            pages_created=result.pages_created,
            error=result.error,
        )

    async def _close_run_with_error(self, claim: _Claim, exc: BaseException) -> None:
        """Catch-all close path for unexpected exceptions inside _run_one."""
        err = f"{type(exc).__name__}: {exc}"
        async with raw_conn() as conn:
            current = await conn.fetchval(
                "SELECT status FROM wiki_synthesis_runs WHERE run_id = $1",
                claim.run_id,
            )
            if current == "cancelled":
                return
            await conn.execute(
                """
                UPDATE wiki_synthesis_runs
                   SET finished_at = NOW(),
                       status = 'failed',
                       error = $2
                 WHERE run_id = $1
                """,
                claim.run_id,
                err,
            )

    async def _mark_cancelled(self, claim: _Claim, *, error: str) -> None:
        """Idempotent UPDATE to status='cancelled'. Skips if already cancelled."""
        async with raw_conn() as conn:
            await conn.execute(
                """
                UPDATE wiki_synthesis_runs
                   SET finished_at = COALESCE(finished_at, NOW()),
                       status = 'cancelled',
                       error = COALESCE(error, $2)
                 WHERE run_id = $1
                   AND status <> 'cancelled'
                """,
                claim.run_id,
                error,
            )

    # ------------------------------------------------------------------
    # Cancel channel
    # ------------------------------------------------------------------

    async def _cancel_listener_loop(self) -> None:
        """LISTEN on WIKI_BOOTSTRAP_CANCEL_CHANNEL with a dedicated conn,
        push every payload onto the cancel queue. Doesn't itself cancel
        tasks; the processor coroutine does.

        Mirrors NotifyListener's reconnect loop but exposes the payload
        (the generic NotifyListener intentionally drops payloads to keep
        its surface minimal).
        """
        backoff = 1.0
        while not self._shutdown.is_set():
            try:
                conn = await asyncpg.connect(self._dsn)
            except (asyncpg.PostgresError, OSError) as exc:
                log.warning(
                    "bootstrap_worker.cancel_listener.connect_failed",
                    error=str(exc),
                    backoff_seconds=backoff,
                )
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._shutdown.wait(), timeout=backoff)
                backoff = min(backoff * 2, 60.0)
                continue

            backoff = 1.0
            try:

                def _on_cancel_notify(_conn, _pid, _channel, payload) -> None:
                    log.info(
                        "bootstrap_worker.cancel_listener.notified",
                        channel=_channel,
                        payload=payload,
                    )
                    # asyncpg invokes this from the conn's read callback;
                    # use put_nowait + a generous queue. The queue is
                    # unbounded because we'd rather buffer than drop.
                    self._cancel_payloads.put_nowait(payload)
                    self._wake_cancel.set()

                await conn.add_listener(WIKI_BOOTSTRAP_CANCEL_CHANNEL, _on_cancel_notify)
                log.info("bootstrap_worker.cancel_listener.ready")
                while not self._shutdown.is_set():
                    try:
                        await asyncio.wait_for(self._shutdown.wait(), timeout=30.0)
                    except TimeoutError:
                        try:
                            await conn.fetchval("SELECT 1")
                        except (asyncpg.PostgresError, OSError) as exc:
                            log.warning(
                                "bootstrap_worker.cancel_listener.lost",
                                error=str(exc),
                            )
                            break
            finally:
                with contextlib.suppress(Exception):
                    await conn.close()

        log.info("bootstrap_worker.cancel_listener.stop")

    async def _process_cancel_payloads(self) -> None:
        """Pull cancel payloads off the queue; cancel matching in-flight tasks."""
        while not self._shutdown.is_set():
            try:
                payload = await self._cancel_payloads.get()
            except asyncio.CancelledError:
                raise
            if payload is None:
                continue
            await self._handle_cancel_payload(payload)

    async def _handle_cancel_payload(self, payload: str) -> None:
        try:
            data = orjson.loads(payload)
        except orjson.JSONDecodeError:
            log.warning("bootstrap_worker.cancel_payload_unparseable", payload=payload)
            return
        run_ids = data.get("run_ids") or []
        if not isinstance(run_ids, list):
            return
        for raw_id in run_ids:
            try:
                run_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            task = self._in_flight.get(run_id)
            if task is None or task.done():
                continue
            log.info(
                "bootstrap_worker.cancelling_task",
                run_id=run_id,
            )
            task.cancel()

    # ------------------------------------------------------------------
    # Done-callback bookkeeping
    # ------------------------------------------------------------------

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        # Find the run_id for this task and pop it. Linear scan is fine
        # at semaphore-bounded concurrency.
        for rid, t in list(self._in_flight.items()):
            if t is task:
                self._in_flight.pop(rid, None)
                break
        # Always release the semaphore slot we acquired in run().
        self._sem.release()


# ---------------------------------------------------------------------------
# Health server (Fly probe)
# ---------------------------------------------------------------------------


def _build_health_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from shared.db import health_check

    app = FastAPI(
        title="prbe-knowledge-wiki-bootstrap health",
        docs_url=None,
        redoc_url=None,
    )

    @app.get("/health")
    async def health() -> JSONResponse:
        db_ok = await health_check()
        body = {
            "status": "ok" if db_ok else "degraded",
            "db": db_ok,
            "time": datetime.now(UTC).isoformat(),
        }
        return JSONResponse(body, status_code=200 if db_ok else 503)

    return app


@contextlib.asynccontextmanager
async def build_http_client() -> AsyncIterator[httpx.AsyncClient]:
    """Shared httpx client used by every crawler.

    Generous timeout because individual crawlers may pull large pages
    of data; per-call retries / rate-limit handling live inside each
    api_client wrapper.
    """
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=5.0)
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        yield client


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run_bootstrap_app_forever() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    # Import handlers package so any decorators run (mirror the other
    # wiki apps; bootstrap reuses the same Normalizer write path inside
    # the wiki tools).
    import services.ingestion.handlers  # noqa: F401

    # LISTEN must run on a direct (non-pooler) DSN. Neon's pgbouncer
    # transaction-pooler resets `LISTEN *` between txns, so a listener
    # holding a pooled conn never receives any NOTIFY.
    listener_dsn = settings.database_url_unpooled or settings.database_url

    health_port = int(os.environ.get("WORKER_HEALTH_PORT", "8082"))
    # IPv4 bind for Fly's `[[http_service.checks]]` IPv4 health probe.
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
        "bootstrap_app.boot",
        environment=settings.environment,
        health_port=health_port,
        timestamp=datetime.now(UTC).isoformat(),
    )

    async with build_http_client() as http:
        worker = BootstrapWorker(
            dsn=listener_dsn,
            http=http,
            settings=settings,
            bearer_resolver_factory=_make_default_bearer_factory(http),
        )
        reclaim_loop = BootstrapReclaimLoop()

        loop = asyncio.get_running_loop()
        gather_future: asyncio.Future | None = None  # type: ignore[type-arg]
        shutdown_started = False

        def handle_signal(signame: str) -> None:
            nonlocal shutdown_started
            if shutdown_started:
                return
            shutdown_started = True
            log.info("bootstrap_app.shutdown_signal", signal=signame)
            worker.shutdown()
            reclaim_loop.shutdown()
            health_server.should_exit = True
            if gather_future is not None and not gather_future.done():
                gather_future.cancel()

        for signame in ("SIGTERM", "SIGINT"):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(getattr(signal, signame), handle_signal, signame)

        gather_future = asyncio.gather(
            worker.run(),
            health_server.serve(),
            reclaim_loop.run(),
        )
        try:
            await gather_future
        except asyncio.CancelledError:
            log.info("bootstrap_app.shutdown_complete")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_bootstrap_app_forever())
