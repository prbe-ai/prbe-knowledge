"""SynthesisWorker — drain triaged rows through the wiki agent loop.

Runs in the prbe-knowledge-wiki-synthesis fly app (2 x 3GB).

Per tick:
  1. SELECT customers with triaged rows AND wiki_generation_enabled.
  2. Fan out per-customer drains via asyncio.gather + global concurrency
     cap (WIKI_AGENT_GLOBAL_CONCURRENCY).
  3. Per customer:
     a. Acquire pg_try_advisory_xact_lock(customer_drain_lock_key) so
        cron + manual button can't double-drain. Lock contended ->
        log 'drain.skip_concurrent', exit.
     b. Open a wiki_synthesis_runs row (stage='synthesis').
     c. Claim triaged rows for this drain (status -> 'synthesizing'),
        ordered by source_ts ASC.
     d. Spawn AgentLoop(WikiAgentRuntime). The runtime owns state +
        tool dispatch; the harness owns turn loop + Gemini calls.
     e. On done() -> commit happens inside the runtime's tool_done
        handler (atomic across all staged pages). Worker just records
        metrics on the run row.
     f. On AgentHaltError -> runtime.discard() + DLQ all 'synthesizing'
        rows with the categorized halt_reason.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from datetime import UTC, datetime

from services.ingestion.handlers.base import ConnectorContext, make_default_context
from services.synthesis import persistence
from services.synthesis.agent_harness import AgentLoop, new_agent_run_id
from services.synthesis.agent_tools import ALL_TOOLS
from services.synthesis.prompts import wiki_agent_system_prompt
from services.synthesis.wiki_agent import WikiAgentRuntime
from shared.constants import (
    WIKI_AGENT_GLOBAL_CONCURRENCY,
    WIKI_AGENT_MODEL,
    WIKI_SYNTHESIS_CLAIM_BATCH,
    WIKI_SYNTHESIS_PERIODIC_WAKE_SECONDS,
)
from shared.db import raw_conn
from shared.embeddings import GeminiEmbedder
from shared.exceptions import AgentHaltError
from shared.logging import get_logger
from shared.storage import ObjectStore, get_store

log = get_logger(__name__)


class SynthesisWorker:
    """Drain triaged rows through the wiki agent loop.

    One instance per machine (2 machines x 3GB on the wiki-synthesis
    fly app). Per-customer advisory lock serializes drains so cron +
    manual button + reclaim wakes don't double-drain a customer.
    """

    def __init__(
        self,
        wake_event: asyncio.Event,
        *,
        ctx: ConnectorContext | None = None,
        store: ObjectStore | None = None,
        embedder: GeminiEmbedder | None = None,
        llm_client: object | None = None,
        periodic_wake_seconds: float = WIKI_SYNTHESIS_PERIODIC_WAKE_SECONDS,
        global_concurrency: int = WIKI_AGENT_GLOBAL_CONCURRENCY,
    ) -> None:
        self._wake = wake_event
        self._ctx = ctx or make_default_context()
        self._store = store or get_store()
        self._embedder = embedder
        self._llm_client = llm_client
        self._periodic = periodic_wake_seconds
        self._global_sem = asyncio.Semaphore(global_concurrency)
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        log.info("synthesis_worker.start")
        while not self._shutdown.is_set():
            woken_by_notify = await self._wait()
            try:
                await self._tick(woken_by_notify=woken_by_notify)
            except Exception:
                log.exception("synthesis_worker.tick_failed")
        log.info("synthesis_worker.stop")

    def shutdown(self) -> None:
        self._shutdown.set()

    async def _wait(self) -> bool:
        shutdown_task = asyncio.create_task(self._shutdown.wait())
        wake_task = asyncio.create_task(self._wake.wait())
        try:
            done, pending = await asyncio.wait(
                {shutdown_task, wake_task},
                timeout=self._periodic,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            return wake_task in done
        finally:
            if self._wake.is_set():
                self._wake.clear()

    async def _tick(self, *, woken_by_notify: bool) -> None:
        if self._shutdown.is_set():
            return
        async with raw_conn() as conn:
            customer_ids = await persistence.list_triaged_customers(conn)
        if not customer_ids:
            return
        kind = "wake" if woken_by_notify else "scheduled"

        async def _drain(cid: str) -> None:
            async with self._global_sem:
                try:
                    await self._drain_customer(cid, run_kind=kind)
                except Exception:
                    log.exception(
                        "synthesis_worker.drain_failed", customer=cid
                    )

        await asyncio.gather(*[_drain(cid) for cid in customer_ids])

    # -----------------------------------------------------------------------
    # Per-customer drain
    # -----------------------------------------------------------------------

    async def _drain_customer(self, customer_id: str, *, run_kind: str) -> None:
        # Per-customer advisory lock: cron + manual button can't
        # double-drain. The lock key is derived from the customer_id so
        # different customers never contend.
        lock_key = self._lock_key(customer_id)
        async with raw_conn() as lock_conn, lock_conn.transaction():
            acquired = await lock_conn.fetchval(
                "SELECT pg_try_advisory_xact_lock($1)", lock_key
            )
            if not acquired:
                log.info(
                    "synthesis_worker.drain_skip_concurrent",
                    customer=customer_id,
                    lock_key=lock_key,
                )
                return
            # Hold the lock for the duration of the drain. Open a
            # txn so the lock auto-releases on commit/rollback.
            run_id = await self._open_run(customer_id, run_kind)
            agent_run_id = new_agent_run_id()
            log.info(
                "synthesis_worker.run_open",
                customer=customer_id,
                run_id=run_id,
                agent_run_id=agent_run_id,
                kind=run_kind,
            )

            # Claim triaged rows in this txn so the lock holds
            # while the agent runs. claim_triaged_rows itself uses
            # FOR UPDATE SKIP LOCKED on a different conn, so this
            # is fine; the advisory lock is purely the
            # serialization gate.
            claimed = await persistence.claim_triaged_rows(
                customer_id, limit=WIKI_SYNTHESIS_CLAIM_BATCH
            )
            if not claimed:
                await self._close_run(
                    run_id, customer_id=customer_id, status="complete"
                )
                return

            pages_updated = pages_created = 0
            halt_reason: str | None = None
            try:
                metrics = await self._run_agent(
                    customer_id=customer_id,
                    agent_run_id=agent_run_id,
                    run_id=run_id,
                    run_kind=run_kind,
                )
                pages_updated = 0  # filled by metrics if exposed
                pages_created = 0
                log.info(
                    "synthesis_worker.run_complete",
                    customer=customer_id,
                    agent_run_id=agent_run_id,
                    turns=metrics.turns,
                    gemini_calls=metrics.gemini_call_count,
                    cache_hit_rate=metrics.cache_hit_rate,
                )
            except AgentHaltError as exc:
                halt_reason = exc.reason
                dlq_count = await persistence.dlq_agent_synthesizing_rows(
                    customer_id, reason=exc.reason
                )
                log.warning(
                    "synthesis_worker.agent_halt",
                    customer=customer_id,
                    agent_run_id=agent_run_id,
                    reason=exc.reason,
                    dlq_count=dlq_count,
                )
            except Exception as exc:
                halt_reason = f"agent.exception: {type(exc).__name__}"
                dlq_count = await persistence.dlq_agent_synthesizing_rows(
                    customer_id, reason=halt_reason
                )
                log.exception(
                    "synthesis_worker.agent_unhandled",
                    customer=customer_id,
                    agent_run_id=agent_run_id,
                    dlq_count=dlq_count,
                )

            status = "failed" if halt_reason else "complete"
            await self._close_run(
                run_id,
                customer_id=customer_id,
                status=status,
                pages_updated=pages_updated,
                pages_created=pages_created,
                error=halt_reason,
            )

    async def _run_agent(
        self,
        *,
        customer_id: str,
        agent_run_id: str,
        run_id: int,
        run_kind: str,
    ) -> object:
        """Build the runtime + harness, then drive the loop to completion."""
        runtime = WikiAgentRuntime(
            customer_id,
            agent_run_id=agent_run_id,
            run_id=run_id,
            run_kind=run_kind,
            ctx=self._ctx,
            store=self._store,
            embedder=self._embedder,
        )
        llm = self._resolve_llm_client()
        from services.synthesis.agent_compactor import call_summarizer

        loop = AgentLoop(
            runtime=runtime,
            llm=llm,
            system_prompt=wiki_agent_system_prompt(datetime.now(UTC)),
            tool_schemas=ALL_TOOLS,
            summarizer=call_summarizer,
            model=WIKI_AGENT_MODEL,
        )
        try:
            return await loop.run()
        except AgentHaltError:
            await runtime.discard()
            raise

    def _resolve_llm_client(self) -> object:
        if self._llm_client is not None:
            return self._llm_client
        # Lazy build the production Gemini wrapper. Imported here to
        # keep the test path (which always passes llm_client) free of
        # the SDK requirement.
        from services.synthesis.gemini_agent_client import GeminiAgentClient

        return GeminiAgentClient()

    # -----------------------------------------------------------------------
    # Run row helpers
    # -----------------------------------------------------------------------

    async def _open_run(self, customer_id: str, run_kind: str) -> int:
        return await persistence.open_run(
            customer_id, kind=run_kind, stage="synthesis"
        )

    async def _close_run(
        self,
        run_id: int,
        *,
        customer_id: str,
        status: str,
        pages_updated: int = 0,
        pages_created: int = 0,
        error: str | None = None,
    ) -> None:
        await persistence.close_run(
            run_id,
            customer_id=customer_id,
            status=status,
            events_total=0,
            events_triaged=0,
            events_kept=0,
            pages_updated=pages_updated,
            pages_created=pages_created,
            error=error,
        )

    # -----------------------------------------------------------------------
    # Advisory lock key derivation
    # -----------------------------------------------------------------------

    def _lock_key(self, customer_id: str) -> int:
        """Hash customer_id into a 64-bit signed int for advisory lock.

        Using a stable hash so the same customer always lands on the
        same lock key across machines / processes / app restarts.
        Mirrors PR #87's BFF rate-limit lock convention.
        """
        digest = hashlib.sha256(customer_id.encode("utf-8")).digest()
        # Take low 8 bytes, interpret as signed 64-bit. Postgres
        # advisory lock keys are bigint = signed 64-bit.
        return int.from_bytes(digest[:8], "big", signed=True)


__all__ = ["SynthesisWorker"]
