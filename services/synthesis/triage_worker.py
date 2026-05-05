"""TriageWorker — drain wiki_synthesis_queue rows from pending → triaged.

Runs in the prbe-knowledge-wiki-worker fly app (4 x 1GB). One process
per machine; each process runs:

  - NotifyListener on `wiki_synthesize_pending` (sets wake_event)
  - TriageWorker.run (waits on wake_event + periodic timer; drains)
  - tiny health endpoint

Per tick:
  1. SELECT customers with pending rows AND wiki_generation_enabled.
  2. Fan out per-customer drains via asyncio.gather + semaphore (cap
     `WIKI_SYNTHESIS_CUSTOMER_CONCURRENCY`).
  3. Per customer:
     a. Open a wiki_synthesis_runs row.
     b. Loop: claim batch → fetch full doc bodies → token-batch →
        fan-out triage calls (cap `WIKI_TRIAGE_BATCH_CONCURRENCY`) →
        per-row mark rejected/triaged → bulk UPDATE+NOTIFY for kept
        rows (atomic — Postgres delivers NOTIFY only on COMMIT).
     c. Close the run row.

Triage reads FULL document bodies via `persistence.fetch_bodies` (joins
chunks). Synthesis is in a separate fly app and wakes on NOTIFY.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime

import asyncpg
from anthropic import AsyncAnthropic

from services.synthesis import persistence
from services.synthesis.models import TriageInput, TriageVerdict
from services.synthesis.triage import (
    call_triage,
    pack_into_batches,
)
from shared.config import get_settings
from shared.constants import (
    WIKI_SYNTHESIS_CLAIM_BATCH,
    WIKI_SYNTHESIS_CUSTOMER_CONCURRENCY,
    WIKI_SYNTHESIS_PERIODIC_WAKE_SECONDS,
    WIKI_TRIAGE_BATCH_CONCURRENCY,
    WIKI_TRIAGE_SCORE_THRESHOLD,
    WIKI_TRIAGED_CHANNEL,
)
from shared.db import raw_conn
from shared.logging import get_logger

log = get_logger(__name__)


class TriageWorker:
    """Drain pending wiki_synthesis_queue rows through triage.

    Single-instance per machine. Multiple machines safely co-drain via
    `FOR UPDATE SKIP LOCKED` in `claim_pending_batch`.
    """

    def __init__(
        self,
        wake_event: asyncio.Event,
        *,
        anthropic_client: AsyncAnthropic | None = None,
        periodic_wake_seconds: float = WIKI_SYNTHESIS_PERIODIC_WAKE_SECONDS,
        notify_channel: str = WIKI_TRIAGED_CHANNEL,
        customer_concurrency: int = WIKI_SYNTHESIS_CUSTOMER_CONCURRENCY,
        batch_concurrency: int = WIKI_TRIAGE_BATCH_CONCURRENCY,
    ) -> None:
        self._wake = wake_event
        self._anthropic_client = anthropic_client
        self._periodic = periodic_wake_seconds
        self._notify_channel = notify_channel
        self._customer_sem = asyncio.Semaphore(customer_concurrency)
        self._batch_sem = asyncio.Semaphore(batch_concurrency)
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        log.info("triage_worker.start")
        while not self._shutdown.is_set():
            woken_by_notify = await self._wait()
            try:
                await self._tick(woken_by_notify=woken_by_notify)
            except Exception:
                log.exception("triage_worker.tick_failed")
        log.info("triage_worker.stop")

    def shutdown(self) -> None:
        self._shutdown.set()

    async def _wait(self) -> bool:
        """Return True if the tick was woken by NOTIFY, False on periodic timer."""
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
            customer_ids = await persistence.list_pending_customers(conn)
        if not customer_ids:
            return
        client = self._resolve_client()
        if client is None:
            log.warning(
                "triage_worker.no_anthropic_key",
                pending_customers=len(customer_ids),
            )
            return
        kind = "wake" if woken_by_notify else "scheduled"

        async def _drain(cid: str) -> None:
            async with self._customer_sem:
                try:
                    await self._drain_customer(cid, client, run_kind=kind)
                except Exception:
                    log.exception("triage_worker.drain_failed", customer=cid)

        await asyncio.gather(*[_drain(cid) for cid in customer_ids])

    def _resolve_client(self) -> AsyncAnthropic | None:
        if self._anthropic_client is not None:
            return self._anthropic_client
        settings = get_settings()
        secret = settings.anthropic_api_key
        if secret is None:
            return None
        key = secret.get_secret_value()
        if not key:
            return None
        return AsyncAnthropic(api_key=key)

    async def _drain_customer(
        self,
        customer_id: str,
        client: AsyncAnthropic,
        *,
        run_kind: str,
    ) -> None:
        run_id = await persistence.open_run(customer_id, kind=run_kind, stage="triage")
        log.info(
            "triage_worker.run_open",
            customer=customer_id,
            run_id=run_id,
            kind=run_kind,
        )
        events_total = events_triaged = events_kept = 0
        run_status = "complete"
        run_error: str | None = None
        try:
            while not self._shutdown.is_set():
                queue_rows = await persistence.claim_pending_batch(
                    customer_id, limit=WIKI_SYNTHESIS_CLAIM_BATCH
                )
                if not queue_rows:
                    break
                events_total += len(queue_rows)

                triage_inputs = await persistence.fetch_bodies(customer_id, queue_rows)
                batches = pack_into_batches(triage_inputs)
                verdicts = await self._call_triage_batches(client, batches, customer_id)
                events_triaged += len(verdicts)

                # v4: every batch failed -> unrecoverable for this drain.
                # DLQ all pending + triaging rows for the customer with a
                # categorized reason so the dashboard surface and the
                # admin reset endpoint can handle it. Distinguishing
                # anthropic vs gemini vs timeout is best-effort; the
                # broad reason "triage.batch_failure" is the safe fallback.
                if batches and not verdicts:
                    reason = self._classify_triage_failure(customer_id, batches)
                    dlq_count = await persistence.dlq_customer_for_triage_failure(
                        customer_id, reason=reason
                    )
                    log.warning(
                        "triage_worker.dlq_customer",
                        customer=customer_id,
                        reason=reason,
                        dlq_count=dlq_count,
                    )
                    run_status = "failed"
                    run_error = reason
                    return

                events_kept += await self._apply_verdicts(
                    customer_id,
                    queue_rows,
                    verdicts,
                )
        except Exception as exc:
            # Unrecoverable customer-level crash (DB error, RLS issue,
            # SDK setup failure). DLQ the whole pending+triaging slice
            # and re-raise; the outer `_tick` catches it.
            run_status = "failed"
            run_error = str(exc)
            try:
                dlq_count = await persistence.dlq_customer_for_triage_failure(
                    customer_id, reason=f"triage.crash: {type(exc).__name__}"
                )
                log.warning(
                    "triage_worker.dlq_customer_on_crash",
                    customer=customer_id,
                    error=str(exc),
                    error_class=type(exc).__name__,
                    dlq_count=dlq_count,
                )
            except Exception:
                # Best-effort DLQ. If the DB itself is gone, at least
                # the run row will record the failure once we re-raise
                # past the finally block.
                log.exception("triage_worker.dlq_failed", customer=customer_id)
            raise
        finally:
            await persistence.close_run(
                run_id,
                customer_id=customer_id,
                status=run_status,
                events_total=events_total,
                events_triaged=events_triaged,
                events_kept=events_kept,
                pages_updated=0,
                pages_created=0,
                error=run_error,
            )
            log.info(
                "triage_worker.run_close",
                customer=customer_id,
                run_id=run_id,
                status=run_status,
                events_total=events_total,
                events_triaged=events_triaged,
                events_kept=events_kept,
            )

    def _classify_triage_failure(
        self,
        customer_id: str,
        batches: list[list[TriageInput]],
    ) -> str:
        """Best-effort classifier for the dlq_reason tag.

        The provider-specific exception type is lost by the time we
        reach this hook (per-batch errors are swallowed inside
        _call_triage_batches). We log per batch; the reason string here
        is the single tag the dashboard shows.
        """
        from shared.constants import WIKI_TRIAGE_MODEL

        model = (WIKI_TRIAGE_MODEL or "").lower()
        if model.startswith("gemini"):
            return "triage.gemini"
        if "haiku" in model or "claude" in model:
            return "triage.anthropic"
        return "triage.batch_failure"

    async def _call_triage_batches(
        self,
        client: AsyncAnthropic,
        batches: list[list[TriageInput]],
        customer_id: str,
    ) -> dict[int, TriageVerdict]:
        """Fire triage calls in parallel within `batch_concurrency`.

        Per-batch errors are isolated: a failed batch marks its rows
        for retry but does not poison sibling batches.
        """
        if not batches:
            return {}
        now = datetime.now(UTC)

        async def _one(batch: list[TriageInput]) -> dict[int, TriageVerdict]:
            async with self._batch_sem:
                try:
                    output = await call_triage(client, batch, now=now)
                except Exception as exc:
                    log.warning(
                        "triage_worker.triage_failed",
                        customer=customer_id,
                        batch_size=len(batch),
                        error=str(exc),
                    )
                    await persistence.mark_batch_triage_error(customer_id, batch, str(exc))
                    return {}
                out: dict[int, TriageVerdict] = {}
                for qid_str, verdict in output.verdicts.items():
                    try:
                        out[int(qid_str)] = verdict
                    except ValueError:
                        log.warning(
                            "triage_worker.verdict_bad_qid",
                            qid=qid_str,
                        )
                return out

        results = await asyncio.gather(*[_one(b) for b in batches])
        merged: dict[int, TriageVerdict] = {}
        for r in results:
            merged.update(r)
        return merged

    async def _apply_verdicts(
        self,
        customer_id: str,
        queue_rows: list[asyncpg.Record],
        verdicts: dict[int, TriageVerdict],
    ) -> int:
        """Apply verdicts: rejected/retry one-shot, triaged batched + notified.

        Returns the number of rows marked triaged. Triaged rows are
        UPDATE'd in a single SQL statement that also fires the NOTIFY,
        all in one transaction — so the synthesis worker's listener
        cannot wake on rows that haven't committed yet.

        v4: triage produces score-only verdicts. Below-threshold rows
        are 'rejected'; at-or-above rows go to 'triaged' for the wiki
        agent. The agent picks pages downstream.
        """
        triaged_verdicts: list[tuple[int, TriageVerdict]] = []
        for row in queue_rows:
            qid = row["queue_id"]
            verdict = verdicts.get(qid)
            if verdict is None:
                await persistence.mark_for_retry(customer_id, qid)
                continue
            if (
                not verdict.important
                or verdict.score < WIKI_TRIAGE_SCORE_THRESHOLD
            ):
                await persistence.mark_rejected(customer_id, qid, verdict)
                continue
            triaged_verdicts.append((qid, verdict))

        if triaged_verdicts:
            await persistence.mark_batch_triaged_and_notify(
                customer_id,
                triaged_verdicts,
                notify_channel=self._notify_channel,
            )
        return len(triaged_verdicts)


__all__ = ["TriageWorker"]
