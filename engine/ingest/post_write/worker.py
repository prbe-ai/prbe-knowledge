"""Drain loop for `node_post_write_queue`.

Lifecycle of a queued row::

  graph_writer.upsert_nodes commits
    -> enqueue_post_write_node(customer_id, node_id) [post-commit hook]
       INSERT ... ON CONFLICT DO UPDATE SET analyzer_status='{}'

  PostWriteWorker._claim_loop polls:
    SELECT (customer_id, node_id) FROM node_post_write_queue
    WHERE (locked_until IS NULL OR locked_until < NOW())
      AND COALESCE((analyzer_status->'auto_merge'->>'attempts')::int, 0) < 3
    ORDER BY enqueued_at FOR UPDATE SKIP LOCKED LIMIT 1
    UPDATE SET locked_until = NOW() + INTERVAL '5 minutes'

  Process (two transactions -- see _process):
    1. Embed the node text via GeminiEmbedder if graph_nodes.embedding IS NULL,
       and drain pending edges waiting on this node   [tx 1, committed]
    2. Run AutoMergeAnalyzer.analyze()                   [tx 2]

  On success: DELETE FROM node_post_write_queue WHERE (customer_id, node_id) = (...)
  On failure: clear locked_until, bump attempts in analyzer_status JSONB; if
              attempts hit 3 the WHERE clause stops picking it back up.
  On deferral (the judge was unreachable, action="deferred"): KEEP the row and
              push locked_until into the future (the claim query reclaims it
              once that passes), doubling per consecutive deferral. An outage
              does NOT spend the 3 attempts; after AUTO_MERGE_MAX_DEFERRALS in a
              row the row is parked as failed, visible with its last error.

Concurrency: 16 tasks per process (POST_WRITE_CONCURRENCY env var).
Runs alongside InferredEdgesWorker — both pull from independent queues
inside the same Fly process; see `inferred_edges/worker.py:run_worker_forever`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import time
import uuid
from datetime import datetime

import asyncpg

from engine.ingest.auto_merge import AutoMergeAnalyzer
from engine.ingest.auto_merge.analyzer import AutoMergeResult
from engine.ingest.graph_writer import drain_pending_edges, reap_expired_pending_edges
from engine.ingest.normalizer import _pg_vector
from engine.shared.constants import (
    AUTO_MERGE_MAX_DEFERRALS,
    AUTO_MERGE_RETRY_MAX_SECONDS,
    AUTO_MERGE_RETRY_SECONDS,
)
from engine.shared.db import raw_conn, with_tenant
from engine.shared.embeddings import get_embedder_v2
from engine.shared.logging import get_logger
from engine.shared.metrics import counter
from engine.shared.tenant_status import active_tenant_sql
from scripts.backfill_graph_node_embeddings import _embedding_text

log = get_logger(__name__)

_MAX_ATTEMPTS = 3
_DEFAULT_CONCURRENCY = int(os.getenv("POST_WRITE_CONCURRENCY", "16"))
_POLL_INTERVAL_SECONDS = 2.0
_LOCK_DURATION = "5 minutes"

# The claim is two index-friendly legs instead of one `locked_until IS NULL OR
# locked_until < NOW()` scan: that OR cannot use the partial index
# idx_node_post_write_queue_pending (enqueued_at WHERE locked_until IS NULL),
# so every claim sorted the whole queue -- nothing on managed, where the queue
# is near empty, but ~2 cores of Postgres at 2 rows/s against research's 220k
# backlog (2026-09-23).
#
# Both legs claim only an ACTIVE tenant's rows (shared.tenant_status): a held
# tenant's rows stay queued, unlocked, until its purge cascades them.
_CLAIM_FRESH = f"""
    SELECT customer_id, node_id, analyzer_status, enqueued_at
    FROM node_post_write_queue
    WHERE locked_until IS NULL
      AND COALESCE((analyzer_status->'auto_merge'->>'attempts')::int, 0) < $1
      AND {active_tenant_sql("node_post_write_queue.customer_id")}
    ORDER BY enqueued_at ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1
"""
# A lease that expired (a worker died mid-process) or a deferral whose delay
# has passed. No index serves it, so it runs when there is no fresh row, and
# first on every _DUE_EVERY-th claim so a long backlog cannot starve retries.
_CLAIM_DUE = f"""
    SELECT customer_id, node_id, analyzer_status, enqueued_at
    FROM node_post_write_queue
    WHERE locked_until < NOW()
      AND COALESCE((analyzer_status->'auto_merge'->>'attempts')::int, 0) < $1
      AND {active_tenant_sql("node_post_write_queue.customer_id")}
    ORDER BY enqueued_at ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1
"""
_DUE_EVERY = 20
# The drain path reaps a tenant's expired pending edges at most this often
# (per process): the sweep is tenant-wide, so once per row was pure repetition.
_REAP_INTERVAL_SECONDS = 600.0
_last_reap: dict[str, float] = {}


class PostWriteWorker:
    """Drain loop for node_post_write_queue."""

    def __init__(
        self,
        *,
        concurrency: int = _DEFAULT_CONCURRENCY,
        worker_id: str | None = None,
        execute_high_confidence: bool = False,
        auto_merge_enabled: bool = True,
        embeddings_enabled: bool = True,
    ) -> None:
        self._concurrency = max(1, concurrency)
        # Both default on. A plane that only needs parked edges linked (the
        # research plane) turns them off: node embeddings are read only by
        # auto-merge's candidate search.
        self._auto_merge_enabled = auto_merge_enabled
        self._embeddings_enabled = embeddings_enabled
        self._worker_id = worker_id or (
            f"post-write-{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        )
        self._shutdown = asyncio.Event()
        self._claims = 0
        self._analyzer = AutoMergeAnalyzer(execute_high_confidence=execute_high_confidence)

    async def run(self) -> None:
        log.info(
            "post_write_worker.start",
            worker_id=self._worker_id,
            concurrency=self._concurrency,
            execute=self._analyzer._execute,
            auto_merge=self._auto_merge_enabled,
            embeddings=self._embeddings_enabled,
        )
        await asyncio.gather(
            *(self._claim_loop() for _ in range(self._concurrency))
        )
        log.info("post_write_worker.stop")

    def shutdown(self) -> None:
        self._shutdown.set()

    async def _claim_loop(self) -> None:
        while not self._shutdown.is_set():
            claimed = await self._claim_one()
            if claimed is None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._shutdown.wait(), timeout=_POLL_INTERVAL_SECONDS
                    )
                continue
            await self._process(claimed)

    async def _claim_one(self) -> asyncpg.Record | None:
        """Claim one pending row via FOR UPDATE SKIP LOCKED.

        Returns a row with (customer_id, node_id, analyzer_status, enqueued_at).
        Atomically flips locked_until to NOW() + 5min so concurrent workers
        skip it.

        Also reclaims rows whose previous lock has expired (locked_until in
        the past) — those got stuck because a worker pod died mid-process,
        couldn't run the success-DELETE or failure-clear, and would
        otherwise never get picked back up.
        """
        self._claims += 1
        legs = (_CLAIM_DUE, _CLAIM_FRESH) if self._claims % _DUE_EVERY == 0 else (_CLAIM_FRESH, _CLAIM_DUE)
        async with raw_conn() as conn, conn.transaction():
            row = None
            for sql in legs:
                row = await conn.fetchrow(sql, _MAX_ATTEMPTS)
                if row is not None:
                    break
            if row is None:
                return None
            await conn.execute(
                f"""
                UPDATE node_post_write_queue
                SET locked_until = NOW() + INTERVAL '{_LOCK_DURATION}'
                WHERE customer_id = $1 AND node_id = $2
                """,
                row["customer_id"],
                row["node_id"],
            )
            return row

    async def _process(self, row: asyncpg.Record) -> None:
        customer_id: str = row["customer_id"]
        node_id: int = row["node_id"]
        # Every write back to the row is conditional on this: an upsert of the
        # node mid-processing re-enqueues it (new enqueued_at, lock cleared),
        # and that fresh row must be processed again, not deleted or stamped
        # with this pass's stale counters.
        claimed_at = row["enqueued_at"]
        status_json = row["analyzer_status"]
        status = json.loads(status_json) if isinstance(status_json, str) else (status_json or {})

        log.info(
            "post_write_worker.processing",
            customer=customer_id,
            node_id=node_id,
        )

        try:
            # Two transactions, on purpose. Writing the embedding locks this
            # node's row, and a high-confidence verdict runs merge_cluster on
            # its OWN connection, which must lock the same row: sharing one
            # transaction made the merge wait on its caller until the statement
            # timed out (5 min), so a brand-new node's first merge always failed.
            async with with_tenant(customer_id) as conn:
                if self._embeddings_enabled:
                    await self._ensure_embedding(conn, node_id)
                await self._drain_pending_edges(conn, customer_id, node_id)
            if self._auto_merge_enabled:
                async with with_tenant(customer_id) as conn:
                    result = await self._analyzer.analyze(conn, customer_id, node_id)
            else:
                result = AutoMergeResult(action="skipped", rationale="auto-merge disabled")

            counter(
                "post_write.processed",
                1,
                customer_id=customer_id,
                action=result.action,
            )
            log.info(
                "post_write_worker.done",
                customer=customer_id,
                node_id=node_id,
                action=result.action,
                primary=result.primary_canonical_id,
                confidence=result.confidence,
                judge_model=result.judge_model,
                p=result.p,
            )
            if result.action == "deferred":
                await self._defer(customer_id, node_id, claimed_at, status, result)
                return
            await self._delete_queue_row(customer_id, node_id, claimed_at)

        except Exception as exc:
            attempts = _auto_merge_status(status).get("attempts", 0) + 1
            log.exception(
                "post_write_worker.process_failed",
                customer=customer_id,
                node_id=node_id,
                attempts=attempts,
                error=str(exc),
            )
            await self._record_failure(customer_id, node_id, claimed_at, attempts, repr(exc))

    async def _drain_pending_edges(
        self, conn: asyncpg.Connection, customer_id: str, node_id: int
    ) -> None:
        """Materialise edges that were parked waiting on this node.

        A node has just been written; any pending_edges row keyed on this
        node's (label, canonical_id) can now resolve. Best-effort: a drain
        failure must not fail the node's post-write processing, since the
        reaper and the next touch of this node both retry.
        """
        row = await conn.fetchrow(
            "SELECT label, canonical_id FROM graph_nodes WHERE node_id = $1",
            node_id,
        )
        if row is None:
            return
        try:
            # A SAVEPOINT, so a failed drain rolls back only itself. Without it
            # the error aborts the whole transaction, which then rolls back the
            # embedding written just before -- silently, since this is caught.
            async with conn.transaction():
                await drain_pending_edges(
                    conn, customer_id, row["label"], row["canonical_id"]
                )
                # Opportunistic TTL sweep for this tenant -- no separate cron --
                # at most once per _REAP_INTERVAL_SECONDS per process.
                now = time.monotonic()
                if now - _last_reap.get(customer_id, float("-inf")) >= _REAP_INTERVAL_SECONDS:
                    await reap_expired_pending_edges(conn, customer_id)
                    _last_reap[customer_id] = now
        except Exception as exc:
            log.warning(
                "post_write_worker.drain_pending_edges_failed",
                customer=customer_id,
                node_id=node_id,
                error=str(exc),
            )

    async def _ensure_embedding(
        self, conn: asyncpg.Connection, node_id: int
    ) -> None:
        """Compute + write graph_nodes.embedding if currently NULL.

        Idempotent: skips if embedding is already populated. Uses the same
        GeminiEmbedder + `_embedding_text` shape as the backfill script so
        new and existing nodes converge on identical embedding semantics.
        """
        row = await conn.fetchrow(
            """
            SELECT label, canonical_id, properties,
                   (embedding IS NULL) AS needs_embedding
            FROM graph_nodes
            WHERE node_id = $1
            """,
            node_id,
        )
        if row is None or not row["needs_embedding"]:
            return

        properties = row["properties"]
        if isinstance(properties, str):
            properties = json.loads(properties or "{}")
        text = _embedding_text(row["label"], row["canonical_id"], properties or {})

        embedder = get_embedder_v2()
        result = await embedder.embed_many([text])
        if not result.embedded:
            log.warning(
                "post_write_worker.embedding_failed",
                node_id=node_id,
                failed_count=len(result.failed),
            )
            return

        vector_str = _pg_vector(result.embedded[0].embedding)
        await conn.execute(
            "UPDATE graph_nodes SET embedding = $1::halfvec(3072) WHERE node_id = $2",
            vector_str,
            node_id,
        )

    async def _delete_queue_row(self, customer_id: str, node_id: int, claimed_at: datetime) -> None:
        async with raw_conn() as conn:
            await conn.execute(
                "DELETE FROM node_post_write_queue WHERE customer_id = $1 AND node_id = $2 AND enqueued_at = $3",
                customer_id,
                node_id,
                claimed_at,
            )

    async def _defer(
        self,
        customer_id: str,
        node_id: int,
        claimed_at: datetime,
        status: dict,
        result,
    ) -> None:
        """Keep the row; make it claimable again only after a backoff.

        The claim query reclaims rows whose `locked_until` has passed, so a
        future lock IS the retry schedule. Deferrals double from the
        analyzer's base delay up to AUTO_MERGE_RETRY_MAX_SECONDS and do not
        touch `attempts` -- an outage is not this node's fault. After
        AUTO_MERGE_MAX_DEFERRALS failed CALLS the row is parked (attempts set
        to the cap, status "failed") so a judge that keeps failing cannot keep
        a node cycling forever. A deferral with the breaker open sent nothing
        and does not count: a long outage backs the queue off to the hourly
        ceiling instead of parking all of it. A new upsert resets all of this.
        """
        prev = _auto_merge_status(status)
        deferrals = prev.get("deferrals", 0) + 1
        failed_calls = prev.get("failed_calls", 0) + (1 if result.judge_called else 0)
        error = (result.error or "judge unavailable")[:240]
        if failed_calls > AUTO_MERGE_MAX_DEFERRALS:
            log.warning(
                "post_write_worker.deferral_cap_reached",
                customer=customer_id,
                node_id=node_id,
                failed_calls=failed_calls - 1,
                error=error,
            )
            await self._write_status(
                customer_id,
                node_id,
                claimed_at,
                {"status": "failed", "attempts": _MAX_ATTEMPTS, "deferrals": deferrals,
                 "failed_calls": failed_calls - 1, "last_error": error},
                delay_seconds=None,
            )
            return
        base = result.retry_after_seconds or AUTO_MERGE_RETRY_SECONDS
        delay = min(base * 2 ** (deferrals - 1), AUTO_MERGE_RETRY_MAX_SECONDS)
        await self._write_status(
            customer_id,
            node_id,
            claimed_at,
            {"status": "deferred", "attempts": prev.get("attempts", 0), "deferrals": deferrals,
             "failed_calls": failed_calls, "last_error": error},
            delay_seconds=delay,
        )

    async def _write_status(
        self,
        customer_id: str,
        node_id: int,
        claimed_at: datetime,
        auto_merge_status: dict,
        *,
        delay_seconds: float | None,
    ) -> None:
        """Set analyzer_status.auto_merge and the lock: NULL (claimable now) or
        NOW() + delay (claimable after it). A no-op if the row was re-enqueued
        since it was claimed."""
        async with raw_conn() as conn:
            await conn.execute(
                """
                UPDATE node_post_write_queue
                SET locked_until = CASE WHEN $3::float8 IS NULL THEN NULL
                                        ELSE NOW() + make_interval(secs => $3::float8) END,
                    analyzer_status = $4::jsonb
                WHERE customer_id = $1 AND node_id = $2 AND enqueued_at = $5
                """,
                customer_id,
                node_id,
                None if delay_seconds is None else float(delay_seconds),
                json.dumps({"auto_merge": auto_merge_status}),
                claimed_at,
            )

    async def _record_failure(
        self,
        customer_id: str,
        node_id: int,
        claimed_at: datetime,
        attempts: int,
        error: str,
    ) -> None:
        # Clear lock so the row CAN be re-tried, but bump attempts. When
        # attempts hits _MAX_ATTEMPTS, the claim WHERE clause stops picking
        # it back up — row stays in queue for visibility but won't process.
        await self._write_status(
            customer_id,
            node_id,
            claimed_at,
            {"status": "failed", "attempts": attempts, "last_error": error[:240]},
            delay_seconds=None,
        )


def _auto_merge_status(status: dict) -> dict:
    """analyzer_status["auto_merge"] with integer counters (0 when absent)."""
    raw = status.get("auto_merge") or {}
    return {
        **raw,
        "attempts": int(raw.get("attempts", 0)),
        "deferrals": int(raw.get("deferrals", 0)),
        "failed_calls": int(raw.get("failed_calls", 0)),
    }


# --------------------------------------------------------------------------- #
# Entry point — runs alongside InferredEdgesWorker
# --------------------------------------------------------------------------- #


async def run_worker_forever() -> None:
    """Run the PostWriteWorker until SIGTERM.

    Mirrors `services/ingestion/inferred_edges/worker.py:run_worker_forever`
    but for the post-write queue. Typically NOT invoked standalone — the
    inferred-edges worker process gather()s both workers into one event loop
    (see post_write integration in inferred_edges/worker.py).
    """
    from engine.shared.config import get_settings
    from engine.shared.db import close_pool, init_pool
    from engine.shared.logging import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    worker = PostWriteWorker(execute_high_confidence=False)
    try:
        await worker.run()
    finally:
        await close_pool()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_worker_forever())
