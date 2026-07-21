"""Drain loop for `node_post_write_queue`.

Lifecycle of a queued row::

  graph_writer.upsert_nodes commits
    -> enqueue_post_write_node(customer_id, node_id) [post-commit hook]
       INSERT ... ON CONFLICT DO UPDATE SET analyzer_status='{}'

  PostWriteWorker._claim_loop polls:
    SELECT (customer_id, node_id) FROM node_post_write_queue
    WHERE (locked_until IS NULL)
      AND ((analyzer_status->'auto_merge'->>'attempts')::int < 3
           OR analyzer_status->'auto_merge'->>'attempts' IS NULL)
    FOR UPDATE SKIP LOCKED LIMIT 1
    UPDATE SET locked_until = NOW() + INTERVAL '5 minutes'

  Process:
    1. Embed the node text via GeminiEmbedder if graph_nodes.embedding IS NULL
    2. Run AutoMergeAnalyzer.analyze(); honor per-customer auto_merge_execute
       toggle if/when added (default suggestion-only for safety in v1)

  On success: DELETE FROM node_post_write_queue WHERE (customer_id, node_id) = (...)
  On failure: clear locked_until, bump attempts in analyzer_status JSONB; if
              attempts hit 3 the WHERE clause stops picking it back up.

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
import uuid

import asyncpg

from engine.ingest.auto_merge import AutoMergeAnalyzer
from engine.ingest.graph_writer import drain_pending_edges, reap_expired_pending_edges
from engine.ingest.normalizer import _pg_vector
from engine.shared.db import raw_conn, with_tenant
from engine.shared.embeddings import get_embedder_v2
from engine.shared.logging import get_logger
from engine.shared.metrics import counter
from scripts.backfill_graph_node_embeddings import _embedding_text

log = get_logger(__name__)

_MAX_ATTEMPTS = 3
_DEFAULT_CONCURRENCY = int(os.getenv("POST_WRITE_CONCURRENCY", "16"))
_POLL_INTERVAL_SECONDS = 2.0
_LOCK_DURATION = "5 minutes"


class PostWriteWorker:
    """Drain loop for node_post_write_queue."""

    def __init__(
        self,
        *,
        concurrency: int = _DEFAULT_CONCURRENCY,
        worker_id: str | None = None,
        execute_high_confidence: bool = False,
    ) -> None:
        self._concurrency = max(1, concurrency)
        self._worker_id = worker_id or (
            f"post-write-{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        )
        self._shutdown = asyncio.Event()
        self._analyzer = AutoMergeAnalyzer(execute_high_confidence=execute_high_confidence)

    async def run(self) -> None:
        log.info(
            "post_write_worker.start",
            worker_id=self._worker_id,
            concurrency=self._concurrency,
            execute=self._analyzer._execute,
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

        Returns a row with (customer_id, node_id, analyzer_status). Atomically
        flips locked_until to NOW() + 5min so concurrent workers skip it.

        Also reclaims rows whose previous lock has expired (locked_until in
        the past) — those got stuck because a worker pod died mid-process,
        couldn't run the success-DELETE or failure-clear, and would
        otherwise never get picked back up.
        """
        async with raw_conn() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT customer_id, node_id, analyzer_status
                FROM node_post_write_queue
                WHERE (locked_until IS NULL OR locked_until < NOW())
                  AND COALESCE(
                      (analyzer_status->'auto_merge'->>'attempts')::int, 0
                  ) < $1
                ORDER BY enqueued_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                _MAX_ATTEMPTS,
            )
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
        status_json = row["analyzer_status"]
        status = json.loads(status_json) if isinstance(status_json, str) else (status_json or {})

        log.info(
            "post_write_worker.processing",
            customer=customer_id,
            node_id=node_id,
        )

        try:
            async with with_tenant(customer_id) as conn:
                await self._ensure_embedding(conn, node_id)
                await self._drain_pending_edges(conn, customer_id, node_id)
                result = await self._analyzer.analyze(conn, customer_id, node_id)

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
            )
            await self._delete_queue_row(customer_id, node_id)

        except Exception as exc:
            attempts = int(status.get("auto_merge", {}).get("attempts", 0)) + 1
            log.exception(
                "post_write_worker.process_failed",
                customer=customer_id,
                node_id=node_id,
                attempts=attempts,
                error=str(exc),
            )
            await self._record_failure(customer_id, node_id, attempts, repr(exc))

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
            await drain_pending_edges(
                conn, customer_id, row["label"], row["canonical_id"]
            )
            # Opportunistic TTL sweep for this tenant -- no separate cron.
            await reap_expired_pending_edges(conn, customer_id)
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

    async def _delete_queue_row(self, customer_id: str, node_id: int) -> None:
        async with raw_conn() as conn:
            await conn.execute(
                "DELETE FROM node_post_write_queue WHERE customer_id = $1 AND node_id = $2",
                customer_id,
                node_id,
            )

    async def _record_failure(
        self,
        customer_id: str,
        node_id: int,
        attempts: int,
        error: str,
    ) -> None:
        # Clear lock so the row CAN be re-tried, but bump attempts. When
        # attempts hits _MAX_ATTEMPTS, the claim WHERE clause stops picking
        # it back up — row stays in queue for visibility but won't process.
        new_status = json.dumps(
            {"auto_merge": {"status": "failed", "attempts": attempts, "last_error": error[:240]}}
        )
        async with raw_conn() as conn:
            await conn.execute(
                """
                UPDATE node_post_write_queue
                SET locked_until = NULL,
                    analyzer_status = $3::jsonb
                WHERE customer_id = $1 AND node_id = $2
                """,
                customer_id,
                node_id,
                new_status,
            )


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
