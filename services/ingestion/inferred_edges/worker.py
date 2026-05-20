"""Side-queue worker for LLM-inferred edge extraction.

Drain loop:
  1. Claim one inferred_edges_queue row via FOR UPDATE SKIP LOCKED, atomically
     incrementing attempts and stamping processing_started_at /
     processing_worker_id in the same UPDATE.
  2. Build bundle -> extract -> upsert edges.
  3. On success: mark done_at. On failure: clear processing_started_at so the
     row becomes re-claimable; attempts is NOT bumped here because it was
     already bumped at claim time (step 1). The WHERE attempts < MAX gate at
     claim time naturally drops rows that have been claimed too many times.

Concurrency: 16 async tasks per worker process (INFERRED_EDGES_CONCURRENCY
env var). Each task independently claims via SKIP LOCKED.

Fly notes (from memory):
  - Health endpoint binds on 0.0.0.0 (IPv4) for Fly health checks.
  - This worker has no internal API server; it's a pure drain-loop process
    with one health endpoint. No 6PN/IPv6 binding required.
  - count in fly.toml needs a follow-up:
        flyctl scale count worker=4 -a prbe-knowledge-side-worker
    (this drainer lives on the generic side-worker fly app; future
     side-queue drainers can land on the same app as additional
     [processes] entries instead of separate fly apps)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import uuid
from datetime import UTC, datetime

import asyncpg

from services.ingestion.graph_writer import upsert_edges, upsert_nodes
from services.ingestion.inferred_edges.bundle import build_bundle
from services.ingestion.inferred_edges.extractor import InferredEdge, extract_edges
from services.ingestion.inferred_edges.prompts.v1 import PROMPT_VERSION
from shared.constants import NodeLabel
from shared.db import get_pool, with_tenant
from shared.logging import get_logger
from shared.metrics import counter, gauge

log = get_logger(__name__)

# Max attempts before a queue row is dropped.
_MAX_ATTEMPTS = 3

# Default concurrency (tasks per process).
_DEFAULT_CONCURRENCY = 16

# Worker poll interval when queue is empty.
_POLL_INTERVAL_SECONDS = 2.0


class InferredEdgesWorker:
    """Async drain loop for inferred_edges_queue."""

    def __init__(
        self,
        concurrency: int = _DEFAULT_CONCURRENCY,
        worker_id: str | None = None,
    ) -> None:
        self._concurrency = max(1, concurrency)
        self._worker_id = worker_id or f"inferred-edges-{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        log.info(
            "inferred_edges_worker.start",
            worker_id=self._worker_id,
            concurrency=self._concurrency,
        )
        await asyncio.gather(
            *(self._claim_loop() for _ in range(self._concurrency))
        )
        log.info("inferred_edges_worker.stop")

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
        """Atomically claim one pending row via FOR UPDATE SKIP LOCKED."""
        async with get_pool().acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT id, customer_id, anchor_doc_id, extractor_id, attempts
                FROM inferred_edges_queue
                WHERE processing_started_at IS NULL
                  AND done_at IS NULL
                  AND attempts < $1
                ORDER BY enqueued_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                _MAX_ATTEMPTS,
            )
            if row is None:
                return None
            await conn.execute(
                """
                UPDATE inferred_edges_queue
                SET processing_started_at = NOW(),
                    processing_worker_id = $1,
                    attempts = attempts + 1
                WHERE id = $2
                """,
                self._worker_id,
                row["id"],
            )
            return row

    async def _process(self, row: asyncpg.Record) -> None:
        queue_id: int = row["id"]
        customer_id: str = row["customer_id"]
        anchor_doc_id: str = row["anchor_doc_id"]

        log.info(
            "inferred_edges_worker.processing",
            queue_id=queue_id,
            customer=customer_id,
            anchor_doc_id=anchor_doc_id,
        )

        try:
            async with with_tenant(customer_id) as conn:
                bundle = await build_bundle(customer_id, anchor_doc_id, conn)

                if not bundle.docs:
                    # Empty bundle -- anchor may have been deleted. Mark done.
                    await _mark_done(queue_id)
                    return

                extraction = await extract_edges(bundle, conn)

                if extraction.cost_usd > 0:
                    gauge(
                        "inferred_edges_llm_cost_per_customer_per_day",
                        extraction.cost_usd,
                        customer_id=customer_id,
                        extractor_id=PROMPT_VERSION,
                    )

                if extraction.bundle_failed:
                    log.warning(
                        "inferred_edges_worker.bundle_failed",
                        queue_id=queue_id,
                        customer=customer_id,
                        reason=extraction.bundle_fail_reason,
                    )
                    await _mark_error(queue_id, extraction.bundle_fail_reason)
                    return

                if extraction.edges:
                    await _upsert_inferred_edges(conn, customer_id, extraction.edges)

                counter(
                    "inferred_edges.extracted",
                    len(extraction.edges),
                    customer_id=customer_id,
                )
                counter(
                    "inferred_edges.dropped",
                    sum(extraction.dropped.values()),
                    customer_id=customer_id,
                )

            await _mark_done(queue_id)

        except Exception as exc:
            log.exception(
                "inferred_edges_worker.process_failed",
                queue_id=queue_id,
                customer=customer_id,
                error=str(exc),
            )
            await _mark_error(queue_id, repr(exc))


async def _upsert_inferred_edges(
    conn: asyncpg.Connection,
    customer_id: str,
    edges: list[InferredEdge],
) -> None:
    """Upsert inferred edges into graph_edges via graph_writer.upsert_edges.

    We need to resolve endpoints to node_ids. The extractor already validated
    that each endpoint exists in graph_nodes for this customer.
    """
    from shared.models import GraphEdgeSpec, GraphNodeSpec

    # Build minimal node specs for endpoint resolution only.
    # (Nodes already exist; upsert_nodes is idempotent.)
    node_specs: list[GraphNodeSpec] = []
    for edge in edges:
        try:
            from_lbl = NodeLabel(edge.from_label)
            to_lbl = NodeLabel(edge.to_label)
        except ValueError:
            log.warning(
                "inferred_edges_worker.unknown_node_label",
                from_label=edge.from_label,
                to_label=edge.to_label,
            )
            continue
        node_specs.append(GraphNodeSpec(label=from_lbl, canonical_id=edge.from_canonical_id))
        node_specs.append(GraphNodeSpec(label=to_lbl, canonical_id=edge.to_canonical_id))

    # Dedupe node specs by (label, canonical_id)
    seen: set[tuple[str, str]] = set()
    deduped_nodes: list[GraphNodeSpec] = []
    for n in node_specs:
        key = (n.label.value, n.canonical_id)
        if key not in seen:
            seen.add(key)
            deduped_nodes.append(n)

    node_ids = await upsert_nodes(
        conn, customer_id, deduped_nodes, PROMPT_VERSION
    )

    edge_specs: list[GraphEdgeSpec] = []
    for edge in edges:
        try:
            from shared.constants import EdgeType as ET
            edge_type_enum = ET(edge.edge_type)
            from_lbl = NodeLabel(edge.from_label)
            to_lbl = NodeLabel(edge.to_label)
        except ValueError:
            continue

        # Persist the LLM's justification on the edge. Without this the
        # `why` field is validated by the extractor and then dropped on the
        # write path -- inferred edges land in graph_edges with empty
        # properties and no audit trail. Stored under `properties.why`;
        # consumed by /knowledge/insights and the dashboard inferred-edges UI.
        # `properties.model` records which LLM produced this edge so we can
        # audit cutover correctness without bumping extractor_id (the prompt
        # + validator pipeline is unchanged across the Haiku -> Flash Lite
        # cutover; only the model differs).
        edge_props: dict[str, str] = {}
        if edge.why:
            edge_props["why"] = edge.why
        if edge.model:
            edge_props["model"] = edge.model
        edge_specs.append(
            GraphEdgeSpec(
                edge_type=edge_type_enum,
                from_label=from_lbl,
                from_canonical_id=edge.from_canonical_id,
                to_label=to_lbl,
                to_canonical_id=edge.to_canonical_id,
                confidence=edge.confidence,
                properties=edge_props,
            )
        )

    if edge_specs:
        await upsert_edges(
            conn,
            customer_id,
            edge_specs,
            node_ids,
            PROMPT_VERSION,
            extractor_id=PROMPT_VERSION,
            extracted_at=edges[0].extracted_at if edges else None,
        )


async def _mark_done(queue_id: int) -> None:
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            UPDATE inferred_edges_queue
            SET done_at = NOW(),
                processing_started_at = NULL,
                error = NULL
            WHERE id = $1
            """,
            queue_id,
        )


async def _mark_error(queue_id: int, error: str) -> None:
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            UPDATE inferred_edges_queue
            SET processing_started_at = NULL,
                error = $2
            WHERE id = $1
            """,
            queue_id,
            error[:4000],
        )


def _build_health_app():
    """Tiny FastAPI app for Fly health checks.

    Binds on 0.0.0.0 (IPv4) per feedback_fly_health_check_ipv4.md.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from shared.db import health_check

    app = FastAPI(
        title="prbe-knowledge inferred-edges worker health",
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


async def run_worker_forever() -> None:
    """Entry point for the inferred-edges worker process."""
    import uvicorn

    from shared.config import get_settings
    from shared.db import init_pool
    from shared.logging import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    concurrency = int(os.environ.get("INFERRED_EDGES_CONCURRENCY", str(_DEFAULT_CONCURRENCY)))
    worker = InferredEdgesWorker(concurrency=concurrency)

    # Run the PostWriteWorker (entity auto-merge) in the same process. Both
    # workers drain independent queues; sharing the event loop avoids spinning
    # up a separate Fly app/Helm deployment for the second analyzer. Gated by
    # POST_WRITE_ENABLED (default off) so a stale env doesn't accidentally
    # fire merges before the customer has flipped auto_merge_execute on.
    from services.ingestion.post_write import PostWriteWorker

    post_write_enabled = os.environ.get("POST_WRITE_ENABLED", "false").lower() == "true"
    post_write_execute = os.environ.get("POST_WRITE_EXECUTE", "false").lower() == "true"
    post_write_worker: PostWriteWorker | None = None
    if post_write_enabled:
        post_write_worker = PostWriteWorker(
            concurrency=int(os.environ.get("POST_WRITE_CONCURRENCY", "16")),
            execute_high_confidence=post_write_execute,
        )

    health_port = int(os.environ.get("INFERRED_EDGES_HEALTH_PORT", "8083"))
    health_config = uvicorn.Config(
        _build_health_app(),
        # IPv4 for Fly health probes (feedback_fly_health_check_ipv4.md)
        host="0.0.0.0",
        port=health_port,
        log_config=None,
        lifespan="off",
        access_log=False,
    )
    health_server = uvicorn.Server(health_config)

    log.info(
        "inferred_edges_worker.boot",
        environment=settings.environment,
        health_port=health_port,
        concurrency=concurrency,
        timestamp=datetime.now(UTC).isoformat(),
    )

    loop = asyncio.get_running_loop()
    gather_future: asyncio.Future | None = None  # type: ignore[type-arg]
    shutdown_started = False

    def handle_signal(signame: str) -> None:
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True
        log.info("inferred_edges_worker.shutdown_signal", signal=signame)
        worker.shutdown()
        if post_write_worker is not None:
            post_write_worker.shutdown()
        health_server.should_exit = True
        if gather_future is not None and not gather_future.done():
            gather_future.cancel()

    for signame in ("SIGTERM", "SIGINT"):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(getattr(signal, signame), handle_signal, signame)

    try:
        coros = [worker.run(), health_server.serve()]
        if post_write_worker is not None:
            log.info("post_write_worker.enabled", execute=post_write_execute)
            coros.append(post_write_worker.run())
        gather_future = asyncio.gather(*coros)
        try:
            await gather_future
        except asyncio.CancelledError:
            log.info("inferred_edges_worker.shutdown_complete")
    finally:
        pass


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_worker_forever())
