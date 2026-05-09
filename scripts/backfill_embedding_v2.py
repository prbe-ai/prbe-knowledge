"""Stage 2 of the OpenAI -> Gemini embedding migration.

Iterates every chunk where embedding_v2 IS NULL and populates it via
gemini-embedding-2-preview. Designed to be run from a one-off Fly machine
(NOT triggered by deploy) against the production Neon DB.

Idempotent: the WHERE embedding_v2 IS NULL filter means a re-run picks up
exactly the chunks the previous run didn't finish. No external cursor or
state file -- the column itself is the cursor.

Cost cap: $100 per run by default. The script estimates input tokens from
content + title length and stops cleanly when the next batch would push
spend past the cap. Use --cost-cap to tune.

Partial run handling: each batch writes its results in a single UPDATE
statement, atomic per-row. A crash mid-run loses no progress; chunks
already updated stay updated; NULL chunks get picked up next run.

Concurrency: single-threaded by design. Multiple instances racing on the
same NULL chunks would waste API calls; the WHERE embedding_v2 IS NULL
guard in the UPDATE prevents corruption but not duplicate work.

Usage::

    .venv/bin/python -m scripts.backfill_embedding_v2
    .venv/bin/python -m scripts.backfill_embedding_v2 --cost-cap 25
    .venv/bin/python -m scripts.backfill_embedding_v2 --customer cust-X
    .venv/bin/python -m scripts.backfill_embedding_v2 --dry-run

Verify completion via::

    SELECT COUNT(*) FROM chunks WHERE embedding_v2 IS NULL;

Once that returns 0, Stage 3 (HNSW index on embedding_v2) is safe to land.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
import time
from typing import Any

import asyncpg

from services.ingestion.normalizer import _pg_vector
from shared.config import get_settings
from shared.constants import EMBEDDING_V2_DIM, EMBEDDING_V2_MODEL
from shared.db import close_pool, init_pool, raw_conn
from shared.embeddings import DocItem, get_embedder_v2
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)

# gemini-embedding-2-preview input pricing (as of 2026-05). Used only for
# the sanity cost cap -- real billing is whatever Gemini actually charges.
# Update when pricing shifts; the constant is a guard rail, not accounting.
GEMINI_2_INPUT_USD_PER_M_TOKENS = 0.15


def _estimate_tokens(content: str, title: str | None) -> int:
    """Rough char/4 estimate for the formatted Gemini input.

    The chunker caps content at ~2048 tokens already, and the title cap in
    `_format_gemini_document` keeps title bounded, so the variance is low.
    Char/4 errs slightly high for English text, which is the safe direction
    for a cost cap.
    """
    title_str = (title or "")[:200]
    formatted_len = len(title_str) + len(content) + 32  # 32 chars for prefix scaffolding
    return max(1, formatted_len // 4)


async def _fetch_batch(
    conn: asyncpg.Connection,
    customer: str | None,
    batch_size: int,
    workers: int = 1,
    worker_id: int = 0,
) -> list[dict[str, Any]]:
    """Pull the next batch of NULL embedding_v2 rows joined with their doc title.

    LATERAL with LIMIT 1: there's no UNIQUE constraint that documents has
    exactly one live (valid_to IS NULL) row per (customer_id, doc_id) -- a
    bad write race or manual data repair could leave two. A plain LEFT JOIN
    in that case duplicates the chunk row, double-billing Gemini for it.
    LATERAL pins the join to a single title pick so each chunk shows up
    exactly once. Orphan chunks (live doc deleted) still get NULL title and
    fall back to the no-title prefix in the embedder.

    ORDER BY chunk_id gives stable iteration so a re-run after a crash hits
    the same rows in the same order.

    workers/worker_id partition: when running multiple processes in
    parallel, each process owns rows where
    `mod(abs(hashtext(chunk_id)), workers) = worker_id`. With
    workers=1 (default) the modulo is always 0 and the predicate matches
    every row, preserving single-process behavior. abs() is needed because
    Postgres' hashtext returns a signed int4 and `mod(-N, M)` is negative,
    which would never equal a non-negative worker_id.
    """
    rows = await conn.fetch(
        """
        SELECT c.chunk_id, c.customer_id, c.content, d.title
        FROM chunks c
        LEFT JOIN LATERAL (
            SELECT title
            FROM documents
            WHERE customer_id = c.customer_id
              AND doc_id = c.doc_id
              AND valid_to IS NULL
            LIMIT 1
        ) d ON TRUE
        WHERE c.embedding_v2 IS NULL
          AND ($1::text IS NULL OR c.customer_id = $1)
          AND mod(abs(hashtext(c.chunk_id)), $3::int) = $4::int
        ORDER BY c.chunk_id
        LIMIT $2
        """,
        customer,
        batch_size,
        workers,
        worker_id,
    )
    return [dict(r) for r in rows]


async def _write_batch(
    conn: asyncpg.Connection,
    chunk_ids: list[str],
    customer_ids: list[str],
    embeddings_v2_pg: list[str],
) -> int:
    """Atomically write a batch of v2 vectors. Returns the row count actually updated.

    The WHERE embedding_v2 IS NULL guard handles a race where a concurrent
    Stage 1 dual-write populated the same chunk between our fetch and our
    UPDATE -- we'd skip clobbering it. NOTE this guard is on the WRITE only;
    by the time we reach it the Gemini API call has already happened, so it
    prevents data inconsistency, not duplicate spend. Cost accounting in
    main() compensates by billing only against successful rows.
    """
    result = await conn.execute(
        """
        UPDATE chunks AS c
        SET embedding_v2 = u.embedding_v2::halfvec,
            embedding_v2_model = $1,
            embedding_v2_dim = $2
        FROM (
            SELECT unnest($3::text[]) AS chunk_id,
                   unnest($4::text[]) AS customer_id,
                   unnest($5::text[]) AS embedding_v2
        ) u
        WHERE c.chunk_id = u.chunk_id
          AND c.customer_id = u.customer_id
          AND c.embedding_v2 IS NULL
        """,
        EMBEDDING_V2_MODEL,
        EMBEDDING_V2_DIM,
        chunk_ids,
        customer_ids,
        embeddings_v2_pg,
    )
    # asyncpg returns "UPDATE N" for executes; parse out N. Fall through to 0
    # on unexpected formats so the script keeps making progress.
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0


def _positive_float(s: str) -> float:
    v = float(s)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {v}")
    return v


def _positive_int(s: str) -> int:
    v = int(s)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {v}")
    return v


# Exit codes.
RC_CLEAN = 0
RC_FAILURES = 1  # poison chunks or other per-row failures
RC_CAP_HIT_REMAINING = 2  # cost cap or max-batches hit while NULL rows remain


async def main(argv: list[str] | None = None, *, close_pool_after: bool = False) -> int:
    """Entry point.

    `close_pool_after` defaults to False so tests that share a pool fixture
    don't get their pool yanked. The CLI wrapper at the bottom passes True.

    Exit codes are operator-meaningful: 0 = done (no NULL rows left, no
    failures), 1 = some chunks failed Gemini and stayed NULL (re-run may
    clear them), 2 = the run stopped early on cost cap or max-batches with
    NULL rows still in the table (operator MUST re-run before Stage 3).
    """
    ap = argparse.ArgumentParser(description="Backfill embedding_v2 column.")
    ap.add_argument(
        "--cost-cap",
        type=_positive_float,
        default=100.0,
        help="Max estimated USD spend before stopping (default: 100, must be > 0)",
    )
    ap.add_argument(
        "--customer",
        default=None,
        help="Limit to a single customer_id (default: all customers)",
    )
    ap.add_argument(
        "--batch-size",
        type=_positive_int,
        default=None,
        help="Override settings.embedding_batch_size (default: 256, must be > 0)",
    )
    ap.add_argument(
        "--max-batches",
        type=_positive_int,
        default=None,
        help="Stop after N batches (for staged dry-runs against prod)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate cost only on the first batch; do not embed or write",
    )
    ap.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help=(
            "Total number of parallel worker processes (default: 1). "
            "Each process must be launched with a distinct --worker-id "
            "in [0, workers). Partitioning is by hashtext(chunk_id) MOD "
            "workers, so processes never embed the same row."
        ),
    )
    ap.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help=(
            "0-indexed id of THIS process within --workers (default: 0). "
            "Must be in [0, workers)."
        ),
    )
    args = ap.parse_args(argv)
    if not 0 <= args.worker_id < args.workers:
        ap.error(
            f"--worker-id must be in [0, {args.workers}); got {args.worker_id}"
        )

    configure_logging()
    await init_pool()
    settings = get_settings()
    batch_size = args.batch_size or settings.embedding_batch_size
    embedder = get_embedder_v2()

    # Co-operative shutdown: Fly machines stop with SIGTERM. Set a flag the
    # main loop checks at batch boundaries so an in-flight batch finishes
    # cleanly (UPDATE commits) before the loop exits. Without this, SIGTERM
    # would tear down asyncio mid-Gemini-call and leak that batch's tokens.
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        # Windows / restricted env may not support add_signal_handler --
        # proceed without graceful shutdown rather than refusing to start.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    log.info(
        "backfill_v2.start",
        cost_cap=args.cost_cap,
        customer=args.customer,
        batch_size=batch_size,
        dry_run=args.dry_run,
        workers=args.workers,
        worker_id=args.worker_id,
    )

    cost = 0.0
    total_updated = 0
    total_failed = 0
    batches = 0
    cap_hit = False
    remaining = 0
    started = time.monotonic()

    try:
        while True:
            if stop.is_set():
                log.warning("backfill_v2.signal_received_stopping_clean")
                cap_hit = True  # treat as "didn't finish" for rc semantics
                break
            if args.max_batches is not None and batches >= args.max_batches:
                log.info("backfill_v2.max_batches_hit", batches=batches)
                cap_hit = True
                break

            async with raw_conn() as conn:
                rows = await _fetch_batch(
                    conn,
                    args.customer,
                    batch_size,
                    workers=args.workers,
                    worker_id=args.worker_id,
                )

            if not rows:
                log.info("backfill_v2.no_more_chunks")
                break

            est_tokens = sum(_estimate_tokens(r["content"], r["title"]) for r in rows)
            est_cost = est_tokens / 1_000_000 * GEMINI_2_INPUT_USD_PER_M_TOKENS

            if cost + est_cost > args.cost_cap:
                log.warning(
                    "backfill_v2.cost_cap_reached",
                    spent_usd=round(cost, 2),
                    next_batch_usd=round(est_cost, 4),
                    cap_usd=args.cost_cap,
                )
                cap_hit = True
                break

            if args.dry_run:
                cost += est_cost
                batches += 1
                log.info(
                    "backfill_v2.dry_run_batch",
                    batch=batches,
                    chunks=len(rows),
                    est_cost_usd=round(est_cost, 4),
                )
                # Dry-run exits after the first batch to avoid pointlessly
                # iterating against the same NULL rows -- nothing was written.
                cap_hit = True  # rc semantics: didn't finish (intentional)
                break

            items = [DocItem(content=r["content"], title=r["title"]) for r in rows]
            outcome = await embedder.embed_documents(items)

            success_chunk_ids: list[str] = []
            success_customers: list[str] = []
            success_pg: list[str] = []
            for emb in outcome.embedded:
                if emb.chunk_index < 0 or emb.chunk_index >= len(rows):
                    continue
                row = rows[emb.chunk_index]
                success_chunk_ids.append(row["chunk_id"])
                success_customers.append(row["customer_id"])
                success_pg.append(_pg_vector(emb.embedding))

            failed_count = len(outcome.failed)
            total_failed += failed_count
            for fail in outcome.failed:
                log.warning(
                    "backfill_v2.chunk_failed",
                    chunk_idx=fail.chunk_index,
                    error=fail.error,
                )

            if success_chunk_ids:
                async with raw_conn() as conn:
                    updated = await _write_batch(
                        conn, success_chunk_ids, success_customers, success_pg
                    )
                total_updated += updated

            # Bill cost only against rows that produced a vector. A 100%
            # API-error batch (Gemini outage) charges 0 against the cap so a
            # transient outage doesn't consume the entire budget on failed
            # calls. Real Gemini billing varies by error type; this errs
            # toward letting the operator keep retrying.
            success_ratio = (
                len(success_chunk_ids) / len(rows) if rows else 0.0
            )
            cost += est_cost * success_ratio
            batches += 1

            log.info(
                "backfill_v2.batch_done",
                batch=batches,
                chunks=len(rows),
                updated=len(success_chunk_ids),
                failed=failed_count,
                cum_cost_usd=round(cost, 4),
            )

        # If we stopped early (cap / max-batches / signal), check whether
        # NULL rows remain in THIS worker's partition so the operator
        # knows to re-run before Stage 3. Each worker only owns its
        # partition; the combined "fully drained" check is the operator's
        # responsibility (run all workers + count globally).
        remaining = 0
        if cap_hit and not args.dry_run:
            async with raw_conn() as conn:
                remaining = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM chunks
                    WHERE embedding_v2 IS NULL
                      AND ($1::text IS NULL OR customer_id = $1)
                      AND mod(abs(hashtext(chunk_id)), $2::int) = $3::int
                    """,
                    args.customer,
                    args.workers,
                    args.worker_id,
                )
            log.info("backfill_v2.remaining_nulls", remaining=remaining)
    finally:
        if close_pool_after:
            await close_pool()

    elapsed = time.monotonic() - started
    log.info(
        "backfill_v2.finished",
        batches=batches,
        chunks_updated=total_updated,
        chunks_failed=total_failed,
        est_cost_usd=round(cost, 2),
        elapsed_s=round(elapsed, 1),
        cap_hit=cap_hit,
    )

    if cap_hit and not args.dry_run and remaining > 0:
        return RC_CAP_HIT_REMAINING
    if total_failed > 0:
        return RC_FAILURES
    return RC_CLEAN


if __name__ == "__main__":
    sys.exit(asyncio.run(main(close_pool_after=True)))
