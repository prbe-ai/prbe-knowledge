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
) -> list[dict[str, Any]]:
    """Pull the next batch of NULL embedding_v2 rows joined with their doc title.

    LEFT JOIN: orphan chunks (live doc deleted) get NULL title and fall back
    to the no-title prefix in the embedder. ORDER BY chunk_id gives stable
    iteration so a re-run after a crash hits the same rows in the same order.
    """
    rows = await conn.fetch(
        """
        SELECT c.chunk_id, c.customer_id, c.content, d.title
        FROM chunks c
        LEFT JOIN documents d
          ON c.customer_id = d.customer_id
         AND c.doc_id = d.doc_id
         AND d.valid_to IS NULL
        WHERE c.embedding_v2 IS NULL
          AND ($1::text IS NULL OR c.customer_id = $1)
        ORDER BY c.chunk_id
        LIMIT $2
        """,
        customer,
        batch_size,
    )
    return [dict(r) for r in rows]


async def _write_batch(
    conn: asyncpg.Connection,
    chunk_ids: list[str],
    customer_ids: list[str],
    embeddings_v2_pg: list[str],
) -> int:
    """Atomically write a batch of v2 vectors. Returns the row count actually updated.

    The WHERE embedding_v2 IS NULL guard prevents a race where a concurrent
    Stage 1 dual-write populated the same chunk between fetch and write --
    we wouldn't want to clobber a fresh ingest-time vector with our backfill
    one (they should be identical, but defensive).
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


async def main(argv: list[str] | None = None, *, close_pool_after: bool = False) -> int:
    """Entry point.

    `close_pool_after` defaults to False so tests that share a pool fixture
    don't get their pool yanked. The CLI wrapper at the bottom passes True.
    """
    ap = argparse.ArgumentParser(description="Backfill embedding_v2 column.")
    ap.add_argument(
        "--cost-cap",
        type=float,
        default=100.0,
        help="Max estimated USD spend before stopping (default: 100)",
    )
    ap.add_argument(
        "--customer",
        default=None,
        help="Limit to a single customer_id (default: all customers)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override settings.embedding_batch_size (default: 256)",
    )
    ap.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop after N batches (for staged dry-runs against prod)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate cost only on the first batch; do not embed or write",
    )
    args = ap.parse_args(argv)

    configure_logging()
    await init_pool()
    settings = get_settings()
    batch_size = args.batch_size or settings.embedding_batch_size
    embedder = get_embedder_v2()

    log.info(
        "backfill_v2.start",
        cost_cap=args.cost_cap,
        customer=args.customer,
        batch_size=batch_size,
        dry_run=args.dry_run,
    )

    cost = 0.0
    total_updated = 0
    total_failed = 0
    batches = 0
    started = time.monotonic()

    try:
        while True:
            if args.max_batches is not None and batches >= args.max_batches:
                log.info("backfill_v2.max_batches_hit", batches=batches)
                break

            async with raw_conn() as conn:
                rows = await _fetch_batch(conn, args.customer, batch_size)

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

            cost += est_cost
            batches += 1

            log.info(
                "backfill_v2.batch_done",
                batch=batches,
                chunks=len(rows),
                updated=len(success_chunk_ids),
                failed=failed_count,
                cum_cost_usd=round(cost, 4),
            )
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
    )
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(close_pool_after=True)))
