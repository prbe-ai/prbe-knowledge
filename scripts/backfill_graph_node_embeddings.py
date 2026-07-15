"""Backfill `graph_nodes.embedding` with gemini-embedding-2 vectors.

Lane A's migration 0082 added `graph_nodes.embedding halfvec(3072)` nullable.
This script populates it for existing rows so the AutoMergeAnalyzer's vector
search has something to query.

Idempotent: WHERE embedding IS NULL guards re-runs. Resumable: no in-memory
cursor; the column IS the cursor. Single-threaded by design (no concurrency
on the same NULL rows).

Embedding text per node:
    f"{label} {canonical_id} {properties.get('name','')} {properties.get('email','')}"
with blanks stripped. Same dim (3072) as chunks.embedding_v2 so the existing
GeminiEmbedder writes in-place.

Usage::

    .venv/bin/python -m scripts.backfill_graph_node_embeddings
    .venv/bin/python -m scripts.backfill_graph_node_embeddings --customer acme
    .venv/bin/python -m scripts.backfill_graph_node_embeddings --dry-run

Verify::

    SELECT COUNT(*) FROM graph_nodes WHERE embedding IS NULL;
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import asyncpg

from engine.ingest.normalizer import _pg_vector
from engine.shared.config import get_settings
from engine.shared.db import close_pool, init_pool
from engine.shared.embeddings import DocItem, get_embedder_v2
from engine.shared.logging import configure_logging, get_logger

log = get_logger(__name__)

BATCH_SIZE = 64


def _embedding_text(label: str, canonical_id: str, properties: dict | None) -> str:
    """Compose the text that gets embedded for a node.

    Order matters: label first (steers the embedding toward type semantics),
    then canonical_id (the primary identifier), then name + email + login
    when present. Empty/missing fields are skipped — joining with single
    spaces avoids producing "lots  of  blanks" that would dilute the vector.
    """
    props = properties or {}
    parts = [label, canonical_id]
    for key in ("name", "display_name", "real_name", "email", "login", "handle"):
        v = props.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " ".join(parts)


async def _fetch_batch(
    conn: asyncpg.Connection,
    customer: str | None,
    batch_size: int,
) -> list[asyncpg.Record]:
    if customer:
        return await conn.fetch(
            """
            SELECT node_id, customer_id, label, canonical_id, properties
            FROM graph_nodes
            WHERE embedding IS NULL AND customer_id = $1
            ORDER BY node_id
            LIMIT $2
            """,
            customer,
            batch_size,
        )
    return await conn.fetch(
        """
        SELECT node_id, customer_id, label, canonical_id, properties
        FROM graph_nodes
        WHERE embedding IS NULL
        ORDER BY node_id
        LIMIT $1
        """,
        batch_size,
    )


async def _embed_and_write(
    conn: asyncpg.Connection,
    rows: list[asyncpg.Record],
    *,
    dry_run: bool,
) -> int:
    """Embed a batch and UPDATE in one statement. Returns count written."""
    embedder = get_embedder_v2()

    docs: list[DocItem] = []
    for r in rows:
        props = r["properties"]
        if isinstance(props, str):
            props = json.loads(props or "{}")
        text = _embedding_text(r["label"], r["canonical_id"], props)
        docs.append(DocItem(doc_id=str(r["node_id"]), content=text, title=None))

    result = await embedder.embed_many([d.content for d in docs])
    if result.failed:
        log.warning(
            "embed_backfill.partial_fail",
            failed=len(result.failed),
            succeeded=len(result.embedded),
        )
    if not result.embedded:
        return 0

    if dry_run:
        for emb in result.embedded[:3]:
            log.info(
                "embed_backfill.dry_run.preview",
                node_id=rows[emb.chunk_index].get("node_id"),
                dim=len(emb.embedding),
            )
        return len(result.embedded)

    node_ids = [rows[emb.chunk_index]["node_id"] for emb in result.embedded]
    vectors = [_pg_vector(emb.embedding) for emb in result.embedded]
    await conn.execute(
        """
        UPDATE graph_nodes AS g
        SET embedding = v.embedding::halfvec(3072)
        FROM unnest($1::bigint[], $2::text[]) AS v(node_id, embedding)
        WHERE g.node_id = v.node_id
        """,
        node_ids,
        vectors,
    )
    return len(node_ids)


async def run(customer: str | None, dry_run: bool, max_batches: int | None) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    total = 0
    batches = 0
    try:
        while True:
            if max_batches and batches >= max_batches:
                log.info("embed_backfill.batch_cap_reached", batches=batches)
                break
            # Use raw connection for cross-tenant SELECT (no RLS scope needed on
            # graph_nodes for this script — script runs as the table owner).
            from engine.shared.db import raw_conn

            async with raw_conn() as conn:
                rows = await _fetch_batch(conn, customer, BATCH_SIZE)
                if not rows:
                    log.info("embed_backfill.done", total=total, batches=batches)
                    break
                written = await _embed_and_write(conn, rows, dry_run=dry_run)
                total += written
                batches += 1
                log.info(
                    "embed_backfill.batch",
                    batch=batches,
                    batch_size=len(rows),
                    written=written,
                    total=total,
                )
    finally:
        await close_pool()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--customer", default=None, help="Scope to one customer_id")
    ap.add_argument("--dry-run", action="store_true", help="Embed but don't UPDATE")
    ap.add_argument("--max-batches", type=int, default=None, help="Cap iterations for testing")
    args = ap.parse_args()
    try:
        asyncio.run(run(args.customer, args.dry_run, args.max_batches))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
