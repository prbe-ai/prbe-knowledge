"""Backfill the kind='metadata' synthetic chunks for documents that
predate the metadata-chunks feature.

Usage:
    .venv/bin/python -m scripts.backfill_metadata_chunks \
        --customer cust-smoke
    .venv/bin/python -m scripts.backfill_metadata_chunks --all-tenants
    .venv/bin/python -m scripts.backfill_metadata_chunks --customer X --dry-run

Idempotent: skips docs that already have a kind='metadata' chunk.
Restartable: re-running picks up from where the prior run stopped.
Embed-first ordering: each metadata chunk is embedded before its row is
INSERTed, so retrieval never sees a NULL embedding mid-backfill.

Resilience: respects the embedder's transient-error handling. A
permanent embedding failure for one doc is logged and the loop continues
(records to failed_chunks, same as the steady-state path).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import AsyncIterator

from services.ingestion.normalizer import (
    _insert_chunk,
    _insert_failed_chunk,
    _metadata_piece,
)
from shared.config import get_settings
from shared.db import close_pool, init_pool, raw_conn, with_tenant
from shared.embeddings import get_embedder
from shared.exceptions import EmbeddingError
from shared.logging import configure_logging, get_logger
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    DocClass,
    DocType,
    Document,
    Permission,
    PrincipalType,
    SourceSystem,
)

log = get_logger(__name__)


async def _list_customers() -> list[str]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT customer_id FROM customers WHERE status = 'active' ORDER BY customer_id"
        )
    return [r["customer_id"] for r in rows]


async def _docs_needing_backfill(customer_id: str, batch_size: int) -> AsyncIterator[Document]:
    """Yield live documents (one batch at a time) that don't yet have a
    metadata chunk. Streamed so the script doesn't load every row into
    memory on large tenants."""
    async with with_tenant(customer_id) as conn:
        last_doc_id: str | None = None
        while True:
            rows = await conn.fetch(
                """
                SELECT d.doc_id, d.version, d.customer_id,
                       d.source_system, d.source_id, d.source_url,
                       d.doc_class, d.doc_type, d.content_type,
                       d.content_hash, d.title, d.body_preview,
                       d.body_size_bytes, d.body_token_count, d.author_id,
                       d.created_at, d.updated_at, d.valid_from, d.ingested_at,
                       d.metadata, d.entities
                FROM documents d
                WHERE d.customer_id = $1
                  AND d.valid_to IS NULL
                  AND ($2::text IS NULL OR d.doc_id > $2)
                  AND NOT EXISTS (
                      SELECT 1 FROM chunks c
                      WHERE c.customer_id = d.customer_id
                        AND c.doc_id = d.doc_id
                        AND c.kind = 'metadata'
                        AND c.valid_to IS NULL
                  )
                ORDER BY d.doc_id
                LIMIT $3
                """,
                customer_id,
                last_doc_id,
                batch_size,
            )
            if not rows:
                break
            for row in rows:
                last_doc_id = row["doc_id"]
                # Reconstruct a minimal Document — we only need the fields
                # _metadata_piece reads (title, source_system, author_id,
                # source_url, body_preview).
                yield Document(
                    doc_id=row["doc_id"],
                    customer_id=row["customer_id"],
                    version=row["version"],
                    source_system=SourceSystem(row["source_system"]),
                    source_id=row["source_id"],
                    source_url=row["source_url"],
                    doc_class=DocClass(row["doc_class"]),
                    doc_type=DocType(row["doc_type"]),
                    content_type=row["content_type"],
                    content_hash=row["content_hash"],
                    title=row["title"],
                    body_preview=row["body_preview"],
                    body_size_bytes=row["body_size_bytes"],
                    body_token_count=row["body_token_count"],
                    author_id=row["author_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    valid_from=row["valid_from"],
                    ingested_at=row["ingested_at"],
                    acl=ACLSnapshot(
                        principals=[
                            ACLPrincipal(
                                principal_type=PrincipalType.WORKSPACE,
                                principal_id=customer_id,
                                permission=Permission.READ,
                            )
                        ],
                        captured_at=row["created_at"],
                    ),
                )


async def _backfill_tenant(
    customer_id: str, batch_size: int, dry_run: bool
) -> tuple[int, int, int]:
    """Returns (skipped, embedded, failed)."""
    embedder = get_embedder()
    embedded = 0
    failed = 0
    skipped = 0

    async for doc in _docs_needing_backfill(customer_id, batch_size):
        piece = _metadata_piece(doc)
        if piece is None:
            # No useful fields to embed — skip silently. (Doc has no title,
            # no author, no URL, no body_preview.)
            skipped += 1
            continue

        if dry_run:
            log.info(
                "backfill_metadata.dry_run",
                customer=customer_id,
                doc_id=doc.doc_id,
                content_preview=piece.content[:100],
            )
            embedded += 1
            continue

        # Embed FIRST. If embed fails permanently, record to failed_chunks
        # and move on. Transient errors raise and cause the loop to surface
        # the exception — operator restarts.
        try:
            embeds = await embedder.embed_many([piece.content])
        except EmbeddingError:
            log.exception(
                "backfill_metadata.embed_failed",
                customer=customer_id,
                doc_id=doc.doc_id,
            )
            raise

        async with with_tenant(customer_id) as conn:
            for match in embeds.embedded:
                await _insert_chunk(conn, doc, piece, match.embedding, kind="metadata")
                embedded += 1
            for fail in embeds.failed:
                await _insert_failed_chunk(conn, doc, fail, piece)
                failed += 1

    return skipped, embedded, failed


async def _amain() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--customer", help="Single customer to backfill")
    group.add_argument(
        "--all-tenants",
        action="store_true",
        help="Iterate every active customer",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be embedded without calling the embedder",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    try:
        if args.all_tenants:
            customers = await _list_customers()
        else:
            customers = [args.customer]

        for customer in customers:
            log.info("backfill_metadata.start", customer=customer, dry_run=args.dry_run)
            skipped, embedded, failed = await _backfill_tenant(
                customer, args.batch_size, args.dry_run
            )
            log.info(
                "backfill_metadata.done",
                customer=customer,
                skipped=skipped,
                embedded=embedded,
                failed=failed,
            )
    finally:
        await close_pool()

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(_amain()))
