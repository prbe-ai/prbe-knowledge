"""Re-embed the metadata chunks for live Claude Code session documents.

Companion to migration 0038_backfill_cc_doc_titles. The migration rewrites
documents.title + identity metadata for live Claude Code session docs;
this script picks up the change and re-embeds the synthetic kind='metadata'
chunk for each so retrieval queries see the new identity-bearing text
(name, email, hostname).

Each doc's chunk swap is atomic: inside one transaction we INSERT the new
chunk row (kind='metadata', valid_from=NOW()) and UPDATE the prior live
row's valid_to=NOW() for the matching doc. There is no observable gap
where the metadata chunk is missing.

Usage:
    .venv/bin/python -m scripts.backfill_cc_metadata_chunks
    .venv/bin/python -m scripts.backfill_cc_metadata_chunks --dry-run
    .venv/bin/python -m scripts.backfill_cc_metadata_chunks --customer cust-X

Idempotent: re-running picks up only docs whose live metadata-chunk
content_hash differs from what _metadata_text(doc) now produces, so a
second run after the first has fully drained is a no-op.

Concurrency: a small asyncio.Semaphore limits in-flight embed calls.
Per-doc errors are logged and skipped — one bad doc never crashes the
loop. Final summary prints scanned / skipped / updated / errored.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from typing import Any

from services.ingestion.normalizer import (
    METADATA_CHUNK_INDEX,
    _chunk_hash,
    _metadata_text,
    _pg_vector,
)
from shared.config import get_settings
from shared.constants import (
    CHUNKER_VERSION,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    DocClass,
    DocType,
    SourceSystem,
)
from shared.db import close_pool, init_pool, raw_conn, with_tenant
from shared.embeddings import Embedder, get_embedder
from shared.logging import configure_logging, get_logger
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    Document,
    Permission,
    PrincipalType,
)

log = get_logger(__name__)


# Bounded concurrency for the OpenAI embed calls. 8 is enough to drain
# the ~100 live CC docs in a few seconds without spiking provider load.
_DEFAULT_CONCURRENCY = 8


def _doc_from_row(row: Any) -> Document:
    """Reconstruct a minimal Document from a documents-row.

    We only need the fields _metadata_text reads: title, source_system,
    author_id, source_url, body_preview, metadata['co_authors']. Other
    Document fields default to plausible values.
    """
    return Document(
        doc_id=row["doc_id"],
        customer_id=row["customer_id"],
        version=row["version"],
        source_system=SourceSystem(row["source_system"]),
        source_id=row["source_id"],
        source_url=row["source_url"] or "",
        doc_class=DocClass(row["doc_class"]),
        doc_type=DocType(row["doc_type"]),
        content_type=row["content_type"] or "text/plain",
        content_hash=row["content_hash"],
        title=row["title"],
        body_preview=row["body_preview"],
        body_size_bytes=row["body_size_bytes"] or 0,
        body_token_count=row["body_token_count"] or 0,
        author_id=row["author_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        valid_from=row["valid_from"],
        ingested_at=row["ingested_at"],
        metadata=row["metadata"] if isinstance(row["metadata"], dict) else {},
        acl=ACLSnapshot(
            principals=[
                ACLPrincipal(
                    principal_type=PrincipalType.USER,
                    principal_id=row["author_id"] or row["customer_id"],
                    permission=Permission.READ,
                )
            ],
            captured_at=row["created_at"],
        ),
    )


async def _list_live_cc_docs(customer: str | None) -> list[dict[str, Any]]:
    """Return every live Claude Code session doc, optionally filtered to
    a single customer. Returned as plain dicts so the caller doesn't need
    to keep a connection alive.

    Documents has neither RLS nor FORCE RLS (see migration
    0036_strip_metadata_body), so a global SELECT under raw_conn() works.
    Per-doc writes still set the tenant GUC via with_tenant().
    """
    async with raw_conn() as conn:
        # asyncpg returns metadata as bytes/str/dict depending on codec; we
        # accept whichever and coerce in _coerce_meta below at row-build time.
        params: list[Any] = []
        where = (
            "source_system = 'claude_code' "
            "AND doc_type = 'claude_code.session' "
            "AND valid_to IS NULL"
        )
        if customer:
            params.append(customer)
            where += " AND customer_id = $1"
        rows = await conn.fetch(
            f"""
            SELECT doc_id, customer_id, version, source_system, source_id,
                   source_url, doc_class, doc_type, content_type,
                   content_hash, title, body_preview, body_size_bytes,
                   body_token_count, author_id, created_at, updated_at,
                   valid_from, ingested_at, metadata
            FROM documents
            WHERE {where}
            ORDER BY customer_id, doc_id
            """,
            *params,
        )
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        meta = d.get("metadata")
        # Decode metadata if asyncpg gave us a raw json text payload.
        if isinstance(meta, (bytes, bytearray, str)):
            import orjson

            try:
                d["metadata"] = orjson.loads(meta)
            except Exception:
                d["metadata"] = {}
        out.append(d)
    return out


async def _process_doc(
    sem: asyncio.Semaphore,
    embedder: Embedder,
    doc_row: dict[str, Any],
    dry_run: bool,
) -> str:
    """Process a single doc row.

    Returns one of: "skipped" / "updated" / "errored" so the caller can
    aggregate.
    """
    customer_id: str = doc_row["customer_id"]
    doc_id: str = doc_row["doc_id"]

    try:
        doc = _doc_from_row(doc_row)
        new_text = _metadata_text(doc)
        if not new_text.strip():
            return "skipped"
        new_hash = _chunk_hash(new_text)

        # Look up the current live metadata chunk under the tenant GUC.
        # chunks has RLS and requires app.current_customer_id to be set.
        async with with_tenant(customer_id) as conn:
            existing = await conn.fetchrow(
                """
                SELECT chunk_id, content_hash
                FROM chunks
                WHERE customer_id = $1 AND doc_id = $2
                  AND kind = 'metadata' AND valid_to IS NULL
                LIMIT 1
                """,
                customer_id,
                doc_id,
            )

        if existing is not None and existing["content_hash"] == new_hash:
            # Already current; idempotent skip.
            return "skipped"

        if dry_run:
            log.info(
                "backfill_cc_metadata.dry_run",
                customer=customer_id,
                doc_id=doc_id,
                old_hash=(existing["content_hash"] if existing else None),
                new_hash=new_hash,
                preview=new_text[:120],
            )
            return "updated"

        # Embed outside any transaction (the long I/O). One semaphore slot
        # per concurrent call so we don't blast the provider with N
        # parallel requests.
        async with sem:
            embeds = await embedder.embed_many([new_text])

        if not embeds.embedded:
            log.warning(
                "backfill_cc_metadata.embed_no_result",
                customer=customer_id,
                doc_id=doc_id,
                failed=[f.error for f in embeds.failed],
            )
            return "errored"

        embedding = embeds.embedded[0].embedding

        # Atomic close+insert inside one txn under the tenant GUC.
        async with with_tenant(customer_id) as conn:
            now = datetime.now(UTC)
            # Insert new row first; identity is (doc_id, content_hash)
            # so a redelivery with the same text is a no-op via ON CONFLICT.
            new_chunk_id = f"{doc_id}:m_{new_hash[:16]}"
            await conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, customer_id,
                    chunk_index, content, content_hash, token_count,
                    embedding, embedding_model, embedding_dim,
                    chunker_version,
                    first_seen_version, last_seen_version, kind,
                    valid_from
                )
                VALUES (
                    $1, $2, $3,
                    $4, $5, $6, $7,
                    $8::halfvec, $9, $10,
                    $11,
                    $12, $12, 'metadata',
                    $13
                )
                ON CONFLICT (doc_id, content_hash) DO UPDATE
                    SET last_seen_version = EXCLUDED.last_seen_version,
                        valid_to = NULL
                """,
                new_chunk_id,
                doc_id,
                customer_id,
                METADATA_CHUNK_INDEX,
                new_text,
                new_hash,
                len(new_text.split()),  # rough token count; embedded value not used downstream for metadata chunks
                _pg_vector(embedding),
                EMBEDDING_MODEL,
                EMBEDDING_DIM,
                CHUNKER_VERSION,
                doc_row["version"],
                now,
            )
            # Close out the prior live metadata row, if it exists and
            # differs from the one we just inserted.
            if existing is not None and existing["content_hash"] != new_hash:
                await conn.execute(
                    """
                    UPDATE chunks
                    SET valid_to = $4
                    WHERE customer_id = $1 AND doc_id = $2
                      AND content_hash = $3 AND kind = 'metadata'
                      AND valid_to IS NULL
                    """,
                    customer_id,
                    doc_id,
                    existing["content_hash"],
                    now,
                )
        return "updated"
    except Exception as exc:
        # Don't crash on one bad doc — log and move on. The summary
        # surfaces the errored count so the operator sees it.
        log.exception(
            "backfill_cc_metadata.doc_failed",
            customer=customer_id,
            doc_id=doc_id,
            error=str(exc),
            error_class=type(exc).__name__,
        )
        return "errored"


async def _amain() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--customer",
        default=None,
        help="Restrict to one customer (default: every active customer's CC docs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be re-embedded without calling the embedder or DB writes",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=_DEFAULT_CONCURRENCY,
        help="Max in-flight embed calls (default 8)",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    scanned = 0
    skipped = 0
    updated = 0
    errored = 0
    try:
        docs = await _list_live_cc_docs(args.customer)
        log.info("backfill_cc_metadata.start", scanned=len(docs), dry_run=args.dry_run)

        embedder = get_embedder()
        sem = asyncio.Semaphore(max(1, args.concurrency))

        # Run with bounded concurrency. asyncio.gather schedules them all
        # concurrently; the semaphore caps live OpenAI calls.
        tasks = [
            asyncio.create_task(_process_doc(sem, embedder, d, args.dry_run))
            for d in docs
        ]
        for i, task in enumerate(asyncio.as_completed(tasks), start=1):
            outcome = await task
            scanned += 1
            if outcome == "skipped":
                skipped += 1
            elif outcome == "updated":
                updated += 1
            else:
                errored += 1
            if i % 10 == 0:
                log.info(
                    "backfill_cc_metadata.progress",
                    scanned=scanned,
                    skipped=skipped,
                    updated=updated,
                    errored=errored,
                )
    finally:
        log.info(
            "backfill_cc_metadata.done",
            scanned=scanned,
            skipped=skipped,
            updated=updated,
            errored=errored,
        )
        await close_pool()

    return 0 if errored == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(_amain()))
