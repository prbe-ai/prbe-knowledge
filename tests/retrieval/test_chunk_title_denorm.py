"""`chunks.title` must never go stale, whichever door the write came through.

Phase 2 denormalizes `documents.title` onto `chunks` so BM25 can match titles
from a single-table index. Denormalization buys speed and takes on a
consistency obligation, and the obligation is the part that rots quietly: a
wrong title does not raise, it just ranks the wrong document.

The obligation is enforced in the DATABASE (migration 0100), not in the
application, because the application is not the only writer. Backfill scripts,
alembic migrations and manual SQL all insert chunks, and none of them will
remember a column added later.

Three write paths, three different ways to get it wrong:

  1. Normal ingest -- normalizer passes the title explicitly.
  2. In-place retitle -- `UPDATE documents SET title=...` rewrites no chunks
     (normalizer.py:1327). This is the path that actually drifts in production:
     content-identical chunks survive a retitle, so without the trigger they
     keep the old title indefinitely.
  3. A writer that does not know the column exists -- inserts chunks with no
     title at all.

The retitle case has a subtlety worth stating: 19,305 chunks on the managed
data plane span MORE THAN ONE document version (last_seen_version >
first_seen_version), so "the document's title" is not a single value for them.
The rule is newest-version-wins, which is what a searcher means by "the title".
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engine.shared.db import raw_conn

_NOW = datetime.now(UTC)


async def _seed_doc(customer_id: str, doc_id: str, title: str, version: int = 1) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at, acl
            ) VALUES (
                $1, $4, $2,
                'manual_upload', $1, 'https://prbe.ai/' || $1,
                'raw_source', 'manual_upload.note', 'text/plain',
                'h-' || $1, $3, 100, 0, $5, $5, $5, $5, '{}'::jsonb
            )
            """,
            doc_id, customer_id, title, version, _NOW,
        )


async def _insert_chunk(
    customer_id: str,
    doc_id: str,
    idx: int,
    body: str,
    *,
    title: str | None,
    first_version: int = 1,
    last_version: int = 1,
) -> None:
    """Insert a chunk, optionally WITHOUT a title (path 3)."""
    async with raw_conn() as conn:
        if title is None:
            await conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, customer_id, chunk_index, content,
                    content_hash, token_count, kind, embedding,
                    first_seen_version, last_seen_version
                ) VALUES ($1,$2,$3,$4,$5,$6,5,'content',
                    array_fill(0::real, ARRAY[3072])::halfvec, $7, $8)
                """,
                f"{doc_id}-c{idx}", doc_id, customer_id, idx, body,
                f"chash-{doc_id}-{idx}", first_version, last_version,
            )
        else:
            await conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, customer_id, chunk_index, content,
                    content_hash, token_count, kind, embedding,
                    first_seen_version, last_seen_version, title
                ) VALUES ($1,$2,$3,$4,$5,$6,5,'content',
                    array_fill(0::real, ARRAY[3072])::halfvec, $7, $8, $9)
                """,
                f"{doc_id}-c{idx}", doc_id, customer_id, idx, body,
                f"chash-{doc_id}-{idx}", first_version, last_version, title,
            )


async def _titles(doc_id: str) -> list[str]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT title FROM chunks WHERE doc_id = $1 ORDER BY chunk_index", doc_id
        )
    return [r["title"] for r in rows]


@pytest.mark.asyncio
async def test_chunk_inserted_without_a_title_is_filled_from_the_document(
    pg_search_db,
) -> None:
    """Path 3: a writer that predates the column.

    This is the one an application-level contract cannot cover. Backfill
    scripts and migrations insert chunks directly; the BEFORE INSERT trigger is
    what makes a title-less chunk unrepresentable rather than merely
    discouraged.
    """
    cust, doc = "cust-title-denorm-fill", "cust-title-denorm-fill-doc"
    await _seed_doc(cust, doc, "Original Title")
    await _insert_chunk(cust, doc, 0, "body text", title=None)

    assert await _titles(doc) == ["Original Title"]


@pytest.mark.asyncio
async def test_in_place_retitle_restamps_existing_chunks(pg_search_db) -> None:
    """Path 2: the drift path, and the reason a test alone was not enough.

    normalizer.py updates documents.title in place and rewrites no chunks, so
    every chunk of the document would keep the stale title until its CONTENT
    changed -- which for an unedited document is never.
    """
    cust, doc = "cust-title-denorm-retitle", "cust-title-denorm-retitle-doc"
    await _seed_doc(cust, doc, "Before")
    for i in range(3):
        await _insert_chunk(cust, doc, i, f"body {i}", title="Before")
    assert await _titles(doc) == ["Before"] * 3

    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE documents SET title = $2 WHERE doc_id = $1", doc, "After"
        )

    assert await _titles(doc) == ["After"] * 3, (
        "an in-place retitle left chunks holding the old title -- the "
        "documents-title trigger from 0100 is not firing"
    )


@pytest.mark.asyncio
async def test_retitle_restamps_chunks_spanning_multiple_versions(pg_search_db) -> None:
    """19,305 chunks on the managed plane outlive a document version.

    A chunk with last_seen_version > first_seen_version is shared across
    versions, so the trigger's `NEW.version BETWEEN first_seen AND last_seen`
    predicate is what decides whether it gets re-stamped. Pinning it here
    because an off-by-one in that range would silently skip exactly the 9% of
    chunks most likely to be stale.
    """
    cust, doc = "cust-title-denorm-span", "cust-title-denorm-span-doc"
    await _seed_doc(cust, doc, "V1 Title", version=1)
    await _insert_chunk(
        cust, doc, 0, "spanning body", title="V1 Title",
        first_version=1, last_version=3,
    )
    await _seed_doc(cust, doc, "V2 Title", version=2)

    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE documents SET title = $2 WHERE doc_id = $1 AND version = 2",
            doc, "V2 Retitled",
        )

    assert await _titles(doc) == ["V2 Retitled"], (
        "a chunk spanning versions 1-3 was not re-stamped when version 2 was "
        "retitled -- check the BETWEEN in chunks_sync_title_from_document"
    )


@pytest.mark.asyncio
async def test_title_is_never_null(pg_search_db) -> None:
    """'' is the honest value for a document with no title; NULL is not.

    A nullable column would push a `coalesce` into every read, and the read
    that forgets it is a silent wrong-answer rather than an error.
    """
    cust, doc = "cust-title-denorm-null", "cust-title-denorm-null-doc"
    await _seed_doc(cust, doc, "")
    await _insert_chunk(cust, doc, 0, "body", title=None)

    async with raw_conn() as conn:
        nulls = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE doc_id = $1 AND title IS NULL", doc
        )
    assert nulls == 0
