"""Regression test for the `_upsert_document` concurrent-writer race (TODOS.md P1).

Prior bug: two concurrent workers normalizing events for the same `(customer_id,
doc_id)` could both read `version=N`, both compute `N+1`, one's INSERT won, the
other silently no-opped via `ON CONFLICT DO NOTHING`. The losing writer's content
was lost.

This test fires both writers via `asyncio.gather` against a real Postgres so the
race is actually exercised (each writer holds its own `with_tenant` connection
and they really do interleave on the unique index). After both complete we
assert two rows exist for the doc_id — versions N and N+1 — and both
content_hashes are preserved.

Without the retry loop in `_upsert_document`, only one row would persist; the
second writer's no-op would not raise, just silently drop content.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime

import pytest

from engine.ingest.normalizer import _upsert_document
from engine.shared.constants import DocClass, DocType, SourceSystem
from engine.shared.db import raw_conn, with_tenant
from engine.shared.models import ACLSnapshot, Document


def _make_doc(*, customer_id: str, doc_id: str, body: str) -> Document:
    """Construct a minimal in-memory Document with a content_hash derived from body.

    Two docs sharing (customer_id, doc_id) but differing in `body` will produce
    distinct content_hashes — exactly the race shape we need.
    """
    now = datetime.now(UTC)
    return Document(
        doc_id=doc_id,
        customer_id=customer_id,
        source_system=SourceSystem.SLACK,
        source_id=doc_id,
        source_url=f"https://example.test/{doc_id}",
        doc_class=DocClass.RAW_SOURCE,
        doc_type=DocType.SLACK_MESSAGE,
        content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        title=f"doc {body[:8]}",
        body_preview=body[:64],
        body_size_bytes=len(body.encode("utf-8")),
        body_token_count=len(body.split()),
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        acl=ACLSnapshot(principals=[], captured_at=now),
        metadata={"body": body},
    )


async def _seed_customer(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'test-hash')
            ON CONFLICT DO NOTHING
            """,
            customer_id,
        )


async def _upsert_in_own_txn(doc: Document) -> bool:
    """Run `_upsert_document` inside its own `with_tenant` connection.

    This mirrors how the Normalizer calls it — Phase B opens one
    `with_tenant(customer_id)` per queue row, then calls `_upsert_document`
    inside that transaction. Two concurrent gather'd callers therefore hold
    two distinct connections + transactions, which is what makes the race
    reproducible (otherwise asyncpg would serialize on a shared connection).
    """
    async with with_tenant(doc.customer_id) as conn:
        return await _upsert_document(conn, doc)


@pytest.mark.asyncio
async def test_concurrent_upsert_preserves_both_writers(live_db) -> None:
    """Two concurrent writers, same doc_id, distinct content → both rows persist.

    Reproduces the TODOS.md P1 race. Without the retry loop in
    `_upsert_document` this test fails: only one document row exists and the
    second writer's content_hash is silently dropped.
    """
    customer_id = "cust-upsert-race"
    doc_id = "slack:T:C:race-1234"
    await _seed_customer(customer_id)

    doc_a = _make_doc(customer_id=customer_id, doc_id=doc_id, body="payload from worker A")
    doc_b = _make_doc(customer_id=customer_id, doc_id=doc_id, body="payload from worker B")
    assert doc_a.content_hash != doc_b.content_hash, (
        "test setup invariant: differing bodies must hash differently"
    )

    # Both fire concurrently. Either may win the version=1 slot; the loser
    # should detect the conflict, re-read inside its own txn, and retry at
    # version=2 — instead of silently dropping its content.
    results = await asyncio.gather(
        _upsert_in_own_txn(doc_a),
        _upsert_in_own_txn(doc_b),
    )
    assert all(results), f"both upserts must report a write; got {results!r}"

    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT version, content_hash
            FROM documents
            WHERE customer_id = $1 AND doc_id = $2
            ORDER BY version
            """,
            customer_id,
            doc_id,
        )

    assert len(rows) == 2, (
        f"expected two document rows (versions 1 and 2) after the concurrent race; "
        f"got {len(rows)}: {[(r['version'], r['content_hash'][:8]) for r in rows]!r}"
    )
    versions = [r["version"] for r in rows]
    assert versions == [1, 2], f"versions should be a contiguous (1, 2); got {versions!r}"

    persisted_hashes = {r["content_hash"] for r in rows}
    expected_hashes = {doc_a.content_hash, doc_b.content_hash}
    assert persisted_hashes == expected_hashes, (
        "both writers' content_hashes must be preserved across the race; "
        f"missing {expected_hashes - persisted_hashes!r}, "
        f"unexpected {persisted_hashes - expected_hashes!r}"
    )

    # The retry loop also bumps `doc.version` on the winning writer's
    # in-memory Document — chunk-write contracts depend on it being correct.
    # The two upsert calls each set their own doc's version; one to 1, one to 2.
    in_memory_versions = sorted([doc_a.version, doc_b.version])
    assert in_memory_versions == [1, 2], (
        f"each writer must mutate its own doc.version to the winning slot; "
        f"got {in_memory_versions!r}"
    )
