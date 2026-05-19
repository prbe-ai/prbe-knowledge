"""Regression: persist_single_document threads visibility through to
both ``documents`` and ``chunks`` rows.

Live Postgres required. Builds a Document directly (no chunker round-
trip beyond what persist_single_document does anyway) and asserts the
``visibility`` column lands as expected on both tables. The default
(APPROVED) is verified independently from the explicit DRAFT case so
existing callers — who never pass a ``visibility`` field — stay
protected from accidental regressions.

The embedder is stubbed out: the chunk-plan path inside
``Normalizer._plan_chunks`` calls the live embedder, which we don't
need (and can't reach in CI). A no-op embedder that returns a single
zero-vector per piece is enough — the test asserts on column values,
not on retrieval semantics.
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio

from services.ingestion.handlers.base import make_default_context
from services.ingestion.normalizer import Normalizer
from shared import db as db_module
from shared.constants import (
    DocClass,
    DocType,
    EMBEDDING_V2_DIM,
    Permission,
    PrincipalType,
    SourceSystem,
    Visibility,
)
from shared.embeddings import EmbeddedChunk, EmbedResult
from shared.models import ACLPrincipal, ACLSnapshot, Document

pytestmark = pytest.mark.asyncio


def _skip_if_no_db() -> None:
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")


def _new_customer_id() -> str:
    return f"viz-test-{uuid.uuid4().hex[:8]}"


async def _seed_customer(customer_id: str) -> None:
    import asyncpg
    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, $2, $3) ON CONFLICT (customer_id) DO NOTHING",
            customer_id, f"test {customer_id}", "h",
        )
    finally:
        await conn.close()


async def _cleanup_customer(customer_id: str) -> None:
    import asyncpg
    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute(
            "DELETE FROM chunks WHERE customer_id = $1", customer_id,
        )
        await conn.execute(
            "DELETE FROM documents WHERE customer_id = $1", customer_id,
        )
        await conn.execute(
            "DELETE FROM customers WHERE customer_id = $1", customer_id,
        )
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def customer_id():
    _skip_if_no_db()
    db_module.reset_pool()
    await db_module.init_pool()
    cid = _new_customer_id()
    await _seed_customer(cid)
    try:
        yield cid
    finally:
        await _cleanup_customer(cid)
        await db_module.close_pool()


class _StubEmbedder:
    """Returns a deterministic zero-vector for every input piece.

    The normalizer only cares that ``EmbedResult.embedded`` has one entry
    per input index; we don't want to call the live Gemini API in tests.
    """

    async def embed_documents(self, items) -> EmbedResult:
        embedded = [
            EmbeddedChunk(chunk_index=i, embedding=[0.0] * EMBEDDING_V2_DIM)
            for i, _ in enumerate(items)
        ]
        return EmbedResult(embedded=embedded, failed=[])


def _build_doc(
    customer_id: str,
    *,
    doc_id: str,
    visibility: Visibility,
) -> Document:
    now = datetime.now(UTC)
    acl = ACLSnapshot(
        principals=[
            ACLPrincipal(
                principal_type=PrincipalType.WORKSPACE,
                principal_id=customer_id,
                permission=Permission.READ,
            ),
        ],
        captured_at=now,
    )
    return Document(
        doc_id=doc_id,
        customer_id=customer_id,
        source_system=SourceSystem.PAGERDUTY,
        source_id=doc_id,
        source_url="",
        doc_class=DocClass.AGENT_ARTIFACT,
        doc_type=DocType.WIKI_POSTMORTEM,
        content_type="text/markdown",
        content_hash=f"hash-{doc_id}",
        title=f"Test artifact {doc_id}",
        body="Body paragraph one.\n\nBody paragraph two.",
        body_preview="Body paragraph one.",
        body_size_bytes=64,
        body_token_count=12,
        parent_doc_id=None,
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        acl=acl,
        metadata={},
        visibility=visibility,
    )


async def test_persist_single_document_defaults_visibility_to_approved(
    customer_id: str,
) -> None:
    """Backwards-compat: Document() without an explicit visibility lands
    as approved on both rows — identical to pre-feature behavior."""
    ctx = make_default_context()
    try:
        normalizer = Normalizer(ctx, embedder=_StubEmbedder())
        doc = _build_doc(
            customer_id,
            doc_id="pd:wiki.postmortem:default-test:v1",
            visibility=Visibility.APPROVED,
        )
        await normalizer.persist_single_document(customer_id, doc)

        import asyncpg
        conn = await asyncpg.connect(dsn=os.environ["DATABASE_URL"])
        try:
            doc_row = await conn.fetchrow(
                "SELECT visibility FROM documents "
                "WHERE customer_id = $1 AND doc_id = $2 "
                "AND valid_to IS NULL",
                customer_id, doc.doc_id,
            )
            chunk_rows = await conn.fetch(
                "SELECT visibility FROM chunks "
                "WHERE customer_id = $1 AND doc_id = $2 "
                "AND valid_to IS NULL",
                customer_id, doc.doc_id,
            )
        finally:
            await conn.close()

        assert doc_row is not None
        assert doc_row["visibility"] == "approved"
        assert len(chunk_rows) > 0
        assert all(r["visibility"] == "approved" for r in chunk_rows)
    finally:
        await ctx.http.aclose()


async def test_persist_single_document_threads_draft_visibility(
    customer_id: str,
) -> None:
    """Explicit DRAFT lands on both tables. Every chunk row (content
    AND metadata) reflects the draft state — the approve path will
    flip all of them in one transaction."""
    ctx = make_default_context()
    try:
        normalizer = Normalizer(ctx, embedder=_StubEmbedder())
        doc = _build_doc(
            customer_id,
            doc_id="pd:wiki.postmortem:draft-test:v1",
            visibility=Visibility.DRAFT,
        )
        await normalizer.persist_single_document(customer_id, doc)

        import asyncpg
        conn = await asyncpg.connect(dsn=os.environ["DATABASE_URL"])
        try:
            doc_row = await conn.fetchrow(
                "SELECT visibility FROM documents "
                "WHERE customer_id = $1 AND doc_id = $2 "
                "AND valid_to IS NULL",
                customer_id, doc.doc_id,
            )
            chunk_rows = await conn.fetch(
                "SELECT visibility, kind FROM chunks "
                "WHERE customer_id = $1 AND doc_id = $2 "
                "AND valid_to IS NULL",
                customer_id, doc.doc_id,
            )
        finally:
            await conn.close()

        assert doc_row is not None
        assert doc_row["visibility"] == "draft"
        assert len(chunk_rows) > 0
        assert all(r["visibility"] == "draft" for r in chunk_rows)
    finally:
        await ctx.http.aclose()
