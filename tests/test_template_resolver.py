"""Tests for the postmortem template resolver.

Live Postgres required (DATABASE_URL must point at a running instance
with migrations through 0085 applied). Each test allocates a unique
customer_id and cleans up after itself.

doc_ref tests seed a wiki document by inserting directly into
``documents`` + ``chunks`` (visibility='approved'). This matches the
real ingestion path's storage shape (documents has no ``body`` column;
content lives in chunks).
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest

from services.post_approval.template_resolver import (
    get_effective_template,
    get_override,
    upsert_override,
)
from shared import db as db_module
from shared.schemas.postmortem_template import TemplateUpsertRequest
from shared.templates.postmortem import DEFAULT_POSTMORTEM_TEMPLATE

pytestmark = pytest.mark.asyncio


def _skip_if_no_db() -> None:
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")


def _new_customer_id() -> str:
    return f"tmpl-test-{uuid.uuid4().hex[:8]}"


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


async def _seed_doc_with_body(
    customer_id: str, doc_id: str, body: str,
) -> None:
    """Insert a single-version approved wiki doc with one content chunk.

    Body lives in chunks.content (no documents.body column in this
    schema); we keep it to a single chunk for simplicity — the resolver
    handles N-chunk reassembly the same way.
    """
    import asyncpg
    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        now = datetime.now(UTC)
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at,
                acl, visibility
            ) VALUES (
                $1, 1, $2,
                'wiki', $1, 'https://example/wiki',
                'wiki_page', 'wiki_page', 'text/markdown',
                $3, $4, $5, 0,
                $6, $6, $6, $6,
                '{}'::jsonb, 'approved'
            )
            """,
            doc_id, customer_id,
            f"hash-{doc_id}", f"Template doc {doc_id}",
            len(body.encode()), now,
        )
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count,
                first_seen_version, last_seen_version,
                visibility
            ) VALUES (
                $1, $2, $3,
                0, $4, $5, 5,
                1, 1,
                'approved'
            )
            """,
            f"{doc_id}:c0:v1", doc_id, customer_id,
            body, f"chunk-hash-{doc_id}",
        )
    finally:
        await conn.close()


async def _seed_doc_with_mixed_visibility(
    customer_id: str, doc_id: str, body: str,
    *, doc_visibility: str, chunk_visibility: str,
) -> None:
    """Insert a single-version wiki doc + one content chunk at the given
    per-row visibilities. Used to verify the resolver gates BOTH sides
    of the join — a doc whose row is ``approved`` but whose chunks are
    still ``draft`` (e.g. mid-flip race) must NOT render.
    """
    import asyncpg
    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        now = datetime.now(UTC)
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at,
                acl, visibility
            ) VALUES (
                $1, 1, $2,
                'wiki', $1, 'https://example/wiki',
                'wiki_page', 'wiki_page', 'text/markdown',
                $3, $4, $5, 0,
                $6, $6, $6, $6,
                '{}'::jsonb, $7
            )
            """,
            doc_id, customer_id,
            f"hash-{doc_id}", f"Template doc {doc_id}",
            len(body.encode()), now, doc_visibility,
        )
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count,
                first_seen_version, last_seen_version,
                visibility
            ) VALUES (
                $1, $2, $3,
                0, $4, $5, 5,
                1, 1,
                $6
            )
            """,
            f"{doc_id}:c0:v1", doc_id, customer_id,
            body, f"chunk-hash-{doc_id}", chunk_visibility,
        )
    finally:
        await conn.close()


async def _cleanup_customer(customer_id: str) -> None:
    import asyncpg
    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute(
            "DELETE FROM customer_postmortem_templates WHERE customer_id = $1",
            customer_id,
        )
        # FK cascade on customers handles documents/chunks deletion.
        await conn.execute(
            "DELETE FROM customers WHERE customer_id = $1", customer_id,
        )
    finally:
        await conn.close()


@pytest.fixture
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


async def test_no_override_returns_default(customer_id: str) -> None:
    resp = await get_effective_template(customer_id)
    assert resp.source == "default"
    assert resp.body_markdown == DEFAULT_POSTMORTEM_TEMPLATE
    assert resp.resolved_ref_doc_id is None


async def test_inline_override_returned(customer_id: str) -> None:
    body = "# Custom Template\n\n{{summary}}\n"
    await upsert_override(
        TemplateUpsertRequest(
            customer_id=customer_id,
            mode="inline",
            body_markdown=body,
        )
    )
    resp = await get_effective_template(customer_id)
    assert resp.source == "inline_override"
    assert resp.body_markdown == body
    assert resp.resolved_ref_doc_id is None


async def test_doc_ref_override_fetches_doc_body(customer_id: str) -> None:
    body = "# Wiki-managed template\n\n## Custom slot: {{summary}}\n"
    await _seed_doc_with_body(customer_id, "wiki:postmortem-template", body)
    await upsert_override(
        TemplateUpsertRequest(
            customer_id=customer_id,
            mode="doc_ref",
            ref_doc_id="wiki:postmortem-template",
        )
    )
    resp = await get_effective_template(customer_id)
    assert resp.source == "doc_ref_override"
    assert resp.body_markdown == body
    assert resp.resolved_ref_doc_id == "wiki:postmortem-template"


async def test_doc_ref_approved_doc_but_draft_chunks_falls_back_to_default(
    customer_id: str,
) -> None:
    """A doc row whose ``visibility='approved'`` but whose chunks are
    still ``visibility='draft'`` (an inconsistent transient state — e.g.
    the approve transaction failed between the documents UPDATE and the
    chunks UPDATE) must NOT render draft chunk content.

    The chunks visibility gate in ``_fetch_doc_body`` matches the
    docs-side gate so the resolver finds 0 matching chunks and falls
    through to ``DEFAULT_POSTMORTEM_TEMPLATE`` rather than leak the
    draft body. Regression for the chunks-side filter; without it the
    template would silently render with draft content.
    """
    doc_id = "wiki:template-ref-mixed"
    # Doc row is approved (so upsert_override's validation passes) but
    # the chunks are still draft.
    await _seed_doc_with_mixed_visibility(
        customer_id, doc_id, body="DRAFT chunk content",
        doc_visibility="approved", chunk_visibility="draft",
    )
    await upsert_override(
        TemplateUpsertRequest(
            customer_id=customer_id,
            mode="doc_ref",
            ref_doc_id=doc_id,
        )
    )

    resp = await get_effective_template(customer_id)
    # The chunks-side visibility filter returns 0 rows -> _fetch_doc_body
    # returns None -> resolver falls back to default.
    assert resp.source == "default"
    assert resp.body_markdown == DEFAULT_POSTMORTEM_TEMPLATE
    assert resp.resolved_ref_doc_id is None
    # And the leaked draft body never reached the rendered template.
    assert "DRAFT chunk content" not in resp.body_markdown


async def test_doc_ref_unresolved_falls_back_to_default(
    customer_id: str,
) -> None:
    # Seed the doc so the upsert validation passes, then delete it to
    # simulate a doc that became unreadable after the override was set.
    await _seed_doc_with_body(customer_id, "wiki:will-vanish", "body")
    await upsert_override(
        TemplateUpsertRequest(
            customer_id=customer_id,
            mode="doc_ref",
            ref_doc_id="wiki:will-vanish",
        )
    )
    # Delete the doc + its chunks so the resolver's lookup misses.
    import asyncpg
    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute(
            "DELETE FROM chunks WHERE customer_id = $1 AND doc_id = $2",
            customer_id, "wiki:will-vanish",
        )
        await conn.execute(
            "DELETE FROM documents WHERE customer_id = $1 AND doc_id = $2",
            customer_id, "wiki:will-vanish",
        )
    finally:
        await conn.close()

    resp = await get_effective_template(customer_id)
    assert resp.source == "default"
    assert resp.body_markdown == DEFAULT_POSTMORTEM_TEMPLATE
    assert resp.resolved_ref_doc_id is None


async def test_get_override_returns_none_when_absent(
    customer_id: str,
) -> None:
    assert await get_override(customer_id) is None


async def test_upsert_doc_ref_validates_target_doc_exists(
    customer_id: str,
) -> None:
    with pytest.raises(ValueError, match="not readable for customer"):
        await upsert_override(
            TemplateUpsertRequest(
                customer_id=customer_id,
                mode="doc_ref",
                ref_doc_id="wiki:does-not-exist",
            )
        )


async def test_upsert_doc_ref_succeeds_when_target_exists(
    customer_id: str,
) -> None:
    await _seed_doc_with_body(customer_id, "wiki:exists", "body content")
    row = await upsert_override(
        TemplateUpsertRequest(
            customer_id=customer_id,
            mode="doc_ref",
            ref_doc_id="wiki:exists",
        )
    )
    assert row.mode == "doc_ref"
    assert row.ref_doc_id == "wiki:exists"
    assert row.body_markdown is None


async def test_upsert_inline_does_not_validate_ref(
    customer_id: str,
) -> None:
    # No seeded doc; inline mode should not touch documents at all.
    row = await upsert_override(
        TemplateUpsertRequest(
            customer_id=customer_id,
            mode="inline",
            body_markdown="# inline body",
        )
    )
    assert row.mode == "inline"
    assert row.body_markdown == "# inline body"
    assert row.ref_doc_id is None
