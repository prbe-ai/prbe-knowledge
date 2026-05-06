"""Integration tests for `persist_links_for_page`.

Verifies the delete-then-insert behavior against the live Postgres
schema (migrations 0044 + 0045 already applied via the live_db
fixture):

  - 3 markdown links + 1 frontmatter link -> 4 rows
  - Re-call with a different set replaces the prior set
  - 'manual' rows survive replacement
  - Re-inserting the same set is idempotent (ON CONFLICT DO NOTHING)
"""

from __future__ import annotations

import pytest

from services.synthesis.wiki_links import (
    ExtractedLink,
    persist_links_for_page,
)
from shared.db import raw_conn, with_tenant

_CUSTOMER = "lane-b-cust"
_SRC_TYPE = "service_card"
_SRC_SLUG = "auth"


async def _setup_customer() -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'lane-b', 'h') ON CONFLICT DO NOTHING",
            _CUSTOMER,
        )


async def _count_rows(*, link_source: str | None = None) -> int:
    async with raw_conn() as conn:
        if link_source is None:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM wiki_links "
                "WHERE customer_id = $1 AND src_wiki_type = $2 AND src_slug = $3",
                _CUSTOMER,
                _SRC_TYPE,
                _SRC_SLUG,
            )
        return await conn.fetchval(
            "SELECT COUNT(*) FROM wiki_links "
            "WHERE customer_id = $1 AND src_wiki_type = $2 AND src_slug = $3 "
            "AND link_source = $4",
            _CUSTOMER,
            _SRC_TYPE,
            _SRC_SLUG,
            link_source,
        )


@pytest.mark.asyncio
async def test_persist_inserts_extracted_links(live_db) -> None:
    await _setup_customer()
    extracted = [
        ExtractedLink(
            dst_wiki_type="person",
            dst_slug="maison",
            link_type="works_at",
            context="ctx-1",
            link_source="markdown",
        ),
        ExtractedLink(
            dst_wiki_type="decision",
            dst_slug="auth-rollback",
            link_type="",
            context="ctx-2",
            link_source="markdown",
        ),
        ExtractedLink(
            dst_wiki_type="event",
            dst_slug="2026-05-05-1on1",
            link_type="",
            context="ctx-3",
            link_source="markdown",
        ),
        ExtractedLink(
            dst_wiki_type="company",
            dst_slug="probe",
            link_type="works_at",
            context="",
            link_source="frontmatter",
        ),
    ]
    async with with_tenant(_CUSTOMER) as conn:
        await persist_links_for_page(
            conn,
            customer_id=_CUSTOMER,
            src_wiki_type=_SRC_TYPE,
            src_slug=_SRC_SLUG,
            extracted=extracted,
        )
    assert await _count_rows() == 4
    assert await _count_rows(link_source="markdown") == 3
    assert await _count_rows(link_source="frontmatter") == 1


@pytest.mark.asyncio
async def test_persist_replaces_prior_extracted_set(live_db) -> None:
    await _setup_customer()
    first = [
        ExtractedLink(
            dst_wiki_type="person",
            dst_slug="maison",
            link_type="works_at",
            context="",
            link_source="markdown",
        ),
        ExtractedLink(
            dst_wiki_type="company",
            dst_slug="probe",
            link_type="works_at",
            context="",
            link_source="frontmatter",
        ),
    ]
    async with with_tenant(_CUSTOMER) as conn:
        await persist_links_for_page(
            conn,
            customer_id=_CUSTOMER,
            src_wiki_type=_SRC_TYPE,
            src_slug=_SRC_SLUG,
            extracted=first,
        )
    assert await _count_rows() == 2

    second = [
        ExtractedLink(
            dst_wiki_type="person",
            dst_slug="someone-else",
            link_type="reports_to",
            context="",
            link_source="markdown",
        ),
    ]
    async with with_tenant(_CUSTOMER) as conn:
        await persist_links_for_page(
            conn,
            customer_id=_CUSTOMER,
            src_wiki_type=_SRC_TYPE,
            src_slug=_SRC_SLUG,
            extracted=second,
        )
    # Old markdown + frontmatter rows replaced by the new single row.
    assert await _count_rows() == 1
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT dst_slug, link_type FROM wiki_links "
            "WHERE customer_id = $1 AND src_wiki_type = $2 AND src_slug = $3",
            _CUSTOMER,
            _SRC_TYPE,
            _SRC_SLUG,
        )
    assert row["dst_slug"] == "someone-else"
    assert row["link_type"] == "reports_to"


@pytest.mark.asyncio
async def test_manual_links_survive_replacement(live_db) -> None:
    await _setup_customer()
    # Pre-seed a manual link directly via SQL — simulates the human-curated path.
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO wiki_links (
                customer_id, src_wiki_type, src_slug,
                dst_wiki_type, dst_slug, link_type, context, link_source
            )
            VALUES ($1, $2, $3, 'person', 'human-edit', 'pinned', '', 'manual')
            """,
            _CUSTOMER,
            _SRC_TYPE,
            _SRC_SLUG,
        )
    extracted = [
        ExtractedLink(
            dst_wiki_type="person",
            dst_slug="maison",
            link_type="works_at",
            context="",
            link_source="markdown",
        ),
    ]
    async with with_tenant(_CUSTOMER) as conn:
        await persist_links_for_page(
            conn,
            customer_id=_CUSTOMER,
            src_wiki_type=_SRC_TYPE,
            src_slug=_SRC_SLUG,
            extracted=extracted,
        )
    assert await _count_rows(link_source="manual") == 1
    assert await _count_rows(link_source="markdown") == 1

    # A second persist call with an empty extracted set must still keep manual.
    async with with_tenant(_CUSTOMER) as conn:
        await persist_links_for_page(
            conn,
            customer_id=_CUSTOMER,
            src_wiki_type=_SRC_TYPE,
            src_slug=_SRC_SLUG,
            extracted=[],
        )
    assert await _count_rows(link_source="manual") == 1
    assert await _count_rows(link_source="markdown") == 0


@pytest.mark.asyncio
async def test_re_persist_same_set_is_idempotent(live_db) -> None:
    """ON CONFLICT DO NOTHING absorbs re-inserts of identical rows."""
    await _setup_customer()
    extracted = [
        ExtractedLink(
            dst_wiki_type="person",
            dst_slug="maison",
            link_type="works_at",
            context="",
            link_source="markdown",
        ),
    ]
    for _ in range(3):
        async with with_tenant(_CUSTOMER) as conn:
            await persist_links_for_page(
                conn,
                customer_id=_CUSTOMER,
                src_wiki_type=_SRC_TYPE,
                src_slug=_SRC_SLUG,
                extracted=extracted,
            )
    assert await _count_rows() == 1
