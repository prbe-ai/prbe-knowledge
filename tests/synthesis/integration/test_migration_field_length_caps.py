"""Migration assertions for the wiki field-length caps (migration 0045).

Two CHECK constraints to keep agent-generated text from overflowing
btree key limits and from misusing summary as a long-form field:

- wiki_timeline_entries.summary <= 1000 chars
- wiki_links.context <= 200 chars
"""

from __future__ import annotations

import pytest
from asyncpg.exceptions import CheckViolationError

from shared.db import raw_conn


@pytest.mark.asyncio
async def test_wiki_timeline_summary_cap_rejects_overlong(live_db) -> None:
    """summary > 1000 chars fails the CHECK constraint."""
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'mig-45-tl', 'h') ON CONFLICT DO NOTHING",
            "mig-45-tl",
        )
        try:
            with pytest.raises(CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO wiki_timeline_entries (
                        customer_id, wiki_type, slug, entry_date,
                        source, summary
                    )
                    VALUES ($1, 'service_card', 'auth', '2026-05-01',
                            'github', $2)
                    """,
                    "mig-45-tl",
                    "x" * 1001,
                )
        finally:
            await conn.execute("DELETE FROM customers WHERE customer_id = $1", "mig-45-tl")


@pytest.mark.asyncio
async def test_wiki_timeline_summary_cap_accepts_at_limit(live_db) -> None:
    """summary at exactly 1000 chars is accepted."""
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'mig-45-tl-ok', 'h') ON CONFLICT DO NOTHING",
            "mig-45-tl-ok",
        )
        try:
            await conn.execute(
                """
                INSERT INTO wiki_timeline_entries (
                    customer_id, wiki_type, slug, entry_date,
                    source, summary
                )
                VALUES ($1, 'service_card', 'auth', '2026-05-01',
                        'github', $2)
                """,
                "mig-45-tl-ok",
                "y" * 1000,
            )
        finally:
            await conn.execute(
                "DELETE FROM wiki_timeline_entries WHERE customer_id = $1",
                "mig-45-tl-ok",
            )
            await conn.execute("DELETE FROM customers WHERE customer_id = $1", "mig-45-tl-ok")


@pytest.mark.asyncio
async def test_wiki_links_context_cap_rejects_overlong(live_db) -> None:
    """context > 200 chars fails the CHECK constraint."""
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'mig-45-ln', 'h') ON CONFLICT DO NOTHING",
            "mig-45-ln",
        )
        try:
            with pytest.raises(CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO wiki_links (
                        customer_id, src_wiki_type, src_slug,
                        dst_wiki_type, dst_slug, link_source, context
                    )
                    VALUES ($1, 'service_card', 'auth',
                            'person', 'maison', 'markdown', $2)
                    """,
                    "mig-45-ln",
                    "x" * 201,
                )
        finally:
            await conn.execute("DELETE FROM customers WHERE customer_id = $1", "mig-45-ln")


@pytest.mark.asyncio
async def test_wiki_links_context_cap_accepts_at_limit(live_db) -> None:
    """context at exactly 200 chars is accepted."""
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'mig-45-ln-ok', 'h') ON CONFLICT DO NOTHING",
            "mig-45-ln-ok",
        )
        try:
            await conn.execute(
                """
                INSERT INTO wiki_links (
                    customer_id, src_wiki_type, src_slug,
                    dst_wiki_type, dst_slug, link_source, context
                )
                VALUES ($1, 'service_card', 'auth',
                        'person', 'maison', 'markdown', $2)
                """,
                "mig-45-ln-ok",
                "y" * 200,
            )
        finally:
            await conn.execute("DELETE FROM wiki_links WHERE customer_id = $1", "mig-45-ln-ok")
            await conn.execute("DELETE FROM customers WHERE customer_id = $1", "mig-45-ln-ok")
