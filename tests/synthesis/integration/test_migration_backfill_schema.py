"""Migration assertions for the wiki bootstrap schema (migration 0043).

Tests the post-upgrade shape: 3 new tables (wiki_links,
wiki_timeline_entries, wiki_raw_data), wiki_synthesis_runs.kind accepts
'bootstrap', wiki_synthesis_runs.source column exists, composite index
idx_wsr_kind_source exists, and the unique-with-NULLS-NOT-DISTINCT
collapses duplicate edges on wiki_links.

Same caveat as test_migration_wiki_v4: doesn't reapply the migration —
just inspects the post-upgrade shape via the live_db fixture's
containerized Postgres.
"""

from __future__ import annotations

import json

import pytest
from asyncpg.exceptions import CheckViolationError, UniqueViolationError

from engine.shared.db import raw_conn


@pytest.mark.asyncio
async def test_wiki_links_columns(live_db) -> None:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'wiki_links'
            """
        )
    cols = {r["column_name"]: (r["data_type"], r["is_nullable"]) for r in rows}
    assert cols["customer_id"][1] == "NO"
    assert cols["src_wiki_type"][1] == "NO"
    assert cols["src_slug"][1] == "NO"
    assert cols["dst_wiki_type"][1] == "NO"
    assert cols["dst_slug"][1] == "NO"
    assert cols["link_type"] == ("text", "NO")
    assert cols["context"] == ("text", "NO")
    assert cols["link_source"] == ("text", "NO")


@pytest.mark.asyncio
async def test_wiki_links_indexes(live_db) -> None:
    async with raw_conn() as conn:
        names = {
            r["indexname"]
            for r in await conn.fetch(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'wiki_links'"
            )
        }
    assert "ix_wiki_links_from" in names
    assert "ix_wiki_links_to" in names


@pytest.mark.asyncio
async def test_wiki_links_link_source_check(live_db) -> None:
    """The link_source CHECK rejects values outside the allowed set."""
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'mig-43-cust', 'h') ON CONFLICT DO NOTHING",
            "mig-43-cust",
        )
        try:
            with pytest.raises(CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO wiki_links (
                        customer_id, src_wiki_type, src_slug,
                        dst_wiki_type, dst_slug, link_source
                    )
                    VALUES ($1, 'service_card', 'auth',
                            'person', 'maison', 'invalid_source')
                    """,
                    "mig-43-cust",
                )
        finally:
            await conn.execute("DELETE FROM customers WHERE customer_id = $1", "mig-43-cust")


@pytest.mark.asyncio
async def test_wiki_links_unique_collapses_duplicates(live_db) -> None:
    """The UNIQUE NULLS NOT DISTINCT collapses re-inserts of the same edge."""
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'mig-43-uq', 'h') ON CONFLICT DO NOTHING",
            "mig-43-uq",
        )
        try:
            await conn.execute(
                """
                INSERT INTO wiki_links (
                    customer_id, src_wiki_type, src_slug,
                    dst_wiki_type, dst_slug, link_type, link_source
                )
                VALUES ($1, 'service_card', 'auth',
                        'person', 'maison', 'works_at', 'markdown')
                """,
                "mig-43-uq",
            )
            with pytest.raises(UniqueViolationError):
                await conn.execute(
                    """
                    INSERT INTO wiki_links (
                        customer_id, src_wiki_type, src_slug,
                        dst_wiki_type, dst_slug, link_type, link_source
                    )
                    VALUES ($1, 'service_card', 'auth',
                            'person', 'maison', 'works_at', 'markdown')
                    """,
                    "mig-43-uq",
                )
        finally:
            await conn.execute("DELETE FROM wiki_links WHERE customer_id = $1", "mig-43-uq")
            await conn.execute("DELETE FROM customers WHERE customer_id = $1", "mig-43-uq")


@pytest.mark.asyncio
async def test_wiki_timeline_entries_shape(live_db) -> None:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'wiki_timeline_entries'
            """
        )
    cols = {r["column_name"]: (r["data_type"], r["is_nullable"]) for r in rows}
    assert cols["entry_date"][0] == "date"
    assert cols["entry_date"][1] == "NO"
    assert cols["source"] == ("text", "NO")
    assert cols["summary"] == ("text", "NO")
    assert cols["detail"] == ("text", "NO")
    assert cols["source_ref"] == ("text", "YES")


@pytest.mark.asyncio
async def test_wiki_timeline_dedup(live_db) -> None:
    """uq_wiki_timeline_dedup blocks duplicate (cust, type, slug, date, summary)."""
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'mig-43-tl', 'h') ON CONFLICT DO NOTHING",
            "mig-43-tl",
        )
        try:
            await conn.execute(
                """
                INSERT INTO wiki_timeline_entries (
                    customer_id, wiki_type, slug, entry_date,
                    source, summary
                )
                VALUES ($1, 'service_card', 'auth', '2026-05-01',
                        'github', 'PR #42 merged')
                """,
                "mig-43-tl",
            )
            with pytest.raises(UniqueViolationError):
                await conn.execute(
                    """
                    INSERT INTO wiki_timeline_entries (
                        customer_id, wiki_type, slug, entry_date,
                        source, summary
                    )
                    VALUES ($1, 'service_card', 'auth', '2026-05-01',
                            'github', 'PR #42 merged')
                    """,
                    "mig-43-tl",
                )
        finally:
            await conn.execute(
                "DELETE FROM wiki_timeline_entries WHERE customer_id = $1",
                "mig-43-tl",
            )
            await conn.execute("DELETE FROM customers WHERE customer_id = $1", "mig-43-tl")


@pytest.mark.asyncio
async def test_wiki_raw_data_shape_and_dedup(live_db) -> None:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'wiki_raw_data'
            """
        )
    cols = {r["column_name"]: (r["data_type"], r["is_nullable"]) for r in rows}
    assert cols["data"][0] == "jsonb"
    assert cols["source_ref"] == ("text", "NO")

    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'mig-43-raw', 'h') ON CONFLICT DO NOTHING",
            "mig-43-raw",
        )
        try:
            await conn.execute(
                """
                INSERT INTO wiki_raw_data (
                    customer_id, wiki_type, slug, source, source_ref, data
                )
                VALUES ($1, 'service_card', 'auth',
                        'github', 'PR-42', $2::jsonb)
                """,
                "mig-43-raw",
                json.dumps({"id": 42, "title": "auth refactor"}),
            )
            with pytest.raises(UniqueViolationError):
                await conn.execute(
                    """
                    INSERT INTO wiki_raw_data (
                        customer_id, wiki_type, slug, source, source_ref, data
                    )
                    VALUES ($1, 'service_card', 'auth',
                            'github', 'PR-42', $2::jsonb)
                    """,
                    "mig-43-raw",
                    json.dumps({"id": 42, "title": "different body"}),
                )
        finally:
            await conn.execute("DELETE FROM wiki_raw_data WHERE customer_id = $1", "mig-43-raw")
            await conn.execute("DELETE FROM customers WHERE customer_id = $1", "mig-43-raw")


@pytest.mark.asyncio
async def test_wsr_kind_accepts_bootstrap(live_db) -> None:
    """The CHECK constraint on wiki_synthesis_runs.kind accepts 'bootstrap'."""
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'mig-43-run', 'h') ON CONFLICT DO NOTHING",
            "mig-43-run",
        )
        try:
            run_id = await conn.fetchval(
                """
                INSERT INTO wiki_synthesis_runs (
                    customer_id, kind, stage, source
                )
                VALUES ($1, 'bootstrap', 'synthesis', 'github')
                RETURNING run_id
                """,
                "mig-43-run",
            )
            assert run_id is not None
            with pytest.raises(CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO wiki_synthesis_runs (customer_id, kind)
                    VALUES ($1, 'made_up_kind')
                    """,
                    "mig-43-run",
                )
        finally:
            await conn.execute(
                "DELETE FROM wiki_synthesis_runs WHERE customer_id = $1",
                "mig-43-run",
            )
            await conn.execute("DELETE FROM customers WHERE customer_id = $1", "mig-43-run")


@pytest.mark.asyncio
async def test_wsr_source_column_and_index(live_db) -> None:
    """source column is nullable; idx_wsr_kind_source backs per-source lookups."""
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'wiki_synthesis_runs'
              AND column_name = 'source'
            """
        )
    assert len(rows) == 1
    assert rows[0]["data_type"] == "text"
    assert rows[0]["is_nullable"] == "YES"

    async with raw_conn() as conn:
        names = {
            r["indexname"]
            for r in await conn.fetch(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'wiki_synthesis_runs'"
            )
        }
    assert "idx_wsr_kind_source" in names
