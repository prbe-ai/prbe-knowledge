"""Retention reaper: deletes exactly the dead-and-old, nothing else.

The predicate is only trustworthy because kb#528 made `valid_to` THE
liveness marker (removal caps the version range too); these tests pin the
three boundaries that must never move: live rows are untouchable whatever
their age, dead-but-recent rows wait out the window, and dry-run deletes
nothing while reporting what it would.
"""

from __future__ import annotations

import pytest

from engine.shared import db as db_module
from scripts.cron_chunk_retention import reap_tenant

pytestmark = pytest.mark.asyncio


async def _seed(customer_id: str) -> None:
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, $1, 'test-hash') ON CONFLICT DO NOTHING
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO documents (customer_id, doc_id, version, source_system,
                                   source_id, source_url, doc_type, content_hash,
                                   created_at, updated_at, valid_from, acl,
                                   title, body_preview)
            VALUES ($1, 'doc-r', 5, 'slack', 'doc-r', 'https://x', 'message',
                    'dh-r', NOW(), NOW(), NOW(), '{}'::jsonb, 't', 'p')
            """,
            customer_id,
        )
        for chunk_id, chash, valid_to_sql in (
            ("ck-live", "h-live", "NULL"),
            ("ck-dead-old", "h-dead-old", "now() - interval '90 days'"),
            ("ck-dead-recent", "h-dead-recent", "now() - interval '2 days'"),
        ):
            await conn.execute(
                f"""
                INSERT INTO chunks (customer_id, chunk_id, doc_id, chunk_index,
                                    content, content_hash, token_count,
                                    first_seen_version, last_seen_version, valid_to)
                VALUES ($1, $2, 'doc-r', 0, 'body', $3, 1, 1,
                        CASE WHEN $2 = 'ck-live' THEN 5 ELSE 2 END,
                        {valid_to_sql})
                """,
                customer_id,
                chunk_id,
                chash,
            )


async def _remaining(customer_id: str) -> set[str]:
    async with db_module.raw_conn() as conn:
        return {
            r["chunk_id"]
            for r in await conn.fetch(
                "SELECT chunk_id FROM chunks WHERE customer_id = $1", customer_id
            )
        }


async def test_reaps_only_dead_and_past_window(live_db) -> None:
    await _seed("cust-reap-1")
    deleted = await reap_tenant("cust-reap-1")
    assert deleted == 1
    assert await _remaining("cust-reap-1") == {"ck-live", "ck-dead-recent"}


async def test_dry_run_deletes_nothing(live_db) -> None:
    await _seed("cust-reap-dry")
    deleted = await reap_tenant("cust-reap-dry", dry_run=True)
    assert deleted == 0
    assert await _remaining("cust-reap-dry") == {
        "ck-live",
        "ck-dead-old",
        "ck-dead-recent",
    }


async def test_batching_terminates_and_deletes_all(live_db, monkeypatch) -> None:
    """Force batch size 1 so the loop must iterate; every reapable row must
    still go, and the loop must stop -- an off-by-one here is either an
    infinite cron or a reaper that quietly leaves a tail behind."""
    import scripts.cron_chunk_retention as mod

    monkeypatch.setattr(mod, "CHUNK_RETENTION_BATCH_SIZE", 1)
    await _seed("cust-reap-batch")
    async with db_module.raw_conn() as conn:
        for i in range(3):
            await conn.execute(
                """
                INSERT INTO chunks (customer_id, chunk_id, doc_id, chunk_index,
                                    content, content_hash, token_count,
                                    first_seen_version, last_seen_version, valid_to)
                VALUES ($1, $2, 'doc-r', 0, 'body', $3, 1, 1, 2,
                        now() - interval '90 days')
                """,
                "cust-reap-batch",
                f"ck-extra-{i}",
                f"h-extra-{i}",
            )
    deleted = await reap_tenant("cust-reap-batch")
    assert deleted == 4  # ck-dead-old + 3 extras, one at a time
    assert await _remaining("cust-reap-batch") == {"ck-live", "ck-dead-recent"}
