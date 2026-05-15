"""bucket_for resolution: pulls customers.r2_bucket when set, falls back to
the legacy ``<R2_BUCKET_PREFIX>-<customer_id>`` formula when NULL.

The fallback is load-bearing during the rollout window between migration
0073 (which adds the column) and the control-plane writer that populates
``r2_bucket`` for every new tenant — until that lands, a freshly mirrored
customer can sit on a NULL row, and the runtime must keep serving uploads.
"""

from __future__ import annotations

import pytest

from shared.config import get_settings
from shared.db import raw_conn
from shared.storage import _reset_bucket_cache_for_tests, get_store


async def _seed_customer(customer_id: str, r2_bucket: str | None) -> None:
    async with raw_conn() as conn:
        await conn.execute("DELETE FROM customers WHERE customer_id = $1", customer_id)
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash, r2_bucket) "
            "VALUES ($1, $2, $3, $4)",
            customer_id,
            "test",
            "hash",
            r2_bucket,
        )


@pytest.mark.asyncio
async def test_bucket_for_uses_db_column_when_set(live_db) -> None:
    _reset_bucket_cache_for_tests()
    cid = "bucket-for-explicit-cust"
    await _seed_customer(cid, "my-explicit-bucket")
    store = get_store()
    assert await store.bucket_for(cid) == "my-explicit-bucket"
    # Cached — second call must not change shape even if we delete the row.
    async with raw_conn() as conn:
        await conn.execute("DELETE FROM customers WHERE customer_id = $1", cid)
    assert await store.bucket_for(cid) == "my-explicit-bucket"


@pytest.mark.asyncio
async def test_bucket_for_falls_back_when_column_null(live_db) -> None:
    _reset_bucket_cache_for_tests()
    cid = "bucket-for-null-cust"
    await _seed_customer(cid, None)
    store = get_store()
    expected = get_settings().bucket_for(cid)
    assert await store.bucket_for(cid) == expected


@pytest.mark.asyncio
async def test_bucket_for_falls_back_when_row_missing(live_db) -> None:
    _reset_bucket_cache_for_tests()
    cid = "bucket-for-missing-cust"
    async with raw_conn() as conn:
        await conn.execute("DELETE FROM customers WHERE customer_id = $1", cid)
    store = get_store()
    expected = get_settings().bucket_for(cid)
    assert await store.bucket_for(cid) == expected
