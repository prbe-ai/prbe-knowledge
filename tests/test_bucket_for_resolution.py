"""bucket_for resolution post-PR-3: reads customers.r2_bucket; raises on
missing row / NULL column (both impossible after migration 0075).

The legacy ``<R2_BUCKET_PREFIX>-<customer_id>`` fallback that 0073 and
0074 propped up is gone — the cluster now guarantees every customers
row has a non-NULL ``r2_bucket`` (the CP mirror sets it at INSERT, and
0075 backfilled any stragglers + made the column NOT NULL). Silently
falling back to a computed bucket name would write to the wrong R2
location; surfacing the bug is the right behavior.
"""

from __future__ import annotations

import pytest

from shared.db import raw_conn
from shared.exceptions import StorageUnavailable
from shared.storage import _reset_bucket_cache_for_tests, get_store


async def _seed_customer(customer_id: str, r2_bucket: str) -> None:
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
    # Cached — second call must return the same value even if we delete
    # the row (immutability assumption, confirmed by NOT NULL).
    async with raw_conn() as conn:
        await conn.execute("DELETE FROM customers WHERE customer_id = $1", cid)
    assert await store.bucket_for(cid) == "my-explicit-bucket"


@pytest.mark.asyncio
async def test_bucket_for_raises_when_row_missing(live_db) -> None:
    """A customer_id that doesn't exist in the DP customers table is a
    real bug — the CP mirror creates the row before any tenant traffic
    can start. Surface it instead of silently writing to a wrong bucket."""
    _reset_bucket_cache_for_tests()
    cid = "bucket-for-missing-cust"
    async with raw_conn() as conn:
        await conn.execute("DELETE FROM customers WHERE customer_id = $1", cid)
    store = get_store()
    with pytest.raises(StorageUnavailable, match="no customers row"):
        await store.bucket_for(cid)
