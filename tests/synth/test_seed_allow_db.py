"""DB tests for set_allow_synth_seed. Depends on `live_db` from
tests/conftest.py — needs `docker compose up -d` and migrations applied
locally."""

from __future__ import annotations

import json
import secrets

import pytest

from engine.shared.db import get_pool, raw_conn
from scripts.synth.seed import set_allow_synth_seed


async def _seed_customer(customer_id: str, metadata: dict | None = None) -> None:
    """Insert a customers row with optional metadata. live_db's truncate
    cleans up at fixture teardown; no per-test DELETE needed."""
    metadata_json = json.dumps(metadata or {})
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash, status, metadata)
            VALUES ($1, $2, 'h', 'active', $3::jsonb)
            ON CONFLICT (customer_id) DO UPDATE SET metadata = EXCLUDED.metadata
            """,
            customer_id, f"test-{customer_id}", metadata_json,
        )


async def _read_metadata(customer_id: str) -> dict:
    """Fetch and JSON-decode customers.metadata."""
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT metadata FROM customers WHERE customer_id = $1", customer_id
        )
    if row is None:
        return {}
    meta = row["metadata"]
    return json.loads(meta) if isinstance(meta, str) else dict(meta)


async def test_sets_flag_to_true(live_db):
    cid = f"cust-prbe-test-set-{secrets.token_hex(4)}"
    await _seed_customer(cid)
    await set_allow_synth_seed(cid, get_pool())
    meta = await _read_metadata(cid)
    assert meta.get("allow_synth_seed") is True


async def test_idempotent(live_db):
    cid = f"cust-prbe-test-idem-{secrets.token_hex(4)}"
    await _seed_customer(cid)
    await set_allow_synth_seed(cid, get_pool())
    await set_allow_synth_seed(cid, get_pool())
    meta = await _read_metadata(cid)
    assert meta == {"allow_synth_seed": True}


async def test_missing_customer_raises(live_db):
    with pytest.raises(ValueError, match="not found"):
        await set_allow_synth_seed("cust-prbe-doesnotexist-zzz", get_pool())


async def test_preserves_other_metadata(live_db):
    cid = f"cust-prbe-test-other-{secrets.token_hex(4)}"
    await _seed_customer(cid, metadata={"existing_key": "value"})
    await set_allow_synth_seed(cid, get_pool())
    meta = await _read_metadata(cid)
    assert meta["existing_key"] == "value"
    assert meta["allow_synth_seed"] is True
