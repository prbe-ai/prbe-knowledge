"""Tests for TenantBootstrap (init + clean) + the new ObjectStore.delete shim."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.synth.bootstrap import (
    CUSTOMER_OWNED_TABLES,
    clean_tenant,
    init_tenant,
)
from scripts.synth.profile import Profile


def _profile(customer_id: str = "cust-eval-test-01") -> Profile:
    raw = {
        "customer_id": customer_id,
        "repos": [{"url": "github.com/x/y", "local_path": "/tmp/y"}],
        "preset": "tiny_test",
        "seed": 42,
        "sources": ["slack", "notion"],
    }
    return Profile(
        customer_id=raw["customer_id"],
        repos=(),
        preset=raw["preset"],
        seed=raw["seed"],
        raw=raw,
    )


def _mock_db() -> AsyncMock:
    db = AsyncMock()
    db.execute = AsyncMock(return_value=None)
    db.transaction = MagicMock()
    db.transaction.return_value.__aenter__ = AsyncMock()
    db.transaction.return_value.__aexit__ = AsyncMock(return_value=False)
    return db


def _mock_bucket() -> AsyncMock:
    bucket = AsyncMock()
    bucket.bucket_for = MagicMock(return_value="prbe-synth-bucket")
    bucket.ensure_bucket = AsyncMock(return_value=None)
    bucket.list_keys = AsyncMock(return_value=[])
    bucket.delete = AsyncMock(return_value=None)
    return bucket


@pytest.mark.asyncio
async def test_init_tenant_inserts_customer_and_tokens() -> None:
    db = _mock_db()
    bucket = _mock_bucket()
    profile = _profile()
    await init_tenant(profile, db, bucket)
    # customers insert + 2 source token inserts (slack, notion)
    assert db.execute.await_count == 3
    calls = [c.args[0] for c in db.execute.await_args_list]
    assert any("INSERT INTO customers" in q for q in calls)
    assert sum("INSERT INTO integration_tokens" in q for q in calls) == 2
    bucket.ensure_bucket.assert_awaited_once()
    # customers.api_key_hash is NOT NULL — the customers insert must include
    # api_key_hash and pass a non-null placeholder so the row commits.
    customers_call = next(
        c for c in db.execute.await_args_list
        if "INSERT INTO customers" in c.args[0]
    )
    assert "api_key_hash" in customers_call.args[0]
    placeholder = customers_call.args[3]
    assert placeholder and len(placeholder) == 64  # sha256 hex digest


@pytest.mark.asyncio
async def test_init_tenant_idempotent_on_repeat() -> None:
    db = _mock_db()
    bucket = _mock_bucket()
    profile = _profile()
    await init_tenant(profile, db, bucket)
    await init_tenant(profile, db, bucket)
    # Both runs are happy-path; ON CONFLICT DO NOTHING keeps things safe.
    # The mock doesn't simulate conflict, but we verify queries always carry
    # the ON CONFLICT clause so a real DB would handle it.
    for call in db.execute.await_args_list:
        assert "ON CONFLICT" in call.args[0]


@pytest.mark.asyncio
async def test_clean_tenant_refuses_non_synth_prefix() -> None:
    db = _mock_db()
    bucket = _mock_bucket()
    with pytest.raises(ValueError, match="refuse to clean non-synthetic"):
        await clean_tenant("prod-customer-01", db, bucket)
    db.execute.assert_not_awaited()
    bucket.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_clean_tenant_accepts_eval_prefix() -> None:
    db = _mock_db()
    bucket = _mock_bucket()
    await clean_tenant("cust-eval-test-01", db, bucket)
    # One DELETE per CUSTOMER_OWNED_TABLES entry.
    assert db.execute.await_count == len(CUSTOMER_OWNED_TABLES)


@pytest.mark.asyncio
async def test_clean_tenant_accepts_synth_prefix() -> None:
    db = _mock_db()
    bucket = _mock_bucket()
    await clean_tenant("cust-synth-test-01", db, bucket)
    assert db.execute.await_count == len(CUSTOMER_OWNED_TABLES)


@pytest.mark.asyncio
async def test_clean_tenant_deletes_only_keys_under_synth_prefix() -> None:
    db = _mock_db()
    bucket = _mock_bucket()
    bucket.list_keys = AsyncMock(return_value=[
        "raw/slack/cust-eval-test-01/synth/doc-1.json",
        "raw/notion/cust-eval-test-01/synth/page-1.json",
        "raw/slack/cust-OTHER-01/synth/doc-x.json",  # belongs to a different tenant
    ])
    await clean_tenant("cust-eval-test-01", db, bucket)
    # Only the two keys belonging to cust-eval-test-01/synth/ are deleted.
    assert bucket.delete.await_count == 2
    deleted_keys = [c.args[1] for c in bucket.delete.await_args_list]
    assert all("cust-eval-test-01/synth/" in k for k in deleted_keys)
