"""DB+R2 tests for seed_tenant. Depends on `live_db` from
tests/conftest.py — needs `docker compose up -d` and migrations applied
locally. Uses the local MinIO at localhost:9000."""

from __future__ import annotations

import secrets
from pathlib import Path

import pytest

from scripts.synth.seed import MissingCanonicalError, seed_tenant
from shared.db import get_pool, raw_conn

# ObjectStore is constructed directly — there is no build_object_store factory
# in shared/storage.py. The CLI (scripts/synth/cli.py::_open_db_and_bucket)
# uses ObjectStore() with no arguments, pulling config from settings.
from shared.storage import ObjectStore


CANONICAL_MINI = Path(__file__).parent.parent / "fixtures" / "canonical-mini"


async def _seed_customer(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash, status)
            VALUES ($1, $2, 'h', 'active')
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id, f"test-{customer_id}",
        )


async def _queue_rows(customer_id: str) -> list[tuple[str, str]]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT source_system, source_event_id FROM ingestion_queue "
            "WHERE customer_id = $1 ORDER BY source_event_id",
            customer_id,
        )
    return [(r["source_system"], r["source_event_id"]) for r in rows]


async def test_seed_happy_path(live_db):
    cid = f"cust-eval-test-seed-{secrets.token_hex(4)}"
    await _seed_customer(cid)
    bucket = ObjectStore()
    result = await seed_tenant(
        customer_id=cid,
        canonical_dir=CANONICAL_MINI,
        db=get_pool(),
        bucket=bucket,
    )
    assert result.envelopes_processed == 2
    assert result.r2_uploaded == 2
    assert result.queued == 2
    rows = await _queue_rows(cid)
    assert {ev for _, ev in rows} == {"std-001", "oncall-001"}


async def test_seed_idempotent(live_db):
    cid = f"cust-eval-test-idem-{secrets.token_hex(4)}"
    await _seed_customer(cid)
    bucket = ObjectStore()
    first = await seed_tenant(cid, CANONICAL_MINI, get_pool(), bucket)
    second = await seed_tenant(cid, CANONICAL_MINI, get_pool(), bucket)
    assert first.queued == 2
    # Second run: R2 PUT overwrites (uploaded > 0), queue ON CONFLICT skips.
    assert second.queued == 0
    assert second.r2_uploaded == 2  # PUTs are unconditional; idempotent overwrite


async def test_seed_missing_canonical_raises(live_db):
    cid = f"cust-eval-test-noncan-{secrets.token_hex(4)}"
    await _seed_customer(cid)
    bucket = ObjectStore()
    with pytest.raises(MissingCanonicalError):
        await seed_tenant(cid, Path("/tmp/does-not-exist"), get_pool(), bucket)
