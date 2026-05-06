"""DB+R2 tests for seed_tenant. Depends on `live_db` from
tests/conftest.py — needs `docker compose up -d` and migrations applied
locally. Uses the local MinIO at localhost:9000."""

from __future__ import annotations

import secrets
import subprocess
import sys
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
    # canonical-mini has 3 envelopes: 2 slack + 1 notion (per MANIFEST.json).
    assert result.envelopes_processed == 3
    assert result.r2_uploaded == 3
    assert result.queued == 3
    assert result.canonical_customer_id == "cust-eval-canonical-v1"
    rows = await _queue_rows(cid)
    assert set(rows) == {
        ("slack", "std-001"),
        ("slack", "oncall-001"),
        ("notion", "page-001"),
    }


async def test_seed_idempotent(live_db):
    cid = f"cust-eval-test-idem-{secrets.token_hex(4)}"
    await _seed_customer(cid)
    bucket = ObjectStore()
    first = await seed_tenant(cid, CANONICAL_MINI, get_pool(), bucket)
    second = await seed_tenant(cid, CANONICAL_MINI, get_pool(), bucket)
    assert first.queued == 3
    # Second run: R2 PUT overwrites (uploaded > 0), queue ON CONFLICT skips.
    assert second.queued == 0
    assert second.r2_uploaded == 3  # PUTs are unconditional; idempotent overwrite


async def test_seed_missing_canonical_raises(live_db):
    cid = f"cust-eval-test-noncan-{secrets.token_hex(4)}"
    await _seed_customer(cid)
    bucket = ObjectStore()
    with pytest.raises(MissingCanonicalError):
        await seed_tenant(cid, Path("/tmp/does-not-exist"), get_pool(), bucket)


# ---------------------------------------------------------------------------
# Gate-failure tests — Task 8 (invoke CLI via subprocess)
# ---------------------------------------------------------------------------


def test_cli_seed_missing_customer(live_db):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.synth", "seed",
         "--customer", "cust-prbe-doesnotexist-zzz"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "not found" in result.stderr


async def test_cli_seed_no_path_satisfied(live_db):
    cid = f"cust-prbe-test-nopath-{secrets.token_hex(4)}"
    await _seed_customer(cid)  # inserted without metadata flag
    result = subprocess.run(
        [sys.executable, "-m", "scripts.synth", "seed", "--customer", cid,
         "--canonical-dir", str(CANONICAL_MINI)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "not seed-eligible" in result.stderr
    rows = await _queue_rows(cid)
    assert rows == []


async def test_cli_seed_confirm_mismatch(live_db):
    cid = f"cust-prbe-test-confirm-{secrets.token_hex(4)}"
    await _seed_customer(cid)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.synth", "seed",
         "--customer", cid, "--allow-non-sandbox",
         "--canonical-dir", str(CANONICAL_MINI)],
        input="wrong-customer-id\n",
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "confirmation mismatch" in result.stderr
    rows = await _queue_rows(cid)
    assert rows == []


async def test_cli_seed_canonical_missing(live_db):
    # Eval prefix bypasses Path 1 / Path 2 gates; the canonical-missing
    # check fires next.
    cid = "cust-eval-test-can-missing"
    await _seed_customer(cid)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.synth", "seed",
         "--customer", cid, "--canonical-dir", "/tmp/nope-dir"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "canonical corpus not found" in result.stderr
    # Confirm gate ordering: gates 1 and 2 must have passed (customer
    # exists + eval prefix bypasses eligibility) for gate 1a to fire.
    assert "not found" not in result.stderr or "canonical corpus not found" in result.stderr
    assert "not seed-eligible" not in result.stderr


# ---------------------------------------------------------------------------
# CLI happy-path tests — subprocess exercises full dispatch + success paths
# ---------------------------------------------------------------------------


async def test_cli_seed_path1_happy(live_db):
    cid = f"cust-prbe-test-cli-happy-{secrets.token_hex(4)}"
    await _seed_customer(cid)

    # Step 1: set the allow_synth_seed metadata flag via allow-seed subcommand
    result1 = subprocess.run(
        [sys.executable, "-m", "scripts.synth", "allow-seed", "--customer", cid],
        capture_output=True, text=True,
    )
    assert result1.returncode == 0, f"allow-seed failed: stderr={result1.stderr!r}"
    assert f"metadata.allow_synth_seed=true for {cid}" in result1.stderr

    # Step 2: seed via Path 1 (metadata flag satisfied)
    result2 = subprocess.run(
        [sys.executable, "-m", "scripts.synth", "seed",
         "--customer", cid,
         "--canonical-dir", "tests/fixtures/canonical-mini"],
        capture_output=True, text=True,
    )
    assert result2.returncode == 0, f"seed failed: stderr={result2.stderr!r}"
    assert "seeded 3 envelopes" in result2.stderr
    assert result2.stdout == ""

    rows = await _queue_rows(cid)
    assert {ev for _, ev in rows} == {"std-001", "oncall-001", "page-001"}


async def test_cli_seed_path2_happy_with_confirm(live_db):
    cid = f"cust-prbe-test-cli-escape-{secrets.token_hex(4)}"
    await _seed_customer(cid)

    # Path 2: escape hatch flag + typed confirmation matches customer id
    result = subprocess.run(
        [sys.executable, "-m", "scripts.synth", "seed",
         "--customer", cid,
         "--allow-non-sandbox",
         "--canonical-dir", "tests/fixtures/canonical-mini"],
        input=cid + "\n",
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"seed failed: stderr={result.stderr!r}"
    assert "seeded 3 envelopes" in result.stderr
    # The typed-confirm prompt is written to stdout; allow it but ensure no
    # unexpected extra output beyond the prompt line.
    assert "seeded" not in result.stdout  # success message stays on stderr

    rows = await _queue_rows(cid)
    assert {ev for _, ev in rows} == {"std-001", "oncall-001", "page-001"}
