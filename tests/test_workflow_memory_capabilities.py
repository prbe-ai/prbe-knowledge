"""Per-tenant workflow-memory capability flags, and seed-on-enable.

The flags are six booleans in `customers.preferences`, read with the same
fail-closed coercion `shared.customer_prefs` uses. That posture is the thing
under test here: a tenant who has not explicitly opted in gets nothing, and no
error path may flip that to "on".

WHAT THIS FILE HAS TO DEFEND, and why each claim needs its own test:

* FAIL-CLOSED means fail-closed on EVERY route into "no value": key absent,
  customer absent, blob not an object, value a string `"true"` rather than a
  real bool, JSON that will not decode, DB unreachable. Five of the six are
  reachable from SQL; the sixth (undecodable JSONB) is not, because the column
  is typed JSONB, so it is exercised at the reader's seam instead.
* A MISSPELLED KEY IS NOT "OFF". Both prior implementations of this pattern in
  the codebase build key strings by concatenation with no registry, so a typo
  reads as `False` forever and looks exactly like a tenant who opted out. Here
  an unknown key must RAISE, and the tests assert that on several typo shapes
  including one that differs from a real key by a single character.
* THE ENVELOPE'S THIRD STATE. `enabled` alone cannot distinguish "nobody
  turned it on" from "the plan does not include it". The shape is asserted
  field-by-field so a later change to the wire format has to come through here.
* SEEDING IS PART OF THE FLIP, NOT A FOLLOW-UP. A tenant with the declared
  input on and zero situations classifies everything as unknown and serves zero
  cards -- indistinguishable from "no rule matched". So the flip and the seed
  are one transaction, and the atomicity test PROVES the rollback by making the
  insert fail inside the database (a BEFORE INSERT trigger that raises), not by
  stubbing out the seeding function.
* THE REGISTRY AND THE MIGRATION CANNOT DRIFT. Migration 0077 hardcodes its key
  list on purpose (migrations are frozen in time and must not import app code),
  which means two lists exist. The last test compares them.

Run with the isolated wfmem database:

    PYTHONPATH=<scratchpad> .venv/bin/pytest tests/test_workflow_memory_capabilities.py \
        -q -p wfmem_isolated_db
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType
from typing import Any

import asyncpg
import pytest
import pytest_asyncio

from shared.db import raw_conn, with_tenant
from shared.wfmem.capabilities import (
    WFMEM_CAPABILITY_KEYS,
    WFMEM_INPUT_DECLARED,
    InputPath,
    OutputSurface,
    capability_envelope,
    input_capability_key,
    is_capability_enabled,
    output_capability_key,
)
from shared.wfmem.situations import SEED_SITUATIONS, enable_capability, seed_situations

TENANT_A = "cust-wfmem-cap-a"
TENANT_B = "cust-wfmem-cap-b"

#: The six cells, spelled out BY HAND on purpose. `capabilities` derives its set
#: from the two axes; a test that derived them the same way would rename the
#: contract in lockstep with any axis rename and assert nothing.
EXPECTED_KEYS = frozenset(
    {
        "wfmem_input_declared",
        "wfmem_input_imported",
        "wfmem_input_mined",
        "wfmem_output_retrieval",
        "wfmem_output_compiled",
        "wfmem_output_midsession",
    }
)


@pytest_asyncio.fixture
async def two_tenants(live_db: None) -> AsyncIterator[tuple[str, str]]:
    """Two customers with an empty `preferences` blob -- the never-configured state.

    Deliberately does NOT run migration 0077's backfill: CI stamps the alembic
    head instead of running migrations, so anything that depended on the
    backfill having executed would pass here and lie there.
    """
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'wfmem-cap-a', 'h-wfmem-cap-a'),
                   ($2, 'wfmem-cap-b', 'h-wfmem-cap-b')
            ON CONFLICT (customer_id) DO NOTHING
            """,
            TENANT_A,
            TENANT_B,
        )
    yield TENANT_A, TENANT_B


async def _set_pref(customer_id: str, key: str, value: Any) -> None:
    """Write one preferences key with a JSON value of the caller's choosing."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            UPDATE customers
               SET preferences = jsonb_set(preferences, ARRAY[$2::text], $3::jsonb, true)
             WHERE customer_id = $1
            """,
            customer_id,
            key,
            json.dumps(value),
        )


async def _set_raw_prefs(customer_id: str, blob: str) -> None:
    """Replace the whole preferences column with an arbitrary JSONB literal."""
    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE customers SET preferences = $2::jsonb WHERE customer_id = $1",
            customer_id,
            blob,
        )


async def _read_prefs(customer_id: str) -> dict[str, Any]:
    async with raw_conn() as conn:
        raw = await conn.fetchval(
            "SELECT preferences FROM customers WHERE customer_id = $1", customer_id
        )
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


async def _count_situations(customer_id: str) -> int:
    """Count with an explicit tenant predicate.

    Not a bare `SELECT count(*)` under `with_tenant`: the dev role is a
    SUPERUSER and bypasses RLS, so the GUC would filter nothing and the
    per-tenant test would pass vacuously.
    """
    async with raw_conn() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM situations WHERE customer_id = $1", customer_id
        )


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------


def test_registry_is_exactly_the_six_cells() -> None:
    assert WFMEM_CAPABILITY_KEYS == EXPECTED_KEYS


def test_registry_is_derived_from_both_axes() -> None:
    """Every axis member must contribute a key, and contribute only one."""
    derived = {input_capability_key(p) for p in InputPath} | {
        output_capability_key(s) for s in OutputSurface
    }
    assert derived == EXPECTED_KEYS
    assert len(InputPath) == 3
    assert len(OutputSurface) == 3
    assert input_capability_key(InputPath.DECLARED) == WFMEM_INPUT_DECLARED


def test_accessors_reject_a_value_off_the_axis() -> None:
    with pytest.raises(ValueError):
        input_capability_key("declaredd")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        output_capability_key("retreival")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Reading a flag: fail-closed on every route into "no value"
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
async def test_unset_key_reads_false(two_tenants: tuple[str, str], key: str) -> None:
    tenant, _ = two_tenants
    assert await is_capability_enabled(tenant, key) is False


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
async def test_key_reads_true_only_after_explicit_boolean_true(
    two_tenants: tuple[str, str], key: str
) -> None:
    tenant, other = two_tenants
    await _set_pref(tenant, key, True)
    assert await is_capability_enabled(tenant, key) is True
    # The flip is per-tenant and per-key: nothing else moved.
    assert await is_capability_enabled(other, key) is False
    for sibling in EXPECTED_KEYS - {key}:
        assert await is_capability_enabled(tenant, sibling) is False


@pytest.mark.parametrize("value", ["true", "True", "1", "yes"])
async def test_string_true_does_not_enable(two_tenants: tuple[str, str], value: str) -> None:
    """`_coerce_bool`'s contract: the value must be a real JSON bool."""
    tenant, _ = two_tenants
    await _set_pref(tenant, WFMEM_INPUT_DECLARED, value)
    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is False


@pytest.mark.parametrize("value", [1, 0, None, {"enabled": True}, ["true"]])
async def test_non_boolean_values_do_not_enable(two_tenants: tuple[str, str], value: Any) -> None:
    tenant, _ = two_tenants
    await _set_pref(tenant, WFMEM_INPUT_DECLARED, value)
    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is False


async def test_explicit_false_reads_false(two_tenants: tuple[str, str]) -> None:
    tenant, _ = two_tenants
    await _set_pref(tenant, WFMEM_INPUT_DECLARED, False)
    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is False


@pytest.mark.parametrize("blob", ['"garbage"', "[1, 2, 3]", "123", "null", "true"])
async def test_non_object_preferences_blob_reads_false(
    two_tenants: tuple[str, str], blob: str
) -> None:
    """A JSONB that is not an object is malformed as far as this reader cares."""
    tenant, _ = two_tenants
    await _set_raw_prefs(tenant, blob)
    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is False


async def test_undecodable_json_reads_false(
    two_tenants: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not reachable from SQL (the column is typed JSONB), so it is faked at the seam.

    The branch still has to hold: asyncpg hands JSONB back as a str with no
    codec registered, and a driver/codec change is exactly the kind of thing
    that would start delivering junk to the coercion.
    """
    tenant, _ = two_tenants
    monkeypatch.setattr("shared.wfmem.capabilities.raw_conn", _fake_raw_conn(returns="{not json"))
    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is False


async def test_missing_customer_reads_false(live_db: None) -> None:
    assert await is_capability_enabled("cust-does-not-exist", WFMEM_INPUT_DECLARED) is False


async def test_null_preferences_reads_false(
    two_tenants: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NULL blob reads False.

    `customers.preferences` is NOT NULL in the schema, so this state is not
    representable in SQL; the None is injected at the reader's seam. The branch
    matters anyway -- `fetchval` returns None for a row that is not there, and
    a future nullable column would land in the same place.
    """
    tenant, _ = two_tenants
    monkeypatch.setattr("shared.wfmem.capabilities.raw_conn", _fake_raw_conn(returns=None))
    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is False


async def test_db_error_reads_false(
    two_tenants: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreachable database is not an accidental grant."""
    tenant, _ = two_tenants
    monkeypatch.setattr(
        "shared.wfmem.capabilities.raw_conn", _fake_raw_conn(raises=RuntimeError("pool is gone"))
    )
    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is False


async def test_empty_customer_id_reads_false(live_db: None) -> None:
    assert await is_capability_enabled("", WFMEM_INPUT_DECLARED) is False


def _fake_raw_conn(returns: Any = None, raises: BaseException | None = None) -> Any:
    """A stand-in for `shared.db.raw_conn` with a one-method connection."""
    from contextlib import asynccontextmanager

    class _Conn:
        async def fetchval(self, *args: Any, **kwargs: Any) -> Any:
            if raises is not None:
                raise raises
            return returns

    @asynccontextmanager
    async def _cm() -> AsyncIterator[_Conn]:
        if raises is not None:
            raise raises
        yield _Conn()

    return _cm


# --------------------------------------------------------------------------
# An unknown key is a programming error, not "off"
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "wfmem_input_declare",  # one character short of a real key
        "wfmem_input_declared ",  # trailing space
        "WFMEM_INPUT_DECLARED",  # wrong case
        "wfmem_output_mined",  # real words, wrong axis pairing
        "wiki_generation_enabled",  # a real pref key, but not one of ours
        "",
    ],
)
async def test_unknown_key_raises(two_tenants: tuple[str, str], key: str) -> None:
    tenant, _ = two_tenants
    with pytest.raises(ValueError) as exc:
        await is_capability_enabled(tenant, key)
    assert key.strip() in str(exc.value) or "unknown" in str(exc.value).lower()


async def test_unknown_key_raises_even_without_a_customer(live_db: None) -> None:
    """The key check comes first: a bad key is a bug regardless of the tenant."""
    with pytest.raises(ValueError):
        await is_capability_enabled("", "wfmem_input_declare")


async def test_unknown_key_raises_in_the_envelope(two_tenants: tuple[str, str]) -> None:
    tenant, _ = two_tenants
    with pytest.raises(ValueError):
        await capability_envelope(tenant, "wfmem_input_declare")


async def test_unknown_key_raises_in_enable(two_tenants: tuple[str, str]) -> None:
    tenant, _ = two_tenants
    with pytest.raises(ValueError):
        await enable_capability(tenant, "wfmem_input_declare")
    assert await _read_prefs(tenant) == {}


# --------------------------------------------------------------------------
# The envelope: three states, not two
# --------------------------------------------------------------------------


async def test_envelope_shape_when_off(two_tenants: tuple[str, str]) -> None:
    tenant, _ = two_tenants
    envelope = await capability_envelope(tenant, WFMEM_INPUT_DECLARED)
    assert set(envelope) == {"enabled", "entitled", "upgrade_url"}
    assert envelope["enabled"] is False
    assert envelope["entitled"] is True
    assert envelope["upgrade_url"] is None


async def test_envelope_shape_when_on(two_tenants: tuple[str, str]) -> None:
    tenant, _ = two_tenants
    await _set_pref(tenant, WFMEM_INPUT_DECLARED, True)
    envelope = await capability_envelope(tenant, WFMEM_INPUT_DECLARED)
    assert envelope == {"enabled": True, "entitled": True, "upgrade_url": None}


# --------------------------------------------------------------------------
# The seed vocabulary
# --------------------------------------------------------------------------


def test_seed_vocabulary_is_twelve_distinct_situations() -> None:
    assert len(SEED_SITUATIONS) == 12
    slugs = [s.slug for s in SEED_SITUATIONS]
    assert len(set(slugs)) == 12
    assert set(slugs) == {
        "launch-run",
        "claim-done",
        "open-pr",
        "deploy",
        "incident-recovery",
        "edit-repo",
        "process-dataset",
        "run-eval",
        "review-code",
        "provision-infra",
        "reproduce-experiment",
        "debug-failing-run",
    }


def test_seed_descriptions_are_classifier_input_not_label_synonyms() -> None:
    """A description that just restates the label gives a classifier nothing."""
    for situation in SEED_SITUATIONS:
        assert situation.label
        assert len(situation.description) >= 60, situation.slug
        assert situation.description.strip().lower() != situation.label.strip().lower()
    descriptions = [s.description for s in SEED_SITUATIONS]
    assert len(set(descriptions)) == 12


async def test_seed_situations_inserts_twelve(two_tenants: tuple[str, str]) -> None:
    tenant, _ = two_tenants
    async with with_tenant(tenant) as conn:
        inserted = await seed_situations(conn, tenant)
    assert inserted == 12
    assert await _count_situations(tenant) == 12
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT slug, label, description FROM situations WHERE customer_id = $1 ORDER BY slug",
            tenant,
        )
    assert [r["slug"] for r in rows] == sorted(s.slug for s in SEED_SITUATIONS)
    by_slug = {s.slug: s for s in SEED_SITUATIONS}
    for row in rows:
        assert row["label"] == by_slug[row["slug"]].label
        assert row["description"] == by_slug[row["slug"]].description


async def test_seed_situations_is_idempotent(two_tenants: tuple[str, str]) -> None:
    tenant, _ = two_tenants
    async with with_tenant(tenant) as conn:
        first = await seed_situations(conn, tenant)
    async with with_tenant(tenant) as conn:
        second = await seed_situations(conn, tenant)
    assert (first, second) == (12, 0)
    assert await _count_situations(tenant) == 12


async def test_seed_situations_is_per_tenant(two_tenants: tuple[str, str]) -> None:
    tenant_a, tenant_b = two_tenants
    async with with_tenant(tenant_a) as conn:
        await seed_situations(conn, tenant_a)
    assert await _count_situations(tenant_a) == 12
    assert await _count_situations(tenant_b) == 0
    # And B seeding later is a full seed, not a partial one deduped against A.
    async with with_tenant(tenant_b) as conn:
        assert await seed_situations(conn, tenant_b) == 12


async def test_seed_situations_uses_the_callers_transaction(
    two_tenants: tuple[str, str],
) -> None:
    """It must not open its own connection -- the caller owns the transaction."""
    tenant, _ = two_tenants
    with pytest.raises(RuntimeError):
        async with with_tenant(tenant) as conn:
            await seed_situations(conn, tenant)
            raise RuntimeError("caller aborts after seeding")
    assert await _count_situations(tenant) == 0


# --------------------------------------------------------------------------
# Enable: flip and seed, or neither
# --------------------------------------------------------------------------


async def test_enable_declared_input_flips_and_seeds(two_tenants: tuple[str, str]) -> None:
    tenant, other = two_tenants
    await enable_capability(tenant, WFMEM_INPUT_DECLARED)
    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is True
    assert await _read_prefs(tenant) == {WFMEM_INPUT_DECLARED: True}
    assert await _count_situations(tenant) == 12
    assert await _count_situations(other) == 0


async def test_enable_declared_input_twice_does_not_duplicate(
    two_tenants: tuple[str, str],
) -> None:
    tenant, _ = two_tenants
    await enable_capability(tenant, WFMEM_INPUT_DECLARED)
    await enable_capability(tenant, WFMEM_INPUT_DECLARED)
    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is True
    assert await _count_situations(tenant) == 12


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS - {"wfmem_input_declared"}))
async def test_enable_other_capabilities_does_not_seed(
    two_tenants: tuple[str, str], key: str
) -> None:
    tenant, _ = two_tenants
    await enable_capability(tenant, key)
    assert await is_capability_enabled(tenant, key) is True
    assert await _count_situations(tenant) == 0


async def test_enable_preserves_other_preference_keys(two_tenants: tuple[str, str]) -> None:
    tenant, _ = two_tenants
    await _set_pref(tenant, "wiki_generation_enabled", True)
    await enable_capability(tenant, WFMEM_INPUT_DECLARED)
    assert await _read_prefs(tenant) == {
        "wiki_generation_enabled": True,
        WFMEM_INPUT_DECLARED: True,
    }


async def test_enable_for_a_missing_customer_raises(live_db: None) -> None:
    with pytest.raises(LookupError):
        await enable_capability("cust-does-not-exist", WFMEM_INPUT_DECLARED)


async def test_enable_rolls_the_flag_back_when_seeding_fails(
    two_tenants: tuple[str, str],
) -> None:
    """Atomicity, proven in the database rather than asserted in a comment.

    A BEFORE INSERT trigger makes the seed fail for real, inside the same
    transaction as the flag flip. If the two were separate transactions the
    tenant would come out of this with the capability on and no vocabulary --
    the exact "classifies everything as unknown, serves nothing" state the
    seed-on-enable decision exists to prevent.
    """
    tenant, _ = two_tenants
    async with raw_conn() as conn:
        await conn.execute(
            """
            CREATE OR REPLACE FUNCTION wfmem_test_block_seed() RETURNS TRIGGER AS $$
            BEGIN
                RAISE EXCEPTION 'seed blocked by test';
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        await conn.execute(
            """
            CREATE TRIGGER wfmem_test_block_seed_trg
                BEFORE INSERT ON situations
                FOR EACH ROW EXECUTE FUNCTION wfmem_test_block_seed();
            """
        )
    try:
        with pytest.raises(asyncpg.PostgresError):
            await enable_capability(tenant, WFMEM_INPUT_DECLARED)
    finally:
        async with raw_conn() as conn:
            await conn.execute("DROP TRIGGER IF EXISTS wfmem_test_block_seed_trg ON situations")
            await conn.execute("DROP FUNCTION IF EXISTS wfmem_test_block_seed()")

    assert await _read_prefs(tenant) == {}, "the flag flip survived a failed seed"
    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is False
    assert await _count_situations(tenant) == 0


# --------------------------------------------------------------------------
# The registry and the migration must not drift
# --------------------------------------------------------------------------


def _load_migration_0077() -> ModuleType:
    versions = Path(__file__).resolve().parents[1] / "db" / "migrations" / "versions"
    matches = sorted(versions.glob("*_0077_*.py"))
    assert len(matches) == 1, f"expected exactly one 0077 migration, found {matches}"
    spec = importlib.util.spec_from_file_location("wfmem_migration_0077", matches[0])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_backfills_exactly_the_registry_keys() -> None:
    module = _load_migration_0077()
    assert frozenset(module.WFMEM_CAPABILITY_KEYS_BACKFILLED) == WFMEM_CAPABILITY_KEYS
    assert len(module.WFMEM_CAPABILITY_KEYS_BACKFILLED) == 6


def test_migration_follows_the_head() -> None:
    module = _load_migration_0077()
    assert module.down_revision == "0076_workflow_memory_store"
