"""Per-tenant workflow-memory capability flags, and seed-on-enable.

The flags are six booleans in `customers.preferences`, read with a fail-closed
coercion of wfmem's own -- STRICTER than `engine.shared.customer_prefs`, which
was changed upstream to honour the jsonb string `"true"` for reasons that do not
apply to these cells (see `_coerce_capability_bool`). That posture is the thing
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
* A RE-SEED MUST NOT REVERT A TENANT'S EDITS. The count-based idempotency
  tests cannot see the difference between `DO NOTHING` and an upsert, so the
  wording is asserted on the ROW after an edit, before the count.
* THERE IS AN OFF-SWITCH AND IT HAS TO BE SAFE. `disable_capability` writes
  false rather than deleting the key (the state stays legible to an operator),
  leaves the vocabulary alone, and survives a corrupted blob -- a tenant you
  need to switch off in a hurry is exactly the one whose row may be a mess.
* THE FLAG HAS OTHER WRITERS. The dashboard PATCHes the column directly, so the
  "enabled with no vocabulary" state is reachable without `enable_capability`.
  `enabled_tenants_missing_situations` is the backstop and is tested against a
  tenant flipped on by raw SQL, one seeded properly, one turned off, and one
  whose rows were deleted -- plus a case that fails for the LEFT JOIN written
  without a tenant predicate.
* THE REGISTRY AND THE MIGRATION CANNOT DRIFT, and the migration is never run
  by CI (which stamps the head). 0115 hardcodes its key list on purpose --
  migrations are frozen in time and must not import app code -- so the tests
  compare the two lists, replay the rendered SQL to check the key-absence and
  non-object guards are still there, and execute a real upgrade/downgrade
  round trip to prove those guards bite.

Run with the isolated wfmem database. `PRBE_TEST_DATABASE_URL` is conftest's own
hook for concurrent checkouts on one machine, and it enforces a localhost host --
these fixtures TRUNCATE, so they must never reach a real deployment:

    PRBE_TEST_DATABASE_URL=postgresql://prbe:prbe@localhost:5432/prbe_knowledge_wfmem \
        .venv/bin/pytest tests/test_workflow_memory_capabilities.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType
from typing import Any

import asyncpg
import pytest
import pytest_asyncio
import structlog.testing

from engine.shared.db import raw_conn, with_tenant
from engine.shared.wfmem.capabilities import (
    WFMEM_CAPABILITY_KEYS,
    WFMEM_INPUT_DECLARED,
    InputPath,
    OutputSurface,
    capability_envelope,
    input_capability_key,
    is_capability_enabled,
    output_capability_key,
)
from engine.shared.wfmem.situations import (
    SEED_SITUATIONS,
    disable_capability,
    enable_capability,
    enabled_tenants_missing_situations,
    seed_situations,
)

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

    Deliberately does NOT run migration 0115's backfill: CI stamps the alembic
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
    """`_coerce_capability_bool`'s contract: the value must be a real JSON bool.

    `"true"` is the case that matters and the reason wfmem no longer shares
    `customer_prefs._coerce_bool`: that function was CHANGED upstream to honour
    the jsonb string, to fix a real wiki outage where the engine's PUT wrote a
    string and the Python gate read it as off. These cells have no writer that
    produces the string shape, so honouring it here would mean enabling a
    capability because something wrote it wrong -- and it would hide the tenant
    from `enabled_tenants_missing_situations`, whose jsonb containment match
    does not accept the string either.
    """
    tenant, _ = two_tenants
    await _set_pref(tenant, WFMEM_INPUT_DECLARED, value)
    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is False


async def test_wrong_typed_cell_is_off_and_loud(two_tenants: tuple[str, str]) -> None:
    """A present-but-wrong-typed cell is False AND warns. Absent is False, silently.

    The warning is the whole difference between "this tenant opted out" and
    "something wrote this cell by a route that does not exist" -- two states that
    are otherwise identical on the wire and in the dashboard. Same reasoning as
    `enabled_tenants_missing_situations`: where we cannot prevent the bad state,
    we make it visible rather than waiting for a customer to report that a
    feature they turned on does nothing.
    """
    tenant, _ = two_tenants

    await _set_pref(tenant, WFMEM_INPUT_DECLARED, "true")
    with structlog.testing.capture_logs() as logs:
        assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is False
    assert [entry for entry in logs if entry["event"] == "wfmem_capabilities.non_boolean_cell"], (
        f"a string-valued cell must warn; captured: {logs}"
    )

    # Explicit false is a supported value and must stay quiet -- migration 0115
    # writes it to every cell, so warning here would fire for every tenant.
    await _set_pref(tenant, WFMEM_INPUT_DECLARED, False)
    with structlog.testing.capture_logs() as logs:
        assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is False
    assert not [
        entry for entry in logs if entry["event"] == "wfmem_capabilities.non_boolean_cell"
    ], f"explicit false is not a misconfiguration; captured: {logs}"


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
    monkeypatch.setattr(
        "engine.shared.wfmem.capabilities.raw_conn", _fake_raw_conn(returns="{not json")
    )
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
    monkeypatch.setattr("engine.shared.wfmem.capabilities.raw_conn", _fake_raw_conn(returns=None))
    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is False


async def test_db_error_reads_false(
    two_tenants: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreachable database is not an accidental grant."""
    tenant, _ = two_tenants
    monkeypatch.setattr(
        "engine.shared.wfmem.capabilities.raw_conn",
        _fake_raw_conn(raises=RuntimeError("pool is gone")),
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


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
async def test_envelope_shape_when_off(two_tenants: tuple[str, str], key: str) -> None:
    tenant, _ = two_tenants
    envelope = await capability_envelope(tenant, key)
    assert set(envelope) == {"enabled", "entitled", "upgrade_url"}
    assert envelope["enabled"] is False
    assert envelope["entitled"] is True
    assert envelope["upgrade_url"] is None


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
async def test_envelope_shape_when_on(two_tenants: tuple[str, str], key: str) -> None:
    tenant, _ = two_tenants
    await _set_pref(tenant, key, True)
    envelope = await capability_envelope(tenant, key)
    assert envelope == {"enabled": True, "entitled": True, "upgrade_url": None}


# --------------------------------------------------------------------------
# The seed vocabulary
# --------------------------------------------------------------------------


def test_seed_vocabulary_is_twelve_labels_plus_one_bucket() -> None:
    """Twelve CLASSIFIER LABELS, and `misc`, which is not one of them.

    The count split matters more than the total. `misc` is where a rule goes
    when nothing fit, and it is excluded from the classifier's candidate set on
    purpose: "anything that does not fit the others" has no situation in it to
    embed, so as a label it would either match nothing or weakly match
    everything -- the second of which files rules under misc BY classification
    rather than for want of a home, quietly making the bucket the vocabulary.
    """
    labels = [s for s in SEED_SITUATIONS if s.classifiable]
    buckets = [s for s in SEED_SITUATIONS if not s.classifiable]

    assert len(labels) == 12
    assert [s.slug for s in buckets] == ["misc"], (
        "exactly one non-classifiable bucket; a second would make 'nothing fit' "
        "a choice between buckets, which is a classification again"
    )

    slugs = [s.slug for s in labels]
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
    assert len(set(descriptions)) == len(SEED_SITUATIONS)


async def test_seed_situations_inserts_the_whole_vocabulary(two_tenants: tuple[str, str]) -> None:
    tenant, _ = two_tenants
    async with with_tenant(tenant) as conn:
        inserted = await seed_situations(conn, tenant)
    assert inserted == len(SEED_SITUATIONS)
    assert await _count_situations(tenant) == len(SEED_SITUATIONS)
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
    assert (first, second) == (len(SEED_SITUATIONS), 0)
    assert await _count_situations(tenant) == len(SEED_SITUATIONS)


async def test_seed_situations_is_per_tenant(two_tenants: tuple[str, str]) -> None:
    tenant_a, tenant_b = two_tenants
    async with with_tenant(tenant_a) as conn:
        await seed_situations(conn, tenant_a)
    assert await _count_situations(tenant_a) == len(SEED_SITUATIONS)
    assert await _count_situations(tenant_b) == 0
    # And B seeding later is a full seed, not a partial one deduped against A.
    async with with_tenant(tenant_b) as conn:
        assert await seed_situations(conn, tenant_b) == len(SEED_SITUATIONS)


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


async def test_reseeding_does_not_revert_a_tenants_edits(two_tenants: tuple[str, str]) -> None:
    """The idempotency clause must be DO NOTHING, not an upsert.

    `seed_situations` promises a re-seed will not silently revert somebody's
    wording, and the count-based idempotency tests cannot see the difference:
    `ON CONFLICT DO UPDATE SET label = EXCLUDED.label, description =
    EXCLUDED.description` also inserts nothing the second time and also returns
    0 rows -- while overwriting every edit the tenant made. That is data loss
    with a green suite, so the property needs an assertion on the CONTENT.
    """
    tenant, _ = two_tenants
    async with with_tenant(tenant) as conn:
        await seed_situations(conn, tenant)

    edited_label = "Kicking off a sweep (our wording)"
    edited_description = "House rules for launching anything that spends GPU hours here."
    async with raw_conn() as conn:
        await conn.execute(
            """
            UPDATE situations
               SET label = $2, description = $3
             WHERE customer_id = $1 AND slug = 'launch-run'
            """,
            tenant,
            edited_label,
            edited_description,
        )

    async with with_tenant(tenant) as conn:
        reseeded = await seed_situations(conn, tenant)

    # CONTENT FIRST, deliberately. An upsert mutation also changes the returned
    # count, so asserting the count first would mask which property caught it --
    # and the count is the weaker claim: a variant that upserts without
    # returning the updated rows keeps the count honest and still loses the
    # edits.
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT label, description FROM situations WHERE customer_id = $1 AND slug = $2",
            tenant,
            "launch-run",
        )
    assert row["label"] == edited_label, "a re-seed overwrote the tenant's label"
    assert row["description"] == edited_description, "a re-seed overwrote the tenant's description"
    assert reseeded == 0
    assert await _count_situations(tenant) == len(SEED_SITUATIONS)


# --------------------------------------------------------------------------
# Enable: flip and seed, or neither
# --------------------------------------------------------------------------


async def test_enable_declared_input_flips_and_seeds(two_tenants: tuple[str, str]) -> None:
    tenant, other = two_tenants
    await enable_capability(tenant, WFMEM_INPUT_DECLARED)
    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is True
    assert await _read_prefs(tenant) == {WFMEM_INPUT_DECLARED: True}
    assert await _count_situations(tenant) == len(SEED_SITUATIONS)
    assert await _count_situations(other) == 0


async def test_enable_declared_input_twice_does_not_duplicate(
    two_tenants: tuple[str, str],
) -> None:
    tenant, _ = two_tenants
    await enable_capability(tenant, WFMEM_INPUT_DECLARED)
    await enable_capability(tenant, WFMEM_INPUT_DECLARED)
    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is True
    assert await _count_situations(tenant) == len(SEED_SITUATIONS)


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
# Disable: the kill switch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
async def test_disable_turns_an_enabled_cell_off(two_tenants: tuple[str, str], key: str) -> None:
    tenant, _ = two_tenants
    await enable_capability(tenant, key)
    assert await is_capability_enabled(tenant, key) is True
    await disable_capability(tenant, key)
    assert await is_capability_enabled(tenant, key) is False


async def test_disable_writes_false_rather_than_deleting_the_key(
    two_tenants: tuple[str, str],
) -> None:
    """The explicit-false invariant migration 0115 establishes must survive a disable.

    Deleting the key reads False too, so no behavioural test can tell the two
    apart -- but an operator reading the row during an incident can, and that is
    the difference between "this tenant is switched off" and "this tenant has
    never been configured".
    """
    tenant, _ = two_tenants
    await enable_capability(tenant, WFMEM_INPUT_DECLARED)
    await disable_capability(tenant, WFMEM_INPUT_DECLARED)
    prefs = await _read_prefs(tenant)
    assert WFMEM_INPUT_DECLARED in prefs, "disable deleted the key instead of writing false"
    assert prefs[WFMEM_INPUT_DECLARED] is False


async def test_disable_leaves_the_vocabulary_intact(two_tenants: tuple[str, str]) -> None:
    tenant, _ = two_tenants
    await enable_capability(tenant, WFMEM_INPUT_DECLARED)
    await disable_capability(tenant, WFMEM_INPUT_DECLARED)
    assert await _count_situations(tenant) == len(SEED_SITUATIONS)


async def test_disable_enable_round_trip_does_not_duplicate_or_revert(
    two_tenants: tuple[str, str],
) -> None:
    """Off, then on again -- the flow the kill switch exists for.

    Also covers the edit case: flipping back must not restore stock wording over
    what the tenant changed while it was off.
    """
    tenant, _ = two_tenants
    await enable_capability(tenant, WFMEM_INPUT_DECLARED)
    await disable_capability(tenant, WFMEM_INPUT_DECLARED)
    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE situations SET label = $2 WHERE customer_id = $1 AND slug = 'deploy'",
            tenant,
            "Shipping it (our wording)",
        )
    await enable_capability(tenant, WFMEM_INPUT_DECLARED)

    assert await is_capability_enabled(tenant, WFMEM_INPUT_DECLARED) is True
    assert await _count_situations(tenant) == len(SEED_SITUATIONS)
    async with raw_conn() as conn:
        label = await conn.fetchval(
            "SELECT label FROM situations WHERE customer_id = $1 AND slug = 'deploy'", tenant
        )
    assert label == "Shipping it (our wording)"


async def test_disable_is_per_tenant(two_tenants: tuple[str, str]) -> None:
    tenant_a, tenant_b = two_tenants
    await enable_capability(tenant_a, WFMEM_INPUT_DECLARED)
    await enable_capability(tenant_b, WFMEM_INPUT_DECLARED)
    await disable_capability(tenant_a, WFMEM_INPUT_DECLARED)
    assert await is_capability_enabled(tenant_a, WFMEM_INPUT_DECLARED) is False
    assert await is_capability_enabled(tenant_b, WFMEM_INPUT_DECLARED) is True


async def test_disable_preserves_other_preference_keys(two_tenants: tuple[str, str]) -> None:
    tenant, _ = two_tenants
    await _set_pref(tenant, "wiki_generation_enabled", True)
    await enable_capability(tenant, "wfmem_output_retrieval")
    await disable_capability(tenant, "wfmem_output_retrieval")
    assert await _read_prefs(tenant) == {
        "wiki_generation_enabled": True,
        "wfmem_output_retrieval": False,
    }


async def test_disable_unknown_key_raises(two_tenants: tuple[str, str]) -> None:
    tenant, _ = two_tenants
    with pytest.raises(ValueError):
        await disable_capability(tenant, "wfmem_input_declare")


async def test_disable_for_a_missing_customer_raises(live_db: None) -> None:
    with pytest.raises(LookupError):
        await disable_capability("cust-does-not-exist", WFMEM_INPUT_DECLARED)


async def test_disable_repairs_a_non_object_preferences_blob(
    two_tenants: tuple[str, str],
) -> None:
    """The kill switch must not be the thing that fails during an incident.

    `jsonb_set` raises `cannot set path in scalar` on a non-object blob. A
    tenant whose preferences got corrupted is exactly a tenant somebody may need
    to switch off in a hurry, so the writer replaces the junk rather than
    erroring -- safe, because a scalar blob holds no keys to lose.
    """
    tenant, _ = two_tenants
    await _set_raw_prefs(tenant, '"garbage"')
    await disable_capability(tenant, WFMEM_INPUT_DECLARED)
    assert await _read_prefs(tenant) == {WFMEM_INPUT_DECLARED: False}


# --------------------------------------------------------------------------
# The backstop: enabled tenants with no vocabulary
# --------------------------------------------------------------------------


async def test_audit_finds_a_tenant_flipped_on_outside_enable_capability(
    two_tenants: tuple[str, str],
) -> None:
    """The state `enable_capability` cannot produce but the dashboard can.

    A bare PATCH of the JSONB column is the realistic route in, so the fixture
    uses exactly that -- a plain UPDATE, no seeding.
    """
    tenant_a, tenant_b = two_tenants
    await _set_pref(tenant_a, WFMEM_INPUT_DECLARED, True)  # flag on, nothing seeded
    await enable_capability(tenant_b, WFMEM_INPUT_DECLARED)  # the correct door

    missing = await enabled_tenants_missing_situations()
    assert tenant_a in missing, "an enabled tenant with no vocabulary went unreported"
    assert tenant_b not in missing, "a correctly seeded tenant was reported as broken"


async def test_audit_ignores_tenants_with_the_capability_off(
    two_tenants: tuple[str, str],
) -> None:
    _, tenant_b = two_tenants
    # Never configured (tenant A), and explicitly off (tenant B): neither is a
    # problem, and neither has a vocabulary.
    await _set_pref(tenant_b, WFMEM_INPUT_DECLARED, False)
    assert await enabled_tenants_missing_situations() == []


async def test_audit_does_not_count_a_string_true_as_enabled(
    two_tenants: tuple[str, str],
) -> None:
    """Same contract as the reader: only a real JSON bool is on.

    A `"true"` string reads as off, so a tenant carrying one is not enabled and
    must not appear here -- otherwise the audit reports work for a tenant who is
    getting nothing anyway.
    """
    tenant_a, _ = two_tenants
    await _set_pref(tenant_a, WFMEM_INPUT_DECLARED, "true")
    assert await enabled_tenants_missing_situations() == []


async def test_audit_reports_a_tenant_whose_vocabulary_was_deleted(
    two_tenants: tuple[str, str],
) -> None:
    """Seeded correctly, then emptied -- a restore that replayed one table."""
    tenant_a, _ = two_tenants
    await enable_capability(tenant_a, WFMEM_INPUT_DECLARED)
    assert await enabled_tenants_missing_situations() == []
    async with raw_conn() as conn:
        await conn.execute("DELETE FROM situations WHERE customer_id = $1", tenant_a)
    assert await enabled_tenants_missing_situations() == [tenant_a]


async def test_audit_is_not_confused_by_another_tenants_vocabulary(
    two_tenants: tuple[str, str],
) -> None:
    """B having situations must not make A look healthy.

    The obvious LEFT JOIN written without a tenant predicate passes every other
    test in this section and fails this one.
    """
    tenant_a, tenant_b = two_tenants
    await _set_pref(tenant_a, WFMEM_INPUT_DECLARED, True)
    await enable_capability(tenant_b, WFMEM_INPUT_DECLARED)
    assert await enabled_tenants_missing_situations() == [tenant_a]


# --------------------------------------------------------------------------
# The registry and the migration must not drift
# --------------------------------------------------------------------------


def _load_migration_0115() -> ModuleType:
    versions = Path(__file__).resolve().parents[1] / "db" / "migrations" / "versions"
    matches = sorted(versions.glob("*_0115_*.py"))
    assert len(matches) == 1, f"expected exactly one 0115 migration, found {matches}"
    spec = importlib.util.spec_from_file_location("wfmem_migration_0115", matches[0])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rendered_sql(direction: str) -> list[str]:
    """The SQL migration 0115 would run, without an alembic context.

    Same trick as tests/test_workflow_memory_isolation.py: swap the module's
    `op` for a recorder so this replays what is IN THE FILE, with the keys
    actually interpolated, rather than substring-matching an f-string template
    that may or may not render the way it reads.
    """
    module = _load_migration_0115()
    collected: list[str] = []

    class _Recorder:
        @staticmethod
        def execute(sql: object) -> None:
            collected.append(str(sql))

    module.op = _Recorder  # type: ignore[attr-defined]
    getattr(module, direction)()
    return collected


def test_migration_backfills_exactly_the_registry_keys() -> None:
    module = _load_migration_0115()
    assert frozenset(module.WFMEM_CAPABILITY_KEYS_BACKFILLED) == WFMEM_CAPABILITY_KEYS
    assert len(module.WFMEM_CAPABILITY_KEYS_BACKFILLED) == 6


def test_migration_follows_the_head() -> None:
    module = _load_migration_0115()
    assert module.down_revision == "0114_workflow_memory_store"
    assert len(module.revision) <= 32, "alembic_version.version_num is varchar(32)"


def test_upgrade_sql_keeps_the_key_absence_guard() -> None:
    """The one clause standing between this backfill and wiping every opt-in.

    Without `WHERE NOT (... ? key)` the UPDATE is unconditional and rewrites
    every tenant's `true` to `false`. Nothing else in the suite would notice:
    the migration is never executed by CI, which stamps the head instead.
    """
    statements = _rendered_sql("upgrade")
    assert len(statements) == 6, f"expected one UPDATE per key, got {len(statements)}"
    for key in sorted(WFMEM_CAPABILITY_KEYS):
        matching = [s for s in statements if f"'{key}'" in s]
        assert len(matching) == 1, f"no single statement for {key}"
        sql = matching[0]
        assert f"WHERE NOT (COALESCE(preferences, '{{}}'::jsonb) ? '{key}')" in sql, (
            f"the key-absence guard is missing for {key}; this UPDATE is unconditional"
        )
        assert "jsonb_set" in sql
        assert "'false'::jsonb" in sql


def test_upgrade_and_downgrade_skip_non_object_blobs() -> None:
    """`jsonb_set` and `-` both raise on a scalar, aborting the whole migration."""
    for direction in ("upgrade", "downgrade"):
        statements = _rendered_sql(direction)
        assert len(statements) == 6
        for sql in statements:
            assert "jsonb_typeof" in sql and "= 'object'" in sql, (
                f"{direction} has no non-object guard: one junk blob aborts the run"
            )


def test_downgrade_sql_removes_exactly_the_six_keys() -> None:
    statements = _rendered_sql("downgrade")
    for key in sorted(WFMEM_CAPABILITY_KEYS):
        matching = [s for s in statements if f"'{key}'" in s]
        assert len(matching) == 1
        assert f"preferences - '{key}'" in matching[0]


async def test_migration_round_trip_against_a_live_database(live_db: None) -> None:
    """Execute the real upgrade/downgrade and assert on the rows.

    Stronger than reading the SQL: it proves the guards BITE rather than merely
    being present. Runs the revision's own functions through an alembic
    operations context on a synchronous connection -- the same code path
    `alembic upgrade` takes, minus the version stamp.
    """
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import create_engine, text

    module = _load_migration_0115()
    url = os.environ["DATABASE_URL"]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url.removeprefix("postgresql://")
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO customers (customer_id, display_name, api_key_hash, preferences)
                    VALUES
                      ('mig-empty', 'e', 'h1', '{}'::jsonb),
                      ('mig-optedin', 'i', 'h2', '{"wfmem_input_declared": true}'::jsonb),
                      ('mig-mixed', 'm', 'h3',
                       '{"wfmem_output_midsession": false, "wiki_generation_enabled": true}'::jsonb),
                      ('mig-scalar', 's', 'h4', '"garbage"'::jsonb)
                    """
                )
            )
            ctx = MigrationContext.configure(conn)

            with Operations.context(ctx):
                module.upgrade()
            after_up = _prefs_by_customer(conn, text)

            assert after_up["mig-empty"] == dict.fromkeys(sorted(WFMEM_CAPABILITY_KEYS), False)
            assert after_up["mig-optedin"][WFMEM_INPUT_DECLARED] is True, (
                "the backfill overwrote a tenant's opt-in"
            )
            assert set(after_up["mig-optedin"]) == WFMEM_CAPABILITY_KEYS
            assert after_up["mig-mixed"]["wiki_generation_enabled"] is True
            assert after_up["mig-mixed"]["wfmem_output_midsession"] is False
            assert set(after_up["mig-mixed"]) == WFMEM_CAPABILITY_KEYS | {"wiki_generation_enabled"}
            assert after_up["mig-scalar"] == "garbage", "a scalar blob was rewritten"

            with Operations.context(ctx):
                module.downgrade()
            after_down = _prefs_by_customer(conn, text)

            assert after_down["mig-empty"] == {}
            assert after_down["mig-optedin"] == {}
            assert after_down["mig-mixed"] == {"wiki_generation_enabled": True}
            assert after_down["mig-scalar"] == "garbage"

            conn.execute(text("DELETE FROM customers WHERE customer_id LIKE 'mig-%'"))
    finally:
        engine.dispose()


def _prefs_by_customer(conn: Any, text: Any) -> dict[str, Any]:
    rows = conn.execute(
        text("SELECT customer_id, preferences FROM customers WHERE customer_id LIKE 'mig-%'")
    ).all()
    return {row[0]: row[1] for row in rows}
