"""`companion_infra`: one fail-closed capability cell, its own accessor.

The wfmem reader (`engine.shared.wfmem.capabilities`) REFUSES any key outside
its six-cell registry -- so the companion cannot simply reuse it with a new key
name; it needs its own registered accessor. The coercion rules are shared on
purpose (a real JSON `true` is the only ON; a string "true" is a wrong-typed
cell and reads OFF), so the two readers cannot drift on what "on" means.

Run with the isolated database:

    PRBE_TEST_DATABASE_URL=postgresql://prbe:prbe@localhost:55442/prbe_knowledge \
        .venv/bin/pytest tests/test_companion_capability.py -q
"""

from __future__ import annotations

import pytest

from engine.shared.companion.capability import (
    COMPANION_INFRA,
    companion_envelope,
    is_companion_enabled,
)
from engine.shared.db import raw_conn


async def _customer(customer_id: str, preferences_sql: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash, preferences) "
            f"VALUES ($1, 'c', 'h-' || $1, {preferences_sql})",
            customer_id,
        )


def test_key_name_is_stable() -> None:
    # Two other modules will key off this string (the research-os flip and the
    # tenant onboarding flow); renaming it is a migration, not a refactor.
    assert COMPANION_INFRA == "companion_infra"


@pytest.mark.asyncio
async def test_absent_tenant_is_off(live_db) -> None:
    assert await is_companion_enabled("no-such-tenant") is False


@pytest.mark.asyncio
async def test_empty_customer_id_is_off(live_db) -> None:
    assert await is_companion_enabled("") is False


@pytest.mark.asyncio
async def test_missing_key_is_off(live_db) -> None:
    await _customer("cust-cap-none", "'{}'::jsonb")
    assert await is_companion_enabled("cust-cap-none") is False


@pytest.mark.asyncio
async def test_real_boolean_true_is_on(live_db) -> None:
    await _customer("cust-cap-on", "jsonb_build_object('companion_infra', true)")
    assert await is_companion_enabled("cust-cap-on") is True
    assert await companion_envelope("cust-cap-on") == {
        "enabled": True,
        "entitled": True,
        "upgrade_url": None,
    }


@pytest.mark.asyncio
async def test_string_true_is_off(live_db) -> None:
    await _customer("cust-cap-str", "jsonb_build_object('companion_infra', 'true')")
    assert await is_companion_enabled("cust-cap-str") is False


@pytest.mark.asyncio
async def test_sibling_wfmem_cells_do_not_leak(live_db) -> None:
    """Every wfmem cell on, companion cell absent: still off."""
    await _customer(
        "cust-cap-wfmem",
        "jsonb_build_object('wfmem_output_retrieval', true, 'wfmem_output_midsession', true)",
    )
    assert await is_companion_enabled("cust-cap-wfmem") is False
