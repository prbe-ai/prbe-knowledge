"""The wiki-enabled gate: SQL predicate and Python coercion must agree.

Both sides answer "is this tenant's wiki on", on different schedules: the
SQL fragment (`wiki_enabled_sql`) drives the queue drain guards, the
nightly reconcile, and the catchup CLI; `is_wiki_generation_enabled`
drives the Normalizer enqueue and the workers' short-circuits. They had
already diverged once when this file was written: the engine's own
PUT /settings writes the STRING "true" (to_jsonb of a text parameter),
which `->> = 'true'` accepted and `value is True` rejected — a tenant
enabled through the product surface drained in SQL and silently stopped
enqueueing in Python. Every row here pins both sides to the same answer.
"""

from __future__ import annotations

import pytest

from engine.shared.customer_prefs import (
    is_wiki_generation_enabled,
    wiki_enabled_sql,
)
from engine.shared.db import raw_conn
from tests.wiki_fixtures import insert_customer

CUSTOMER = "prefs-parity-cust"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prefs_json", "expected"),
    [
        ('{"wiki_generation_enabled": true}', True),
        # The shape the engine PUT writes. MUST be ON on both sides.
        ('{"wiki_generation_enabled": "true"}', True),
        ('{"wiki_generation_enabled": false}', False),
        ('{"wiki_generation_enabled": "false"}', False),
        ('{"wiki_generation_enabled": null}', False),
        ('{"wiki_generation_enabled": 1}', False),
        ("{}", False),
    ],
)
async def test_sql_predicate_and_python_gate_agree(
    live_db: None, prefs_json: str, expected: bool
) -> None:
    await insert_customer(CUSTOMER, preferences=prefs_json)
    async with raw_conn() as conn:
        sql_side = await conn.fetchval(
            f"SELECT {wiki_enabled_sql()} FROM customers WHERE customer_id = $1",
            CUSTOMER,
        )
    py_side = await is_wiki_generation_enabled(CUSTOMER)
    assert bool(sql_side) is expected, "SQL side disagrees with the table above"
    assert py_side is expected, "Python side disagrees with the table above"


def test_wiki_enabled_sql_rejects_non_identifiers() -> None:
    """Defense in depth: the fragment interpolates its column argument, so
    a caller can never be allowed to pass anything but an identifier."""
    for good in ("preferences", "c.preferences", "cust_2.prefs"):
        assert "->>" in wiki_enabled_sql(good)
    for bad in ("preferences; DROP TABLE customers", "p'x", "a.b.c", ""):
        with pytest.raises(ValueError):
            wiki_enabled_sql(bad)
