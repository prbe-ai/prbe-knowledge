"""Legal hold and tenant state: may Probe delete this tenant's data right now?

A legal hold is set in research-os and mirrored into this database as
`customers.metadata.legal_hold` (a reason string; the key absent means no
hold). While it is set, nothing Probe runs on its own may destroy that
tenant's data. Customer-initiated deletes still happen upstream; what a hold
stops is the engine's background destroyers finishing the job later, the
tombstone purge first among them.

Two rules, and both are checked AT EXECUTION TIME, per committed batch, not
once when a job starts: a hold set while a run is in flight must stop the next
delete, not the next run.

  1. `status` must be 'active'. A terminated tenant is purged wholesale by
     research-os (its purge reaper drains every table); an engine job deleting
     pieces of it in parallel would race that and could run during the hold
     window the reaper is deliberately waiting out.
  2. No legal hold.

FAIL CLOSED on the hold's shape. Any value other than JSON null counts as a
hold -- an empty string, `false`, an object. The contract is "a reason string,
or the key absent", and the two ways to get that wrong are not symmetric:
reading a malformed hold as "no hold" destroys evidence, which cannot be
undone; reading it as a hold keeps data a little longer, which is visible and
fixable. JSON null is the one other value treated as "no hold", because it is
the natural way to clear a key without removing it.
"""

from __future__ import annotations

from typing import Final

import asyncpg

#: The key research-os mirrors a hold into, on `customers.metadata`.
LEGAL_HOLD_KEY: Final = "legal_hold"


def legal_hold_sql(alias: str = "customers") -> str:
    """SQL boolean, TRUE when the `customers` row under `alias` is on hold."""
    return (
        f"(coalesce(jsonb_typeof({alias}.metadata -> '{LEGAL_HOLD_KEY}'), 'null')"
        " <> 'null')"
    )


def purge_eligible_tenant_sql(alias: str = "customers") -> str:
    """SQL boolean, TRUE when background destroyers may act on this tenant."""
    return f"({alias}.status = 'active' AND NOT {legal_hold_sql(alias)})"


async def purge_blocked_reason(
    conn: asyncpg.Connection, customer_id: str, *, lock: bool = False
) -> str | None:
    """Why a background destroyer must not touch `customer_id` now, or None.

    Returns 'missing' (no customers row), 'status:<status>', or 'legal_hold'.

    `lock=True` takes FOR SHARE on the customers row, which a caller holding a
    transaction open across its deletes should use: the UPDATE that sets a hold
    then waits for that transaction to commit, so once the hold is visible no
    later delete of this caller can still commit. Without the lock a hold
    committed mid-batch lets that one batch finish.
    """
    row = await conn.fetchrow(
        f"""
        SELECT c.status, {legal_hold_sql("c")} AS held
        FROM customers c
        WHERE c.customer_id = $1
        {"FOR SHARE" if lock else ""}
        """,
        customer_id,
    )
    if row is None:
        return "missing"
    if row["status"] != "active":
        return f"status:{row['status']}"
    if row["held"]:
        return LEGAL_HOLD_KEY
    return None
