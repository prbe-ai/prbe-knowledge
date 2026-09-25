"""Which tenants the engine may do work for: ACTIVE ones, and nobody else.

When research-os terminates a team it keeps the team's data for a fixed hold
before purging it, and the kb `customers` row says so: `status = 'terminated'`
for the hold, then `'deleted'` once the purge is due. The managed plane's soft
delete also writes `'deleted'`. During that time NOTHING may be processed for
the tenant -- no mining (a paid model call), no embedding, no polling its
sources with its credentials, no new writes. Before this module the engine had
no notion of it: the idle-session sweep enumerated every `customers` row and
re-mined held sessions, and the queue claim took any pending row.

So every automatic producer that enumerates tenants reads ACTIVE_TENANTS_SQL,
every queue claim carries `active_tenant_sql(...)`, and the ingest doors refuse
new writes (`refusal_for`). A held tenant's queued rows are not touched: they
stay pending, are simply never claimed, and go with the tenant when research-os
purges it (every tenant table cascades from `customers`). If a tenant is ever
set back to active, its work resumes where it stopped.

Purge, retention and index-maintenance paths deliberately do NOT use this: they
must reach every tenant, held ones included (scripts/cron_chunk_retention.py,
the pg_search guardian's partition drop, kb/purge_routes.py).

COST. A claim walks its queue index in order, so a held tenant's rows at the
head are stepped over on every claim; the per-row check is a memoized
primary-key probe. Measured on a local copy (300k done + 2k live pending
ingestion rows, 20 tenants): the ingestion claim is 0.18 ms with no held rows,
0.36 ms with 500, 3.5 ms with 10k, 9 ms with 50k. The post-write claim is
14 ms at 40k held rows and 67 ms at 200k -- about twice the cost of that
worker's existing un-indexed "due" leg over the same table.
"""

from __future__ import annotations

from typing import Any

from engine.shared.constants import CustomerStatus
from engine.shared.db import get_pool

#: The machine-readable reason an ingest door gives for a 409.
TENANT_NOT_ACTIVE = "tenant_not_active"

_ACTIVE = CustomerStatus.ACTIVE.value

#: Every tenant work may be done for, in a stable order. `customers` has no RLS.
ACTIVE_TENANTS_SQL = (
    f"SELECT customer_id FROM customers WHERE status = '{_ACTIVE}' ORDER BY customer_id"
)


def active_tenant_sql(customer_col: str) -> str:
    """A predicate that is true when `customer_col`'s tenant is ACTIVE.

    One primary-key probe per row. The alias is deliberately unusual: a query
    that already calls some table `c` and passes `c.customer_id` would
    otherwise have that reference captured by the subquery's own alias and
    compare the customers row with itself -- true for every row.
    """
    return (
        "EXISTS (SELECT 1 FROM customers active_tenant"
        f" WHERE active_tenant.customer_id = {customer_col}"
        f" AND active_tenant.status = '{_ACTIVE}')"
    )


async def refusal_for(customer_id: str) -> dict[str, Any] | None:
    """The 409 body for a write to a tenant that exists and is NOT active.

    None when the tenant is active -- or has no `customers` row at all, which
    keeps each door's existing behaviour for an unknown tenant (research-os
    seeds the row before its first push; a missing one fails further in).
    """
    async with get_pool().acquire() as conn:
        status = await conn.fetchval(
            "SELECT status FROM customers WHERE customer_id = $1", customer_id
        )
    if status is None or status == _ACTIVE:
        return None
    return {"reason": TENANT_NOT_ACTIVE, "status": status}
