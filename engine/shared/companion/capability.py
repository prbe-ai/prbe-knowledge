"""`companion_infra`: the one per-tenant capability cell for the companion.

A boolean in `customers.preferences`, read fail-closed with the same coercion
the workflow-memory cells use (`engine.shared.wfmem.capabilities`): the only ON
is a real JSON `true`; absent, wrong-typed, unreadable and missing-tenant all
read OFF. The coercer is shared on purpose so the two readers cannot disagree
about what "on" means.

WHY NOT REUSE THE WFMEM READER. `is_capability_enabled` there REFUSES any key
outside its six-cell registry -- loud on purpose, so a typo cannot be laundered
into "this tenant opted out". That is exactly right for wfmem and exactly why
the companion needs its own registered accessor rather than a seventh key
smuggled into a registry derived from two axes it does not belong to.

READ-ONLY, like its sibling. The writer is an operator step for now
(`UPDATE customers SET preferences = preferences || '{"companion_infra": true}'`
under the tenant's consent); when the research-os toggle lands it writes a real
boolean through `to_jsonb($1::boolean)`, never a formatted literal.
"""

from __future__ import annotations

from engine.shared.db import raw_conn
from engine.shared.logging import get_logger
from engine.shared.wfmem.capabilities import coerce_capability_bool

log = get_logger(__name__)

#: The cell name. Two other places will key off it (the research-os flip and
#: tenant onboarding); renaming it is a data migration, not a refactor.
COMPANION_INFRA = "companion_infra"


async def is_companion_enabled(customer_id: str) -> bool:
    """True iff the tenant has explicitly opted into the companion transport.

    Fail-closed on every data path: missing customer, missing key, non-object
    blob, non-boolean value, JSON decode failure, DB error. Callers gate real
    work on this, so a False must be safe to act on when the database is down.
    """
    if not customer_id:
        return False
    try:
        async with raw_conn() as conn:
            raw = await conn.fetchval(
                "SELECT preferences FROM customers WHERE customer_id = $1",
                customer_id,
            )
    except Exception as exc:
        log.warning(
            "companion_capability.read_failed",
            customer=customer_id,
            error=str(exc),
            error_class=type(exc).__name__,
        )
        return False
    return coerce_capability_bool(raw, COMPANION_INFRA)


async def companion_envelope(customer_id: str) -> dict[str, object]:
    """The house three-state shape: `enabled` / `entitled` / `upgrade_url`.

    Same contract as `wfmem.capabilities.capability_envelope`: `entitled` is
    hardcoded True until the entitlement layer exists, and `upgrade_url` stays
    None because there is nowhere to send anyone yet. Off-because-nobody-turned-
    it-on and off-because-not-entitled must stay distinguishable on the wire.
    """
    return {
        "enabled": await is_companion_enabled(customer_id),
        "entitled": True,
        "upgrade_url": None,
    }
