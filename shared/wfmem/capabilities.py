"""Per-tenant capability flags for workflow memory.

Six booleans in `customers.preferences` -- the same JSONB column and the same
fail-closed coercion `shared.customer_prefs` uses, because these flags gate the
same kind of thing: work a tenant has to explicitly ask for. A missing key, a
missing customer, a blob that is not an object, a string `"true"` instead of a
bool, an unreadable database: every one of those is **off**.

TWO AXES, NOT ONE SWITCH
------------------------
Workflow memory has three ways knowledge gets IN and three surfaces it comes
OUT of, and a tenant can want any of them independently -- importing an
existing runbook is a different consent than mining transcripts, and a
mid-session nudge is a different intrusion than a card in a search result::

    wfmem_input_declared      wfmem_output_retrieval
    wfmem_input_imported      wfmem_output_compiled
    wfmem_input_mined         wfmem_output_midsession

So the six cells are two axes of three, NOT a 3x3 grid, and this module offers
two accessors rather than one `capability_key(input, output)`. A combined
accessor would advertise nine cells, six of which do not exist.

WHY A REGISTRY AT ALL
---------------------
Both existing implementations of this pattern in our code -- the
`is_wiki_generation_enabled` gate here and `is_enrichment_enabled` in
prbe-orchestrator -- build the pref key by string concatenation with no
registry at all. Under a fail-closed reader that is a quiet trap: a misspelled
key is not an error, it is a permanent `False`, and it is indistinguishable
from a tenant who chose not to opt in. Nobody notices until someone asks why
the feature "never turned on" for one customer.

This module is READ-ONLY, like `shared.customer_prefs`. The writers --
`enable_capability` (which also seeds) and `disable_capability` (the kill
switch) -- live in `shared.wfmem.situations`, because enabling touches the
situations table and the import can only point one way.

Here the key set is DERIVED from the two axes (so it cannot be half-updated by
hand), the accessors go through the enums (so an off-axis value raises at
construction), and `is_capability_enabled` REFUSES a key that is not in the
registry. The only thing allowed to read as off is a genuinely absent tenant
key.
"""

from __future__ import annotations

from enum import StrEnum

# The coercion, imported rather than copied. These six cells live in the same
# JSONB column, are written by the same dashboard PATCH, and must answer to the
# same rules; a second local copy is a second place for the contract to drift.
# Private-by-underscore and imported anyway: the alternative is a duplicate
# whose divergence would be silent.
from shared.customer_prefs import _coerce_bool
from shared.db import raw_conn
from shared.logging import get_logger

log = get_logger(__name__)


class InputPath(StrEnum):
    """How a clause gets INTO the store."""

    #: A human wrote the rule down on purpose.
    DECLARED = "declared"
    #: Lifted from a document the team already maintains (runbook, README, wiki).
    IMPORTED = "imported"
    #: Inferred from observed behaviour -- transcripts, PRs, run outcomes.
    MINED = "mined"


class OutputSurface(StrEnum):
    """Where a clause comes OUT."""

    #: Returned alongside search results, on request.
    RETRIEVAL = "retrieval"
    #: Baked into a compiled procedure handed to an agent at task start.
    COMPILED = "compiled"
    #: Injected while a session is already running.
    MIDSESSION = "midsession"


_INPUT_PREFIX = "wfmem_input_"
_OUTPUT_PREFIX = "wfmem_output_"


def input_capability_key(path: InputPath | str) -> str:
    """Pref key for one input path. Raises ValueError on an off-axis value."""
    return f"{_INPUT_PREFIX}{InputPath(path).value}"


def output_capability_key(surface: OutputSurface | str) -> str:
    """Pref key for one output surface. Raises ValueError on an off-axis value."""
    return f"{_OUTPUT_PREFIX}{OutputSurface(surface).value}"


#: The whole registry, derived from the axes. Migration 0077 backfills exactly
#: this set (hardcoded there, because a migration must not import app code that
#: can change under it); a test compares the two lists.
WFMEM_CAPABILITY_KEYS: frozenset[str] = frozenset(
    {input_capability_key(path) for path in InputPath}
    | {output_capability_key(surface) for surface in OutputSurface}
)

#: Named because two other modules key off it: seeding hangs off this cell, and
#: the dashboard's onboarding flow flips it first.
WFMEM_INPUT_DECLARED = input_capability_key(InputPath.DECLARED)


def require_capability_key(key: str) -> str:
    """Return `key`, or raise ValueError if it is not one of the six.

    Loud on purpose. A typo here is a programming error and must not be
    laundered into "this tenant opted out".
    """
    if key not in WFMEM_CAPABILITY_KEYS:
        known = ", ".join(sorted(WFMEM_CAPABILITY_KEYS))
        raise ValueError(f"unknown workflow-memory capability key {key!r}; known keys: {known}")
    return key


async def is_capability_enabled(customer_id: str, key: str) -> bool:
    """True iff the tenant has explicitly opted into this capability.

    Fail-closed on every data path: missing customer, missing key, non-object
    blob, non-boolean value, JSON decode failure, DB error. Callers gate real
    work on this, so a False must be safe to act on even when the database is
    unreachable.

    NOT fail-closed on a bad `key` -- that raises. See `require_capability_key`.
    """
    require_capability_key(key)
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
            "wfmem_capabilities.read_failed",
            customer=customer_id,
            key=key,
            error=str(exc),
            error_class=type(exc).__name__,
        )
        return False
    return _coerce_bool(raw, key)


async def capability_envelope(customer_id: str, key: str) -> dict[str, object]:
    """The three-state answer for one capability.

    THE SHAPE IS A HOUSE PATTERN, not an invention here: it mirrors
    `WikiGenerationSettingsOut` in research-os `app/wiki/schemas.py`
    (`enabled` / `entitled: bool = True` / `upgrade_url: str | None = None`),
    which reached it for the same reason. THREE STATES, NOT TWO, and `enabled`
    alone cannot spell them: on, off-because-nobody-turned-it-on, and
    off-because-the-plan-does-not-include-it. The third reads identically to
    the second on the wire, and a reader who cannot tell them apart concludes
    the switch is broken -- renders a toggle, flips it, nothing happens, and
    files "the feature is broken" rather than "we are not paying for this".

    `entitled` is hardcoded True today and that is deliberate, not a stub left
    in by accident: the entitlement layer is Phase 3. Note what the default is
    doing over there -- True so that a client built BEFORE the gate still reads
    an entitled team correctly -- and that is exactly why the field goes in now
    rather than with the gate. A field that arrives later has to arrive
    everywhere at once; a field that is already there, already defaulted the
    permissive way, cannot break a client by showing up.

    So when Phase 3 lands, only the VALUES change: `entitled` starts coming
    back False for some tenants and `upgrade_url` starts carrying a link. No
    client needs a new field to handle it, and until then `upgrade_url` stays
    None -- there is nowhere to send an entitled tenant, and a client that
    renders any string it is given would otherwise link to nowhere.
    """
    return {
        "enabled": await is_capability_enabled(customer_id, key),
        "entitled": True,
        "upgrade_url": None,
    }
