"""The wiki is decommissioned: every write refuses, every read keeps serving.

Generation stopped fleet-wide on 2026-08-18 (research-os deploy.yml sets
`engine.wiki.enabled=false`, which removes the triage, synthesis and backfill
workers), and the feature is being removed in favour of first-class notes.

WHY THIS EXISTS HERE TOO, when research-os already refuses these writes at its
own proxy. Refusing only at the proxy assumes the proxy is the only caller,
and it is not: this API is reachable by anything holding
`INTERNAL_KNOWLEDGE_API_KEY` -- the dashboard through the prbe-backend BFF,
operational scripts, and any service someone points at it later. A refusal
that lives only in the consumer stops the callers you listed, not the callers
you have.

THE DESTRUCTIVE ONES ARE WHY THIS SHIPPED FIRST. `/backfill/trigger` and
`/bootstrap/trigger` wipe a tenant's compiled pages before re-crawling, and
the workers that would rebuild them no longer exist. So the wipe half of a
rebuild still ran and the rebuild half never came: a tenant that clicked the
button would have been left with an empty wiki and a run row that never
finishes.

WHAT DELIBERATELY KEEPS WORKING: every read, and `POST /backfill/undo`. Undo
restores compiled pages that have no live version, so it can only put back
what a wipe took -- and for a tenant already caught by the gap above, it is
the only way home. It retires with the rest of the surface once the sweep for
such tenants has run.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

#: Marks an endpoint as a decommissioned wiki write, for the route-table
#: invariant in `tests/test_wiki_decommissioned.py`. A hand-kept list of paths
#: is satisfied by the paths we knew about on the day and says nothing about
#: the next route someone adds.
DECOMMISSIONED_ATTR = "__wiki_decommissioned__"

WIKI_DECOMMISSIONED_DETAIL = (
    "the team wiki is decommissioned and no longer accepts writes. Nothing was "
    "stored. Generation stopped fleet-wide on 2026-08-18 and the pages are "
    "scheduled for deletion; the triage, synthesis and backfill workers are no "
    "longer deployed, so a rebuild would wipe pages nothing can re-derive. "
    "Agent-written prose belongs in notes now (research-os: PATCH "
    "/v1/projects/{project_id} with `notes_append`). Reads still serve, and "
    "POST /api/wiki/backfill/undo is still open as the recovery path."
)


def decommissioned_wiki_write[F: Callable[..., Any]](func: F) -> F:
    """Replace a wiki write handler with a permanent 410.

    Replaces the BODY rather than adding a dependency so that the checks
    already in front of these routes keep answering first: the shared-secret
    check on the router, `_require_customer`, and FastAPI's own body
    validation. A dependency is solved before any of the request is validated,
    which would turn a malformed call into "gone" and lose the caller the one
    diagnosis they can act on.

    The original body stays in the file, unreachable, for this phase only: the
    compare-and-swap, the per-page advisory lock and the idempotency-key
    contract are written down in those handlers, and that is what the notes
    work reads when it decides which of them to carry over. Phase 3 deletes
    handlers and decorator together.
    """

    @functools.wraps(func)
    async def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise HTTPException(status_code=410, detail=WIKI_DECOMMISSIONED_DETAIL)

    setattr(_refuse, DECOMMISSIONED_ATTR, True)
    return _refuse  # type: ignore[return-value]


def is_decommissioned(endpoint: Any) -> bool:
    """Does this endpoint refuse as a decommissioned wiki write?"""
    return bool(getattr(endpoint, DECOMMISSIONED_ATTR, False))
