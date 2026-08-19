"""Every wiki write refuses, and no future route can quietly miss the cutover.

An invariant over the live route table rather than a list of paths: a list is
satisfied by the routes that existed when it was written, so the eleventh route
someone adds is exactly the one it will not catch. The reads are asserted in the
other direction too -- research-os proxies them for its own deprecation window,
so a refusal that spread to a read would break stale clients here rather than in
the repo that owns the decision.
"""

from __future__ import annotations

import pytest
from fastapi.routing import APIRoute

from kb.wiki_decommission import is_decommissioned
from kb.wiki_routes import router as wiki_router

_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Mutating routes that deliberately still work, with the reason.
#:
#: THE SCOPE HERE IS NARROWER THAN research-os's ON PURPOSE. research-os
#: refuses every wiki write, which covers every caller that is not holding
#: `INTERNAL_KNOWLEDGE_API_KEY` -- that is the agent-facing cutover, and it is
#: complete. What this repo adds is the half research-os cannot reach: the
#: routes that DESTROY or SPEND. A rebuild wipes compiled pages and the
#: workers that would re-derive them are gone; a synthesis trigger starts LLM
#: work with nothing to consume it. Page writes are left callable at the engine
#: layer because nothing external can reach them any more, and because they are
#: how this repo's tests set up the state that reads, references and undo are
#: still tested against. Phase 3 deletes the file and the distinction with it.
_RECOVERY_ROUTES = {
    "/api/wiki/backfill/undo": (
        "the recovery path: restores compiled pages that have no live version, "
        "so it can only put back what a wipe took. With the synthesis workers "
        "gone it is the only way home for a tenant whose rebuild wiped its "
        "pages and then had nothing to re-derive them with."
    ),
    "/api/wiki/pages/{wiki_type}/{slug}": (
        "page writes: unreachable from outside because research-os refuses "
        "them, and still the seam this repo's own tests build state with"
    ),
    "/api/wiki/pages/{wiki_type}/{slug}/append": "page write; see above",
    "/api/wiki/pages/{wiki_type}/{slug}/settings": "page write; see above",
    "/api/wiki/pages/{wiki_type}/{slug}/revert": "page write; see above",
    "/api/wiki/synthesize/dlq/reset": (
        "an operator recovery tool that only moves dead-lettered rows back to "
        "pending. It neither destroys pages nor starts LLM work -- with no "
        "workers to drain the queue it is inert -- so refusing it would buy "
        "nothing and take away a lever during the wind-down."
    ),
}


def _routes() -> list[APIRoute]:
    routes = [r for r in wiki_router.routes if isinstance(r, APIRoute)]
    assert routes, "no wiki routes found -- the router moved and this test went blind"
    return routes


@pytest.mark.parametrize(
    "path,method",
    [
        pytest.param(r.path, m, id=f"{m} {r.path}")
        for r in _routes()
        for m in sorted(r.methods & _WRITE_METHODS)
    ],
)
def test_every_wiki_write_refuses(path: str, method: str) -> None:
    route = next(r for r in _routes() if r.path == path and method in r.methods)
    if path in _RECOVERY_ROUTES:
        assert not is_decommissioned(route.endpoint), (
            f"{method} {path} is listed as a recovery route "
            f"({_RECOVERY_ROUTES[path]}) but refuses. Remove it from "
            "_RECOVERY_ROUTES or drop the decorator."
        )
        return
    assert is_decommissioned(route.endpoint), (
        f"{method} {path} is a wiki write with no `@decommissioned_wiki_write`. "
        "The wiki no longer has workers: a write lands in a page scheduled for "
        "deletion, and a rebuild wipes pages nothing can re-derive. Add the "
        "decorator, or add the route to _RECOVERY_ROUTES with the reason."
    )


@pytest.mark.parametrize(
    "path",
    sorted({r.path for r in _routes() if "GET" in r.methods}),
)
def test_wiki_reads_keep_serving(path: str) -> None:
    """research-os is still proxying these for its deprecation window."""
    route = next(r for r in _routes() if r.path == path and "GET" in r.methods)
    assert not is_decommissioned(route.endpoint), (
        f"GET {path} refuses. Reads stay up until research-os closes its "
        "deprecation window -- an agent whose instructions still name the wiki "
        "must get its pages, not a 410."
    )


def test_the_refusal_says_what_is_still_possible() -> None:
    """A refusal that omits undo strands the tenants who most need it."""
    from kb.wiki_decommission import WIKI_DECOMMISSIONED_DETAIL

    assert "notes_append" in WIKI_DECOMMISSIONED_DETAIL
    assert "backfill/undo" in WIKI_DECOMMISSIONED_DETAIL
    assert "Nothing was stored" in WIKI_DECOMMISSIONED_DETAIL
