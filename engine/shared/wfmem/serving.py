"""Reading clauses out, and recording that we did.

THE ONE READ PATH. Both surfaces that hand clause bodies to a person or an agent
come through here: the `/pull-rules` skill (a human asking what the team has
captured) and, from Stage 4, `probe_procedures` (an agent asking what applies to
the situation it is in). They differ in who asks and how hard the ranking works,
NOT in code. Two read paths would drift on the two things that must never
diverge -- the visibility predicate and the ledger write -- and they would drift
quietly, because both bugs look like "slightly different results" rather than
like an error.

WHY THE LEDGER WRITE LIVES HERE AND NOT AT THE ENDPOINT. Exposure is
unbackfillable. Taint-excluded support and every confidence number later derived
from it read `serve_ledger`, and a delivery that did not write a row is not a
gap that can be repaired -- the fact is simply gone. Putting the write next to
the SELECT, in the same transaction, is what makes "served but unlogged"
unreachable rather than merely discouraged. A caller cannot forget it, because
there is no call that returns clauses without it.

WHAT DOES *NOT* GET A LEDGER ROW: a query that returns zero clauses. The ledger
answers "was this person's agent shown this clause before they produced that
evidence", and an empty result shows nobody anything -- there is no id to join
on, and every column the table has (`clause_ids`, `situation_id` as the thing
they were served under) is meaningless without at least one clause. Logging
empties would turn an exposure log into a query log and force every downstream
join to remember to exclude them. If we later want serve-ATTEMPT denominators --
how often somebody asked and got nothing, which is a real question about whether
this feature works -- they belong in a metrics path, not in the audit table.
`serve_id` is None in that case, and callers must treat that as "nothing was
exposed", never as "the write failed".

RLS. Every query here runs inside `with_tenant`. That is load-bearing rather
than decorative: `clauses`, `clause_situation_edges` and `serve_ledger` all carry
FORCE ROW LEVEL SECURITY, so under the production role a read without the tenant
GUC returns zero rows and a write is refused outright. Note that the dev/CI role
is a SUPERUSER and BYPASSES all of it, which is why a test that believes it is
proving isolation has to `SET LOCAL ROLE prbe_rls_test` first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from engine.shared.db import with_tenant
from engine.shared.logging import get_logger
from engine.shared.wfmem.situations import FALLBACK_SITUATION_SLUG
from engine.shared.wfmem.visibility import HUMAN_SOURCE_CLASSES, VISIBILITY_PREDICATE

log = get_logger(__name__)

#: The delivery channel this module records. The column accepts four
#: (`retrieved`, `strip`, `compiled`, `injected`) because the design names four
#: delivery surfaces; Phase 0 ships exactly one of them, and the other three
#: stay unimplemented rather than being approximated by this one.
CHANNEL_RETRIEVED = "retrieved"

#: Default cards per response. Three because a serving surface competes for the
#: agent's attention with the actual task -- the design's budget is <=3 cards at
#: <=250 tokens each. `/pull-rules` overrides it: a human asking "what have we
#: captured" wants a listing, not a top-3.
DEFAULT_LIMIT = 3

#: Hard ceiling regardless of what a caller asks for. A tenant's whole corpus in
#: one response is a context-window problem for the agent on the other end, and
#: the read is unpaginated by design in v0.
MAX_LIMIT = 50

#: Epistemic strength, strongest first. Rank position is what `ORDER BY` uses;
#: anything not named here sorts after everything named.
#:
#: PROVISIONAL, and honestly so: v0 writes ONLY `declared`, so every other entry
#: is an ordering nobody has exercised. The ladder is in the CHECK constraint
#: from the start so Phase 1 needs no migration, and this list is the matching
#: half of that bet.
#:
#: The two at the bottom are the considered part. `contested` and `stale` are not
#: weak evidence, they are ACTIVE WARNINGS -- a rule somebody disputed, a rule
#: that has rotted -- and leading a three-card response with one buries the two
#: cards a person could act on. They are still served, because hiding a
#: contested rule is how one silently becomes policy again.
STATUS_RANK: tuple[str, ...] = (
    "expert_confirmed",
    "intervention_validated",
    "declared",
    "documented",
    "success_associated",
    "observed_convention",
    "exception",
    "anti_pattern",
    "agent_proposed",
    "contested",
    "stale",
)


@dataclass(frozen=True)
class ServedClause:
    """One clause as a surface renders it.

    Deliberately NOT the whole row. `salience`, `binding_health` and `owner_ref`
    are ranking and maintenance inputs with no reader on the other end, and a
    dataclass that mirrors the table invites an endpoint to serialize the lot.
    """

    id: UUID
    kind: str
    body: str
    status: str
    author_ref: str
    version: int
    binding: dict[str, Any]
    scope: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    #: Who published this on their own authority, or None if it is visible
    #: because two humans back it. CARRIED ALL THE WAY TO THE SURFACE on
    #: purpose: the house rule is label, never gate, and a rule one person
    #: declared and published themselves is a different claim from one the team
    #: demonstrably follows. Drop this field and both render identically, which
    #: is the failure the whole two-human guard exists to prevent -- arrived at
    #: by the feature meant to work around it.
    shared_by: str | None = None
    #: How many DISTINCT untainted humans back it. 1 on a published-but-
    #: uncorroborated rule, which is the number a reader needs to see next to
    #: `shared_by` for that label to mean anything.
    human_backers: int = 0
    #: True when this came out of the `misc` bucket because the situation the
    #: caller asked about had no rules of its own. MUST reach the surface: it is
    #: the difference between "your team decided this about what you are doing"
    #: and "nothing here fit, so here is a rule nobody could file". Render them
    #: alike and the fallback stops being a safety net and becomes a source of
    #: confident irrelevant advice -- which is how an agent learns to skip the
    #: whole surface.
    from_fallback: bool = False


@dataclass(frozen=True)
class Serving:
    """What was served, and the audit row that records it."""

    clauses: list[ServedClause]
    #: `serve_ledger.id`, or None when nothing was served. None means "nothing
    #: was exposed", NOT "the write failed" -- see the module docstring.
    serve_id: int | None
    situation_id: UUID | None


# `scope` is a JSONB object whose keys narrow who a rule applies to; more keys is
# more specific. In v0 every clause is `{}` (workspace-wide is the default and
# narrowing is the explicit act), so this term is uniformly 0 today and exists so
# the ordering is already right the first time somebody narrows one. Counting
# keys is a deliberately crude proxy -- it says a rule with `{repo, path}` beats
# one with `{repo}`, which is the property that matters, and it does not pretend
# to rank two different single-key scopes against each other.
_SCOPE_SPECIFICITY = """
    (SELECT count(*) FROM jsonb_object_keys(c.scope))
"""

# Status strength, as a position in the $3 array rather than a CASE ladder or a
# stored SQL function. Passing the vocabulary as a parameter keeps ONE ordering
# in the codebase -- `STATUS_RANK` above -- where a CASE in SQL would be a second
# copy that drifts, and a stored function would put it a migration away from the
# list it has to agree with. `array_position` returns NULL for a status not in
# the list, and COALESCE to a value past the end sorts it last: a status added to
# the CHECK constraint but forgotten here degrades to "ranked last", never to a
# dropped row.
_STATUS_ORDER = """
    COALESCE(array_position($3::text[], c.status), array_length($3::text[], 1) + 1)
"""

# The same COUNT the visibility predicate runs, selected rather than only
# tested. It is the number that makes `shared_by` legible: "published by
# @mahit, 1 human behind it" and "published by @mahit, 4 humans behind it" are
# very different instructions, and without it a surface can only say the first
# half. Postgres evaluates the subquery once per returned row, and the row
# count here is capped at MAX_LIMIT.
_HUMAN_BACKERS = """
    (
        SELECT COUNT(DISTINCT e.author_ref)
          FROM clause_evidence e
         WHERE e.clause_id = c.id
           AND e.author_ref IS NOT NULL
           AND e.source_class = ANY($2::text[])
           AND NOT e.exposure_tainted
    )
"""

_SELECT_COLUMNS = f"""
    c.id, c.kind, c.body, c.status, c.author_ref, c.version,
    c.binding, c.scope, c.created_at, c.updated_at,
    c.shared_by,
    {_HUMAN_BACKERS} AS human_backers
"""


async def serve_clauses(
    customer_id: str,
    *,
    actor_ref: str,
    situation_id: UUID | None = None,
    session_id: str | None = None,
    limit: int = DEFAULT_LIMIT,
    trigger: str | None = None,
) -> Serving:
    """Clauses `actor_ref` may see, ranked, plus the ledger row recording it.

    `situation_id=None` serves across every situation. That is `/pull-rules`
    asking "what have we captured", not a classifier failure -- a caller that
    classified and got `unknown` must serve NOTHING rather than calling here with
    None, because serving a broadly-scoped rule into a situation we could not
    identify is the exact failure the `unknown` escape hatch exists to prevent.
    The distinction is the caller's to make and cannot be recovered here.

    `actor_ref` is both the visibility subject and what the ledger records. One
    argument, not two, on purpose: a surface that could pass a different viewer
    than it logs would produce an audit trail attributing one person's exposure
    to another, and there is no legitimate reason to want that.
    """
    if not actor_ref:
        # Not defensive noise. `actor_ref` is what makes the ledger joinable at
        # all, and the visibility predicate's single-author escape hatch keys on
        # it -- an empty one silently matches no clause's `author_ref`, so the
        # caller would get a quietly narrower result AND an unattributable audit
        # row. Both failures are invisible in the response.
        raise ValueError("serve_clauses() requires a non-empty actor_ref")

    limit = max(0, min(limit, MAX_LIMIT))
    if limit == 0:
        return Serving(clauses=[], serve_id=None, situation_id=situation_id)

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            _build_query(situation_id is not None),
            *_query_args(actor_ref, customer_id, situation_id, limit),
        )
        clauses = [_to_served(row) for row in rows]

        # THE FALLBACK TIER, and note how narrow it is: only when the situation
        # was named AND matched nothing at all.
        #
        # Topping up a PARTIAL result was the tempting version and it is wrong.
        # `misc` holds the rules nothing fit, so mixing them beside real matches
        # dilutes an answer that was already correct, and at limit=3 it would
        # spend a third of the agent's attention on rules we know do not apply.
        # Empty is different: there is nothing to dilute, and "we have no rules
        # for this, but here is what never found a home" beats silence -- which
        # is what the author of an unfiled rule gets today.
        #
        # Never fires for an unscoped listing: that already returns everything,
        # `misc` included, so re-querying the bucket would duplicate every rule
        # in it.
        #
        # The `!= situation_id` guard is an OPTIMISATION, not a correctness fix,
        # and saying so matters because the obvious reading is wrong. Asking for
        # `misc` directly cannot double-count: this branch only runs when the
        # first query came back empty, so a bucket with rules never reaches it
        # and a bucket without them has nothing to duplicate. The guard just
        # skips a second query guaranteed to return what the first one did.
        # (A mutation test that deleted it stayed green, which is how this
        # comment got corrected rather than left as a plausible-sounding lie.)
        if not clauses and situation_id is not None:
            fallback_id = await _fallback_situation_id(conn, customer_id)
            if fallback_id is not None and fallback_id != situation_id:
                rows = await conn.fetch(
                    _build_query(True),
                    *_query_args(actor_ref, customer_id, fallback_id, limit),
                )
                clauses = [_to_served(row, from_fallback=True) for row in rows]

        serve_id = None
        if clauses:
            serve_id = await _record_serve(
                conn,
                customer_id=customer_id,
                clause_ids=[c.id for c in clauses],
                # The ledger records the situation the caller ASKED about, not
                # the bucket the rows came from. Exposure is "this person was
                # shown this while doing X", and X is the question they asked --
                # rewriting it to `misc` would make every fallback serve look
                # like somebody browsing the junk drawer.
                situation_id=situation_id,
                session_id=session_id,
                actor_ref=actor_ref,
                trigger=trigger,
            )

    return Serving(clauses=clauses, serve_id=serve_id, situation_id=situation_id)


async def _fallback_situation_id(conn: Any, customer_id: str) -> UUID | None:
    """The tenant's `misc` row, or None when they predate 0118's seeding."""
    return await conn.fetchval(
        "SELECT id FROM situations WHERE customer_id = $1 AND slug = $2",
        customer_id,
        FALLBACK_SITUATION_SLUG,
    )


def _build_query(filter_by_situation: bool) -> str:
    """The served-clause SELECT.

    `VISIBILITY_PREDICATE` is interpolated rather than reimplemented -- it is
    written against an aliased `clauses c` and expects the viewer as $1 and the
    human source classes as $2, which is why those two keep those positions here.
    Reimplementing it inline is how the read path and the guard drift apart.

    THE EXPLICIT `c.customer_id = $4` IS NOT REDUNDANT WITH RLS, and it is worth
    saying why since most reads in this codebase omit it. Two reasons, and the
    second is the one that decided it:

    Defence in depth on the one path that hands rule BODIES to a person. RLS is
    the real guarantee and is proven table-by-table in the isolation suite; this
    is the belt to its braces, on the surface where a cross-tenant slip is the
    §7 non-negotiable rather than a wrong number.

    And it makes the tenancy test mean something. The dev and CI role `prbe` is a
    SUPERUSER and BYPASSES row security entirely, so a serving query relying on
    RLS alone returns every tenant's clauses under the test role -- which is
    exactly what the first run of `test_another_tenants_clauses_are_never_served`
    did. A test that can only pass by first switching to `prbe_rls_test` is a
    test people stop writing. RLS keeps being proven where it belongs, in the
    isolation suite, under the role that actually exercises it.

    The situation filter is EXISTS over `clause_situation_edges` rather than a
    JOIN: a clause attached to a situation twice would otherwise be returned
    twice and consume two of the three cards with one rule.

    Its alias is `cse`, not `e`. `VISIBILITY_PREDICATE` already binds `e` to
    `clause_evidence` in its own subquery -- the two do not actually collide,
    since each is scoped to the subquery it appears in, but one letter meaning
    two tables in a query somebody has to debug at 2am is a gift to nobody.
    """
    situation_clause = (
        """
          AND EXISTS (
              SELECT 1 FROM clause_situation_edges cse
               WHERE cse.clause_id = c.id
                 AND cse.situation_id = $5
          )
        """
        if filter_by_situation
        else ""
    )
    limit_param = "$6" if filter_by_situation else "$5"
    return f"""
        SELECT {_SELECT_COLUMNS}
          FROM clauses c
         WHERE c.customer_id = $4
               AND {VISIBILITY_PREDICATE}
               {situation_clause}
         ORDER BY {_SCOPE_SPECIFICITY} DESC,
                  {_STATUS_ORDER} ASC,
                  c.updated_at DESC,
                  c.id ASC
         LIMIT {limit_param}
    """


def _query_args(
    actor_ref: str, customer_id: str, situation_id: UUID | None, limit: int
) -> tuple[Any, ...]:
    """Positional args. $1/$2 are fixed by `VISIBILITY_PREDICATE`'s own contract."""
    base: tuple[Any, ...] = (
        actor_ref,
        list(HUMAN_SOURCE_CLASSES),
        list(STATUS_RANK),
        customer_id,
    )
    if situation_id is not None:
        return (*base, situation_id, limit)
    return (*base, limit)


def _to_served(row: Any, *, from_fallback: bool = False) -> ServedClause:
    return ServedClause(
        id=row["id"],
        kind=row["kind"],
        body=row["body"],
        status=row["status"],
        author_ref=row["author_ref"],
        version=row["version"],
        binding=_as_dict(row["binding"]),
        scope=_as_dict(row["scope"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        shared_by=row["shared_by"],
        human_backers=row["human_backers"] or 0,
        from_fallback=from_fallback,
    )


def _as_dict(raw: Any) -> dict[str, Any]:
    """JSONB comes back as a dict or as text depending on codec registration.

    Both shapes are real in this codebase, and a caller that got a `str` where it
    expected a mapping fails at `.get(...)` far from here.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def _record_serve(
    conn: Any,
    *,
    customer_id: str,
    clause_ids: list[UUID],
    situation_id: UUID | None,
    session_id: str | None,
    actor_ref: str,
    trigger: str | None,
) -> int:
    """Append one exposure row and return its id.

    Same transaction as the SELECT above, which is the point: a failure here
    rolls the read back too, so the caller cannot end up holding clause bodies
    that no ledger row accounts for. That is the right direction to fail -- an
    error the caller sees beats a silent hole in an unbackfillable audit trail.
    """
    return await conn.fetchval(
        """
        INSERT INTO serve_ledger
            (customer_id, clause_ids, situation_id, session_id,
             actor_ref, channel, route, mode, trigger)
        VALUES ($1, $2::uuid[], $3, $4, $5, $6, 'n/a', 'live', $7)
        RETURNING id
        """,
        customer_id,
        clause_ids,
        situation_id,
        session_id,
        actor_ref,
        CHANNEL_RETRIEVED,
        trigger,
    )
