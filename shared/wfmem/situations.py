"""The situation vocabulary, and the enable-flip that seeds it.

A situation is "what the person is doing right now" -- the classifier's label
space and the join key every clause hangs off. A tenant with an empty
vocabulary classifies every session as unknown and serves zero cards, and that
failure is INVISIBLE: zero cards because there is no vocabulary looks exactly
like zero cards because no rule matched. Nothing errors, nothing logs, the
feature is just quietly dead.

So onboarding is not a separate step the enable path can forget. Turning the
declared-input capability on through `enable_capability` seeds the tenant's
vocabulary IN THE SAME TRANSACTION as the flag flip: either the tenant ends up
with the capability and twelve situations, or with neither.

THE INTENDED DOOR, NOT THE ONLY ONE. The capability is a key in a JSONB column
on a shared product table, and other writers reach it: the dashboard PATCHes
`customers.preferences`, and an operator flips a cell by hand during an
incident. Neither goes through `enable_capability`, so neither seeds -- and the
result is the silent-dead-feature state above, arrived at by a route this module
cannot intercept. Nothing here pretends otherwise. `enabled_tenants_missing
_situations` exists exactly for that: it names the tenants who are in that state
so somebody can look, instead of waiting for a customer to report that a feature
they turned on does nothing.

DEPENDENCY FOR STAGE 4, recorded here so it is not forgotten: `probe_procedures`
must report "no situations configured" DISTINCTLY from "no rules matched". Those
are the same empty response today, which is the whole reason this failure is
invisible; one extra branch at the serving edge is what converts it into
something a reader can see. Not implemented in this stage.

This module also holds BOTH WRITERS of the capability cells --
`enable_capability` and `disable_capability` -- while `shared.wfmem.capabilities`
stays read-only. The split is the seeding: enabling touches `situations`, so the
writers live where the situations are, and a reader importing `capabilities`
does not drag the write path in behind it.

The twelve are a starting point, not a fixed taxonomy -- the table is per-tenant
and editable, and Phase 1's mining is expected to add to it. They are the
moments where a team's practice is actually load-bearing (the ones where doing
it wrong costs money, data, or an incident), which is why they skew toward
"about to spend something irreversible" rather than covering the working day
evenly.
"""

from __future__ import annotations

from typing import NamedTuple

import asyncpg

from shared.db import raw_conn, with_tenant
from shared.wfmem.capabilities import WFMEM_INPUT_DECLARED, require_capability_key


class SeedSituation(NamedTuple):
    """One row of the starting vocabulary."""

    slug: str
    label: str
    #: Written FOR AN LLM CLASSIFIER, not for a UI tooltip. It has to say what
    #: the person is doing and about to do, in terms a transcript would show,
    #: because the classifier's whole job is telling these twelve apart -- and
    #: several of them are adjacent (editing a repo vs. debugging a failing
    #: test; deploying vs. recovering from what the deploy broke). A
    #: description that restates the label gives it nothing to discriminate on.
    description: str


SEED_SITUATIONS: tuple[SeedSituation, ...] = (
    SeedSituation(
        slug="launch-run",
        label="Launching a training run",
        description=(
            "The person is about to start a training, fine-tuning or sweep job: choosing a "
            "config and base checkpoint, sizing hardware, and issuing the command that "
            "begins spending compute. Nothing has been trained yet."
        ),
    ),
    SeedSituation(
        slug="claim-done",
        label="Claiming work is finished",
        description=(
            "The person is telling someone the work is complete, OUTSIDE any review flow: "
            "closing or resolving a ticket, replying 'done' or 'shipped' in a thread, "
            "marking a task complete, or writing the summary that ends a session. The claim "
            "itself is the artifact and colleagues will act on it without re-checking. "
            "Contrast open-pr, where the same assertion is part of a change being submitted."
        ),
    ),
    SeedSituation(
        slug="open-pr",
        label="Opening a pull request",
        description=(
            "The change is written and the person is packaging it for review: picking the "
            "base branch, writing the title and body, and choosing who should look at it. "
            "The code is not being changed at this moment, it is being explained."
        ),
    ),
    SeedSituation(
        slug="deploy",
        label="Deploying or landing a change",
        description=(
            "An already-reviewed change is being put where other people depend on it -- "
            "merging to the main branch, releasing, rolling out, or promoting a build. The "
            "next state of a shared environment is being decided."
        ),
    ),
    SeedSituation(
        slug="incident-recovery",
        label="Recovering from a failure or incident",
        description=(
            "Something shared is already broken and the person is restoring service under "
            "time pressure: reading alerts, rolling back, or applying a mitigation. Unlike "
            "debugging, the immediate goal is to stop the bleeding, not to explain it."
        ),
    ),
    SeedSituation(
        slug="edit-repo",
        label="Editing code in a repository",
        description=(
            "The person is changing source files in a checkout -- reading the surrounding "
            "code, writing or refactoring, deciding where the change belongs. Ordinary "
            "forward work on a codebase, with nothing yet failing and nothing yet shipped."
        ),
    ),
    SeedSituation(
        slug="process-dataset",
        label="Processing or transforming a dataset",
        description=(
            "Raw data is being turned into something a job can consume: filtering, "
            "deduplicating, splitting, tokenizing, relabelling, or moving it between stores. "
            "The output will be treated as ground truth by everything downstream."
        ),
    ),
    SeedSituation(
        slug="run-eval",
        label="Running an evaluation or benchmark",
        description=(
            "An existing model, agent or system is being measured against a benchmark, "
            "scorer or held-out set, and the numbers that come back will be read as evidence "
            "for a decision. No weights are being updated."
        ),
    ),
    SeedSituation(
        slug="review-code",
        label="Reviewing someone's code",
        description=(
            "The person is reading a change somebody else proposed and deciding whether it "
            "should land -- leaving comments, requesting changes, or approving. They are the "
            "gate, not the author."
        ),
    ),
    SeedSituation(
        slug="provision-infra",
        label="Provisioning infrastructure or environments",
        description=(
            "Machines, clusters, buckets, queues or credentials are being created, resized "
            "or torn down so that other work can run on them later. The thing being changed "
            "is the substrate, not the code or the data."
        ),
    ),
    SeedSituation(
        slug="reproduce-experiment",
        label="Reproducing a past experiment or paper",
        description=(
            "A NAMED PRIOR RESULT is on screen and being matched: a specific earlier run id, "
            "a number a teammate posted, a table from a paper. The person pulls up that "
            "reference's recorded configuration, re-runs from it, and compares the two "
            "outputs against each other. The observable act is fetching and citing the "
            "reference and diffing against it -- an eval whose output is a verdict on "
            "somebody else's number rather than a fresh measurement of their own system."
        ),
    ),
    SeedSituation(
        slug="debug-failing-run",
        label="Debugging a failing run or test",
        description=(
            "A specific job, test or run is failing or producing numbers that cannot be "
            "right, and the person is isolating the cause from logs, metrics and stack "
            "traces. Understanding comes before any fix; nothing shared is on fire."
        ),
    ),
)


async def seed_situations(conn: asyncpg.Connection, customer_id: str) -> int:
    """Insert the starting vocabulary for `customer_id`; return rows inserted.

    Idempotent: a slug the tenant already has is left exactly as it is -- label
    and description included, because a tenant may have edited them and a seed
    re-run must not silently revert somebody's wording. Re-running on a fully
    seeded tenant returns 0.

    Takes the CALLER'S connection and never opens its own. The caller owns the
    transaction: `enable_capability` needs the seed and the flag flip to fail
    together, which is impossible if this function commits on its own.

    CONTRACT ON THE CONNECTION: `customer_id` must match the tenant GUC bound on
    `conn` (i.e. the argument to `with_tenant`). This function does not check --
    under the production role the WITH CHECK half of the `situations` policy
    refuses the mismatch, but the dev/CI role is a SUPERUSER and bypasses RLS,
    so a mismatched pair would silently write another tenant's vocabulary there.
    `enable_capability` always passes the pair it opened the transaction with;
    any other caller must do the same.
    """
    rows = await conn.fetch(
        """
        INSERT INTO situations (customer_id, slug, label, description)
        SELECT $1, s.slug, s.label, s.description
          FROM unnest($2::text[], $3::text[], $4::text[]) AS s(slug, label, description)
        ON CONFLICT (customer_id, slug) DO NOTHING
        RETURNING id
        """,
        customer_id,
        [s.slug for s in SEED_SITUATIONS],
        [s.label for s in SEED_SITUATIONS],
        [s.description for s in SEED_SITUATIONS],
    )
    return len(rows)


async def enable_capability(customer_id: str, key: str) -> None:
    """Turn one capability on, seeding the situation vocabulary where required.

    One transaction. For `wfmem_input_declared` the flip and the seed stand or
    fall together: a tenant with the capability on and no vocabulary is the
    silent-dead-feature state described in this module's docstring, and a
    two-statement version reaches it every time the second statement fails.

    Runs inside `with_tenant`, so the tenant GUC is bound for the situations
    insert. That is not decoration -- `situations` has FORCE ROW LEVEL SECURITY
    and the WITH CHECK half of its policy rejects the insert outright when the
    GUC is unset, under any role that does not bypass RLS (which the production
    role does not).

    Raises ValueError on an unknown key and LookupError on an unknown customer;
    the latter would otherwise be a silent no-op for the five non-seeding keys.

    Its opposite is `disable_capability`, which writes false and leaves the
    vocabulary alone.
    """
    require_capability_key(key)
    if not customer_id:
        raise ValueError("enable_capability() requires a non-empty customer_id")

    async with with_tenant(customer_id) as conn:
        await _write_capability_cell(conn, customer_id, key, True)
        if key == WFMEM_INPUT_DECLARED:
            await seed_situations(conn, customer_id)


async def disable_capability(customer_id: str, key: str) -> None:
    """Turn one capability off. The kill switch.

    WRITES `false`, it does not delete the key. Migration 0077 exists to make
    every cell explicit -- so the dashboard shows off because it IS off, not
    because of a reader's fallback -- and a disable that removed the key would
    walk that back one tenant at a time, leaving the same ambiguity the
    migration was written to clear. The reader answers False either way; the
    operator staring at a row during an incident does not.

    DELIBERATELY DOES NOT TOUCH `situations`. Same reasoning as the seed's
    conflict clause: turning the input off stops new writes, and a tenant may
    have spent real time editing that vocabulary. Destroying it here would make
    the off-switch unusable for its main purpose -- flip it off, work out what
    went wrong, flip it back -- because flipping back would silently restore
    stock wording over the tenant's edits.

    Raises ValueError on an unknown key and LookupError on an unknown customer,
    matching `enable_capability`. Single statement, so no transaction of its own
    is needed, but it runs in one anyway via `with_tenant`.
    """
    require_capability_key(key)
    if not customer_id:
        raise ValueError("disable_capability() requires a non-empty customer_id")

    async with with_tenant(customer_id) as conn:
        await _write_capability_cell(conn, customer_id, key, False)


async def _write_capability_cell(
    conn: asyncpg.Connection, customer_id: str, key: str, value: bool
) -> None:
    """Set one capability cell to a real JSON bool. Raises LookupError if absent.

    The value goes through `to_jsonb($3::boolean)` rather than an interpolated
    literal: the reader only accepts a real bool, and a parameter cannot drift
    into the string `"true"` the way a formatted literal can.

    The CASE repairs a non-object `preferences` instead of failing on it.
    `jsonb_set` raises `cannot set path in scalar` for a scalar or array blob,
    which would take the kill switch out exactly when someone needs it, and a
    non-object blob has no keys to lose -- unlike migration 0077, which skips
    those rows because a bulk backfill has no business rewriting them.
    """
    status = await conn.execute(
        """
        UPDATE customers
           SET preferences = jsonb_set(
                   CASE WHEN jsonb_typeof(preferences) = 'object'
                        THEN preferences
                        ELSE '{}'::jsonb
                   END,
                   ARRAY[$2::text],
                   to_jsonb($3::boolean),
                   true
               )
         WHERE customer_id = $1
        """,
        customer_id,
        key,
        value,
    )
    if status.rsplit(" ", 1)[-1] == "0":
        raise LookupError(f"no such customer: {customer_id!r}")


async def enabled_tenants_missing_situations() -> list[str]:
    """Tenants with the declared input ON and an EMPTY vocabulary.

    The detection half of the seed invariant. `enable_capability` cannot produce
    this state -- the flip and the seed are one transaction -- but it is
    reachable by every other route into the column: a hand-written UPDATE during
    an incident, the dashboard's preferences PATCH, a restore that replayed
    `customers` without `situations`. On screen the result is a tenant for whom
    the feature is on and silently returns nothing, so it needs a query that
    names it rather than a person noticing.

    Two round trips PER TENANT, on purpose, rather than one LEFT JOIN. `customers`
    has no row security but `situations` has FORCE RLS, so under the production
    role a join running without a tenant GUC sees zero situations for everyone
    and reports every enabled tenant as broken -- a false alarm on exactly the
    query whose whole value is being trustworthy. The membership test therefore
    runs inside `with_tenant` per candidate. The candidate list is small (it is
    tenants with the capability on), and this is an audit path, not a hot one.

    Matches with jsonb containment, so a string `"true"` is not a match -- the
    same rule `is_capability_enabled` applies.
    """
    async with raw_conn() as conn:
        candidates = await conn.fetch(
            """
            SELECT customer_id
              FROM customers
             WHERE preferences @> jsonb_build_object($1::text, true)
             ORDER BY customer_id
            """,
            WFMEM_INPUT_DECLARED,
        )

    missing: list[str] = []
    for record in candidates:
        customer_id = record["customer_id"]
        async with with_tenant(customer_id) as conn:
            has_any = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM situations WHERE customer_id = $1)",
                customer_id,
            )
        if not has_any:
            missing.append(customer_id)
    return missing
