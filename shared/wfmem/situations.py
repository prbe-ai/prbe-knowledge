"""The situation vocabulary, and the enable-flip that seeds it.

A situation is "what the person is doing right now" -- the classifier's label
space and the join key every clause hangs off. A tenant with an empty
vocabulary classifies every session as unknown and serves zero cards, and that
failure is INVISIBLE: zero cards because there is no vocabulary looks exactly
like zero cards because no rule matched. Nothing errors, nothing logs, the
feature is just quietly dead.

So onboarding is not a separate step that someone can forget. Turning the
declared-input capability on seeds the tenant's vocabulary IN THE SAME
TRANSACTION as the flag flip: either the tenant ends up with the capability and
twelve situations, or with neither.

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

from shared.db import with_tenant
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
            "The person is about to tell someone the work is complete -- writing a summary, "
            "closing a ticket, or reporting a result as verified -- and what they assert now "
            "is what the rest of the team will act on."
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
            "The person is trying to obtain a result somebody already reported -- an earlier "
            "run, a teammate's number, a published claim -- from its recorded configuration, "
            "and the target number is known before they start."
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

    There is deliberately no `disable_capability` counterpart that deletes the
    vocabulary. Turning the input off should stop new writes, not destroy rows a
    tenant may have spent time editing.
    """
    require_capability_key(key)
    if not customer_id:
        raise ValueError("enable_capability() requires a non-empty customer_id")

    async with with_tenant(customer_id) as conn:
        status = await conn.execute(
            """
            UPDATE customers
               SET preferences = jsonb_set(
                       COALESCE(preferences, '{}'::jsonb),
                       ARRAY[$2::text],
                       'true'::jsonb,
                       true
                   )
             WHERE customer_id = $1
            """,
            customer_id,
            key,
        )
        if status.rsplit(" ", 1)[-1] == "0":
            raise LookupError(f"no such customer: {customer_id!r}")
        if key == WFMEM_INPUT_DECLARED:
            await seed_situations(conn, customer_id)
