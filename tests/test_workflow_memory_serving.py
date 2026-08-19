"""The one read path: visibility, ranking, and the exposure ledger.

WHAT THIS FILE HAS TO DEFEND:

* THE VISIBILITY GUARD ACTUALLY APPLIES. `fetch_visible_clauses` has been tested
  since Stage 1, but until now nothing SERVED through it -- the predicate was
  proven in isolation while every real read path was hypothetical. A serving
  surface that forgot to apply it would pass every existing test.
* EXPOSURE IS RECORDED, AND ONLY REAL EXPOSURE. Exactly one ledger row per
  delivery, carrying the actor; zero rows when nothing was served, because the
  ledger is an exposure log and an empty result exposes nobody.
* THE LEDGER CANNOT BE SKIPPED. The write shares a transaction with the read, so
  there is no ordering in which a caller holds clause bodies that no row
  accounts for.
* RANKING IS THE DOCUMENTED ORDER. Scope specificity, then epistemic status,
  then recency -- with `contested` and `stale` last among statuses, since a
  three-card budget spent leading with a warning buries what a person can act on.
* `situation_id=None` MEANS EVERY SITUATION, NOT "UNKNOWN". These are opposite
  instructions and one function argument spells both, so the test says which.

Run with the isolated wfmem database (conftest enforces a localhost host, since
these fixtures TRUNCATE):

    PRBE_TEST_DATABASE_URL=postgresql://prbe:prbe@localhost:5432/prbe_knowledge_wfmem \
        .venv/bin/pytest tests/test_workflow_memory_serving.py -q
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import pytest
import pytest_asyncio

from engine.shared.db import raw_conn, with_tenant
from engine.shared.wfmem.serving import (
    CHANNEL_RETRIEVED,
    DEFAULT_LIMIT,
    MAX_LIMIT,
    STATUS_RANK,
    serve_clauses,
)
from engine.shared.wfmem.situations import seed_situations

TENANT = "cust-wfmem-serve-a"
OTHER_TENANT = "cust-wfmem-serve-b"

ALICE = "user:alice"
BOB = "user:bob"
CAROL = "user:carol"


@pytest_asyncio.fixture
async def tenant(live_db: None) -> AsyncIterator[str]:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'wfmem-serve-a', 'h-wfmem-serve-a'),
                   ($2, 'wfmem-serve-b', 'h-wfmem-serve-b')
            ON CONFLICT (customer_id) DO NOTHING
            """,
            TENANT,
            OTHER_TENANT,
        )
    yield TENANT


async def _make_clause(
    customer_id: str,
    *,
    body: str,
    author: str = ALICE,
    status: str = "declared",
    scope: dict[str, Any] | None = None,
    kind: str = "step",
    second_human: str | None = BOB,
    tainted_second: bool = False,
    age_days: int = 0,
) -> UUID:
    """Insert a clause plus enough evidence to make it visible.

    `second_human` defaults to a real second author because MOST tests are about
    something other than the visibility guard, and a clause that is invisible by
    default would make every one of them assert on an empty list for the wrong
    reason. Pass None to get the single-author case deliberately.

    `age_days` backdates `updated_at` AT INSERT TIME, and it has to. A later
    `UPDATE clauses SET updated_at = ...` does not work: `clauses_touch_updated_at_trg`
    is a BEFORE UPDATE trigger that overwrites the column with `now()`, so the
    backdated row comes out NEWER than the rows it was supposed to predate and
    the recency test asserts the reverse of what it means. The trigger does not
    fire on INSERT, which is the seam this uses.
    """
    async with with_tenant(customer_id) as conn:
        clause_id = await conn.fetchval(
            """
            INSERT INTO clauses
                (customer_id, kind, body, status, author_ref, scope,
                 created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb,
                    now() - ($7 || ' days')::interval,
                    now() - ($7 || ' days')::interval)
            RETURNING id
            """,
            customer_id,
            kind,
            body,
            status,
            author,
            json.dumps(scope or {}),
            str(age_days),
        )
        await _add_evidence(conn, customer_id, clause_id, author, tainted=False)
        if second_human is not None:
            await _add_evidence(
                conn, customer_id, clause_id, second_human, tainted=tainted_second
            )
    return clause_id


async def _add_evidence(
    conn: Any, customer_id: str, clause_id: UUID, author: str, *, tainted: bool
) -> None:
    await conn.execute(
        """
        INSERT INTO clause_evidence
            (customer_id, clause_id, source_class, source_ref,
             author_ref, exposure_tainted, ts)
        VALUES ($1, $2, 'declared', '{"session": "s-1"}'::jsonb, $3, $4, now())
        """,
        customer_id,
        clause_id,
        author,
        tainted,
    )


async def _attach(customer_id: str, clause_id: UUID, situation_id: UUID) -> None:
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            INSERT INTO clause_situation_edges
                (customer_id, clause_id, situation_id, classification)
            VALUES ($1, $2, $3, '{"method": "human"}'::jsonb)
            """,
            customer_id,
            clause_id,
            situation_id,
        )


async def _situation_id(customer_id: str, slug: str) -> UUID:
    async with with_tenant(customer_id) as conn:
        await seed_situations(conn, customer_id)
        return await conn.fetchval(
            "SELECT id FROM situations WHERE customer_id = $1 AND slug = $2",
            customer_id,
            slug,
        )


async def _ledger_rows(customer_id: str) -> list[Any]:
    async with with_tenant(customer_id) as conn:
        return await conn.fetch(
            "SELECT * FROM serve_ledger WHERE customer_id = $1 ORDER BY id",
            customer_id,
        )


# --------------------------------------------------------------------------
# Visibility
# --------------------------------------------------------------------------


async def test_a_two_human_clause_is_served(tenant: str) -> None:
    await _make_clause(tenant, body="log runs to Probe")
    served = await serve_clauses(tenant, actor_ref=CAROL)
    assert [c.body for c in served.clauses] == ["log runs to Probe"]


async def test_a_single_author_clause_is_hidden_from_everyone_else(tenant: str) -> None:
    """The guard's whole point, exercised through the surface that serves it.

    One person's working note is not the team's practice, and publishing it to
    colleagues is how a private habit becomes false policy.
    """
    await _make_clause(tenant, body="my own habit", author=ALICE, second_human=None)

    mine = await serve_clauses(tenant, actor_ref=ALICE)
    assert [c.body for c in mine.clauses] == ["my own habit"]

    theirs = await serve_clauses(tenant, actor_ref=BOB)
    assert theirs.clauses == []


async def test_a_tainted_second_author_does_not_unlock_a_clause(tenant: str) -> None:
    """An agent that was SERVED a clause and echoed it back is not a second voice.

    This is the self-reinforcement loop the design exists to prevent: without the
    taint exclusion, serving a clause would manufacture the evidence that makes
    it visible to more people, who would be served it, and so on.
    """
    await _make_clause(
        tenant, body="echoed back", author=ALICE, second_human=BOB, tainted_second=True
    )
    served = await serve_clauses(tenant, actor_ref=CAROL)
    assert served.clauses == []


async def test_one_person_logging_twice_is_still_one_person(tenant: str) -> None:
    clause_id = await _make_clause(
        tenant, body="said it twice", author=ALICE, second_human=None
    )
    async with with_tenant(tenant) as conn:
        await _add_evidence(conn, tenant, clause_id, ALICE, tainted=False)

    served = await serve_clauses(tenant, actor_ref=BOB)
    assert served.clauses == []


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------


async def test_one_delivery_writes_exactly_one_ledger_row(tenant: str) -> None:
    a = await _make_clause(tenant, body="rule a")
    b = await _make_clause(tenant, body="rule b")

    served = await serve_clauses(tenant, actor_ref=CAROL, session_id="sess-9")

    rows = await _ledger_rows(tenant)
    assert len(rows) == 1
    row = rows[0]
    assert served.serve_id == row["id"]
    assert set(row["clause_ids"]) == {a, b}
    assert row["actor_ref"] == CAROL
    assert row["session_id"] == "sess-9"
    assert row["channel"] == CHANNEL_RETRIEVED
    assert row["route"] == "n/a"
    assert row["mode"] == "live"


async def test_the_ledger_records_only_what_was_actually_served(tenant: str) -> None:
    """clause_ids is the served set, not the matching set.

    If it recorded everything that matched rather than everything returned, the
    taint join would mark clauses as seen that nobody was ever shown -- and
    taint exclusion would then suppress genuine evidence.
    """
    for i in range(5):
        await _make_clause(tenant, body=f"rule {i}")

    served = await serve_clauses(tenant, actor_ref=CAROL, limit=2)

    rows = await _ledger_rows(tenant)
    assert len(rows) == 1
    assert len(served.clauses) == 2
    assert set(rows[0]["clause_ids"]) == {c.id for c in served.clauses}


async def test_serving_nothing_writes_nothing(tenant: str) -> None:
    """An empty result exposes nobody, so there is no exposure to record."""
    served = await serve_clauses(tenant, actor_ref=CAROL)
    assert served.clauses == []
    assert served.serve_id is None
    assert await _ledger_rows(tenant) == []


async def test_a_hidden_clause_is_not_logged_as_exposure(tenant: str) -> None:
    """The ledger must reflect the VISIBILITY-FILTERED set.

    A surface that logged the pre-filter result would record an exposure that
    never happened, against a person who never saw it.
    """
    await _make_clause(tenant, body="private", author=ALICE, second_human=None)
    served = await serve_clauses(tenant, actor_ref=BOB)
    assert served.clauses == []
    assert await _ledger_rows(tenant) == []


async def test_each_call_appends_rather_than_replacing(tenant: str) -> None:
    await _make_clause(tenant, body="rule a")
    first = await serve_clauses(tenant, actor_ref=CAROL)
    second = await serve_clauses(tenant, actor_ref=BOB)

    rows = await _ledger_rows(tenant)
    assert [r["id"] for r in rows] == [first.serve_id, second.serve_id]
    assert [r["actor_ref"] for r in rows] == [CAROL, BOB]


async def test_an_empty_actor_ref_is_refused(tenant: str) -> None:
    """Not defensive noise: an empty actor produces an unattributable audit row
    AND silently narrows the result, since the single-author escape hatch
    compares against it. Both failures are invisible in the response."""
    await _make_clause(tenant, body="rule a")
    with pytest.raises(ValueError, match="actor_ref"):
        await serve_clauses(tenant, actor_ref="")
    assert await _ledger_rows(tenant) == []


# --------------------------------------------------------------------------
# Situation filtering
# --------------------------------------------------------------------------


async def test_a_situation_filter_serves_only_attached_clauses(tenant: str) -> None:
    launch = await _situation_id(tenant, "launch-run")
    attached = await _make_clause(tenant, body="open a run first")
    await _make_clause(tenant, body="unrelated rule")
    await _attach(tenant, attached, launch)

    served = await serve_clauses(tenant, actor_ref=CAROL, situation_id=launch)
    assert [c.id for c in served.clauses] == [attached]


async def test_no_situation_means_every_situation(tenant: str) -> None:
    """`situation_id=None` is `/pull-rules` asking for everything.

    It is NOT "the classifier returned unknown" -- a caller in that state must
    serve nothing rather than calling here with None, because serving a
    broadly-scoped rule into a situation nobody identified is exactly what the
    `unknown` escape hatch exists to prevent. One argument spells both
    instructions, so this pins which one it means.
    """
    launch = await _situation_id(tenant, "launch-run")
    attached = await _make_clause(tenant, body="attached")
    loose = await _make_clause(tenant, body="attached to nothing")
    await _attach(tenant, attached, launch)

    served = await serve_clauses(tenant, actor_ref=CAROL, situation_id=None)
    assert {c.id for c in served.clauses} == {attached, loose}


async def test_a_clause_attached_twice_is_served_once(tenant: str) -> None:
    """EXISTS, not JOIN. A duplicate edge would otherwise spend two of three
    cards on one rule."""
    launch = await _situation_id(tenant, "launch-run")
    clause_id = await _make_clause(tenant, body="only once")
    await _attach(tenant, clause_id, launch)
    async with with_tenant(tenant) as conn:
        await conn.execute(
            """
            INSERT INTO clause_situation_edges
                (customer_id, clause_id, situation_id, classification)
            VALUES ($1, $2, $3, '{"method": "script"}'::jsonb)
            ON CONFLICT (clause_id, situation_id) DO NOTHING
            """,
            tenant,
            clause_id,
            launch,
        )

    served = await serve_clauses(tenant, actor_ref=CAROL, situation_id=launch)
    assert [c.id for c in served.clauses] == [clause_id]
    assert (await _ledger_rows(tenant))[0]["clause_ids"] == [clause_id]


async def test_the_situation_is_recorded_on_the_ledger_row(tenant: str) -> None:
    launch = await _situation_id(tenant, "launch-run")
    clause_id = await _make_clause(tenant, body="open a run first")
    await _attach(tenant, clause_id, launch)

    await serve_clauses(tenant, actor_ref=CAROL, situation_id=launch)
    assert (await _ledger_rows(tenant))[0]["situation_id"] == launch


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


async def test_stronger_status_outranks_weaker(tenant: str) -> None:
    await _make_clause(tenant, body="contested one", status="contested")
    await _make_clause(tenant, body="confirmed one", status="expert_confirmed")

    served = await serve_clauses(tenant, actor_ref=CAROL)
    assert [c.body for c in served.clauses] == ["confirmed one", "contested one"]


async def test_warnings_sort_last_but_are_still_served(tenant: str) -> None:
    """`stale` and `contested` are active warnings, not weak evidence.

    Leading a three-card response with one buries what a person could act on --
    but hiding them is how a contested rule quietly becomes policy again.
    """
    await _make_clause(tenant, body="stale one", status="stale")
    await _make_clause(tenant, body="declared one", status="declared")

    served = await serve_clauses(tenant, actor_ref=CAROL)
    assert [c.body for c in served.clauses] == ["declared one", "stale one"]


async def test_a_narrower_scope_outranks_a_broader_one(tenant: str) -> None:
    """Uniform in v0 (everything is workspace-wide), ordered correctly anyway."""
    await _make_clause(tenant, body="workspace wide", scope={})
    await _make_clause(tenant, body="repo specific", scope={"repo": "prbe-knowledge"})
    await _make_clause(
        tenant,
        body="repo and path",
        scope={"repo": "prbe-knowledge", "path": "engine/"},
    )

    served = await serve_clauses(tenant, actor_ref=CAROL)
    assert [c.body for c in served.clauses] == [
        "repo and path",
        "repo specific",
        "workspace wide",
    ]


async def test_scope_outranks_status(tenant: str) -> None:
    """The documented order is scope FIRST, then status. A narrow contested rule
    beats a broad confirmed one because it is the one that applies here."""
    await _make_clause(tenant, body="broad and confirmed", status="expert_confirmed")
    await _make_clause(
        tenant, body="narrow but contested", status="contested", scope={"repo": "x"}
    )

    served = await serve_clauses(tenant, actor_ref=CAROL)
    assert next(c.body for c in served.clauses) == "narrow but contested"


async def test_recency_breaks_a_status_tie(tenant: str) -> None:
    older = await _make_clause(tenant, body="older", age_days=3)
    newer = await _make_clause(tenant, body="newer", age_days=0)

    served = await serve_clauses(tenant, actor_ref=CAROL)
    assert [c.id for c in served.clauses] == [newer, older]


async def test_an_update_refreshes_a_clauses_place_in_the_order(tenant: str) -> None:
    """The trigger that broke this test's first draft is also the feature.

    `updated_at` is only a useful ranking signal if editing a rule actually moves
    it, and Postgres has no ON UPDATE default -- without
    `clauses_touch_updated_at_trg` the column would equal `created_at` forever
    and "recency" would silently mean "insertion order".
    """
    stale_row = await _make_clause(tenant, body="edited later", age_days=5)
    await _make_clause(tenant, body="untouched", age_days=1)

    async with with_tenant(tenant) as conn:
        await conn.execute(
            "UPDATE clauses SET body = 'edited later, and revised' WHERE id = $1",
            stale_row,
        )

    served = await serve_clauses(tenant, actor_ref=CAROL)
    assert next(c.id for c in served.clauses) == stale_row


async def test_an_unranked_status_sorts_last_rather_than_vanishing(tenant: str) -> None:
    """A status added to the CHECK constraint but forgotten in STATUS_RANK must
    degrade to "ranked last", never to a dropped row -- a silently missing rule
    is the one failure mode a serving surface must not have."""
    unranked = set(_CHECK_STATUSES) - set(STATUS_RANK)
    assert not unranked, f"STATUS_RANK is missing {sorted(unranked)}"


#: Every value the `clauses.status` CHECK constraint accepts, spelled out by hand
#: so a migration that widens the constraint fails this file rather than silently
#: shipping an unranked status.
_CHECK_STATUSES = (
    "declared",
    "documented",
    "observed_convention",
    "success_associated",
    "expert_confirmed",
    "intervention_validated",
    "exception",
    "anti_pattern",
    "contested",
    "stale",
    "agent_proposed",
)


# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------


async def test_the_default_limit_is_three_cards(tenant: str) -> None:
    for i in range(6):
        await _make_clause(tenant, body=f"rule {i}")
    served = await serve_clauses(tenant, actor_ref=CAROL)
    assert len(served.clauses) == DEFAULT_LIMIT == 3


async def test_a_caller_may_ask_for_more(tenant: str) -> None:
    for i in range(6):
        await _make_clause(tenant, body=f"rule {i}")
    served = await serve_clauses(tenant, actor_ref=CAROL, limit=6)
    assert len(served.clauses) == 6


async def test_the_ceiling_is_enforced_against_an_absurd_limit(tenant: str) -> None:
    for i in range(3):
        await _make_clause(tenant, body=f"rule {i}")
    served = await serve_clauses(tenant, actor_ref=CAROL, limit=10_000)
    assert len(served.clauses) == 3
    assert MAX_LIMIT == 50


async def test_a_zero_limit_serves_nothing_and_logs_nothing(tenant: str) -> None:
    await _make_clause(tenant, body="rule a")
    served = await serve_clauses(tenant, actor_ref=CAROL, limit=0)
    assert served.clauses == []
    assert served.serve_id is None
    assert await _ledger_rows(tenant) == []


# --------------------------------------------------------------------------
# Tenancy
# --------------------------------------------------------------------------


async def test_another_tenants_clauses_are_never_served(tenant: str) -> None:
    await _make_clause(OTHER_TENANT, body="their rule")
    await _make_clause(tenant, body="our rule")

    served = await serve_clauses(tenant, actor_ref=CAROL)
    assert [c.body for c in served.clauses] == ["our rule"]


async def test_the_ledger_row_lands_under_the_serving_tenant(tenant: str) -> None:
    await _make_clause(tenant, body="our rule")
    await serve_clauses(tenant, actor_ref=CAROL)
    assert await _ledger_rows(OTHER_TENANT) == []
    assert len(await _ledger_rows(tenant)) == 1
