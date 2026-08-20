"""Publishing a rule without waiting for a second human to agree.

THE CASE THIS EXISTS FOR. A clause is visible to the team only once two distinct
untainted humans back it. That guard is right for mined and inferred rules -- one
person's working note is not the team's practice -- but it makes the most
important case impossible: a lead sits down and declares twenty existing team
rules, and every one is invisible to everybody, silently, because a lead is one
person. Nothing errors. The feature just looks dead, which is exactly what the
external tenant's onboarding session would have produced on day one.

WHAT THIS FILE HAS TO DEFEND:

* PUBLISHING WORKS, and a single-author published rule really does reach a
  stranger.
* IT IS ATTRIBUTED. `shared_by` names who did it. Unilateral publication is an
  act of authority over what a team is told to do, and an unattributable one is
  worse than none -- nobody to ask when the rule turns out wrong.
* IT IS NOT THE DEFAULT, and no path turns it on by accident.
* IT IS LABELLED, NOT LAUNDERED. A published single-author rule is still
  reported as having one human behind it. If publishing made a rule
  indistinguishable from one the team demonstrably follows, it would have
  defeated the guard rather than routed around it.
* FIRST PUBLISHER WINS. A second publish does not reassign responsibility for a
  decision somebody else made.
* IT CAN BE WITHDRAWN, without destroying the clause or its evidence.

Run with the isolated wfmem database (these fixtures TRUNCATE):

    PRBE_TEST_DATABASE_URL=postgresql://prbe:prbe@localhost:5432/prbe_knowledge_wfmem \
        .venv/bin/pytest tests/test_workflow_memory_publication.py -q
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import pytest
import pytest_asyncio

from engine.shared.db import raw_conn, with_tenant
from engine.shared.wfmem.declaring import (
    DeclarationRefused,
    Relation,
    declare,
    publish_clause,
    unpublish_clause,
)
from engine.shared.wfmem.serving import serve_clauses
from engine.shared.wfmem.structuring import ClauseDraft
from engine.shared.wfmem.visibility import fetch_visible_clauses

TENANT = "cust-wfmem-pub"
OTHER_TENANT = "cust-wfmem-pub-b"

LEAD = "user:lead"
COLLEAGUE = "user:colleague"
STRANGER = "user:stranger"


def _draft(body: str = "open a Probe run before the first GPU step") -> ClauseDraft:
    return ClauseDraft(kind="step", body=body, semantic_action=None, binding={}, scope={})


@pytest_asyncio.fixture
async def tenant(live_db: None) -> AsyncIterator[str]:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'wfmem-pub', 'h-wfmem-pub'),
                   ($2, 'wfmem-pub-b', 'h-wfmem-pub-b')
            ON CONFLICT (customer_id) DO NOTHING
            """,
            TENANT,
            OTHER_TENANT,
        )
    yield TENANT


async def _row(customer_id: str, clause_id: UUID) -> Any:
    async with with_tenant(customer_id) as conn:
        return await conn.fetchrow(
            "SELECT * FROM clauses WHERE customer_id = $1 AND id = $2", customer_id, clause_id
        )


# --------------------------------------------------------------------------
# The gap it closes
# --------------------------------------------------------------------------


async def test_without_publishing_a_solo_rule_reaches_nobody(tenant: str) -> None:
    """The behaviour that makes publishing necessary, stated as a test.

    This is not a bug -- it is the two-human guard doing its job. It is here so
    that if somebody ever weakens the guard, they break this and have to think
    about it rather than discovering the feature became a broadcast channel.
    """
    await declare(tenant, _draft(), actor_ref=LEAD, source_ref={"session": "s1"})
    served = await serve_clauses(tenant, actor_ref=STRANGER)
    assert served.clauses == []


async def test_publishing_at_declare_time_reaches_a_stranger(tenant: str) -> None:
    result = await declare(
        tenant, _draft(), actor_ref=LEAD, source_ref={"session": "s1"}, publish=True
    )

    assert result.shared_by == LEAD
    served = await serve_clauses(tenant, actor_ref=STRANGER)
    assert [c.id for c in served.clauses] == [result.clause_id]


async def test_a_batch_of_solo_declarations_is_the_onboarding_case(tenant: str) -> None:
    """Twenty rules, one lead, all live. The scenario the design's §8 mitigation
    depends on -- 'run a live onboarding session declaring their existing rules'
    -- which without publishing would have produced twenty invisible clauses."""
    for i in range(20):
        await declare(
            tenant,
            _draft(body=f"team rule {i}"),
            actor_ref=LEAD,
            source_ref={"session": "onboarding"},
            publish=True,
        )

    served = await serve_clauses(tenant, actor_ref=STRANGER, limit=50)
    assert len(served.clauses) == 20


async def test_publishing_is_not_the_default(tenant: str) -> None:
    result = await declare(tenant, _draft(), actor_ref=LEAD, source_ref={})
    assert result.shared_by is None
    assert (await _row(tenant, result.clause_id))["shared_by"] is None


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------


async def test_publication_records_who_and_when(tenant: str) -> None:
    result = await declare(
        tenant, _draft(), actor_ref=LEAD, source_ref={}, publish=True
    )
    row = await _row(tenant, result.clause_id)
    assert row["shared_by"] == LEAD
    assert row["shared_at"] is not None


async def test_the_pair_cannot_be_half_written(tenant: str) -> None:
    """A `shared_at` with no `shared_by` is an unattributable publication, and a
    `shared_by` with no timestamp cannot be ordered against later evidence.
    Both are the "who decided this, and when" question half-answered, so the
    CHECK refuses them at the schema."""
    result = await declare(tenant, _draft(), actor_ref=LEAD, source_ref={})

    # SEPARATE TRANSACTIONS, one per violation. The first CheckViolation aborts
    # its transaction, so a second statement inside the same one comes back as
    # InFailedSQLTransactionError -- which would pass a `raises(Exception)` and
    # prove nothing about the constraint.
    async with with_tenant(tenant) as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "UPDATE clauses SET shared_at = now() WHERE id = $1", result.clause_id
            )

    async with with_tenant(tenant) as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "UPDATE clauses SET shared_by = $2 WHERE id = $1", result.clause_id, LEAD
            )


async def test_publishing_an_existing_clause_later(tenant: str) -> None:
    """Declaring and publishing are different decisions made at different times.

    Somebody writes a note for themselves, and weeks later decides the team
    should follow it. Routing that through declare would mean retyping the rule
    and producing a SECOND clause instead of publishing the first.
    """
    result = await declare(tenant, _draft(), actor_ref=LEAD, source_ref={})
    assert (await serve_clauses(tenant, actor_ref=STRANGER)).clauses == []

    publisher = await publish_clause(tenant, result.clause_id, actor_ref=LEAD)

    assert publisher == LEAD
    served = await serve_clauses(tenant, actor_ref=STRANGER)
    assert [c.id for c in served.clauses] == [result.clause_id]
    async with with_tenant(tenant) as conn:
        assert await conn.fetchval("SELECT count(*) FROM clauses") == 1


async def test_the_first_publisher_keeps_the_attribution(tenant: str) -> None:
    """A later caller must not quietly inherit responsibility for a decision
    somebody else made. The column answers "who put this in front of the team",
    and that is the first person to do it."""
    result = await declare(tenant, _draft(), actor_ref=LEAD, source_ref={})
    await publish_clause(tenant, result.clause_id, actor_ref=LEAD)

    again = await publish_clause(tenant, result.clause_id, actor_ref=COLLEAGUE)

    assert again == LEAD
    assert (await _row(tenant, result.clause_id))["shared_by"] == LEAD


async def test_merging_with_publish_does_not_steal_attribution(tenant: str) -> None:
    first = await declare(
        tenant, _draft(), actor_ref=LEAD, source_ref={}, publish=True
    )
    merged = await declare(
        tenant,
        _draft(),
        actor_ref=COLLEAGUE,
        source_ref={},
        relation=Relation.MERGE,
        related_clause_id=first.clause_id,
        publish=True,
    )
    assert merged.shared_by == LEAD


async def test_publishing_must_record_who_did_it(tenant: str) -> None:
    result = await declare(tenant, _draft(), actor_ref=LEAD, source_ref={})
    with pytest.raises(DeclarationRefused, match="who did it"):
        await publish_clause(tenant, result.clause_id, actor_ref="")


async def test_publishing_a_missing_clause_reports_rather_than_raises(tenant: str) -> None:
    assert await publish_clause(tenant, uuid4(), actor_ref=LEAD) is None


async def test_another_tenants_clause_cannot_be_published(tenant: str) -> None:
    theirs = await declare(OTHER_TENANT, _draft(), actor_ref=LEAD, source_ref={})
    assert await publish_clause(tenant, theirs.clause_id, actor_ref=LEAD) is None
    assert (await _row(OTHER_TENANT, theirs.clause_id))["shared_by"] is None


# --------------------------------------------------------------------------
# Labelled, not laundered
# --------------------------------------------------------------------------


async def test_a_published_solo_rule_still_reports_one_human(tenant: str) -> None:
    """THE POINT OF KEEPING PUBLICATION OFF THE STATUS LADDER.

    If publishing made a rule indistinguishable from one the team demonstrably
    follows, it would have DEFEATED the two-human guard rather than routed
    around it. A reader has to be able to tell "one person says so" from "four
    people do this", and that is what these two fields together are for.
    """
    result = await declare(
        tenant, _draft(), actor_ref=LEAD, source_ref={}, publish=True
    )
    served = await serve_clauses(tenant, actor_ref=STRANGER)
    clause = served.clauses[0]

    assert clause.id == result.clause_id
    assert clause.shared_by == LEAD
    assert clause.human_backers == 1
    assert clause.status == "declared"


async def test_a_corroborated_rule_reports_its_backers(tenant: str) -> None:
    first = await declare(tenant, _draft(), actor_ref=LEAD, source_ref={})
    await declare(
        tenant,
        _draft(),
        actor_ref=COLLEAGUE,
        source_ref={},
        relation=Relation.MERGE,
        related_clause_id=first.clause_id,
    )

    served = await serve_clauses(tenant, actor_ref=STRANGER)
    clause = served.clauses[0]
    assert clause.human_backers == 2
    assert clause.shared_by is None


async def test_a_tainted_backer_is_not_counted(tenant: str) -> None:
    """The count shown to a reader must use the same rule the guard does.

    An agent that was served a clause and echoed it back is not an independent
    voice, and a surface that counted it would report "2 humans" for a rule one
    person wrote and a machine repeated.
    """
    result = await declare(
        tenant, _draft(), actor_ref=LEAD, source_ref={}, publish=True
    )
    async with with_tenant(tenant) as conn:
        await conn.execute(
            """
            INSERT INTO clause_evidence
                (customer_id, clause_id, source_class, source_ref,
                 author_ref, exposure_tainted, ts)
            VALUES ($1, $2, 'declared', '{}'::jsonb, $3, TRUE, now())
            """,
            tenant,
            result.clause_id,
            COLLEAGUE,
        )

    served = await serve_clauses(tenant, actor_ref=STRANGER)
    assert served.clauses[0].human_backers == 1


# --------------------------------------------------------------------------
# Withdrawal
# --------------------------------------------------------------------------


async def test_a_publication_can_be_withdrawn(tenant: str) -> None:
    """Publishing is unilateral, so the undo matters.

    Without it the only ways to walk back a rule the team turns out to disagree
    with are deleting the clause -- destroying its evidence and history -- or
    leaving false policy in front of everybody.
    """
    result = await declare(
        tenant, _draft(), actor_ref=LEAD, source_ref={}, publish=True
    )
    assert len((await serve_clauses(tenant, actor_ref=STRANGER)).clauses) == 1

    assert await unpublish_clause(tenant, result.clause_id) is True

    assert (await serve_clauses(tenant, actor_ref=STRANGER)).clauses == []
    row = await _row(tenant, result.clause_id)
    assert row["shared_by"] is None and row["shared_at"] is None
    # The clause and its evidence survive; only the publication was withdrawn.
    async with with_tenant(tenant) as conn:
        assert await conn.fetchval("SELECT count(*) FROM clauses") == 1
        assert await conn.fetchval("SELECT count(*) FROM clause_evidence") == 1


async def test_withdrawing_leaves_a_genuinely_corroborated_rule_visible(tenant: str) -> None:
    """Withdrawal returns the clause to the ORDINARY rule, it does not hide it.

    If two humans came to back it while it was published, it stays visible --
    which is correct, and not something a caller should have to work out.
    """
    first = await declare(
        tenant, _draft(), actor_ref=LEAD, source_ref={}, publish=True
    )
    await declare(
        tenant,
        _draft(),
        actor_ref=COLLEAGUE,
        source_ref={},
        relation=Relation.MERGE,
        related_clause_id=first.clause_id,
    )

    await unpublish_clause(tenant, first.clause_id)

    served = await serve_clauses(tenant, actor_ref=STRANGER)
    assert [c.id for c in served.clauses] == [first.clause_id]


async def test_withdrawing_an_unpublished_clause_changes_nothing(tenant: str) -> None:
    result = await declare(tenant, _draft(), actor_ref=LEAD, source_ref={})
    assert await unpublish_clause(tenant, result.clause_id) is False


async def test_another_tenants_publication_cannot_be_withdrawn(tenant: str) -> None:
    theirs = await declare(
        OTHER_TENANT, _draft(), actor_ref=LEAD, source_ref={}, publish=True
    )
    assert await unpublish_clause(tenant, theirs.clause_id) is False
    assert (await _row(OTHER_TENANT, theirs.clause_id))["shared_by"] == LEAD


# --------------------------------------------------------------------------
# The predicate itself
# --------------------------------------------------------------------------


async def test_the_visibility_helper_agrees_with_the_serving_path(tenant: str) -> None:
    """`fetch_visible_clauses` and `serve_clauses` share VISIBILITY_PREDICATE and
    must not drift -- the helper is what other readers will reach for."""
    published = await declare(
        tenant, _draft(body="published"), actor_ref=LEAD, source_ref={}, publish=True
    )
    await declare(tenant, _draft(body="private"), actor_ref=LEAD, source_ref={})

    async with with_tenant(tenant) as conn:
        visible = await fetch_visible_clauses(conn, STRANGER)
    served = await serve_clauses(tenant, actor_ref=STRANGER)

    assert [row["id"] for row in visible] == [published.clause_id]
    assert [c.id for c in served.clauses] == [published.clause_id]


async def test_publication_does_not_leak_across_tenants(tenant: str) -> None:
    await declare(
        OTHER_TENANT, _draft(body="theirs"), actor_ref=LEAD, source_ref={}, publish=True
    )
    served = await serve_clauses(tenant, actor_ref=STRANGER)
    assert served.clauses == []


async def test_the_ledger_records_a_published_serve(tenant: str) -> None:
    """A published rule reaching a stranger is exposure like any other."""
    result = await declare(
        tenant, _draft(), actor_ref=LEAD, source_ref={}, publish=True
    )
    served = await serve_clauses(tenant, actor_ref=STRANGER, session_id="sess-1")

    async with with_tenant(tenant) as conn:
        row = await conn.fetchrow("SELECT * FROM serve_ledger WHERE id = $1", served.serve_id)
    assert row["actor_ref"] == STRANGER
    assert list(row["clause_ids"]) == [result.clause_id]


def test_the_schema_mirror_carries_the_publication_columns() -> None:
    """CI applies db/schema.sql and stamps alembic -- it never runs the chain.

    A migration whose columns are missing from the mirror passes every local
    test and produces a database without them in CI and on a fresh deploy.
    """
    from pathlib import Path

    schema = Path(__file__).resolve().parents[1] / "db" / "schema.sql"
    text = schema.read_text()
    assert "shared_by           TEXT" in text
    assert "ck_clauses_publication_is_attributed" in text
    assert "clauses_published_idx" in text
