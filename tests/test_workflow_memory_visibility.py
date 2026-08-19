"""The single-author visibility guard."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

import pytest
import pytest_asyncio

from shared.db import raw_conn, with_tenant
from shared.wfmem.visibility import fetch_visible_clauses

TENANT = "cust-wfmem-vis"
ALICE = "user:alice"
BOB = "user:bob"

#: `exposure_tainted` has NO DEFAULT on purpose (see db/schema.sql): a writer
#: that forgets the taint computation must hit a constraint violation rather
#: than silently record the evidence as clean. So every insert below states it.
POINTER = '{"session":"s","span":[0,1]}'


@pytest_asyncio.fixture
async def tenant(live_db) -> AsyncIterator[str]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'wfmem-vis', 'h-wfmem-vis') ON CONFLICT DO NOTHING",
            TENANT,
        )
    yield TENANT
    async with raw_conn() as conn:
        await conn.execute("DELETE FROM customers WHERE customer_id = $1", TENANT)


async def _clause_with_authors(body: str, owner: str, evidence_authors: list[str]) -> UUID:
    async with with_tenant(TENANT) as conn:
        clause_id = await conn.fetchval(
            """
            INSERT INTO clauses (customer_id, kind, body, status, author_ref)
            VALUES ($1, 'step', $2, 'declared', $3) RETURNING id
            """,
            TENANT,
            body,
            owner,
        )
        for author in evidence_authors:
            await conn.execute(
                f"""
                INSERT INTO clause_evidence
                    (customer_id, clause_id, source_class, source_ref, author_ref,
                     exposure_tainted, ts)
                VALUES ($1, $2, 'declared', '{POINTER}'::jsonb, $3, false, now())
                """,
                TENANT,
                clause_id,
                author,
            )
    return clause_id


@pytest.mark.asyncio
async def test_single_author_clause_is_hidden_from_others(tenant):
    await _clause_with_authors("alice's private habit", ALICE, [ALICE])

    async with with_tenant(TENANT) as conn:
        seen_by_bob = await fetch_visible_clauses(conn, BOB)
        seen_by_alice = await fetch_visible_clauses(conn, ALICE)

    assert [r["body"] for r in seen_by_bob] == []
    assert [r["body"] for r in seen_by_alice] == ["alice's private habit"]


@pytest.mark.asyncio
async def test_second_human_makes_it_visible(tenant):
    await _clause_with_authors("a shared convention", ALICE, [ALICE, BOB])

    async with with_tenant(TENANT) as conn:
        seen_by_carol = await fetch_visible_clauses(conn, "user:carol")

    assert [r["body"] for r in seen_by_carol] == ["a shared convention"]


@pytest.mark.asyncio
async def test_repeated_evidence_from_one_author_does_not_unlock(tenant):
    """Distinctness, not count: one person logging twice is still one person."""
    await _clause_with_authors("alice again and again", ALICE, [ALICE, ALICE, ALICE])

    async with with_tenant(TENANT) as conn:
        seen_by_bob = await fetch_visible_clauses(conn, BOB)

    assert seen_by_bob == []


@pytest.mark.asyncio
async def test_agent_evidence_does_not_count_as_a_second_human(tenant):
    """The guarantee is 'until a second HUMAN appears'. Agent-derived evidence
    must not unlock a clause -- otherwise an agent that read alice's rule and
    then behaved accordingly publishes it to the team, which is exactly the
    self-reinforcement loop taint-censoring exists to prevent."""
    async with with_tenant(TENANT) as conn:
        clause_id = await conn.fetchval(
            """
            INSERT INTO clauses (customer_id, kind, body, status, author_ref)
            VALUES ($1, 'step', 'alice plus two robots', 'declared', $2) RETURNING id
            """,
            TENANT,
            ALICE,
        )
        for source_class, author in (
            ("declared", ALICE),
            ("agent_transcript", "agent:claude-1"),
            ("agent_transcript", "agent:claude-2"),
            ("run_outcome", "agent:runner"),
        ):
            await conn.execute(
                f"""
                INSERT INTO clause_evidence
                    (customer_id, clause_id, source_class, source_ref, author_ref,
                     exposure_tainted, ts)
                VALUES ($1, $2, $3, '{POINTER}'::jsonb, $4, false, now())
                """,
                TENANT,
                clause_id,
                source_class,
                author,
            )

        seen_by_bob = await fetch_visible_clauses(conn, BOB)

    assert seen_by_bob == [], "agent evidence must not unlock a single-human clause"


@pytest.mark.asyncio
async def test_tainted_human_evidence_does_not_count_as_a_second_human(tenant):
    """Taint-excluded support, the other half of the same loop. Bob's evidence
    is from a real human and a human source_class, but it is marked
    exposure_tainted: it was produced in a session that had already been SERVED
    this clause. An echo of what we told you is not an independent second voice,
    so ONE untainted human voice remains and the clause stays alice's."""
    async with with_tenant(TENANT) as conn:
        clause_id = await conn.fetchval(
            """
            INSERT INTO clauses (customer_id, kind, body, status, author_ref)
            VALUES ($1, 'step', 'alice plus a tainted echo', 'declared', $2) RETURNING id
            """,
            TENANT,
            ALICE,
        )
        for author, tainted in ((ALICE, False), (BOB, True)):
            await conn.execute(
                f"""
                INSERT INTO clause_evidence
                    (customer_id, clause_id, source_class, source_ref, author_ref,
                     exposure_tainted, ts)
                VALUES ($1, $2, 'human_message', '{POINTER}'::jsonb, $3, $4, now())
                """,
                TENANT,
                clause_id,
                author,
                tainted,
            )

        seen_by_bob = await fetch_visible_clauses(conn, BOB)
        seen_by_alice = await fetch_visible_clauses(conn, ALICE)

    assert seen_by_bob == [], "tainted evidence must not unlock a single-human clause"
    assert [r["body"] for r in seen_by_alice] == ["alice plus a tainted echo"]
