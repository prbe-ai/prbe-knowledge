"""The mailbox core: enqueue idempotency, pending, actor claims, ack outcomes.

Everything the audit reproduced against the v2 draft is a test here:

* SAME KEY, SAME PAYLOAD IS A NO-OP that hands back the original card. Same key,
  different payload is a CONFLICT, never a silent replace -- rows are immutable.
  Same key after the original expired is `expired`: the caller mints a new key.
* ACTOR-KEYED DEDUPE IS PER RECIPIENT and works despite the NULL session_id.
* PENDING excludes expired cards, cards with ANY delivery row (emitted or
  terminal), ids the client already knows about, and cards under a live lease.
* ONE WINNER PER CLAIM. Two concurrent actor claims on one card: exactly one
  gets it; the loser gets None, not a second copy.
* ACK IS IDEMPOTENT ON attempt_id, and a second attempt on the same card is a
  second row -- repeated emissions must stay visible in the fault catalog.

Run with the isolated database:

    PRBE_TEST_DATABASE_URL=postgresql://prbe:prbe@localhost:55442/prbe_knowledge \
        .venv/bin/pytest tests/test_companion_mailbox.py -q
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio

from engine.shared.companion.mailbox import (
    PREFIX,
    Card,
    EnqueueConflict,
    EnqueueRefused,
    ack,
    claim_for_actor,
    deliveries,
    enqueue,
    pending,
)
from engine.shared.db import raw_conn, with_tenant

TENANT = "cust-companion-mb"
ALICE = "user:00000000-0000-0000-0000-00000000a11c"
BOB = "user:00000000-0000-0000-0000-000000000b0b"


@pytest_asyncio.fixture
async def tenant(live_db) -> AsyncIterator[str]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'mb', 'h-mb') ON CONFLICT (customer_id) DO NOTHING",
            TENANT,
        )
    yield TENANT


async def _enqueue(**kw) -> tuple[Card, bool]:
    base = dict(
        recipient=ALICE,
        session_id="sess-1",
        class_="seam",
        body="run make check before claiming done",
        dedupe_key="k1",
        ttl_seconds=600,
    )
    base.update(kw)
    return await enqueue(TENANT, **base)


# --------------------------------------------------------------------------
# enqueue
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_applies_prefix_once(tenant) -> None:
    card, created = await _enqueue()
    assert created is True
    assert card.body == PREFIX + "run make check before claiming done"
    again, _ = await _enqueue(body=PREFIX + "twice-prefixed?", dedupe_key="k2")
    assert again.body == PREFIX + "twice-prefixed?"


@pytest.mark.asyncio
async def test_prefix_counts_toward_the_cap(tenant) -> None:
    room = 4000 - len(PREFIX)
    ok, _ = await _enqueue(body="x" * room, dedupe_key="fits")
    assert len(ok.body) == 4000
    with pytest.raises(EnqueueRefused):
        await _enqueue(body="x" * (room + 1), dedupe_key="overflows")


@pytest.mark.asyncio
async def test_identical_retry_returns_original(tenant) -> None:
    first, created1 = await _enqueue()
    second, created2 = await _enqueue()
    assert created1 is True and created2 is False
    assert second.mailbox_id == first.mailbox_id
    assert second.trial_id == first.trial_id


@pytest.mark.asyncio
async def test_different_payload_same_key_conflicts(tenant) -> None:
    await _enqueue()
    with pytest.raises(EnqueueConflict) as exc:
        await _enqueue(body="something else")
    assert exc.value.reason == "conflict"
    with pytest.raises(EnqueueConflict):
        await _enqueue(ttl_seconds=601)


@pytest.mark.asyncio
async def test_expired_key_is_expired_not_reused(tenant) -> None:
    async with with_tenant(TENANT) as conn:
        await conn.execute(
            """
            INSERT INTO companion_mailbox
                (customer_id, recipient, session_id, class, body, dedupe_key, source,
                 trial_id, created_at, expires_at)
            VALUES ($1, $2, 'sess-1', 'seam', $3, 'k1', 'driver', $4,
                    now() - interval '2 hours', now() - interval '1 hour')
            """,
            TENANT,
            ALICE,
            PREFIX + "run make check before claiming done",
            uuid4(),
        )
    with pytest.raises(EnqueueConflict) as exc:
        await _enqueue()
    assert exc.value.reason == "expired"


@pytest.mark.asyncio
async def test_actor_keyed_dedupe_is_per_recipient(tenant) -> None:
    a1, c1 = await _enqueue(session_id=None)
    a2, c2 = await _enqueue(session_id=None)
    b1, c3 = await _enqueue(session_id=None, recipient=BOB)
    assert (c1, c2, c3) == (True, False, True)
    assert a1.mailbox_id == a2.mailbox_id != b1.mailbox_id


@pytest.mark.asyncio
async def test_structural_refusals(tenant) -> None:
    with pytest.raises(EnqueueRefused):
        await _enqueue(class_="decision-push", session_id=None)
    with pytest.raises(EnqueueRefused):
        await _enqueue(ttl_seconds=0)
    with pytest.raises(EnqueueRefused):
        await _enqueue(ttl_seconds=86_401)
    with pytest.raises(EnqueueRefused):
        await _enqueue(body="   ")
    with pytest.raises(EnqueueRefused):
        await _enqueue(session_id="")
    with pytest.raises(EnqueueRefused):
        await _enqueue(recipient="ingest:not-a-user")
    with pytest.raises(EnqueueRefused):
        await _enqueue(class_="rumor")
    with pytest.raises(EnqueueRefused):
        await _enqueue(intended_seam="telepathy")


# --------------------------------------------------------------------------
# pending
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_is_ordered_and_filtered(tenant) -> None:
    older, _ = await _enqueue(dedupe_key="a")
    newer, _ = await _enqueue(dedupe_key="b")
    await _enqueue(dedupe_key="c", session_id="sess-2")  # other session: excluded
    await _enqueue(dedupe_key="d", recipient=BOB)  # other person: excluded
    got = await pending(TENANT, session_id="sess-1", recipient=ALICE)
    assert [c.mailbox_id for c in got] == [older.mailbox_id, newer.mailbox_id]
    got = await pending(TENANT, session_id="sess-1", recipient=ALICE, exclude={older.mailbox_id})
    assert [c.mailbox_id for c in got] == [newer.mailbox_id]


@pytest.mark.asyncio
async def test_pending_excludes_delivered_and_terminal(tenant) -> None:
    emitted, _ = await _enqueue(dedupe_key="a")
    canceled, _ = await _enqueue(dedupe_key="b")
    fresh, _ = await _enqueue(dedupe_key="c")
    await ack(
        TENANT,
        mailbox_id=emitted.mailbox_id,
        attempt_id=uuid4(),
        seam="stop",
        outcome="emitted",
        receiving_instance="dev:cc",
    )
    await ack(
        TENANT,
        mailbox_id=canceled.mailbox_id,
        attempt_id=uuid4(),
        seam="user-prompt",
        outcome="canceled",
        receiving_instance="dev:cc",
    )
    got = await pending(TENANT, session_id="sess-1", recipient=ALICE)
    assert [c.mailbox_id for c in got] == [fresh.mailbox_id]


@pytest.mark.asyncio
async def test_pending_excludes_expired(tenant) -> None:
    async with with_tenant(TENANT) as conn:
        await conn.execute(
            """
            INSERT INTO companion_mailbox
                (customer_id, recipient, session_id, class, body, dedupe_key, source,
                 trial_id, created_at, expires_at)
            VALUES ($1, $2, 'sess-1', 'seam', 'old', 'old', 'driver', $3,
                    now() - interval '2 hours', now() - interval '1 hour')
            """,
            TENANT,
            ALICE,
            uuid4(),
        )
    assert await pending(TENANT, session_id="sess-1", recipient=ALICE) == []


# --------------------------------------------------------------------------
# actor lane: claims
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_single_winner_under_concurrency(tenant) -> None:
    card, _ = await _enqueue(session_id=None)
    results = await asyncio.gather(
        claim_for_actor(TENANT, recipient=ALICE, claimed_by="ingest:one"),
        claim_for_actor(TENANT, recipient=ALICE, claimed_by="ingest:two"),
    )
    winners = [r for r in results if r is not None]
    assert len(winners) == 1 and winners[0].mailbox_id == card.mailbox_id
    # The loser cannot take it while the lease is live.
    assert await claim_for_actor(TENANT, recipient=ALICE, claimed_by="ingest:two") is None


@pytest.mark.asyncio
async def test_claimed_card_is_not_pending_for_the_actor_lane(tenant) -> None:
    card, _ = await _enqueue(session_id=None)
    assert (await claim_for_actor(TENANT, recipient=ALICE, claimed_by="ingest:one")) is not None
    # pending() is the session lane; an actor-keyed card never appears there.
    assert await pending(TENANT, session_id="sess-1", recipient=ALICE) == []
    # ack releases the lease and the card is terminal for everyone.
    await ack(
        TENANT,
        mailbox_id=card.mailbox_id,
        attempt_id=uuid4(),
        seam="mcp-rider",
        outcome="emitted",
        receiving_instance="mcp",
        delivering_credential="ingest:one",
    )
    assert await claim_for_actor(TENANT, recipient=ALICE, claimed_by="ingest:one") is None


@pytest.mark.asyncio
async def test_expired_lease_can_be_reclaimed(tenant) -> None:
    card, _ = await _enqueue(session_id=None)
    assert (
        await claim_for_actor(TENANT, recipient=ALICE, claimed_by="ingest:one", lease_seconds=1)
    ) is not None
    async with with_tenant(TENANT) as conn:
        await conn.execute("UPDATE companion_claims SET lease_until = now() - interval '1 second'")
    won = await claim_for_actor(TENANT, recipient=ALICE, claimed_by="ingest:two")
    assert won is not None and won.mailbox_id == card.mailbox_id


# --------------------------------------------------------------------------
# ack
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ack_idempotent_on_attempt_id_but_attempts_stay_distinct(tenant) -> None:
    card, _ = await _enqueue()
    attempt = uuid4()
    first_id, created1 = await ack(
        TENANT,
        mailbox_id=card.mailbox_id,
        attempt_id=attempt,
        seam="stop",
        outcome="emitted",
        receiving_instance="dev:cc",
        receipt_to_emission_ms=12,
    )
    second_id, created2 = await ack(
        TENANT,
        mailbox_id=card.mailbox_id,
        attempt_id=attempt,
        seam="stop",
        outcome="emitted",
        receiving_instance="dev:cc",
        receipt_to_emission_ms=12,
    )
    third_id, created3 = await ack(
        TENANT,
        mailbox_id=card.mailbox_id,
        attempt_id=uuid4(),
        seam="stop",
        outcome="emitted",
        receiving_instance="dev:cc",
    )
    assert (created1, created2, created3) == (True, False, True)
    assert first_id == second_id != third_id
    rows = await deliveries(TENANT, session_id="sess-1")
    assert [r["attempt_id"] for r in rows].count(attempt) == 1 and len(rows) == 2


@pytest.mark.asyncio
async def test_ack_refuses_unknown_seam_outcome_or_card(tenant) -> None:
    card, _ = await _enqueue()
    with pytest.raises(ValueError):
        await ack(
            TENANT,
            mailbox_id=card.mailbox_id,
            attempt_id=uuid4(),
            seam="telepathy",
            outcome="emitted",
            receiving_instance="x",
        )
    with pytest.raises(ValueError):
        await ack(
            TENANT,
            mailbox_id=card.mailbox_id,
            attempt_id=uuid4(),
            seam="stop",
            outcome="maybe",
            receiving_instance="x",
        )
    with pytest.raises(ValueError):
        await ack(
            TENANT,
            mailbox_id=uuid4(),
            attempt_id=uuid4(),
            seam="stop",
            outcome="emitted",
            receiving_instance="x",
        )


@pytest.mark.asyncio
async def test_deliveries_readback_by_recipient_covers_actor_lane(tenant) -> None:
    card, _ = await _enqueue(session_id=None)
    await ack(
        TENANT,
        mailbox_id=card.mailbox_id,
        attempt_id=uuid4(),
        seam="mcp-rider",
        outcome="emitted",
        receiving_instance="mcp",
        evidence={"nonce_seen": True},
    )
    by_recipient = await deliveries(TENANT, recipient=ALICE)
    assert len(by_recipient) == 1 and by_recipient[0]["evidence"] == {"nonce_seen": True}
    assert by_recipient[0]["session_id"] is None
    assert await deliveries(TENANT, session_id="sess-1") == []
