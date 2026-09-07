"""The companion HTTP surface: gate, bounds, poll, ack, readback.

WHAT THIS FILE HAS TO DEFEND:

* THE GATE IS REAL AND FAILS CLOSED. A tenant without `companion_infra` gets an
  empty result from every route and NOTHING is written -- not a card, not an
  ack. Off is not 403: every response carries the three-state envelope.
* THE PREFIX ARRIVES ON THE WIRE. Actuators do zero formatting, so the card a
  poll returns must already carry `[probe companion] `.
* KNOWN CARDS ARE NOT RE-OFFERED. A client that reports an id in any state
  never receives it again -- this is the half of the delivery state machine
  the server owns.
* THE WAIT IS A WAIT, NOT A HELD CONNECTION. A poll with nothing pending
  returns empty after `wait_seconds`; a poll with a card returns at once.
* ACK IS IDEMPOTENT OVER HTTP on `attempt_id`.

Run with the isolated database:

    PRBE_TEST_DATABASE_URL=postgresql://prbe:prbe@localhost:55442/prbe_knowledge \
        .venv/bin/pytest tests/test_companion_endpoints.py -q
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from engine.shared.companion.mailbox import PREFIX
from engine.shared.db import close_pool, raw_conn

TENANT = "cust-companion-http"
ALICE = "user:00000000-0000-0000-0000-00000000a11c"
INTERNAL_KEY = "test-internal-key"
HEADERS = {"X-Internal-Knowledge-Key": INTERNAL_KEY, "X-Prbe-Customer": TENANT}


@pytest_asyncio.fixture(autouse=True)
async def _internal_key(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    from pydantic import SecretStr

    from engine.shared.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "internal_knowledge_api_key", SecretStr(INTERNAL_KEY))
    yield


@pytest_asyncio.fixture
async def tenant(live_db: None) -> AsyncIterator[str]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'companion-http', 'h-companion-http') ON CONFLICT (customer_id) DO NOTHING",
            TENANT,
        )
    yield TENANT


@pytest_asyncio.fixture
async def enabled(tenant: str) -> AsyncIterator[str]:
    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE customers SET preferences = coalesce(preferences, '{}'::jsonb) "
            "|| jsonb_build_object('companion_infra', true) WHERE customer_id = $1",
            TENANT,
        )
    yield TENANT


async def _request(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    from engine.retrieval.main import app

    await close_pool()
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        return await client.request(
            method, path, json=body, headers=HEADERS if headers is None else headers
        )


def _enqueue_body(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "recipient": ALICE,
        "session_id": "sess-http",
        "class": "seam",
        "body": "run make check before claiming done",
        "dedupe_key": "k1",
        "ttl_seconds": 600,
    }
    base.update(kw)
    return base


@pytest.mark.asyncio
async def test_unauthenticated_is_401(tenant: str) -> None:
    resp = await _request("POST", "/companion/enqueue", _enqueue_body(), headers={})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_gate_off_writes_nothing_and_is_not_403(tenant: str) -> None:
    resp = await _request("POST", "/companion/enqueue", _enqueue_body())
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["capability"] == {"enabled": False, "entitled": True, "upgrade_url": None}
    assert payload["card"] is None and payload["created"] is False
    async with raw_conn() as conn:
        assert await conn.fetchval("SELECT count(*) FROM companion_mailbox") == 0
    poll = await _request(
        "POST", "/companion/poll", {"recipient": ALICE, "session_id": "sess-http"}
    )
    assert poll.status_code == 200 and poll.json()["cards"] == []
    assert poll.json()["config"]["enabled"] is False


@pytest.mark.asyncio
async def test_enqueue_then_poll_returns_prefixed_card(enabled: str) -> None:
    resp = await _request("POST", "/companion/enqueue", _enqueue_body())
    assert resp.status_code == 200, resp.text
    card = resp.json()["card"]
    assert resp.json()["created"] is True
    assert card["body"] == PREFIX + "run make check before claiming done"
    assert card["class"] == "seam" and card["session_id"] == "sess-http"

    started = time.monotonic()
    poll = await _request(
        "POST",
        "/companion/poll",
        {"recipient": ALICE, "session_id": "sess-http", "wait_seconds": 10},
    )
    assert poll.status_code == 200, poll.text
    assert time.monotonic() - started < 3.0, "a poll with a pending card must not wait"
    cards = poll.json()["cards"]
    assert [c["mailbox_id"] for c in cards] == [card["mailbox_id"]]
    config = poll.json()["config"]
    assert (
        config["enabled"] is True and config["max_cards_per_emission"] >= 1 and "version" in config
    )


@pytest.mark.asyncio
async def test_poll_excludes_known_cards(enabled: str) -> None:
    first = (await _request("POST", "/companion/enqueue", _enqueue_body(dedupe_key="a"))).json()[
        "card"
    ]
    second = (await _request("POST", "/companion/enqueue", _enqueue_body(dedupe_key="b"))).json()[
        "card"
    ]
    poll = await _request(
        "POST",
        "/companion/poll",
        {
            "recipient": ALICE,
            "session_id": "sess-http",
            "known": [{"mailbox_id": first["mailbox_id"], "state": "claimed"}],
        },
    )
    assert [c["mailbox_id"] for c in poll.json()["cards"]] == [second["mailbox_id"]]


@pytest.mark.asyncio
async def test_empty_poll_waits_then_returns_empty(enabled: str) -> None:
    started = time.monotonic()
    poll = await _request(
        "POST",
        "/companion/poll",
        {"recipient": ALICE, "session_id": "sess-http", "wait_seconds": 1},
    )
    assert poll.status_code == 200 and poll.json()["cards"] == []
    assert time.monotonic() - started >= 0.9


@pytest.mark.asyncio
async def test_poll_wait_is_bounded(enabled: str) -> None:
    resp = await _request(
        "POST",
        "/companion/poll",
        {"recipient": ALICE, "session_id": "sess-http", "wait_seconds": 26},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_enqueue_conflict_is_409_and_identical_retry_is_200(enabled: str) -> None:
    first = await _request("POST", "/companion/enqueue", _enqueue_body())
    retry = await _request("POST", "/companion/enqueue", _enqueue_body())
    assert retry.status_code == 200 and retry.json()["created"] is False
    assert retry.json()["card"]["mailbox_id"] == first.json()["card"]["mailbox_id"]
    conflict = await _request("POST", "/companion/enqueue", _enqueue_body(body="something else"))
    assert conflict.status_code == 409, conflict.text
    assert conflict.json()["detail"]["reason"] == "conflict"


@pytest.mark.asyncio
async def test_structural_refusals_are_422(enabled: str) -> None:
    for bad in (
        _enqueue_body(**{"class": "decision-push", "session_id": None}),
        _enqueue_body(ttl_seconds=0),
        _enqueue_body(body="x" * 4001),
        _enqueue_body(recipient="ingest:nope"),
        _enqueue_body(intended_seam="telepathy"),
    ):
        resp = await _request("POST", "/companion/enqueue", bad)
        assert resp.status_code == 422, (bad, resp.text)


@pytest.mark.asyncio
async def test_ack_is_idempotent_over_http_and_readback_sees_it(enabled: str) -> None:
    card = (await _request("POST", "/companion/enqueue", _enqueue_body())).json()["card"]
    attempt = str(uuid4())
    ack_body = {
        "mailbox_id": card["mailbox_id"],
        "attempt_id": attempt,
        "seam": "stop",
        "outcome": "emitted",
        "receiving_instance": "dev:claude-code",
        "harness_version": "2.1.236",
        "session_state": "active",
        "receipt_to_emission_ms": 12,
        "evidence": {"nonce_seen": True},
    }
    one = await _request("POST", "/companion/ack", ack_body)
    two = await _request("POST", "/companion/ack", ack_body)
    assert one.status_code == 200, one.text
    assert one.json()["created"] is True and two.json()["created"] is False
    assert one.json()["delivery_id"] == two.json()["delivery_id"]

    poll = await _request(
        "POST", "/companion/poll", {"recipient": ALICE, "session_id": "sess-http"}
    )
    assert poll.json()["cards"] == [], "an emitted card is no longer pending"

    rows = await _request("GET", "/companion/deliveries?session_id=sess-http")
    assert rows.status_code == 200, rows.text
    got = rows.json()["deliveries"]
    assert (
        len(got) == 1
        and got[0]["attempt_id"] == attempt
        and got[0]["evidence"] == {"nonce_seen": True}
    )


@pytest.mark.asyncio
async def test_ack_unknown_card_is_404_and_bad_enum_is_422(enabled: str) -> None:
    base = {
        "mailbox_id": str(uuid4()),
        "attempt_id": str(uuid4()),
        "seam": "stop",
        "outcome": "emitted",
        "receiving_instance": "x",
    }
    assert (await _request("POST", "/companion/ack", base)).status_code == 404
    assert (
        await _request("POST", "/companion/ack", {**base, "seam": "telepathy"})
    ).status_code == 422


@pytest.mark.asyncio
async def test_claim_lane_over_http(enabled: str) -> None:
    card = (
        await _request(
            "POST", "/companion/enqueue", _enqueue_body(session_id=None, dedupe_key="actor-1")
        )
    ).json()["card"]
    won = await _request("POST", "/companion/claim", {"recipient": ALICE, "claimed_by": "user:mcp"})
    assert won.status_code == 200 and won.json()["card"]["mailbox_id"] == card["mailbox_id"]
    again = await _request(
        "POST", "/companion/claim", {"recipient": ALICE, "claimed_by": "user:mcp-2"}
    )
    assert again.json()["card"] is None
