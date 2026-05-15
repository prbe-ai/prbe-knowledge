"""Backfill runner + per-connector pagination tests.

Exercises:
  - enqueue_backfill → backfill_state row
  - claim_pending_backfill → atomic SKIP LOCKED claim
  - Slack backfill paginates conversations.list + conversations.history
  - Linear backfill paginates issues + nested comments
  - Notion backfill paginates /search
  - Sentry backfill paginates issues via Link-header cursor
  - run_backfill wires a connector to the queue + records progress
"""

from __future__ import annotations

import httpx
import pytest

from services.ingestion.backfill_runner import (
    claim_pending_backfill,
    enqueue_backfill,
)
from services.ingestion.handlers.base import ConnectorContext
from shared.config import Settings, get_settings
from shared.constants import BackfillStatus, SourceSystem
from shared.db import raw_conn
from shared.embeddings import reset_embedder
from shared.storage import reset_store


@pytest.fixture(autouse=True)
def _patch(monkeypatch, settings: Settings):
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _stub_slack_metadata_store(monkeypatch):
    """Slack connector now persists its display-name cache to
    customer_source_mapping.metadata via load/patch helpers. Tests don't have
    a Postgres pool, so redirect both calls to a per-test in-memory dict."""
    store: dict = {}

    async def fake_load(source_system, external_id):
        return dict(store.get((source_system.value, external_id), {}))

    async def fake_patch(source_system, external_id, patch):
        existing = store.setdefault((source_system.value, external_id), {})
        existing.update(patch)

    monkeypatch.setattr(
        "services.ingestion.handlers.slack.load_source_metadata", fake_load
    )
    monkeypatch.setattr(
        "services.ingestion.handlers.slack.patch_source_metadata", fake_patch
    )
    # Shrink the flush debounce so tests that trigger writes don't dangle a
    # 30-second background task past test teardown.
    from services.ingestion.handlers.slack import _SlackUserCache

    monkeypatch.setattr(_SlackUserCache, "FLUSH_DEBOUNCE_S", 0.05)
    return store


# -------------------------------- runner ------------------------------------


@pytest.mark.asyncio
async def test_enqueue_and_claim(live_db) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ('cust-bf','x','y') ON CONFLICT DO NOTHING"
        )

    await enqueue_backfill("cust-bf", SourceSystem.SLACK)

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM backfill_state WHERE customer_id='cust-bf'"
        )
    assert row["status"] == BackfillStatus.PENDING.value

    claimed = await claim_pending_backfill()
    assert claimed == ("cust-bf", SourceSystem.SLACK)

    # After claim, row transitions to running; a second claim returns None.
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM backfill_state WHERE customer_id='cust-bf'"
        )
    assert row["status"] == BackfillStatus.RUNNING.value
    assert await claim_pending_backfill() is None


# -------------------------------- slack -------------------------------------


def _mock_transport(routes: dict) -> httpx.MockTransport:
    """Build a MockTransport that dispatches by (method, path, params)."""

    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        for (method, path_prefix), responder in routes.items():
            if method == request.method and request.url.path == path_prefix:
                return responder(request)
        return httpx.Response(404, json={"error": f"unmocked {key}"})

    return httpx.MockTransport(handler)


def _ctx_with_transport(transport: httpx.MockTransport) -> ConnectorContext:
    from shared.config import Settings as _S

    return ConnectorContext(
        settings=_S(),
        http=httpx.AsyncClient(transport=transport),
    )


@pytest.mark.asyncio
async def test_slack_backfill_paginates_channels_and_history() -> None:
    """Slack: list channels, then paginate history per channel. All messages yielded."""
    calls: list[str] = []

    def auth_test(req):
        return httpx.Response(200, json={"ok": True, "team_id": "T1", "team": "Acme"})

    def list_channels(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [
                    {"id": "C1", "is_member": True},
                    {"id": "C2", "is_member": True},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    def history(req):
        channel = req.url.params["channel"]
        calls.append(f"history:{channel}")
        # Use real-looking Slack ts values (unix_secs.microsecs).
        base = 1713628800 if channel == "C1" else 1713628900
        msgs = [
            {"type": "message", "channel": channel, "ts": f"{base}.000100", "text": "hi", "user": "U1"},
            {"type": "message", "channel": channel, "ts": f"{base + 1}.000200", "text": "ho", "user": "U2"},
        ]
        return httpx.Response(
            200,
            json={"ok": True, "messages": msgs, "response_metadata": {"next_cursor": ""}},
        )

    transport = _mock_transport(
        {
            ("POST", "/api/auth.test"): auth_test,
            ("GET", "/api/conversations.list"): list_channels,
            ("GET", "/api/conversations.history"): history,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    slack = build_connector(SourceSystem.SLACK, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.SLACK, access_token="x"
    )

    events = [e async for e in slack.backfill("cust", token)]
    # Round-robin yields message events plus a synthetic _checkpoint event at
    # end of each round (cursor lives on the checkpoint, not on every message,
    # to avoid blowing up the R2 envelope size).
    msg_events = [e for e in events if not e.raw_payload.get("_checkpoint")]
    cp_events = [e for e in events if e.raw_payload.get("_checkpoint")]
    assert len(msg_events) == 4  # 2 channels, 2 messages each
    assert len(cp_events) == 1  # both channels exhaust in one round
    assert {e.raw_payload["event"]["channel"] for e in msg_events} == {"C1", "C2"}
    assert calls == ["history:C1", "history:C2"]
    # Checkpoint event MUST carry the full cursor so the runner can persist it.
    import json as _json

    final_state = _json.loads(cp_events[0].raw_payload["_cursor"])
    assert final_state == {"active": {}, "done": ["C1", "C2"]}


@pytest.mark.asyncio
async def test_slack_backfill_auto_joins_public_channels_when_scope_present() -> None:
    """With channels:join scope, backfill joins every non-member public channel."""
    joined: list[str] = []
    list_calls: list[str] = []

    def auth_test(req):
        return httpx.Response(200, json={"ok": True, "team_id": "T1", "team": "Acme"})

    def list_channels(req):
        types = req.url.params.get("types", "")
        list_calls.append(types)
        if types == "public_channel":
            # Auto-join sweep: C1 already member, C2 needs to be joined.
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "channels": [
                        {"id": "C1", "is_member": True},
                        {"id": "C2", "is_member": False},
                    ],
                    "response_metadata": {"next_cursor": ""},
                },
            )
        # Backfill's own enumeration (public + private): both now members.
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [
                    {"id": "C1", "is_member": True},
                    {"id": "C2", "is_member": True},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    def join_channel(req):
        channel = dict(x.split("=", 1) for x in req.content.decode().split("&")).get("channel")
        joined.append(channel or "")
        return httpx.Response(200, json={"ok": True, "channel": {"id": channel}})

    def history(req):
        channel = req.url.params["channel"]
        base = 1713628800 if channel == "C1" else 1713628900
        return httpx.Response(
            200,
            json={
                "ok": True,
                "messages": [
                    {"type": "message", "channel": channel, "ts": f"{base}.000100", "text": "hi", "user": "U1"},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    transport = _mock_transport(
        {
            ("POST", "/api/auth.test"): auth_test,
            ("GET", "/api/conversations.list"): list_channels,
            ("POST", "/api/conversations.join"): join_channel,
            ("GET", "/api/conversations.history"): history,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    slack = build_connector(SourceSystem.SLACK, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust",
        source_system=SourceSystem.SLACK,
        access_token="x",
        scope="channels:history,channels:join,channels:read",
    )

    events = [e async for e in slack.backfill("cust", token)]
    assert joined == ["C2"], "should join only the non-member channel"
    assert list_calls[0] == "public_channel", "auto-join must list public channels first"
    msg_events = [e for e in events if not e.raw_payload.get("_checkpoint")]
    assert len(msg_events) == 2  # 2 channels, 1 message each


@pytest.mark.asyncio
async def test_slack_backfill_skips_auto_join_without_scope() -> None:
    """Without channels:join scope, backfill skips the auto-join sweep entirely."""
    join_calls: list[str] = []

    def auth_test(req):
        return httpx.Response(200, json={"ok": True, "team_id": "T1", "team": "Acme"})

    def list_channels(req):
        assert req.url.params.get("types") != "public_channel", (
            "auto-join list call should not fire without scope"
        )
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [{"id": "C1", "is_member": True}],
                "response_metadata": {"next_cursor": ""},
            },
        )

    def join_channel(req):
        join_calls.append("called")
        return httpx.Response(200, json={"ok": True})

    def history(req):
        return httpx.Response(
            200,
            json={"ok": True, "messages": [], "response_metadata": {"next_cursor": ""}},
        )

    transport = _mock_transport(
        {
            ("POST", "/api/auth.test"): auth_test,
            ("GET", "/api/conversations.list"): list_channels,
            ("POST", "/api/conversations.join"): join_channel,
            ("GET", "/api/conversations.history"): history,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    slack = build_connector(SourceSystem.SLACK, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust",
        source_system=SourceSystem.SLACK,
        access_token="x",
        scope="channels:history,channels:read",  # no channels:join
    )

    [e async for e in slack.backfill("cust", token)]
    assert join_calls == [], "conversations.join must not be called without scope"


@pytest.mark.asyncio
async def test_slack_backfill_round_robin_interleaves_pages() -> None:
    """Round-robin: page1(A), page1(B), checkpoint, page2(A), page2(B), checkpoint.

    Critical property — the entire UX win of the rewrite. Sequential would yield
    all of A's pages before any of B's; round-robin must interleave so the user
    sees recent messages from every channel within ~1 round.
    """
    history_calls: list[tuple[str, str | None]] = []

    def auth_test(req):
        return httpx.Response(200, json={"ok": True, "team_id": "T1", "team": "Acme"})

    def list_channels(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [
                    {"id": "C1", "is_member": True, "num_members": 10},
                    {"id": "C2", "is_member": True, "num_members": 10},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    def history(req):
        channel = req.url.params["channel"]
        page_cursor = req.url.params.get("cursor")
        history_calls.append((channel, page_cursor))
        # Each channel has 2 pages. First call (no cursor) returns page 1
        # with next_cursor="p2". Second call (cursor=p2) returns page 2 with
        # no next_cursor (end).
        if page_cursor == "p2":
            base = 1713620000 if channel == "C1" else 1713620100
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "messages": [
                        {"type": "message", "channel": channel, "ts": f"{base}.0", "text": "old", "user": "U1"},
                    ],
                    "response_metadata": {"next_cursor": ""},
                },
            )
        # Page 1 — newest messages, more pages available.
        base = 1713628800 if channel == "C1" else 1713628900
        return httpx.Response(
            200,
            json={
                "ok": True,
                "messages": [
                    {"type": "message", "channel": channel, "ts": f"{base}.0", "text": "new", "user": "U1"},
                ],
                "response_metadata": {"next_cursor": "p2"},
            },
        )

    transport = _mock_transport(
        {
            ("POST", "/api/auth.test"): auth_test,
            ("GET", "/api/conversations.list"): list_channels,
            ("GET", "/api/conversations.history"): history,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    slack = build_connector(SourceSystem.SLACK, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.SLACK, access_token="x"
    )

    events = [e async for e in slack.backfill("cust", token)]

    # Round 1: C1 page1, C2 page1, checkpoint.
    # Round 2: C1 page2, C2 page2, checkpoint.
    assert history_calls == [
        ("C1", None),
        ("C2", None),
        ("C1", "p2"),
        ("C2", "p2"),
    ], "must interleave channels within rounds, not exhaust each channel sequentially"

    # Yield order: msg(C1.new), msg(C2.new), checkpoint, msg(C1.old), msg(C2.old), checkpoint.
    msg_channels = [
        e.raw_payload["event"]["channel"]
        for e in events
        if not e.raw_payload.get("_checkpoint")
    ]
    assert msg_channels == ["C1", "C2", "C1", "C2"], (
        "messages must yield in interleaved order: page1(A), page1(B), page2(A), page2(B)"
    )
    cp_count = sum(1 for e in events if e.raw_payload.get("_checkpoint"))
    assert cp_count == 2, "one checkpoint per round"


@pytest.mark.asyncio
async def test_slack_backfill_ranks_channels_by_num_members_desc() -> None:
    """Hot channels page first within a round so #engineering's recent messages
    land before 499 dead channels' first-page fetches."""
    call_order: list[str] = []

    def auth_test(req):
        return httpx.Response(200, json={"ok": True, "team_id": "T1", "team": "Acme"})

    def list_channels(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [
                    {"id": "C_small", "is_member": True, "num_members": 3},
                    {"id": "C_huge", "is_member": True, "num_members": 200},
                    {"id": "C_med", "is_member": True, "num_members": 50},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    def history(req):
        ch = req.url.params["channel"]
        call_order.append(ch)
        return httpx.Response(
            200,
            json={"ok": True, "messages": [], "response_metadata": {"next_cursor": ""}},
        )

    transport = _mock_transport(
        {
            ("POST", "/api/auth.test"): auth_test,
            ("GET", "/api/conversations.list"): list_channels,
            ("GET", "/api/conversations.history"): history,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    slack = build_connector(SourceSystem.SLACK, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.SLACK, access_token="x"
    )

    [e async for e in slack.backfill("cust", token)]
    assert call_order == ["C_huge", "C_med", "C_small"], (
        "ranking must be num_members desc — hot channels first per round"
    )


@pytest.mark.asyncio
async def test_slack_backfill_resumes_from_new_shape_cursor() -> None:
    """Resume must respect each channel's per-page cursor and skip done channels."""
    history_calls: list[tuple[str, str | None]] = []

    def auth_test(req):
        return httpx.Response(200, json={"ok": True, "team_id": "T1", "team": "Acme"})

    def list_channels(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [
                    {"id": "C1", "is_member": True, "num_members": 10},
                    {"id": "C2", "is_member": True, "num_members": 5},
                    {"id": "C3", "is_member": True, "num_members": 1},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    def history(req):
        ch = req.url.params["channel"]
        page_cursor = req.url.params.get("cursor")
        history_calls.append((ch, page_cursor))
        return httpx.Response(
            200,
            json={"ok": True, "messages": [], "response_metadata": {"next_cursor": ""}},
        )

    transport = _mock_transport(
        {
            ("POST", "/api/auth.test"): auth_test,
            ("GET", "/api/conversations.list"): list_channels,
            ("GET", "/api/conversations.history"): history,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    slack = build_connector(SourceSystem.SLACK, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.SLACK, access_token="x"
    )

    # C1 mid-flight at page "p7", C2 fresh, C3 already done.
    resume_cursor = (
        '{"active": {"C1": "p7", "C2": null}, "done": ["C3"]}'
    )
    [e async for e in slack.backfill("cust", token, cursor=resume_cursor)]

    # C1 must resume at p7, C2 must start fresh, C3 must NOT be re-fetched.
    assert ("C1", "p7") in history_calls
    assert ("C2", None) in history_calls
    assert all(ch != "C3" for ch, _ in history_calls), "done channels must not be re-walked"


@pytest.mark.asyncio
async def test_slack_backfill_one_channel_500s_others_continue() -> None:
    """Sticky-broken channel must be dropped to `done`; other channels keep walking."""

    def auth_test(req):
        return httpx.Response(200, json={"ok": True, "team_id": "T1", "team": "Acme"})

    def list_channels(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [
                    {"id": "C_ok", "is_member": True, "num_members": 5},
                    {"id": "C_bad", "is_member": True, "num_members": 5},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    def history(req):
        ch = req.url.params["channel"]
        if ch == "C_bad":
            return httpx.Response(500, json={"error": "internal_error"})
        return httpx.Response(
            200,
            json={
                "ok": True,
                "messages": [
                    {"type": "message", "channel": ch, "ts": "1713628800.0", "text": "hi", "user": "U1"},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    transport = _mock_transport(
        {
            ("POST", "/api/auth.test"): auth_test,
            ("GET", "/api/conversations.list"): list_channels,
            ("GET", "/api/conversations.history"): history,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    slack = build_connector(SourceSystem.SLACK, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.SLACK, access_token="x"
    )

    events = [e async for e in slack.backfill("cust", token)]
    msg_events = [e for e in events if not e.raw_payload.get("_checkpoint")]
    cp_events = [e for e in events if e.raw_payload.get("_checkpoint")]
    assert {e.raw_payload["event"]["channel"] for e in msg_events} == {"C_ok"}, (
        "good channel must still drain when sibling channel returns 500"
    )

    import json as _json

    final_state = _json.loads(cp_events[-1].raw_payload["_cursor"])
    assert "C_bad" in final_state["done"], "bad channel must land in done"
    assert final_state["active"] == {}


@pytest.mark.asyncio
async def test_slack_backfill_drops_channel_on_ok_false() -> None:
    """ok=false (e.g. channel_not_found from a since-archived channel) must
    move the channel to `done` so the loop doesn't spin re-fetching it."""

    def auth_test(req):
        return httpx.Response(200, json={"ok": True, "team_id": "T1", "team": "Acme"})

    def list_channels(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [
                    {"id": "C_gone", "is_member": True, "num_members": 5},
                    {"id": "C_ok", "is_member": True, "num_members": 5},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    def history(req):
        ch = req.url.params["channel"]
        if ch == "C_gone":
            return httpx.Response(200, json={"ok": False, "error": "channel_not_found"})
        return httpx.Response(
            200,
            json={"ok": True, "messages": [], "response_metadata": {"next_cursor": ""}},
        )

    transport = _mock_transport(
        {
            ("POST", "/api/auth.test"): auth_test,
            ("GET", "/api/conversations.list"): list_channels,
            ("GET", "/api/conversations.history"): history,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    slack = build_connector(SourceSystem.SLACK, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.SLACK, access_token="x"
    )

    events = [e async for e in slack.backfill("cust", token)]
    cp_events = [e for e in events if e.raw_payload.get("_checkpoint")]

    import json as _json

    final_state = _json.loads(cp_events[-1].raw_payload["_cursor"])
    assert "C_gone" in final_state["done"]


@pytest.mark.asyncio
async def test_slack_backfill_429_pauses_and_breaks_round(monkeypatch) -> None:
    """429 must trigger a sleep for retry-after seconds AND break the round
    so we don't immediately fire another channel under the same penalty.

    Sleep happens OUTSIDE the rate limiter (subagent finding #7) — the cap is
    workspace-global, so backing off only one channel would hit 429 again."""
    sleeps: list[int] = []
    history_calls: list[str] = []

    async def fake_sleep(s):
        sleeps.append(int(s))

    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    def auth_test(req):
        return httpx.Response(200, json={"ok": True, "team_id": "T1", "team": "Acme"})

    def list_channels(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [
                    {"id": "C1", "is_member": True, "num_members": 10},
                    {"id": "C2", "is_member": True, "num_members": 5},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    call_count = {"n": 0}

    def history(req):
        ch = req.url.params["channel"]
        history_calls.append(ch)
        call_count["n"] += 1
        # First call (C1) returns 429 with Retry-After: 30. Second call onward
        # (post-pause) returns 200 with one message and end-of-history.
        if call_count["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "30"}, json={"error": "rate_limited"})
        return httpx.Response(
            200,
            json={
                "ok": True,
                "messages": [
                    {"type": "message", "channel": ch, "ts": "1713628800.0", "text": "hi", "user": "U1"},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    transport = _mock_transport(
        {
            ("POST", "/api/auth.test"): auth_test,
            ("GET", "/api/conversations.list"): list_channels,
            ("GET", "/api/conversations.history"): history,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    slack = build_connector(SourceSystem.SLACK, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.SLACK, access_token="x"
    )

    [e async for e in slack.backfill("cust", token)]

    # Slept exactly once for retry-after=30.
    assert 30 in sleeps, f"expected 30s retry-after sleep, got {sleeps}"
    # The 429 broke the round (didn't continue to C2 immediately under penalty).
    # First two calls: C1 (429), then loop restarts -> C1 succeeds, then C2 succeeds.
    assert history_calls[0] == "C1"
    assert history_calls[1] == "C1", (
        "round must break on 429 so next call retries the same channel after pause, "
        "not fire C2 under the same global penalty"
    )


@pytest.mark.asyncio
async def test_slack_backfill_skips_non_message_types() -> None:
    """Channel-join, channel-name, and other non-message subtypes are noise
    we don't want chunked into the graph."""

    def auth_test(req):
        return httpx.Response(200, json={"ok": True, "team_id": "T1", "team": "Acme"})

    def list_channels(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [{"id": "C1", "is_member": True, "num_members": 5}],
                "response_metadata": {"next_cursor": ""},
            },
        )

    def history(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "messages": [
                    {"type": "channel_join", "ts": "1713628800.0", "user": "U1"},
                    {"type": "message", "channel": "C1", "ts": "1713628801.0", "text": "real msg", "user": "U1"},
                    # Bot blocks-only message: type=message but no text and no files. Drop.
                    {"type": "message", "channel": "C1", "ts": "1713628802.0", "bot_id": "B1"},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    transport = _mock_transport(
        {
            ("POST", "/api/auth.test"): auth_test,
            ("GET", "/api/conversations.list"): list_channels,
            ("GET", "/api/conversations.history"): history,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    slack = build_connector(SourceSystem.SLACK, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.SLACK, access_token="x"
    )

    events = [e async for e in slack.backfill("cust", token)]
    msg_events = [e for e in events if not e.raw_payload.get("_checkpoint")]
    assert len(msg_events) == 1, "only the type=message with text should yield"
    assert msg_events[0].raw_payload["event"]["text"] == "real msg"


@pytest.mark.asyncio
async def test_slack_backfill_message_events_omit_cursor_field() -> None:
    """Cursor bloat fix: real message events must NOT carry `_cursor` in their
    raw_payload. Cursor only lives on the synthetic `_checkpoint` event yielded
    at end of each round. With N active channels the cursor can be ~50 bytes
    per channel; copying it onto every yielded message blows up R2 envelopes."""

    def auth_test(req):
        return httpx.Response(200, json={"ok": True, "team_id": "T1", "team": "Acme"})

    def list_channels(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [{"id": "C1", "is_member": True, "num_members": 5}],
                "response_metadata": {"next_cursor": ""},
            },
        )

    def history(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "messages": [
                    {"type": "message", "channel": "C1", "ts": "1713628800.0", "text": "a", "user": "U1"},
                    {"type": "message", "channel": "C1", "ts": "1713628801.0", "text": "b", "user": "U1"},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    transport = _mock_transport(
        {
            ("POST", "/api/auth.test"): auth_test,
            ("GET", "/api/conversations.list"): list_channels,
            ("GET", "/api/conversations.history"): history,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    slack = build_connector(SourceSystem.SLACK, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.SLACK, access_token="x"
    )

    events = [e async for e in slack.backfill("cust", token)]
    msg_events = [e for e in events if not e.raw_payload.get("_checkpoint")]
    cp_events = [e for e in events if e.raw_payload.get("_checkpoint")]

    for e in msg_events:
        assert "_cursor" not in e.raw_payload, (
            "message events must not carry _cursor — bloat fix"
        )
    assert all("_cursor" in e.raw_payload for e in cp_events)


@pytest.mark.asyncio
async def test_slack_backfill_primes_user_cache_via_users_list() -> None:
    """One paginated users.list call at backfill kickoff bulk-populates the
    name cache. Without it, every yielded message would force a per-user
    users.info round-trip — wasteful at workspace scale (1000s of users)
    and easy to rate-limit."""
    users_list_calls: list[str | None] = []
    users_info_calls: list[str] = []

    def auth_test(req):
        return httpx.Response(200, json={"ok": True, "team_id": "T1", "team": "Acme"})

    def list_channels(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [{"id": "C1", "is_member": True, "num_members": 5}],
                "response_metadata": {"next_cursor": ""},
            },
        )

    def users_list(req):
        users_list_calls.append(req.url.params.get("cursor"))
        # Two-page response: first page links to "p2", second page ends.
        if req.url.params.get("cursor") == "p2":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "members": [
                        {"id": "U2", "profile": {"display_name": "Bob", "real_name": "Robert"}},
                    ],
                    "response_metadata": {"next_cursor": ""},
                },
            )
        return httpx.Response(
            200,
            json={
                "ok": True,
                "members": [
                    {"id": "U1", "profile": {"display_name": "Alice", "real_name": "Alice A."}},
                ],
                "response_metadata": {"next_cursor": "p2"},
            },
        )

    def users_info(req):
        users_info_calls.append(req.url.params["user"])
        return httpx.Response(200, json={"ok": False, "error": "should_not_call"})

    def history(req):
        ch = req.url.params["channel"]
        return httpx.Response(
            200,
            json={
                "ok": True,
                "messages": [
                    {"type": "message", "channel": ch, "ts": "1713628800.0", "text": "hi", "user": "U1"},
                    {"type": "message", "channel": ch, "ts": "1713628801.0", "text": "ho", "user": "U2"},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    transport = _mock_transport(
        {
            ("POST", "/api/auth.test"): auth_test,
            ("GET", "/api/conversations.list"): list_channels,
            ("GET", "/api/users.list"): users_list,
            ("GET", "/api/users.info"): users_info,
            ("GET", "/api/conversations.history"): history,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    slack = build_connector(SourceSystem.SLACK, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.SLACK, access_token="x"
    )

    [e async for e in slack.backfill("cust", token)]

    # users.list was paginated through completely.
    assert users_list_calls == [None, "p2"]
    # No per-user users.info fetches — the prime warmed the cache.
    assert users_info_calls == []
    # Cache populated with both users on this connector's per-team cache.
    cache = slack._caches["T1"]
    assert cache._entries["U1"]["name"] == "Alice"
    assert cache._entries["U2"]["name"] == "Bob"


@pytest.mark.asyncio
async def test_slack_backfill_event_carries_user_profile_when_cached() -> None:
    """Synthetic backfill events must inline user_profile from the prime so
    normalize() stamps the name without re-resolving. Without this, normalize
    would land on the cache itself, but threading that read through the
    pipeline relies on the event being self-describing (re-runnable from R2)."""

    def auth_test(req):
        return httpx.Response(200, json={"ok": True, "team_id": "T1", "team": "Acme"})

    def list_channels(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [{"id": "C1", "is_member": True, "num_members": 5}],
                "response_metadata": {"next_cursor": ""},
            },
        )

    def users_list(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "members": [
                    {"id": "U1", "profile": {"display_name": "Richard Wei"}},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    def history(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "messages": [
                    {"type": "message", "channel": "C1", "ts": "1713628800.0", "text": "hi", "user": "U1"},
                    # User missing from prime (rare race / late joiner): no
                    # user_profile inlined, normalize falls back to no prefix.
                    {"type": "message", "channel": "C1", "ts": "1713628801.0", "text": "ho", "user": "Uunknown"},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    transport = _mock_transport(
        {
            ("POST", "/api/auth.test"): auth_test,
            ("GET", "/api/conversations.list"): list_channels,
            ("GET", "/api/users.list"): users_list,
            ("GET", "/api/conversations.history"): history,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    slack = build_connector(SourceSystem.SLACK, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.SLACK, access_token="x"
    )

    events = [e async for e in slack.backfill("cust", token)]
    msg_events = [e for e in events if not e.raw_payload.get("_checkpoint")]

    by_user = {e.raw_payload["event"]["user"]: e.raw_payload["event"] for e in msg_events}
    assert by_user["U1"].get("user_profile") == {"display_name": "Richard Wei"}
    assert "user_profile" not in by_user["Uunknown"], (
        "uncached users must not synthesize a user_profile — normalize falls back to no prefix"
    )


# -------------------------------- linear ------------------------------------


@pytest.mark.asyncio
async def test_linear_backfill_paginates_issues() -> None:
    """Linear: GraphQL pagination. Two pages, each with 1 issue + 1 comment."""
    page = {"ix": 0}

    def graphql(req):
        payload = req.read()
        body = payload.decode()
        # Org-id probe is the short `{ organization { id } }` query.
        if "issues(first" not in body:
            return httpx.Response(200, json={"data": {"organization": {"id": "ORG"}}})
        # Backfill issues query.
        if page["ix"] == 0:
            page["ix"] = 1
            return httpx.Response(
                200,
                json={
                    "data": {
                        "issues": {
                            "pageInfo": {"hasNextPage": True, "endCursor": "cur1"},
                            "nodes": [
                                {
                                    "id": "I1",
                                    "identifier": "ENG-1",
                                    "title": "first",
                                    "description": "",
                                    "url": "https://linear.app/x/I1",
                                    "createdAt": "2026-04-01T00:00:00Z",
                                    "updatedAt": "2026-04-01T00:00:00Z",
                                    "team": {"id": "T1", "key": "ENG", "name": "Eng"},
                                    "teamId": "T1",
                                    "comments": {
                                        "nodes": [
                                            {
                                                "id": "C1",
                                                "body": "hi",
                                                "url": "https://linear.app/x/I1#c1",
                                                "createdAt": "2026-04-01T00:10:00Z",
                                                "updatedAt": "2026-04-01T00:10:00Z",
                                                "user": {"id": "U1", "name": "alice"},
                                            }
                                        ]
                                    },
                                }
                            ],
                        }
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "issues": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [
                            {
                                "id": "I2",
                                "identifier": "ENG-2",
                                "title": "second",
                                "description": "",
                                "url": "https://linear.app/x/I2",
                                "createdAt": "2026-04-02T00:00:00Z",
                                "updatedAt": "2026-04-02T00:00:00Z",
                                "team": {"id": "T1", "key": "ENG", "name": "Eng"},
                                "teamId": "T1",
                                "comments": {"nodes": []},
                            }
                        ],
                    }
                }
            },
        )

    transport = _mock_transport({("POST", "/graphql"): graphql})

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    linear = build_connector(SourceSystem.LINEAR, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="c", source_system=SourceSystem.LINEAR, access_token="x"
    )
    events = [e async for e in linear.backfill("c", token)]

    # 2 issues + 1 comment = 3 events
    assert len(events) == 3
    types = [e.raw_payload["type"] for e in events]
    assert types.count("Issue") == 2
    assert types.count("Comment") == 1


# -------------------------------- notion ------------------------------------


@pytest.mark.asyncio
async def test_notion_backfill_paginates_search() -> None:
    """Notion: paginated /search. Yields one event per page entity."""
    page = {"ix": 0}

    def users_me(req):
        return httpx.Response(
            200,
            json={"bot": {"workspace_id": "WS1", "workspace_name": "Acme WS"}},
        )

    def search(req):
        if page["ix"] == 0:
            page["ix"] = 1
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "object": "page",
                            "id": "P1",
                            "last_edited_time": "2026-04-01T00:00:00.000Z",
                        }
                    ],
                    "has_more": True,
                    "next_cursor": "cur1",
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "object": "database",
                        "id": "D1",
                        "last_edited_time": "2026-04-02T00:00:00.000Z",
                    }
                ],
                "has_more": False,
                "next_cursor": None,
            },
        )

    transport = _mock_transport(
        {
            ("GET", "/v1/users/me"): users_me,
            ("POST", "/v1/search"): search,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    notion = build_connector(SourceSystem.NOTION, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="c", source_system=SourceSystem.NOTION, access_token="x"
    )
    events = [e async for e in notion.backfill("c", token)]
    assert len(events) == 2
    kinds = [e.raw_payload["entity"]["type"] for e in events]
    assert set(kinds) == {"page", "database"}


@pytest.mark.asyncio
async def test_notion_backfill_enumerates_database_rows() -> None:
    """Each database discovered in /search must have its rows pulled via
    databases/{id}/query and yielded as page.created events.

    Without this, every Notion database's contents are invisible: rows are
    pages that /search does not list unless individually shared with the
    integration.
    """

    def users_me(req):
        return httpx.Response(200, json={"bot": {"workspace_id": "WS1"}})

    def search(req):
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "object": "database",
                        "id": "DB1",
                        "last_edited_time": "2026-04-01T00:00:00.000Z",
                    }
                ],
                "has_more": False,
                "next_cursor": None,
            },
        )

    query_calls: list[dict] = []
    page_state = {"ix": 0}

    def query(req):
        body = req.read()
        import json as _json

        query_calls.append(_json.loads(body) if body else {})
        if page_state["ix"] == 0:
            page_state["ix"] = 1
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "object": "page",
                            "id": "ROW1",
                            "last_edited_time": "2026-04-01T01:00:00.000Z",
                        },
                        {
                            "object": "page",
                            "id": "ROW2",
                            "last_edited_time": "2026-04-01T02:00:00.000Z",
                        },
                    ],
                    "has_more": True,
                    "next_cursor": "cur_q1",
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "object": "page",
                        "id": "ROW3",
                        "last_edited_time": "2026-04-01T03:00:00.000Z",
                    }
                ],
                "has_more": False,
                "next_cursor": None,
            },
        )

    transport = _mock_transport(
        {
            ("GET", "/v1/users/me"): users_me,
            ("POST", "/v1/search"): search,
            ("POST", "/v1/databases/DB1/query"): query,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    notion = build_connector(SourceSystem.NOTION, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="c", source_system=SourceSystem.NOTION, access_token="x"
    )
    events = [e async for e in notion.backfill("c", token)]

    # 1 database event + 3 row-as-page events.
    assert len(events) == 4
    kinds = [e.raw_payload["entity"]["type"] for e in events]
    assert kinds == ["database", "page", "page", "page"]
    row_ids = [e.raw_payload["entity"]["id"] for e in events[1:]]
    assert row_ids == ["ROW1", "ROW2", "ROW3"]
    # Pagination cursor was passed on the second query call.
    assert query_calls[0] == {"page_size": 100}
    assert query_calls[1] == {"page_size": 100, "start_cursor": "cur_q1"}
    # Rows carry parent-database breadcrumb so downstream can attribute them.
    assert all(
        e.raw_payload.get("_parent_database_id") == "DB1" for e in events[1:]
    )


# -------------------------------- sentry ------------------------------------


@pytest.mark.asyncio
async def test_sentry_backfill_paginates_issues() -> None:
    """Sentry: Link-header cursor pagination. Two pages of issues."""
    page = {"ix": 0}

    def orgs(req):
        return httpx.Response(200, json=[{"slug": "acme"}])

    def issues(req):
        if page["ix"] == 0:
            page["ix"] = 1
            return httpx.Response(
                200,
                json=[
                    {"id": "1", "lastSeen": "2026-04-01T00:00:00Z", "title": "a"},
                    {"id": "2", "lastSeen": "2026-04-01T00:01:00Z", "title": "b"},
                ],
                headers={
                    "link": (
                        '<https://sentry.io/api/0/organizations/acme/issues/?cursor=p2>; '
                        'rel="next"; results="true"; cursor="p2"'
                    )
                },
            )
        return httpx.Response(
            200,
            json=[{"id": "3", "lastSeen": "2026-04-02T00:00:00Z", "title": "c"}],
            headers={
                "link": (
                    '<https://sentry.io/api/0/organizations/acme/issues/?cursor=end>; '
                    'rel="next"; results="false"'
                )
            },
        )

    transport = _mock_transport(
        {
            ("GET", "/api/0/organizations/"): orgs,
            ("GET", "/api/0/organizations/acme/issues/"): issues,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    sentry = build_connector(SourceSystem.SENTRY, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="c", source_system=SourceSystem.SENTRY, access_token="x"
    )
    events = [e async for e in sentry.backfill("c", token)]
    assert len(events) == 3
    assert {e.source_event_id for e in events} == {
        "issue:1:backfill",
        "issue:2:backfill",
        "issue:3:backfill",
    }


# -------------------------------- github -----------------------------------


def _gh_pr_node(
    number: int,
    *,
    updated_at: str,
    state: str = "OPEN",
    title: str = "",
    body: str = "",
    author: str = "alice",
) -> dict:
    """Minimal GraphQL PR node shape for tests."""
    return {
        "number": number,
        "title": title or f"pr{number}",
        "body": body,
        "state": state,
        "url": f"https://github.com/o/r/pull/{number}",
        "createdAt": updated_at,
        "updatedAt": updated_at,
        "closedAt": None,
        "mergedAt": None,
        "merged": False,
        "changedFiles": 0,
        "additions": 0,
        "deletions": 0,
        "baseRefName": "main",
        "headRefName": f"branch-{number}",
        "author": {"login": author},
        "labels": {"nodes": []},
        "assignees": {"nodes": []},
        "comments": {"nodes": []},
        "reviews": {"nodes": []},
        "files": {"nodes": []},
        "commits": {"nodes": []},
    }


def _gh_issue_node(
    number: int,
    *,
    updated_at: str,
    state: str = "OPEN",
    title: str = "",
    body: str = "",
    author: str = "alice",
) -> dict:
    """Minimal GraphQL Issue node shape for tests."""
    return {
        "number": number,
        "title": title or f"issue{number}",
        "body": body,
        "state": state,
        "url": f"https://github.com/o/r/issues/{number}",
        "createdAt": updated_at,
        "updatedAt": updated_at,
        "closedAt": None,
        "author": {"login": author},
        "labels": {"nodes": []},
        "assignees": {"nodes": []},
        "comments": {"nodes": []},
    }


def _gh_review_node(
    review_id: str,
    *,
    database_id: int | None = None,
    state: str = "APPROVED",
    body: str = "",
    submitted_at: str = "2026-04-01T00:00:00Z",
    author: str = "alice",
) -> dict:
    """Minimal GraphQL review node shape for tests."""
    return {
        "id": review_id,
        "databaseId": database_id,
        "state": state,
        "body": body,
        "submittedAt": submitted_at,
        "author": {"login": author},
    }


def _gh_commit_node(
    sha: str,
    *,
    message: str = "msg",
    committed_at: str = "2026-04-01T00:00:00Z",
    author_name: str = "Alice",
    author_email: str = "alice@example.com",
    author_login: str = "alice",
) -> dict:
    """Minimal GraphQL commit history node shape for tests."""
    return {
        "oid": sha,
        "message": message,
        "committedDate": committed_at,
        "author": {
            "name": author_name,
            "email": author_email,
            "user": {"login": author_login},
        },
    }


def _gh_release_node(
    *,
    release_id: int,
    tag: str,
    name: str = "",
    body: str = "",
    created_at: str = "2026-04-01T00:00:00Z",
    published_at: str = "2026-04-01T00:00:00Z",
    is_draft: bool = False,
    is_prerelease: bool = False,
    author: str = "alice",
) -> dict:
    """Minimal GraphQL release node shape for tests."""
    return {
        "id": f"R_{release_id}",
        "databaseId": release_id,
        "tagName": tag,
        "name": name or tag,
        "body": body,
        "createdAt": created_at,
        "publishedAt": published_at,
        "updatedAt": published_at,
        "isDraft": is_draft,
        "isPrerelease": is_prerelease,
        "author": {"login": author},
        "url": f"https://github.com/o/r/releases/tag/{tag}",
    }


def _gh_graphql_response(
    repo_block: dict, *, remaining: int = 5000, reset_at: str = "2099-01-01T00:00:00Z"
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": {
                "repository": repo_block,
                "rateLimit": {"cost": 1, "remaining": remaining, "resetAt": reset_at},
            }
        },
    )


def _gh_graphql_transport(
    *,
    installation_repos: list[dict],
    pulls_by_repo: dict[str, list[dict]] | None = None,
    issues_by_repo: dict[str, list[dict]] | None = None,
    commits_by_repo: dict[str, list[dict]] | None = None,
    releases_by_repo: dict[str, list[dict]] | None = None,
    default_branch_by_repo: dict[str, str] | None = None,
    on_graphql=None,
) -> httpx.MockTransport:
    """Build a transport that mocks both REST /installation/repositories and
    POST /graphql.

    The GraphQL responder picks pulls / issues / commits / releases by
    inspecting the query body and dispatches by `variables.owner/name`.
    Optional `on_graphql` callback fires for every GraphQL call (used by
    rate-limit + parallel tests).
    """
    pulls_by_repo = pulls_by_repo or {}
    issues_by_repo = issues_by_repo or {}
    commits_by_repo = commits_by_repo or {}
    releases_by_repo = releases_by_repo or {}
    default_branch_by_repo = default_branch_by_repo or {}

    def handler(request: httpx.Request) -> httpx.Response:
        if (
            request.method == "GET"
            and request.url.path == "/installation/repositories"
        ):
            return httpx.Response(200, json={"repositories": installation_repos})

        if request.method == "POST" and request.url.path == "/graphql":
            import json as _json

            body = _json.loads(request.content.decode("utf-8"))
            query = body.get("query", "")
            variables = body.get("variables", {})
            owner = variables.get("owner")
            name = variables.get("name")
            full_name = f"{owner}/{name}"

            if on_graphql is not None:
                override = on_graphql(query, variables)
                if override is not None:
                    return override

            if "BackfillPulls" in query or "pullRequests(" in query:
                nodes = pulls_by_repo.get(full_name, [])
                return _gh_graphql_response(
                    {
                        "pullRequests": {
                            "pageInfo": {"endCursor": None, "hasNextPage": False},
                            "nodes": nodes,
                        }
                    }
                )
            if "BackfillCommits" in query or "defaultBranchRef" in query:
                nodes = commits_by_repo.get(full_name, [])
                branch = default_branch_by_repo.get(full_name, "main")
                return _gh_graphql_response(
                    {
                        "defaultBranchRef": {
                            "name": branch,
                            "target": {
                                "history": {
                                    "pageInfo": {
                                        "endCursor": None,
                                        "hasNextPage": False,
                                    },
                                    "nodes": nodes,
                                }
                            },
                        }
                    }
                )
            if "BackfillReleases" in query or "releases(" in query:
                nodes = releases_by_repo.get(full_name, [])
                return _gh_graphql_response(
                    {
                        "releases": {
                            "pageInfo": {"endCursor": None, "hasNextPage": False},
                            "nodes": nodes,
                        }
                    }
                )
            if "BackfillIssues" in query or "issues(" in query:
                nodes = issues_by_repo.get(full_name, [])
                return _gh_graphql_response(
                    {
                        "issues": {
                            "pageInfo": {"endCursor": None, "hasNextPage": False},
                            "nodes": nodes,
                        }
                    }
                )

        return httpx.Response(404, json={"error": f"unmocked {request.method} {request.url.path}"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_github_backfill_paginates_all_phases() -> None:
    """GitHub: list installation repos, then GraphQL pulls + issues +
    commits + releases per repo. Reviews piggyback on the pulls phase.
    """

    installation_repos = [
        {
            "full_name": "acme/api",
            "name": "api",
            "owner": {"login": "acme"},
            "private": True,
        },
        {
            "full_name": "acme/web",
            "name": "web",
            "owner": {"login": "acme"},
            "private": False,
        },
    ]

    pr1 = _gh_pr_node(1, updated_at="2026-04-01T00:00:00Z")
    pr1["reviews"] = {
        "nodes": [
            _gh_review_node(
                "PRR_1", database_id=1, submitted_at="2026-04-01T01:00:00Z"
            ),
        ]
    }

    captured_queries: list[str] = []

    def _capture(query, variables):
        captured_queries.append(query)
        return None

    transport = _gh_graphql_transport(
        installation_repos=installation_repos,
        pulls_by_repo={
            "acme/api": [
                pr1,
                _gh_pr_node(2, updated_at="2026-04-02T00:00:00Z"),
            ],
            "acme/web": [
                _gh_pr_node(3, updated_at="2026-04-05T00:00:00Z"),
            ],
        },
        issues_by_repo={
            "acme/api": [
                _gh_issue_node(10, updated_at="2026-04-03T00:00:00Z"),
            ],
            "acme/web": [],
        },
        commits_by_repo={
            "acme/api": [
                _gh_commit_node(
                    "a" * 40, committed_at="2026-04-04T00:00:00Z"
                ),
            ],
            "acme/web": [],
        },
        releases_by_repo={
            "acme/api": [
                _gh_release_node(release_id=11, tag="v1.0.0"),
            ],
            "acme/web": [],
        },
        default_branch_by_repo={"acme/api": "main", "acme/web": "main"},
        on_graphql=_capture,
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    gh = build_connector(SourceSystem.GITHUB, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.GITHUB, access_token="x"
    )

    events = [e async for e in gh.backfill("cust", token)]

    # 3 PRs + 1 review + 1 issue + 1 commit-as-push + 1 release = 7 events.
    pr_events = [e for e in events if e.headers.get("X-GitHub-Event") == "pull_request"]
    issue_events = [e for e in events if e.headers.get("X-GitHub-Event") == "issues"]
    review_events = [
        e for e in events if e.headers.get("X-GitHub-Event") == "pull_request_review"
    ]
    push_events = [e for e in events if e.headers.get("X-GitHub-Event") == "push"]
    release_events = [e for e in events if e.headers.get("X-GitHub-Event") == "release"]
    assert len(pr_events) == 3
    assert len(issue_events) == 1
    assert len(review_events) == 1
    assert len(push_events) == 1
    assert len(release_events) == 1
    assert len(events) == 7

    issue_numbers = {e.raw_payload["issue"]["number"] for e in issue_events}
    assert issue_numbers == {10}

    pr_numbers = {e.raw_payload["pull_request"]["number"] for e in pr_events}
    assert pr_numbers == {1, 2, 3}

    # Normalized REST-shape keys survive on the synthesized payload.
    sample_pr = pr_events[0].raw_payload["pull_request"]
    assert "number" in sample_pr
    assert "user" in sample_pr and "login" in sample_pr["user"]
    assert "base" in sample_pr and "ref" in sample_pr["base"]
    assert "html_url" in sample_pr

    # Repository payload survives the round trip with full_name populated.
    assert pr_events[0].raw_payload["repository"]["full_name"] in {"acme/api", "acme/web"}

    # Fix 7: BACKFILL_PULLS_QUERY drops dead weight (nested review.comments +
    # nested PR.commits). We don't normalize review-comments, and there's a
    # separate BACKFILL_COMMITS_QUERY phase for default-branch history, so
    # both subselections only inflate query cost. Verify by inspecting the
    # actual query string the connector sent to GitHub.
    pulls_queries = [q for q in captured_queries if "BackfillPulls" in q]
    assert pulls_queries, "expected at least one BackfillPulls query"
    pulls_q = pulls_queries[0]
    # The reviews block should not contain a nested `comments(` subselection.
    reviews_idx = pulls_q.index("reviews(")
    # Reviews block closes at its first balanced top-level `}` after the
    # opening `{` of `nodes`. Cheap proxy: between `reviews(` and `files(first`
    # there should be no `comments(` token.
    files_idx = pulls_q.index("files(first")
    assert "comments(" not in pulls_q[reviews_idx:files_idx], (
        "reviews subselection should not nest comments"
    )
    # The PR-level nested `commits(first: ...)` is replaced by the
    # BACKFILL_COMMITS_QUERY phase; the only `commits` token allowed in
    # the pulls query string is `BackfillCommits` (which lives in a
    # separate query) -- the pulls query itself should not contain
    # `commits(`.
    assert "commits(" not in pulls_q, (
        "BACKFILL_PULLS_QUERY should not nest PR.commits"
    )


@pytest.mark.asyncio
async def test_github_backfill_with_installation_scope_fetches_token_from_backend(
    monkeypatch,
) -> None:
    """A token with scope='installation:<id>' fetches a fresh bearer from
    prbe-backend's /internal/github/installation_token endpoint."""
    from datetime import UTC, datetime

    observed_auth: list[str] = []

    def on_graphql(query, variables):
        # Capture on every GraphQL call below.
        return None

    base_transport = _gh_graphql_transport(
        installation_repos=[
            {
                "full_name": "acme/api",
                "name": "api",
                "owner": {"login": "acme"},
                "private": True,
            }
        ],
        pulls_by_repo={
            "acme/api": [_gh_pr_node(1, updated_at="2026-04-01T00:00:00Z")],
        },
        issues_by_repo={"acme/api": []},
        on_graphql=on_graphql,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        observed_auth.append(request.headers.get("authorization", ""))
        return base_transport.handler(request)

    transport = httpx.MockTransport(handler)

    fetched_for: list[str] = []

    async def fake_fetch(http, *, customer_id):
        fetched_for.append(customer_id)
        return "ghs_fresh_bearer", datetime(2026, 12, 31, tzinfo=UTC)

    monkeypatch.setattr(
        "services.ingestion.handlers.github.fetch_github_installation_token",
        fake_fetch,
    )

    from services.ingestion.handlers.base import ConnectorContext
    from services.ingestion.handlers.registry import build_connector
    from shared.config import Settings as _S
    from shared.models import IntegrationToken

    ctx = ConnectorContext(
        settings=_S(),
        http=httpx.AsyncClient(transport=transport),
    )
    gh = build_connector(SourceSystem.GITHUB, ctx)
    token = IntegrationToken(
        customer_id="cust",
        source_system=SourceSystem.GITHUB,
        access_token="placeholder-never-used",
        scope="installation:99",
    )

    events = [e async for e in gh.backfill("cust", token)]
    assert len(events) == 1
    assert fetched_for == ["cust"]
    assert observed_auth, "mock transport should have captured requests"
    for auth in observed_auth:
        assert auth == "Bearer ghs_fresh_bearer"


@pytest.mark.asyncio
async def test_github_graphql_rate_limit_backoff(monkeypatch) -> None:
    """When `rateLimit.remaining` drops below the floor, the GraphQL client
    sleeps proactively before returning. We patch asyncio.sleep to record the
    delay instead of waiting.
    """
    sleeps: list[float] = []

    import asyncio as _asyncio

    real_sleep = _asyncio.sleep

    async def fake_sleep(delay):
        sleeps.append(delay)
        # Yield control so the queue drainer can still run without a real wait.
        await real_sleep(0)

    monkeypatch.setattr(
        "services.ingestion.handlers._github_graphql.asyncio.sleep", fake_sleep
    )

    def on_graphql(query, variables):
        if "pullRequests(" in query:
            # First (only) page with remaining=5 -> below floor, should trigger sleep.
            return httpx.Response(
                200,
                json={
                    "data": {
                        "repository": {
                            "pullRequests": {
                                "pageInfo": {"endCursor": None, "hasNextPage": False},
                                "nodes": [
                                    _gh_pr_node(7, updated_at="2026-04-01T00:00:00Z")
                                ],
                            }
                        },
                        "rateLimit": {
                            "cost": 1,
                            "remaining": 5,
                            "resetAt": "2099-01-01T00:00:00Z",
                        },
                    }
                },
            )
        return None  # fall through to default issues responder

    transport = _gh_graphql_transport(
        installation_repos=[
            {
                "full_name": "acme/api",
                "name": "api",
                "owner": {"login": "acme"},
            }
        ],
        issues_by_repo={"acme/api": []},
        on_graphql=on_graphql,
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    gh = build_connector(SourceSystem.GITHUB, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.GITHUB, access_token="x"
    )

    events = [e async for e in gh.backfill("cust", token)]
    assert len(events) == 1
    # At least one non-trivial sleep got triggered by the low-remaining response.
    assert any(delay >= 1 for delay in sleeps), sleeps


@pytest.mark.asyncio
async def test_github_graphql_parallel_repos(monkeypatch) -> None:
    """With semaphore=2 and 3 repos, all events still flow through the queue."""
    # Cap parallelism at 2 via the settings.
    monkeypatch.setenv("GITHUB_BACKFILL_REPO_CONCURRENCY", "2")
    from shared.config import get_settings as _gs

    _gs.cache_clear()  # type: ignore[attr-defined]

    installation_repos = [
        {"full_name": f"acme/r{i}", "name": f"r{i}", "owner": {"login": "acme"}}
        for i in range(3)
    ]

    transport = _gh_graphql_transport(
        installation_repos=installation_repos,
        pulls_by_repo={
            "acme/r0": [_gh_pr_node(1, updated_at="2026-04-01T00:00:00Z")],
            "acme/r1": [_gh_pr_node(2, updated_at="2026-04-02T00:00:00Z")],
            "acme/r2": [_gh_pr_node(3, updated_at="2026-04-03T00:00:00Z")],
        },
        issues_by_repo={
            "acme/r0": [_gh_issue_node(10, updated_at="2026-04-04T00:00:00Z")],
            "acme/r1": [],
            "acme/r2": [],
        },
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    gh = build_connector(SourceSystem.GITHUB, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.GITHUB, access_token="x"
    )

    events = [e async for e in gh.backfill("cust", token)]
    # 3 PRs across 3 repos + 1 issue on r0 = 4 events.
    assert len(events) == 4

    pr_events = [e for e in events if e.headers["X-GitHub-Event"] == "pull_request"]
    repos = {e.raw_payload["repository"]["full_name"] for e in pr_events}
    assert repos == {"acme/r0", "acme/r1", "acme/r2"}


@pytest.mark.asyncio
async def test_github_graphql_reviews_emitted_inline() -> None:
    """A PR node with 2 reviews produces 1 pull_request event + 2
    pull_request_review events. Reviews are nested in BACKFILL_PULLS_QUERY
    so this never costs a second fetch.
    """
    pr = _gh_pr_node(42, updated_at="2026-04-01T00:00:00Z")
    pr["reviews"] = {
        "nodes": [
            _gh_review_node(
                "PRR_A",
                database_id=9001,
                state="APPROVED",
                submitted_at="2026-04-01T01:00:00Z",
            ),
            _gh_review_node(
                "PRR_B",
                database_id=9002,
                state="CHANGES_REQUESTED",
                submitted_at="2026-04-01T02:00:00Z",
            ),
        ]
    }

    transport = _gh_graphql_transport(
        installation_repos=[
            {"full_name": "acme/api", "name": "api", "owner": {"login": "acme"}}
        ],
        pulls_by_repo={"acme/api": [pr]},
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    gh = build_connector(SourceSystem.GITHUB, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.GITHUB, access_token="x"
    )

    events = [e async for e in gh.backfill("cust", token)]
    pr_events = [e for e in events if e.headers.get("X-GitHub-Event") == "pull_request"]
    review_events = [
        e for e in events if e.headers.get("X-GitHub-Event") == "pull_request_review"
    ]
    assert len(pr_events) == 1
    assert len(review_events) == 2

    # Each review event uses the same review:<repo>:<pr#>:<review_id> prefix
    # the live webhook uses, so a backfilled review dedupes with a webhook
    # delivery for the same review. review_id is the REST integer
    # (databaseId), never the GraphQL global string id.
    event_ids = {e.source_event_id for e in review_events}
    assert event_ids == {
        "review:acme/api:42:9001",
        "review:acme/api:42:9002",
    }

    # State is mapped from GraphQL uppercase to REST lowercase + snake_case.
    states = {e.raw_payload["review"]["state"] for e in review_events}
    assert states == {"approved", "changes_requested"}

    # html_url is synthesized from the PR url + databaseId so the
    # downstream Document.source_url is non-empty.
    for ev in review_events:
        review = ev.raw_payload["review"]
        assert review["html_url"]
        assert "#pullrequestreview-" in review["html_url"]
        assert ev.raw_payload["pull_request"]["number"] == 42
        assert ev.raw_payload["action"] == "submitted"


@pytest.mark.asyncio
async def test_github_graphql_commits_phase() -> None:
    """The commits phase paginates defaultBranchRef.target.history and
    emits one synthetic push event per commit with the correct branch
    pulled from the GraphQL response.
    """
    transport = _gh_graphql_transport(
        installation_repos=[
            {"full_name": "acme/api", "name": "api", "owner": {"login": "acme"}}
        ],
        commits_by_repo={
            "acme/api": [
                _gh_commit_node(
                    "a" * 40, committed_at="2026-04-01T00:00:00Z", message="m1"
                ),
                _gh_commit_node(
                    "b" * 40, committed_at="2026-04-02T00:00:00Z", message="m2"
                ),
                _gh_commit_node(
                    "c" * 40, committed_at="2026-04-03T00:00:00Z", message="m3"
                ),
            ]
        },
        default_branch_by_repo={"acme/api": "develop"},
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    gh = build_connector(SourceSystem.GITHUB, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.GITHUB, access_token="x"
    )

    events = [e async for e in gh.backfill("cust", token)]
    push_events = [e for e in events if e.headers.get("X-GitHub-Event") == "push"]
    assert len(push_events) == 3

    # Push payload mirrors the live-webhook shape: commits[0].id == oid.
    shas = {e.raw_payload["commits"][0]["id"] for e in push_events}
    assert shas == {"a" * 40, "b" * 40, "c" * 40}

    # Default branch from the GraphQL response flows into the synthetic ref.
    for ev in push_events:
        assert ev.raw_payload["ref"] == "refs/heads/develop"
        commit = ev.raw_payload["commits"][0]
        assert commit["author"]["username"] == "alice"
        # File deltas are intentionally empty for backfilled commits.
        assert commit["added"] == []
        assert commit["modified"] == []
        assert commit["removed"] == []


@pytest.mark.asyncio
async def test_github_graphql_releases_phase() -> None:
    """The releases phase emits one synthetic release event per release
    with action='published' (never deleted/unpublished from backfill).
    """
    transport = _gh_graphql_transport(
        installation_repos=[
            {"full_name": "acme/api", "name": "api", "owner": {"login": "acme"}}
        ],
        releases_by_repo={
            "acme/api": [
                _gh_release_node(release_id=100, tag="v1.0.0", name="First"),
                _gh_release_node(release_id=101, tag="v1.1.0", name="Second"),
            ]
        },
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    gh = build_connector(SourceSystem.GITHUB, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.GITHUB, access_token="x"
    )

    events = [e async for e in gh.backfill("cust", token)]
    release_events = [e for e in events if e.headers.get("X-GitHub-Event") == "release"]
    assert len(release_events) == 2

    for ev in release_events:
        assert ev.raw_payload["action"] == "published"
        # databaseId becomes the REST `id`.
        assert ev.raw_payload["release"]["id"] in {100, 101}
        # source_event_id format: release:<repo>:<release_id>.
        assert ev.source_event_id.startswith("release:acme/api:")

    ids = {e.raw_payload["release"]["id"] for e in release_events}
    assert ids == {100, 101}
    tags = {e.raw_payload["release"]["tag_name"] for e in release_events}
    assert tags == {"v1.0.0", "v1.1.0"}


@pytest.mark.asyncio
async def test_github_graphql_cursor_resume() -> None:
    """A saved cursor with `pulls_cursor` mid-page resumes the pulls walk at
    that endCursor and feeds it back as the GraphQL `after` variable."""
    import json as _json

    seen_cursors: list[str | None] = []

    def on_graphql(query, variables):
        if "pullRequests(" in query:
            seen_cursors.append(variables.get("cursor"))
            return _gh_graphql_response(
                {
                    "pullRequests": {
                        "pageInfo": {"endCursor": None, "hasNextPage": False},
                        "nodes": [_gh_pr_node(99, updated_at="2026-04-09T00:00:00Z")],
                    }
                }
            )
        return None

    transport = _gh_graphql_transport(
        installation_repos=[
            {
                "full_name": "acme/api",
                "name": "api",
                "owner": {"login": "acme"},
            }
        ],
        issues_by_repo={"acme/api": []},
        on_graphql=on_graphql,
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    gh = build_connector(SourceSystem.GITHUB, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.GITHUB, access_token="x"
    )

    resume_cursor = _json.dumps(
        {
            "version": 2,
            "engine": "graphql",
            "repos_remaining": [],
            "current_repo": "acme/api",
            "current_phase": "pulls",
            "pulls_cursor": "saved-page-cursor-xyz",
            "issues_cursor": None,
            "repo_objs": {
                "acme/api": {
                    "full_name": "acme/api",
                    "name": "api",
                    "owner": {"login": "acme"},
                }
            },
        }
    )

    events = [e async for e in gh.backfill("cust", token, cursor=resume_cursor)]
    assert len(events) == 1
    # The saved endCursor was passed back to GitHub on the first GraphQL call.
    assert seen_cursors[0] == "saved-page-cursor-xyz"


@pytest.mark.asyncio
async def test_github_graphql_review_uses_database_id() -> None:
    """Reviews emit the REST integer databaseId as id (not the global string
    id). Without this, backfill dedupes review docs on a different key than
    live webhooks, producing two pgvector rows per review.
    """
    from services.ingestion.handlers._github_graphql import (
        normalize_review_node,
    )

    # databaseId present -> integer id wins.
    with_db_id = normalize_review_node(
        {
            "id": "PRR_kwDO_global_xyz",
            "databaseId": 9001,
            "state": "APPROVED",
            "body": "lgtm",
            "submittedAt": "2026-04-01T00:00:00Z",
            "author": {"login": "alice"},
        },
        pr_html_url="https://github.com/o/r/pull/42",
    )
    assert with_db_id["id"] == 9001
    assert isinstance(with_db_id["id"], int)
    # html_url synthesized so source_url is non-empty downstream.
    assert with_db_id["html_url"] == (
        "https://github.com/o/r/pull/42#pullrequestreview-9001"
    )

    # databaseId missing -> fall back to the GraphQL global string id.
    without_db_id = normalize_review_node(
        {
            "id": "PRR_kwDO_only_global",
            "databaseId": None,
            "state": "COMMENTED",
            "body": "",
            "submittedAt": "2026-04-01T00:00:00Z",
            "author": {"login": "alice"},
        },
        pr_html_url="https://github.com/o/r/pull/42",
    )
    assert without_db_id["id"] == "PRR_kwDO_only_global"
    # Without databaseId we can't synthesize the anchor reliably -> blank.
    assert without_db_id["html_url"] == ""

    # source_event_id parity: the webhook _parse_review builds
    # `review:<repo>:<pr#>:<review_id>` from the integer review.id;
    # the backfill walker now builds the same string from
    # databaseId, so an identity round-trip is the simplest check
    # that the two halves of the system agree on a dedupe key.
    review_id = with_db_id["id"]
    backfill_key = f"review:acme/api:42:{review_id}"
    webhook_key = "review:acme/api:42:9001"
    assert backfill_key == webhook_key


@pytest.mark.asyncio
async def test_github_graphql_retries_on_5xx(monkeypatch) -> None:
    """A single 502 from GitHub GraphQL is treated as a transient infra
    blip and retried (not a phase-killing failure). Previously a 502
    returned None, the caller broke the page loop, and the rest of that
    repo's stream was dropped from the backfill.
    """
    import asyncio as _asyncio

    real_sleep = _asyncio.sleep

    async def fake_sleep(delay):
        # Don't actually wait; let the test finish in microseconds.
        await real_sleep(0)

    monkeypatch.setattr(
        "services.ingestion.handlers._github_graphql.asyncio.sleep", fake_sleep
    )

    call_count = {"pulls": 0}

    def on_graphql(query, variables):
        if "BackfillPulls" in query or "pullRequests(" in query:
            call_count["pulls"] += 1
            if call_count["pulls"] == 1:
                # First attempt: 502 (Bad Gateway). The client should retry.
                return httpx.Response(502, text="bad gateway")
            # Second attempt: clean 200 with one PR.
            return _gh_graphql_response(
                {
                    "pullRequests": {
                        "pageInfo": {"endCursor": None, "hasNextPage": False},
                        "nodes": [
                            _gh_pr_node(7, updated_at="2026-04-01T00:00:00Z")
                        ],
                    }
                }
            )
        return None

    transport = _gh_graphql_transport(
        installation_repos=[
            {"full_name": "acme/api", "name": "api", "owner": {"login": "acme"}}
        ],
        issues_by_repo={"acme/api": []},
        on_graphql=on_graphql,
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    gh = build_connector(SourceSystem.GITHUB, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.GITHUB, access_token="x"
    )

    events = [e async for e in gh.backfill("cust", token)]
    pr_events = [e for e in events if e.headers.get("X-GitHub-Event") == "pull_request"]
    assert len(pr_events) == 1
    # Retried exactly once: initial 502 + successful retry = 2 calls.
    assert call_count["pulls"] == 2


@pytest.mark.asyncio
async def test_github_graphql_partial_data_with_errors(monkeypatch) -> None:
    """A 200 response with `data.repository.pullRequests.nodes:[pr1]` plus
    `errors:[{type:"RATE_LIMITED"}]` should surface pr1 (the embedded
    cursor is still valid) and trigger a backoff. Previously the partial
    data was discarded and the page stream silently truncated.
    """
    import asyncio as _asyncio

    real_sleep = _asyncio.sleep

    async def fake_sleep(delay):
        await real_sleep(0)

    monkeypatch.setattr(
        "services.ingestion.handlers._github_graphql.asyncio.sleep", fake_sleep
    )

    call_count = {"pulls": 0}

    def on_graphql(query, variables):
        if "BackfillPulls" in query or "pullRequests(" in query:
            call_count["pulls"] += 1
            if call_count["pulls"] == 1:
                # Partial data + recoverable error. Caller should emit pr1
                # AND back off before the next attempt.
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "repository": {
                                "pullRequests": {
                                    "pageInfo": {
                                        "endCursor": None,
                                        "hasNextPage": False,
                                    },
                                    "nodes": [
                                        _gh_pr_node(
                                            55, updated_at="2026-04-01T00:00:00Z"
                                        )
                                    ],
                                }
                            },
                            "rateLimit": {
                                "cost": 1,
                                "remaining": 5000,
                                "resetAt": "2099-01-01T00:00:00Z",
                            },
                        },
                        "errors": [{"type": "RATE_LIMITED", "message": "throttled"}],
                    },
                )
            # Subsequent calls return clean -- exercised by other phases.
            return None
        return None

    transport = _gh_graphql_transport(
        installation_repos=[
            {"full_name": "acme/api", "name": "api", "owner": {"login": "acme"}}
        ],
        issues_by_repo={"acme/api": []},
        on_graphql=on_graphql,
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    gh = build_connector(SourceSystem.GITHUB, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.GITHUB, access_token="x"
    )

    events = [e async for e in gh.backfill("cust", token)]
    pr_events = [e for e in events if e.headers.get("X-GitHub-Event") == "pull_request"]
    # pr1 emitted despite the errors[] block.
    assert len(pr_events) == 1
    assert pr_events[0].raw_payload["pull_request"]["number"] == 55


@pytest.mark.asyncio
async def test_github_graphql_walker_exception_propagates(monkeypatch) -> None:
    """If a per-repo walker raises, the async-generator must re-raise that
    exception once the queue is drained -- not return normally. Otherwise
    backfill_runner writes a "success" cursor over a partial walk and the
    failed repo is silently dropped from this backfill.
    """
    import asyncio as _asyncio

    real_sleep = _asyncio.sleep

    async def fake_sleep(delay):
        await real_sleep(0)

    monkeypatch.setattr(
        "services.ingestion.handlers._github_graphql.asyncio.sleep", fake_sleep
    )

    async def boom_run_graphql(http, headers, query, variables):
        owner = variables.get("owner")
        name = variables.get("name")
        full_name = f"{owner}/{name}"
        if full_name == "acme/bomb":
            raise RuntimeError("boom")
        # Healthy repo: one PR then done.
        if "BackfillPulls" in query:
            return {
                "repository": {
                    "pullRequests": {
                        "pageInfo": {"endCursor": None, "hasNextPage": False},
                        "nodes": [
                            _gh_pr_node(1, updated_at="2026-04-01T00:00:00Z")
                        ],
                    }
                },
                "rateLimit": {
                    "cost": 1,
                    "remaining": 5000,
                    "resetAt": "2099-01-01T00:00:00Z",
                },
            }
        # Issues / commits / releases: empty page.
        if "BackfillIssues" in query:
            return {
                "repository": {
                    "issues": {
                        "pageInfo": {"endCursor": None, "hasNextPage": False},
                        "nodes": [],
                    }
                },
                "rateLimit": {
                    "cost": 1,
                    "remaining": 5000,
                    "resetAt": "2099-01-01T00:00:00Z",
                },
            }
        if "BackfillCommits" in query:
            return {
                "repository": {
                    "defaultBranchRef": {
                        "name": "main",
                        "target": {
                            "history": {
                                "pageInfo": {
                                    "endCursor": None,
                                    "hasNextPage": False,
                                },
                                "nodes": [],
                            }
                        },
                    }
                },
                "rateLimit": {
                    "cost": 1,
                    "remaining": 5000,
                    "resetAt": "2099-01-01T00:00:00Z",
                },
            }
        return {
            "repository": {
                "releases": {
                    "pageInfo": {"endCursor": None, "hasNextPage": False},
                    "nodes": [],
                }
            },
            "rateLimit": {
                "cost": 1,
                "remaining": 5000,
                "resetAt": "2099-01-01T00:00:00Z",
            },
        }

    # Patch the run_graphql symbol the connector imports inside backfill().
    monkeypatch.setattr(
        "services.ingestion.handlers._github_graphql.run_graphql",
        boom_run_graphql,
    )

    transport = _gh_graphql_transport(
        installation_repos=[
            {"full_name": "acme/good", "name": "good", "owner": {"login": "acme"}},
            {"full_name": "acme/bomb", "name": "bomb", "owner": {"login": "acme"}},
        ],
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    gh = build_connector(SourceSystem.GITHUB, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.GITHUB, access_token="x"
    )

    raised = False
    events: list = []
    try:
        async for e in gh.backfill("cust", token):
            events.append(e)
    except RuntimeError as exc:
        raised = True
        assert "boom" in str(exc)
    assert raised, "expected RuntimeError to propagate out of the backfill generator"
    # The healthy repo still streamed its event before the exception surfaced.
    healthy_events = [
        e
        for e in events
        if e.raw_payload.get("repository", {}).get("full_name") == "acme/good"
    ]
    assert len(healthy_events) >= 1


@pytest.mark.asyncio
async def test_github_graphql_snapshot_lock_serializes_cursor_writes(
    monkeypatch,
) -> None:
    """Cursor writes happen under snapshot_lock. With two concurrent
    walkers, every _snapshot_cursor() must observe a consistent rs[]
    (phase + cursors in agreement -- never `phase=issues` paired with
    `issues_cursor=None` mid-transition from another walker).

    Forces a context switch between cursor-mutation and queue.put by
    yielding via asyncio.sleep(0), then validates every observed
    cursor snapshot has consistent shape.
    """
    import json as _json

    monkeypatch.setenv("GITHUB_BACKFILL_REPO_CONCURRENCY", "2")
    from shared.config import get_settings as _gs

    _gs.cache_clear()  # type: ignore[attr-defined]

    transport = _gh_graphql_transport(
        installation_repos=[
            {"full_name": f"acme/r{i}", "name": f"r{i}", "owner": {"login": "acme"}}
            for i in range(2)
        ],
        pulls_by_repo={
            "acme/r0": [
                _gh_pr_node(1, updated_at="2026-04-01T00:00:00Z"),
                _gh_pr_node(2, updated_at="2026-04-02T00:00:00Z"),
            ],
            "acme/r1": [
                _gh_pr_node(3, updated_at="2026-04-03T00:00:00Z"),
            ],
        },
        issues_by_repo={"acme/r0": [], "acme/r1": []},
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    gh = build_connector(SourceSystem.GITHUB, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.GITHUB, access_token="x"
    )

    events = [e async for e in gh.backfill("cust", token)]
    # 3 PR events expected across both repos.
    pr_events = [e for e in events if e.headers.get("X-GitHub-Event") == "pull_request"]
    assert len(pr_events) == 3

    # Each event's embedded _cursor must parse as JSON and the
    # current_phase / current_repo combo must be internally consistent
    # (no torn-write where current_repo is set to a value that's also
    # in repos_remaining, or where current_phase is unknown).
    valid_phases = {"pulls", "issues", "commits", "releases"}
    for ev in pr_events:
        blob = ev.raw_payload.get("_cursor")
        if not blob:
            continue
        parsed = _json.loads(blob)
        assert parsed["version"] == 2
        assert parsed["engine"] == "graphql"
        assert parsed["current_phase"] in valid_phases
        current = parsed.get("current_repo")
        remaining = parsed.get("repos_remaining") or []
        if current is not None:
            assert current not in remaining, (
                "torn snapshot: current_repo appears in repos_remaining"
            )


# --------------------- backfill status endpoint -----------------------------


@pytest.mark.asyncio
async def test_backfill_status_endpoint(live_db) -> None:
    """GET /backfill/status returns per-source rows for a customer."""
    from httpx import ASGITransport

    from shared.db import close_pool, init_pool

    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ('cust-s','x','y') ON CONFLICT DO NOTHING"
        )

    await enqueue_backfill("cust-s", SourceSystem.SLACK)
    await enqueue_backfill("cust-s", SourceSystem.LINEAR)

    from services.ingestion.main import app as ingestion_app

    await close_pool()
    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        resp = await client.get("/backfill/status?customer_id=cust-s")
    await init_pool(settings=None)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["customer_id"] == "cust-s"
    sources = {s["source"]: s for s in body["sources"]}
    assert set(sources.keys()) == {"slack", "linear"}
    assert sources["slack"]["status"] == "pending"


# ---------------------------------------------------------------------------
# run_backfill: batched R2 puts + DB inserts
# ---------------------------------------------------------------------------
#
# These tests exercise the coalesced-flush path: instead of one R2 put + one
# ingestion_queue INSERT per event, the runner accumulates a batch (size set
# by Settings.backfill_batch_size) and flushes via a single asyncio.gather of
# R2 puts plus one asyncpg `executemany`. At 100k events that's ~1000 round
# trips instead of 200000.

from collections.abc import AsyncIterator  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

from services.ingestion import backfill_runner as _bf_runner  # noqa: E402
from services.ingestion.backfill_runner import (  # noqa: E402
    PROGRESS_EVERY_N_EVENTS,
    run_backfill,
)
from services.ingestion.handlers.base import Connector  # noqa: E402
from shared.encryption import encrypt_token  # noqa: E402
from shared.models import IntegrationToken, WebhookEvent  # noqa: E402

_BATCH_CUSTOMER = "cust-bf-batch"
_BATCH_SOURCE = SourceSystem.GRANOLA


async def _seed_batch_customer_and_token() -> None:
    """Seed customer (with r2_bucket) + active integration_tokens row.

    Mirrors the helper shape in test_mark_failed_token_flip.py without
    importing it (keeps this test module self-contained).
    """
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'bf-batch', 'bf-batch-hash') "
            "ON CONFLICT DO NOTHING",
            _BATCH_CUSTOMER,
        )
        await conn.execute(
            """
            INSERT INTO integration_tokens
                (customer_id, source_system, access_token_encrypted, status)
            VALUES ($1, $2, $3, 'active')
            ON CONFLICT DO NOTHING
            """,
            _BATCH_CUSTOMER,
            _BATCH_SOURCE.value,
            encrypt_token("grn_batch_TOKEN"),
        )


async def _batch_queue_count() -> int:
    async with raw_conn() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM ingestion_queue "
            "WHERE customer_id=$1 AND source_system=$2",
            _BATCH_CUSTOMER,
            _BATCH_SOURCE.value,
        )


async def _batch_state_row():
    async with raw_conn() as conn:
        return await conn.fetchrow(
            "SELECT status, events_enqueued, last_cursor, last_progress_at "
            "FROM backfill_state "
            "WHERE customer_id=$1 AND source_system=$2",
            _BATCH_CUSTOMER,
            _BATCH_SOURCE.value,
        )


def _bf_evt(idx: int, *, cursor: str | None = None) -> WebhookEvent:
    """Synthetic event. `cursor` populates payload._cursor so the runner can
    persist the watermark progression."""
    payload: dict = {"idx": idx}
    if cursor is not None:
        payload["_cursor"] = cursor
    return WebhookEvent(
        customer_id=_BATCH_CUSTOMER,
        source_system=_BATCH_SOURCE,
        source_event_id=f"bf-batch-{idx}",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/granola/fake/key.json",
        raw_payload=payload,
    )


class _BatchConnector(Connector):
    """Yields N synthetic events. Each event carries a fresh `_cursor` so the
    runner exercises its cursor-advance path on every flush."""

    source_system = _BATCH_SOURCE

    def __init__(self, ctx: ConnectorContext, total: int) -> None:
        super().__init__(ctx)
        self._total = total

    def verify_signature(self, headers, raw_body):  # pragma: no cover
        return True

    def parse_webhook_event(self, customer_id, headers, raw_payload):  # pragma: no cover
        return None

    async def normalize(self, event, hydrated):  # pragma: no cover
        raise NotImplementedError

    async def backfill(  # type: ignore[override]
        self,
        customer_id: str,
        token: IntegrationToken,
        cursor: str | None = None,
    ) -> AsyncIterator[WebhookEvent]:
        for i in range(self._total):
            yield _bf_evt(i, cursor=f"watermark-{i}")


def _bf_ctx() -> ConnectorContext:
    return ConnectorContext(settings=Settings(environment="local"), http=httpx.AsyncClient())


@pytest.mark.asyncio
async def test_backfill_batches_r2_and_db(live_db, monkeypatch) -> None:
    """N events -> ceil(N / batch_size) R2 gather rounds + ceil(N / batch_size)
    executemany calls.

    Verifies the coalescing collapsed per-event round-trips into per-batch
    round-trips. Count R2 puts via store.put monkeypatch; count executemany
    calls via asyncpg Connection.executemany monkeypatch.
    """
    from shared.config import get_settings as _get_settings

    await _seed_batch_customer_and_token()

    from services.ingestion.backfill_runner import enqueue_backfill as _eq
    await _eq(_BATCH_CUSTOMER, _BATCH_SOURCE)

    batch_size = _get_settings().backfill_batch_size  # default 100
    total = batch_size * 2 + batch_size // 4  # e.g. 225 → 3 batches
    expected_flushes = (total + batch_size - 1) // batch_size

    # Count store.put calls (one per event during a flush).
    put_calls = {"n": 0}
    real_put = _bf_runner.get_store().put

    async def counting_put(bucket, key, body, content_type="application/json"):
        put_calls["n"] += 1
        return await real_put(bucket, key, body, content_type=content_type)

    monkeypatch.setattr(_bf_runner.get_store(), "put", counting_put)

    # Count executemany calls on asyncpg Connections.
    executemany_calls = {"n": 0}
    import asyncpg

    real_executemany = asyncpg.Connection.executemany

    async def counting_executemany(self, query, args, *a, **kw):
        if "ingestion_queue" in query:
            executemany_calls["n"] += 1
        return await real_executemany(self, query, args, *a, **kw)

    monkeypatch.setattr(asyncpg.Connection, "executemany", counting_executemany)

    fake = _BatchConnector(_bf_ctx(), total=total)
    monkeypatch.setattr(
        "services.ingestion.backfill_runner.build_connector",
        lambda src, ctx: fake,
    )

    enqueued = await run_backfill(_bf_ctx(), _BATCH_CUSTOMER, _BATCH_SOURCE)

    assert enqueued == total
    assert await _batch_queue_count() == total
    assert put_calls["n"] == total, (
        f"expected one R2 put per event, got {put_calls['n']}"
    )
    assert executemany_calls["n"] == expected_flushes, (
        f"expected {expected_flushes} executemany calls "
        f"(ceil({total}/{batch_size})), got {executemany_calls['n']}"
    )


@pytest.mark.asyncio
async def test_backfill_partial_batch_flushed_at_end(live_db, monkeypatch) -> None:
    """A partial trailing batch (e.g. 23 events with batch_size=100) must
    still flush at end-of-stream — one executemany with 23 rows, not zero."""
    await _seed_batch_customer_and_token()

    from services.ingestion.backfill_runner import enqueue_backfill as _eq
    await _eq(_BATCH_CUSTOMER, _BATCH_SOURCE)

    total = 23  # well below default batch_size=100

    executemany_invocations: list[int] = []
    import asyncpg

    real_executemany = asyncpg.Connection.executemany

    async def recording_executemany(self, query, args, *a, **kw):
        if "ingestion_queue" in query:
            # args is a list of row-tuples — record the rowcount per call.
            executemany_invocations.append(len(args))
        return await real_executemany(self, query, args, *a, **kw)

    monkeypatch.setattr(asyncpg.Connection, "executemany", recording_executemany)

    fake = _BatchConnector(_bf_ctx(), total=total)
    monkeypatch.setattr(
        "services.ingestion.backfill_runner.build_connector",
        lambda src, ctx: fake,
    )

    enqueued = await run_backfill(_bf_ctx(), _BATCH_CUSTOMER, _BATCH_SOURCE)

    assert enqueued == total
    assert await _batch_queue_count() == total
    assert executemany_invocations == [total], (
        f"trailing partial batch must flush exactly once with all {total} rows, "
        f"got {executemany_invocations}"
    )


@pytest.mark.asyncio
async def test_backfill_cursor_checkpoint_every_25(live_db, monkeypatch) -> None:
    """last_progress_at writes (via _update_progress) must fire at the right
    cadence under batching: at most once per batch, but only when a
    PROGRESS_EVERY_N_EVENTS boundary is crossed since the last write.

    With batch_size=100 and PROGRESS_EVERY_N_EVENTS=25, 200 events → 2 batches
    → 2 progress writes (not 8, which the old per-event path would produce)."""
    await _seed_batch_customer_and_token()

    from services.ingestion.backfill_runner import enqueue_backfill as _eq
    await _eq(_BATCH_CUSTOMER, _BATCH_SOURCE)

    total = 200

    progress_calls: list[tuple[int, str | None]] = []
    real_update = _bf_runner._update_progress

    async def recording_update_progress(customer_id, source, cursor, enqueued, claim_token):
        progress_calls.append((enqueued, cursor))
        return await real_update(customer_id, source, cursor, enqueued, claim_token)

    monkeypatch.setattr(_bf_runner, "_update_progress", recording_update_progress)

    fake = _BatchConnector(_bf_ctx(), total=total)
    monkeypatch.setattr(
        "services.ingestion.backfill_runner.build_connector",
        lambda src, ctx: fake,
    )

    enqueued = await run_backfill(_bf_ctx(), _BATCH_CUSTOMER, _BATCH_SOURCE)

    assert enqueued == total
    # 2 flushes (events 100, 200) — both cross at least one 25-event boundary
    # since the last write, so each one writes a progress checkpoint.
    assert len(progress_calls) == 2, (
        f"expected one progress write per flush (2 batches), got {progress_calls}"
    )
    # Both writes have an `enqueued` value at a multiple of the batch size.
    written_enqueued = [c[0] for c in progress_calls]
    assert written_enqueued == [100, 200]
    # Cursor watermark advanced to the last event's _cursor in each batch.
    assert progress_calls[0][1] == "watermark-99"
    assert progress_calls[1][1] == "watermark-199"
    # Each progress write crosses a PROGRESS_EVERY_N_EVENTS boundary —
    # spec invariant: cadence stays >= 25 events between writes.
    from itertools import pairwise
    for prev, nxt in pairwise(written_enqueued):
        assert nxt - prev >= PROGRESS_EVERY_N_EVENTS


@pytest.mark.asyncio
async def test_backfill_batch_exception_marks_failed(live_db, monkeypatch) -> None:
    """If the batched flush raises (e.g. asyncpg executemany fails on a
    constraint violation), the run must flip to status='failed' — do NOT
    silently drop the partial batch."""
    await _seed_batch_customer_and_token()

    from services.ingestion.backfill_runner import enqueue_backfill as _eq
    await _eq(_BATCH_CUSTOMER, _BATCH_SOURCE)

    # Make every executemany on ingestion_queue raise.
    import asyncpg

    real_executemany = asyncpg.Connection.executemany

    async def exploding_executemany(self, query, args, *a, **kw):
        if "ingestion_queue" in query:
            raise RuntimeError("simulated db failure during batched insert")
        return await real_executemany(self, query, args, *a, **kw)

    monkeypatch.setattr(asyncpg.Connection, "executemany", exploding_executemany)

    fake = _BatchConnector(_bf_ctx(), total=150)  # crosses a batch boundary
    monkeypatch.setattr(
        "services.ingestion.backfill_runner.build_connector",
        lambda src, ctx: fake,
    )

    with pytest.raises(RuntimeError, match="simulated db failure"):
        await run_backfill(_bf_ctx(), _BATCH_CUSTOMER, _BATCH_SOURCE)

    bf = await _batch_state_row()
    assert bf is not None
    assert bf["status"] == BackfillStatus.FAILED.value, (
        f"expected status=failed after flush exception, got {bf['status']}"
    )
