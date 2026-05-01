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
    databases/{id}/query and yielded as page.updated events.

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


@pytest.mark.asyncio
async def test_github_backfill_paginates_repos_pulls_and_issues() -> None:
    """GitHub: list installation repos, then pulls + issues per repo. Issue-shaped PRs filtered."""

    def installation_repos(req):
        return httpx.Response(
            200,
            json={
                "repositories": [
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
            },
        )

    def api_pulls(req):
        return httpx.Response(
            200,
            json=[
                {"number": 1, "updated_at": "2026-04-01T00:00:00Z", "title": "pr1"},
                {"number": 2, "updated_at": "2026-04-02T00:00:00Z", "title": "pr2"},
            ],
        )

    def api_issues(req):
        return httpx.Response(
            200,
            json=[
                {"number": 10, "updated_at": "2026-04-03T00:00:00Z", "title": "issue"},
                # Issue-shaped PR — must be filtered out by the backfill.
                {
                    "number": 11,
                    "updated_at": "2026-04-04T00:00:00Z",
                    "title": "pr-as-issue",
                    "pull_request": {"url": "..."},
                },
            ],
        )

    def web_pulls(req):
        return httpx.Response(
            200,
            json=[
                {"number": 3, "updated_at": "2026-04-05T00:00:00Z", "title": "pr3"},
            ],
        )

    def web_issues(req):
        return httpx.Response(200, json=[])

    transport = _mock_transport(
        {
            ("GET", "/installation/repositories"): installation_repos,
            ("GET", "/repos/acme/api/pulls"): api_pulls,
            ("GET", "/repos/acme/api/issues"): api_issues,
            ("GET", "/repos/acme/web/pulls"): web_pulls,
            ("GET", "/repos/acme/web/issues"): web_issues,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    gh = build_connector(SourceSystem.GITHUB, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.GITHUB, access_token="x"
    )

    events = [e async for e in gh.backfill("cust", token)]

    # 3 PRs + 1 issue = 4 events; the issue-shaped PR was filtered.
    assert len(events) == 4

    pr_events = [e for e in events if e.headers.get("X-GitHub-Event") == "pull_request"]
    issue_events = [e for e in events if e.headers.get("X-GitHub-Event") == "issues"]
    assert len(pr_events) == 3
    assert len(issue_events) == 1

    # Confirm the issue-shaped PR (number=11) was filtered.
    issue_numbers = {e.raw_payload["issue"]["number"] for e in issue_events}
    assert issue_numbers == {10}

    pr_numbers = {e.raw_payload["pull_request"]["number"] for e in pr_events}
    assert pr_numbers == {1, 2, 3}


@pytest.mark.asyncio
async def test_github_backfill_with_installation_scope_mints_token(monkeypatch) -> None:
    """A token with scope='installation:<id>' mints a fresh bearer via github_auth."""
    from datetime import UTC, datetime

    observed_auth: list[str] = []

    def installation_repos(req):
        observed_auth.append(req.headers["authorization"])
        return httpx.Response(
            200,
            json={
                "repositories": [
                    {
                        "full_name": "acme/api",
                        "name": "api",
                        "owner": {"login": "acme"},
                        "private": True,
                    }
                ]
            },
        )

    def api_pulls(req):
        observed_auth.append(req.headers["authorization"])
        return httpx.Response(
            200,
            json=[{"number": 1, "updated_at": "2026-04-01T00:00:00Z", "title": "pr"}],
        )

    def api_issues(req):
        observed_auth.append(req.headers["authorization"])
        return httpx.Response(200, json=[])

    transport = _mock_transport(
        {
            ("GET", "/installation/repositories"): installation_repos,
            ("GET", "/repos/acme/api/pulls"): api_pulls,
            ("GET", "/repos/acme/api/issues"): api_issues,
        }
    )

    minted_for: list[str] = []

    async def fake_mint(http, app_id, private_key_pem, installation_id):
        minted_for.append(installation_id)
        return "ghs_fresh_bearer", datetime(2026, 12, 31, tzinfo=UTC)

    monkeypatch.setattr(
        "services.ingestion.handlers.github.mint_installation_token", fake_mint
    )

    from services.ingestion.handlers.base import ConnectorContext
    from services.ingestion.handlers.registry import build_connector
    from shared.config import Settings as _S
    from shared.models import IntegrationToken

    ctx = ConnectorContext(
        settings=_S(
            github_app_id="1",
            github_app_private_key="dummy-key",  # type: ignore[arg-type]
        ),
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
    assert minted_for == ["99"]
    assert observed_auth, "mock transport should have captured requests"
    for auth in observed_auth:
        assert auth == "Bearer ghs_fresh_bearer"


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
