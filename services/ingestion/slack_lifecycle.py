"""Slack Events API lifecycle hooks that schedule backfills.

Message events go through the normal webhook -> queue -> normalize path. A few
Slack lifecycle events are control-plane signals instead: they make a new
channel visible to the bot, so we queue a channel-scoped backfill and skip
normal document ingestion.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from services.ingestion.backfill_runner import enqueue_slack_channel_backfill
from services.ingestion.handlers.base import ConnectorContext
from shared.constants import SourceSystem
from shared.customer_mapping import load_source_metadata, patch_source_metadata
from shared.logging import get_logger
from shared.models import IntegrationToken
from shared.tokens import load_token

log = get_logger(__name__)

_SLACK_API = "https://slack.com/api"
_AUTO_JOIN_SCOPE = "channels:join"


async def handle_slack_lifecycle_event(
    ctx: ConnectorContext,
    customer_id: str,
    raw_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    event = raw_payload.get("event")
    if not isinstance(event, Mapping):
        return None

    event_type = event.get("type")
    if event_type == "channel_created":
        return await _handle_channel_created(ctx, customer_id, raw_payload, event)
    if event_type == "member_joined_channel":
        return await _handle_member_joined_channel(ctx, customer_id, raw_payload, event)
    if event_type == "message" and event.get("subtype") == "channel_join":
        return await _handle_channel_join_message(ctx, customer_id, raw_payload, event)
    return None


async def _handle_channel_created(
    ctx: ConnectorContext,
    customer_id: str,
    raw_payload: Mapping[str, Any],
    event: Mapping[str, Any],
) -> dict[str, Any]:
    token = await load_token(customer_id, SourceSystem.SLACK)
    if token is None:
        return _result("channel_created", None, False, "missing_active_token")

    channel = event.get("channel")
    if not isinstance(channel, Mapping):
        return _result("channel_created", None, False, "missing_channel")

    channel_id = channel.get("id")
    if not isinstance(channel_id, str) or not channel_id:
        return _result("channel_created", None, False, "missing_channel_id")

    if bool(channel.get("is_private")):
        return _result("channel_created", channel_id, False, "private_channel")

    if not _scope_has(token, _AUTO_JOIN_SCOPE):
        return _result("channel_created", channel_id, False, "missing_channels_join_scope")

    joined = await _join_public_channel(ctx, token, channel_id)
    if not joined:
        return _result("channel_created", channel_id, False, "join_failed")

    queued = await enqueue_slack_channel_backfill(customer_id, channel_id)
    log.info(
        "slack.lifecycle.channel_created_backfill",
        customer=customer_id,
        team=_team_id_from(raw_payload, event),
        channel=channel_id,
        queued=queued.queued,
        reason=queued.reason,
    )
    return _result("channel_created", channel_id, queued.queued, queued.reason)


async def _handle_member_joined_channel(
    ctx: ConnectorContext,
    customer_id: str,
    raw_payload: Mapping[str, Any],
    event: Mapping[str, Any],
) -> dict[str, Any]:
    return await _schedule_if_joined_user_is_bot(
        ctx,
        customer_id,
        raw_payload,
        event,
        event_name="member_joined_channel",
    )


async def _handle_channel_join_message(
    ctx: ConnectorContext,
    customer_id: str,
    raw_payload: Mapping[str, Any],
    event: Mapping[str, Any],
) -> dict[str, Any] | None:
    result = await _schedule_if_joined_user_is_bot(
        ctx,
        customer_id,
        raw_payload,
        event,
        event_name="channel_join",
    )
    if result["reason"] == "non_bot_user_joined":
        return None
    return result


async def _schedule_if_joined_user_is_bot(
    ctx: ConnectorContext,
    customer_id: str,
    raw_payload: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    event_name: str,
) -> dict[str, Any]:
    channel_id = event.get("channel")
    if not isinstance(channel_id, str) or not channel_id:
        return _result(event_name, None, False, "missing_channel_id")

    joined_user = event.get("user")
    if not isinstance(joined_user, str) or not joined_user:
        return _result(event_name, channel_id, False, "missing_user_id")

    token = await load_token(customer_id, SourceSystem.SLACK)
    if token is None:
        return _result(event_name, channel_id, False, "missing_active_token")

    team_id = _team_id_from(raw_payload, event)
    bot_user_id = await _bot_user_id(ctx, token, team_id)
    if not bot_user_id:
        return _result(event_name, channel_id, False, "bot_user_unknown")
    if joined_user != bot_user_id:
        return _result(event_name, channel_id, False, "non_bot_user_joined")

    queued = await enqueue_slack_channel_backfill(customer_id, channel_id)
    log.info(
        "slack.lifecycle.bot_joined_channel_backfill",
        customer=customer_id,
        team=team_id,
        channel=channel_id,
        queued=queued.queued,
        reason=queued.reason,
    )
    return _result(event_name, channel_id, queued.queued, queued.reason)


async def _join_public_channel(
    ctx: ConnectorContext,
    token: IntegrationToken,
    channel_id: str,
) -> bool:
    try:
        resp = await ctx.http.post(
            f"{_SLACK_API}/conversations.join",
            data={"channel": channel_id},
            headers={"Authorization": f"Bearer {token.access_token}"},
        )
    except httpx.HTTPError as exc:
        log.warning(
            "slack.lifecycle.channel_join_http_error",
            channel=channel_id,
            error=str(exc),
        )
        return False

    if resp.status_code != 200:
        log.warning(
            "slack.lifecycle.channel_join_non_200",
            channel=channel_id,
            status=resp.status_code,
        )
        return False

    body = resp.json()
    if body.get("ok"):
        return True
    if body.get("error") == "already_in_channel":
        return True

    log.info(
        "slack.lifecycle.channel_join_not_ok",
        channel=channel_id,
        error=body.get("error"),
    )
    return False


async def _bot_user_id(
    ctx: ConnectorContext,
    token: IntegrationToken,
    team_id: str | None,
) -> str | None:
    if team_id:
        metadata = await load_source_metadata(SourceSystem.SLACK, team_id)
        cached = metadata.get("bot_user_id")
        if isinstance(cached, str) and cached:
            return cached

    try:
        resp = await ctx.http.post(
            f"{_SLACK_API}/auth.test",
            headers={"Authorization": f"Bearer {token.access_token}"},
        )
    except httpx.HTTPError as exc:
        log.warning("slack.lifecycle.auth_test_http_error", error=str(exc))
        return None

    if resp.status_code != 200:
        log.warning("slack.lifecycle.auth_test_non_200", status=resp.status_code)
        return None
    body = resp.json()
    if not body.get("ok"):
        log.warning("slack.lifecycle.auth_test_not_ok", error=body.get("error"))
        return None

    bot_user_id = body.get("user_id")
    resolved_team_id = body.get("team_id") or team_id
    if isinstance(bot_user_id, str) and isinstance(resolved_team_id, str):
        await patch_source_metadata(
            SourceSystem.SLACK,
            resolved_team_id,
            {"bot_user_id": bot_user_id, "bot_id": body.get("bot_id")},
        )
        return bot_user_id
    return None


def _scope_has(token: IntegrationToken, required: str) -> bool:
    scopes = {scope.strip() for scope in (token.scope or "").split(",")}
    return required in scopes


def _team_id_from(
    raw_payload: Mapping[str, Any],
    event: Mapping[str, Any],
) -> str | None:
    if isinstance(raw_payload.get("team_id"), str):
        return raw_payload["team_id"]
    if isinstance(event.get("team"), str):
        return event["team"]
    team = raw_payload.get("team")
    if isinstance(team, Mapping) and isinstance(team.get("id"), str):
        return team["id"]
    return None


def _result(
    event_type: str,
    channel_id: str | None,
    backfill_queued: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": "accepted" if backfill_queued else "ignored",
        "event_type": event_type,
        "channel_id": channel_id,
        "backfill_queued": backfill_queued,
        "reason": reason,
    }
