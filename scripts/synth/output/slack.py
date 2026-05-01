"""SlackWrapper — serialize a SynthDoc to a Slack Events API event_callback envelope.

The output must round-trip through SlackConnector.parse_webhook_event:
  parse_webhook_event(customer_id, {}, json.loads(wrap(doc)))
must return a WebhookParseResult with source_event_id == f"{channel}:{ts}".

Envelope shape (matches fixtures/slack/message_simple.json):
{
  "type": "event_callback",
  "team_id": "T-SYNTH",
  "api_app_id": "A-SYNTH",
  "event_id": "<source_event_id>",
  "event_time": <unix_int>,
  "event": {
    "type": "message",
    "channel": "<channel>",
    "user": "<persona_slug>",
    "text": "<text>",
    "ts": "<unix>.<6digits>",
    "thread_ts": "<parent_ts>"   # only present when thread_parent_id is not None
  }
}
"""

from __future__ import annotations

from datetime import datetime

import orjson

from scripts.synth.output.base import SynthDoc

_SYNTH_TEAM_ID = "T-SYNTH"
_SYNTH_APP_ID = "A-SYNTH"


def _ts_str(dt: datetime) -> str:
    """Convert a datetime to Slack's "<unix_seconds>.<6digits>" string format."""
    unix = int(dt.timestamp())
    return f"{unix}.000000"


def _user_slug(doc: SynthDoc) -> str:
    """Derive a stable pseudo-user id from the first persona canonical_id."""
    if not doc.personas:
        return "U-SYNTH-unknown"
    raw = doc.personas[0].replace(":", "-").replace("@", "").upper()
    return f"U-{raw}"


def wrap(doc: SynthDoc) -> bytes:
    """Produce a Slack Events API event_callback envelope as JSON bytes."""
    channel = doc.channel or "#general"
    ts = _ts_str(doc.occurred_at)

    event: dict = {
        "type": "message",
        "channel": channel,
        "user": _user_slug(doc),
        "text": doc.text,
        "ts": ts,
        "team": _SYNTH_TEAM_ID,
    }

    # Only include thread_ts when this doc is a reply (thread_parent_id set).
    if doc.thread_parent_id is not None:
        # The thread_ts is derived from occurred_at minus 1 second (the parent
        # was posted 1s earlier in the synthetic timeline).
        parent_unix = int(doc.occurred_at.timestamp()) - 1
        event["thread_ts"] = f"{parent_unix}.000000"

    payload = {
        "type": "event_callback",
        "team_id": _SYNTH_TEAM_ID,
        "api_app_id": _SYNTH_APP_ID,
        "event_id": doc.source_event_id,
        "event_time": int(doc.occurred_at.timestamp()),
        "event": event,
    }

    return orjson.dumps(payload)
