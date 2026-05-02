"""LinearWrapper — serialize a SynthDoc to a Linear GraphQL webhook envelope.

Linear sends webhooks with this shape (per fixtures/linear/issue_create.json):
{
  "type": "Issue",
  "action": "create",
  "createdAt": "<iso8601>",
  "organizationId": "<org>",
  "webhookId": "<wh_id>",
  "url": "<linear issue url>",
  "data": {
    "id": "<uuid>",
    "identifier": "<TEAM-N>",
    "title": "<first line of doc.text>",
    "description": "<full doc.text>",
    "priority": 1,
    "url": "<linear issue url>",
    "teamId": "<team_id>",
    "team": {"id": "<team_id>", "key": "ENG", "name": "Engineering"},
    "creatorId": "<user_id>",
    "creator": {"id": "<user_id>", "name": "<name>", "email": "<email>"},
    "assigneeId": "<user_id>",
    "assignee": {"id": "<user_id>", "name": "<name>", "email": "<email>"},
    "state": {"id": "state_inprogress", "name": "In Progress", "type": "started"},
    "createdAt": "<iso8601>",
    "updatedAt": "<iso8601>"
  }
}

doc.text first line → data.title
doc.text full text → data.description
doc.personas[0] → creator
doc.personas[1] (if present) → assignee (falls back to creator if only one persona)
doc.source_event_id → data.id (deterministic UUID-shaped string, sha256-derived)
doc.services_mentioned[0] → team name
"""

from __future__ import annotations

import hashlib

import orjson

from scripts.synth.output.base import SynthDoc

_SYNTH_ORG_ID = "org-synth"
_SYNTH_WEBHOOK_ID = "wh-synth"
_SYNTH_BASE_URL = "https://linear.app/prbe/issue"


def _stable_int(s: str, mod: int) -> int:
    """Deterministic int from a string, stable across processes (vs hash() which is PYTHONHASHSEED-seeded)."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % mod


def _persona_to_user(persona_id: str) -> dict:
    """Derive a stable Linear user object from a canonical persona id."""
    slug = persona_id.replace("gh:", "").replace("@", "")
    uid = "user_" + slug
    return {
        "id": uid,
        "name": slug.capitalize(),
        "email": f"{slug}@example.com",
    }


def _issue_id_from_event_id(source_event_id: str) -> str:
    """Derive a deterministic UUID-shaped id from source_event_id using sha256."""
    h = hashlib.sha256(source_event_id.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _title_from_text(text: str) -> str:
    """Return the first non-empty line of text, truncated to 200 chars."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return "Untitled"


def _number_from_event_id(source_event_id: str) -> int:
    """Derive a stable issue number from source_event_id using sha256."""
    return _stable_int(source_event_id, 9000) + 100


def _iso(doc: SynthDoc) -> str:
    return doc.occurred_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def wrap(doc: SynthDoc) -> bytes:
    """Produce a Linear webhook issue_create envelope as JSON bytes."""
    creator = _persona_to_user(doc.personas[0]) if doc.personas else _persona_to_user("gh:synth-bot")
    assignee = _persona_to_user(doc.personas[1]) if len(doc.personas) > 1 else creator

    team_name = doc.services_mentioned[0].replace("-", " ").title() if doc.services_mentioned else "Engineering"
    team_id = "team_" + (doc.services_mentioned[0] if doc.services_mentioned else "eng")
    team_key = team_name[:3].upper()

    issue_id = _issue_id_from_event_id(doc.source_event_id)
    number = _number_from_event_id(doc.source_event_id)
    identifier = f"{team_key}-{number}"
    issue_url = f"{_SYNTH_BASE_URL}/{identifier}"
    iso = _iso(doc)

    payload = {
        "type": "Issue",
        "action": "create",
        "createdAt": iso,
        "organizationId": _SYNTH_ORG_ID,
        "webhookId": _SYNTH_WEBHOOK_ID,
        "url": issue_url,
        "data": {
            "id": issue_id,
            "identifier": identifier,
            "title": _title_from_text(doc.text),
            "description": doc.text,
            "priority": 1,
            "url": issue_url,
            "teamId": team_id,
            "team": {"id": team_id, "key": team_key, "name": team_name},
            "creatorId": creator["id"],
            "creator": creator,
            "assigneeId": assignee["id"],
            "assignee": assignee,
            "state": {"id": "state_inprogress", "name": "In Progress", "type": "started"},
            "createdAt": iso,
            "updatedAt": iso,
        },
    }
    return orjson.dumps(payload)
