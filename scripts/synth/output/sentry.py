"""SentryWrapper — serialize a SynthDoc to a Sentry issue_created webhook envelope.

Plan 3 emits SENTRY docs for the INCIDENT archetype via the templated path
(no LLM call, per spec §7.2). This wrapper serializes the DocSpec content
into the issue_created shape.

Envelope shape decisions:
  - All archetypes → action=created + issue_created envelope (default, unambiguous).
  - If future archetypes need event_alert, add a secondary wrap_event_alert()
    helper — the current public wrap() always produces issue_created.
    event_alert is reachable in Plan 4 if needed; keeping a single envelope
    type here avoids routing complexity and matches the templated INCIDENT path.

Shape (matches fixtures/sentry/issue_created.json):
{
  "action": "created",
  "installation": {"uuid": "<inst_uuid>"},
  "data": {
    "issue": {
      "id": "<issue_id>",
      "shortId": "<short_id>",
      "title": "<first line of doc.text>",
      "culprit": "<service>.<module_hint>",
      "level": "error",
      "status": "unresolved",
      "platform": "python",
      "firstSeen": "<iso8601>",
      "lastSeen": "<iso8601>",
      "url": "<sentry issue url>",
      "permalink": "<sentry issue url>",
      "assignedTo": {"type": "user", "username": "<persona>", "email": "<email>"},
      "project": {"id": "<proj_id>", "slug": "<service>-api", "name": "<Service> API"},
      "metadata": {"type": "Error", "filename": "<service>/handler.py", "function": "handle"}
    }
  },
  "actor": {"type": "application", "id": "sentry", "name": "Sentry"},
  "organization": {"slug": "prbe", "name": "PRBE"},
  "project": {"id": "<proj_id>", "slug": "<service>-api", "name": "<Service> API",
               "platform": "python", "team": {"id": "t-1", "slug": "<service>"}}
}
"""

from __future__ import annotations

import hashlib

import orjson

from scripts.synth.output.base import SynthDoc

_SYNTH_INSTALLATION_UUID = "inst-synth-0001"
_SYNTH_ORG_SLUG = "prbe"
_SYNTH_ORG_NAME = "PRBE"
_SENTRY_BASE_URL = "https://sentry.io/organizations/prbe/issues"


def _stable_int(s: str, mod: int) -> int:
    """Deterministic int from a string, stable across processes (vs hash() which is PYTHONHASHSEED-seeded)."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % mod


def _persona_to_user(persona_id: str) -> dict:
    """Derive a stable Sentry user object from a canonical persona id."""
    slug = persona_id.replace("gh:", "").replace("@", "")
    return {
        "type": "user",
        "id": str(_stable_int(persona_id, 10_000)),
        "username": slug,
        "email": f"{slug}@example.com",
    }


def _title_from_text(text: str) -> str:
    """Return the first non-empty line of text, truncated to 200 chars."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return "UnknownError"


def _culprit_from_doc(doc: SynthDoc) -> str:
    """Derive a culprit string like 'payments.handler.handle_event' from service + doc id."""
    service = doc.services_mentioned[0] if doc.services_mentioned else "app"
    # Replace hyphens for Python module-style culprit strings.
    module = service.replace("-", "_")
    return f"{module}.handler.handle_event"


def _issue_id_from_doc(doc: SynthDoc) -> str:
    """Derive a stable numeric issue id string from source_event_id using sha256."""
    return str(_stable_int(doc.source_event_id, 9_000_000_000) + 1_000_000_000)


def _short_id_from_doc(doc: SynthDoc) -> str:
    """Derive a stable short id like 'PRBE-N' from source_event_id using sha256."""
    n = _stable_int(doc.source_event_id, 900) + 100
    return f"PRBE-{n}"


def _project_from_doc(doc: SynthDoc) -> dict:
    """Build a project object from the first service mentioned."""
    service = doc.services_mentioned[0] if doc.services_mentioned else "app"
    slug = f"{service}-api"
    proj_id = str(_stable_int(slug, 9000) + 1000)
    return {
        "id": proj_id,
        "slug": slug,
        "name": f"{service.replace('-', ' ').title()} API",
        "platform": "python",
        "team": {"id": "t-1", "slug": service},
    }


def _iso(doc: SynthDoc) -> str:
    return doc.occurred_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def wrap(doc: SynthDoc) -> bytes:
    """Produce a Sentry issue_created webhook envelope as JSON bytes.

    Always emits action=created in the issue_created shape regardless of
    archetype — this is the only envelope type produced by the templated
    INCIDENT path in Plan 3 spec §7.2.
    """
    issue_id = _issue_id_from_doc(doc)
    short_id = _short_id_from_doc(doc)
    title = _title_from_text(doc.text)
    culprit = _culprit_from_doc(doc)
    iso = _iso(doc)
    project = _project_from_doc(doc)
    assigned_to = _persona_to_user(doc.personas[0]) if doc.personas else None
    service = doc.services_mentioned[0] if doc.services_mentioned else "app"

    issue_url = f"{_SENTRY_BASE_URL}/{issue_id}/"

    issue_obj: dict = {
        "id": issue_id,
        "shortId": short_id,
        "title": title,
        "culprit": culprit,
        "level": "error",
        "status": "unresolved",
        "platform": "python",
        "firstSeen": iso,
        "lastSeen": iso,
        "url": issue_url,
        "permalink": issue_url,
        "project": {k: v for k, v in project.items() if k != "team"},
        "metadata": {
            "type": "Error",
            "filename": f"{service.replace('-', '_')}/handler.py",
            "function": "handle_event",
        },
    }
    if assigned_to is not None:
        issue_obj["assignedTo"] = assigned_to

    payload = {
        "action": "created",
        "installation": {"uuid": _SYNTH_INSTALLATION_UUID},
        "data": {"issue": issue_obj},
        "actor": {"type": "application", "id": "sentry", "name": "Sentry"},
        "organization": {"slug": _SYNTH_ORG_SLUG, "name": _SYNTH_ORG_NAME},
        "project": project,
    }
    return orjson.dumps(payload)
