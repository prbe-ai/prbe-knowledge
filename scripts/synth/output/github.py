"""GitHubWrapper — serialize a SynthDoc to a GitHub webhook envelope.

Supports two event shapes based on the archetype + text heuristic:
  - pull_request.opened  (INCIDENT post-mortem PR, LAUNCH announcement PR)
  - issues.opened        (BIG_REFACTOR RFC, BIG_REFACTOR migration ticket)

Shape selection heuristic (_event_kind_for_doc):
  - BIG_REFACTOR archetype → "issues" (RFC and migration tickets both land as issues)
  - All other archetypes (INCIDENT, LAUNCH, ...) → "pull_request"

Both shapes match the fixture envelopes at:
  fixtures/github/pr_opened.json
  fixtures/github/issue_opened.json

The wrapper never imports connector code directly — round-trip correctness
is validated by checking the JSON shape against the fixture files in tests.
"""

from __future__ import annotations

import hashlib

import orjson

from scripts.synth.output.base import SynthDoc


def _stable_int(s: str, mod: int) -> int:
    """Deterministic int from a string, stable across processes (vs hash() which is PYTHONHASHSEED-seeded)."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % mod

_SYNTH_REPO_OWNER = "prbe"
_SYNTH_INSTALLATION_ID = 99001


def _event_kind_for_doc(doc: SynthDoc) -> str:
    """Return "pull_request" or "issues" based on archetype and text content.

    BIG_REFACTOR scenarios (both RFC texts starting with 'RFC:' and plain
    migration-ticket texts) are routed to the issues shape because GitHub
    issues model the planning/tracking lifecycle better than PRs for this
    archetype.

    All other archetypes (INCIDENT post-mortems, LAUNCH announcements)
    become pull_request.opened events.
    """
    if doc.archetype == "BIG_REFACTOR":
        return "issues"
    return "pull_request"


def _user_from_doc(doc: SynthDoc) -> dict:
    """Build a minimal GitHub user object from the first persona."""
    login = doc.personas[0].replace("gh:", "") if doc.personas else "synth-bot"
    return {"login": login, "id": _stable_int(login, 100_000)}


def _repo_from_doc(doc: SynthDoc) -> dict:
    """Build a minimal repository object from the first service mentioned."""
    repo_name = doc.services_mentioned[0] if doc.services_mentioned else "synth-repo"
    return {
        "id": _stable_int(repo_name, 100_000),
        "name": repo_name,
        "full_name": f"{_SYNTH_REPO_OWNER}/{repo_name}",
        "private": False,
        "html_url": f"https://github.com/{_SYNTH_REPO_OWNER}/{repo_name}",
        "owner": {"login": _SYNTH_REPO_OWNER, "id": 1},
    }


def _title_from_doc(doc: SynthDoc) -> str:
    """Extract a one-line title from the doc text (first non-empty line)."""
    for line in doc.text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return doc.id


def _number_from_doc(doc: SynthDoc) -> int:
    """Derive a stable pseudo-issue/PR number from the doc id."""
    return _stable_int(doc.source_event_id, 9000) + 1


def _iso(doc: SynthDoc) -> str:
    return doc.occurred_at.strftime("%Y-%m-%dT%H:%M:%SZ")


def _wrap_pull_request(doc: SynthDoc) -> dict:
    user = _user_from_doc(doc)
    number = _number_from_doc(doc)
    branch = doc.source_event_id.replace(":", "-").replace("_", "-")
    return {
        "action": "opened",
        "number": number,
        "pull_request": {
            "id": _stable_int(doc.id, 1_000_000),
            "number": number,
            "state": "open",
            "title": _title_from_doc(doc),
            "body": doc.text,
            "html_url": (
                f"https://github.com/{_SYNTH_REPO_OWNER}/"
                f"{doc.services_mentioned[0] if doc.services_mentioned else 'synth-repo'}"
                f"/pull/{number}"
            ),
            "created_at": _iso(doc),
            "updated_at": _iso(doc),
            "user": user,
            "base": {"ref": "main", "sha": "0000000"},
            "head": {"ref": branch[:60], "sha": "fffffff"},
            "changed_files": 1,
            "additions": 10,
            "deletions": 2,
            "merged": False,
        },
        "repository": _repo_from_doc(doc),
        "sender": user,
        "installation": {"id": _SYNTH_INSTALLATION_ID},
    }


def _wrap_issue(doc: SynthDoc) -> dict:
    user = _user_from_doc(doc)
    number = _number_from_doc(doc)
    return {
        "action": "opened",
        "issue": {
            "id": _stable_int(doc.id, 1_000_000),
            "number": number,
            "state": "open",
            "title": _title_from_doc(doc),
            "body": doc.text,
            "html_url": (
                f"https://github.com/{_SYNTH_REPO_OWNER}/"
                f"{doc.services_mentioned[0] if doc.services_mentioned else 'synth-repo'}"
                f"/issues/{number}"
            ),
            "created_at": _iso(doc),
            "updated_at": _iso(doc),
            "user": user,
            "labels": [],
        },
        "repository": _repo_from_doc(doc),
        "sender": user,
        "installation": {"id": _SYNTH_INSTALLATION_ID},
    }


def wrap(doc: SynthDoc) -> bytes:
    """Produce a GitHub webhook envelope as JSON bytes.

    Dispatches to pull_request.opened or issues.opened based on
    _event_kind_for_doc(doc).
    """
    kind = _event_kind_for_doc(doc)
    payload = _wrap_issue(doc) if kind == "issues" else _wrap_pull_request(doc)
    return orjson.dumps(payload)
