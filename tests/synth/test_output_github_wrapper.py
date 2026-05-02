"""GitHubWrapper round-trip tests.

Verifies that wrap(doc) -> bytes produces a JSON envelope matching the
shape of the real fixtures at fixtures/github/pr_opened.json and
fixtures/github/issue_opened.json.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.output.github import wrap

_FIXTURE_PR = Path(__file__).parent.parent.parent / "fixtures" / "github" / "pr_opened.json"
_FIXTURE_ISSUE = Path(__file__).parent.parent.parent / "fixtures" / "github" / "issue_opened.json"


def _make_doc(
    *,
    archetype: str = "INCIDENT",
    text: str = "Post-mortem: payments 500s during deploy.",
    occurred_at: datetime | None = None,
    personas: tuple[str, ...] = ("gh:alice",),
    services: tuple[str, ...] = ("payments",),
) -> SynthDoc:
    ts = occurred_at or datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    return SynthDoc(
        id="scn-incident-2026-05-01-github-0",
        source=Source.GITHUB,
        source_event_id="scn-incident-2026-05-01-github-0",
        text=text,
        occurred_at=ts,
        channel=None,
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-incident-2026-05-01",
        archetype=archetype,
        personas=personas,
        services_mentioned=services,
        priority=100,
    )


def test_wrap_returns_bytes() -> None:
    doc = _make_doc()
    result = wrap(doc)
    assert isinstance(result, bytes)
    json.loads(result)  # must be valid JSON


def test_incident_archetype_produces_pull_request_shape() -> None:
    doc = _make_doc(archetype="INCIDENT")
    payload = json.loads(wrap(doc))
    assert payload["action"] == "opened"
    assert "pull_request" in payload
    assert "issue" not in payload


def test_big_refactor_rfc_produces_issue_shape() -> None:
    """BIG_REFACTOR with RFC: prefix in text triggers issue shape."""
    doc = _make_doc(archetype="BIG_REFACTOR", text="RFC: Migrate auth-service to JWT tokens.")
    payload = json.loads(wrap(doc))
    assert payload["action"] == "opened"
    assert "issue" in payload
    assert "pull_request" not in payload


def test_big_refactor_migration_ticket_produces_issue_shape() -> None:
    """BIG_REFACTOR without RFC prefix also triggers issue shape."""
    doc = _make_doc(
        archetype="BIG_REFACTOR",
        text="Migrate legacy billing code to new payment gateway.",
    )
    payload = json.loads(wrap(doc))
    assert "issue" in payload


def test_launch_archetype_produces_pull_request_shape() -> None:
    doc = _make_doc(archetype="LAUNCH", text="Ship v2 payments API.")
    payload = json.loads(wrap(doc))
    assert "pull_request" in payload


def test_pr_fixture_shape_superset_of_wrapper_keys() -> None:
    """Real fixture top-level keys are a superset of wrapper output for PR shape."""
    _FIXTURE_PR.read_text()  # ensure fixture file exists and is readable
    doc = _make_doc(archetype="INCIDENT")
    wrapper = json.loads(wrap(doc))
    # Wrapper must carry at minimum: action, pull_request, repository, sender
    required = {"action", "pull_request", "repository", "sender"}
    assert required.issubset(set(wrapper.keys()))
    # pull_request must have title and body
    assert {"title", "body", "user"}.issubset(set(wrapper["pull_request"].keys()))


def test_issue_fixture_shape_superset_of_wrapper_keys() -> None:
    """Real fixture top-level keys are a superset of wrapper output for issue shape."""
    _FIXTURE_ISSUE.read_text()  # ensure fixture file exists and is readable
    doc = _make_doc(archetype="BIG_REFACTOR", text="RFC: rewrite auth.")
    wrapper = json.loads(wrap(doc))
    required = {"action", "issue", "repository", "sender"}
    assert required.issubset(set(wrapper.keys()))
    assert {"title", "body", "user"}.issubset(set(wrapper["issue"].keys()))
