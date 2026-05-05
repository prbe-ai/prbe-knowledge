"""ON_CALL_HANDOFF archetype builder tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scripts.synth.archetypes.base import Source
from scripts.synth.archetypes.oncall import (
    ON_CALL_HANDOFF,
    _notion_body,
    _person_slug,
    build_oncall_specs,
)
from scripts.synth.ownership import OwnershipIndex
from scripts.synth.scenarios import TimeWindow
from scripts.synth.world_model import (
    ChannelHint,
    Person,
    RepoSummary,
    SectionHint,
    Service,
    ServiceKind,
    Topic,
    TopicKind,
    WorldModel,
)


def _build_oncall_world(
    *,
    n_people: int = 4,
    n_issues: int = 3,
    window_end: datetime | None = None,
) -> WorldModel:
    now = window_end or datetime(2026, 5, 4, 12, 0, 0, tzinfo=UTC)  # Monday
    people = tuple(
        Person(
            canonical_id=f"gh:eng{i}",
            gh_username=f"eng{i}",
            display_name=f"Engineer {i}",
            email_aliases=(f"eng{i}@example.com",),
            role_hint=None,
            repos_active_in=("github.com/prbe-ai/prbe",),
            activity_score=float(n_people - i),
        )
        for i in range(n_people)
    )
    # Some ISSUE topics in last week, some COMMIT topics
    issues = tuple(
        Topic(
            text=f"payments-api returning 500s run {j}",
            kind=TopicKind.ISSUE,
            repo_url="github.com/prbe-ai/prbe",
            ts=now - timedelta(days=j + 1),
            mentioned_services=("payments-api",),
            mentioned_people=(),
            weight=1.0 / (j + 1),
        )
        for j in range(n_issues)
    )
    channels = (
        ChannelHint(name="#oncall", suggested_topic=None, related_services=()),
    )
    sections = (
        SectionHint(title="Engineering > On-call rotation", related_services=()),
    )
    return WorldModel(
        repos=(RepoSummary(url="github.com/prbe-ai/prbe", sha="abc", default_branch="main"),),
        people=people,
        services=(Service(name="payments-api", qualified="payments-api",
                          repo_url="github.com/prbe-ai/prbe", kind=ServiceKind.API,
                          description=None, owners=(), recent_activity=5.0, deploy_target=None),),
        topic_pool=issues,
        channels=channels,
        notion_sections=sections,
        time_anchors=(),
        dep_graph=(),
        company_name="prbe",
        seed=42,
        extracted_at=now,
        sha_set={"github.com/prbe-ai/prbe": "abc"},
    )


def _make_ownership(world: WorldModel) -> OwnershipIndex:
    services_by_person = {p.canonical_id: ("payments-api",) for p in world.people}
    people_by_service = {"payments-api": tuple(p.canonical_id for p in world.people)}
    return OwnershipIndex(services_by_person=services_by_person, people_by_service=people_by_service)


def test_oncall_archetype_metadata() -> None:
    assert ON_CALL_HANDOFF.name == "ON_CALL_HANDOFF"
    assert ON_CALL_HANDOFF.cadence.value == "weekly"
    assert Source.SLACK in ON_CALL_HANDOFF.sources_used
    assert Source.NOTION in ON_CALL_HANDOFF.sources_used


def test_one_monday_produces_one_scenario_with_three_docs() -> None:
    """One Monday in window → 1 ScenarioSpec with 3 DocSpecs (slack-0, slack-1, notion-0)."""
    # Window: exactly one Monday (2026-04-27)
    world = _build_oncall_world(window_end=datetime(2026, 4, 28, tzinfo=UTC))
    ownership = _make_ownership(world)
    window = TimeWindow(end=datetime(2026, 4, 28, tzinfo=UTC), days=7)
    specs = build_oncall_specs(world, ownership, window, seed=42, top_n=4)
    assert len(specs) == 1
    assert len(specs[0].doc_specs) == 3


def test_doc_spec_sources() -> None:
    """Two Slack docs and one Notion doc per handoff."""
    world = _build_oncall_world(window_end=datetime(2026, 4, 28, tzinfo=UTC))
    ownership = _make_ownership(world)
    window = TimeWindow(end=datetime(2026, 4, 28, tzinfo=UTC), days=7)
    specs = build_oncall_specs(world, ownership, window, seed=42, top_n=4)
    sources = [d.source for d in specs[0].doc_specs]
    assert sources.count(Source.SLACK) == 2
    assert sources.count(Source.NOTION) == 1


def test_thread_parent_id_wiring() -> None:
    """Slack reply (slack-1) has thread_parent_id == slack-0's id."""
    world = _build_oncall_world(window_end=datetime(2026, 4, 28, tzinfo=UTC))
    ownership = _make_ownership(world)
    window = TimeWindow(end=datetime(2026, 4, 28, tzinfo=UTC), days=7)
    specs = build_oncall_specs(world, ownership, window, seed=42, top_n=4)
    slack_docs = [d for d in specs[0].doc_specs if d.source == Source.SLACK]
    parent = slack_docs[0]
    reply = slack_docs[1]
    assert reply.thread_parent_id == parent.id


def test_zero_incident_week_emits_quiet_week_text() -> None:
    """When no issues in lookback window, handoff text contains 'Quiet week'."""
    # Build world with no topics in window range
    now = datetime(2026, 4, 28, tzinfo=UTC)
    world = _build_oncall_world(window_end=now, n_issues=0)
    ownership = _make_ownership(world)
    window = TimeWindow(end=now, days=7)
    specs = build_oncall_specs(world, ownership, window, seed=42, top_n=4)
    if specs:
        slack_doc = next(d for d in specs[0].doc_specs if d.source == Source.SLACK)
        assert "Quiet week" in slack_doc.text


def test_person_slug_empty_guard_no_bare_at_mention() -> None:
    """Malformed canonical_id 'email:@example.com' must not produce a bare '@' mention.

    Before the fix, _person_slug returned "" and _notion_body emitted "Outgoing: @".
    After the fix, the raw canonical_id is used as a fallback instead.
    """
    from datetime import date as _date
    body = _notion_body(_date(2026, 5, 4), [], "email:@example.com")
    mention_line = body.split("\n")[1]
    # Must NOT be the broken empty-slug form.
    assert mention_line != "Outgoing: @"
    # Must contain the raw canonical_id as fallback.
    assert "email:@example.com" in mention_line


def test_person_slug_strips_gh_prefix() -> None:
    assert _person_slug("gh:alice") == "alice"


def test_person_slug_strips_email_prefix_to_local_part() -> None:
    assert _person_slug("email:alice@example.com") == "alice"


def test_person_slug_returns_canonical_id_when_no_prefix() -> None:
    assert _person_slug("alice") == "alice"


def test_person_slug_falls_back_to_canonical_when_local_part_is_empty() -> None:
    """Malformed canonical_id like 'email:@example.com' returns full canonical_id (the empty-slug guard)."""
    assert _person_slug("email:@example.com") == "email:@example.com"


def test_rotation_determinism() -> None:
    """Same inputs always produce the same outgoing/incoming pair."""
    world = _build_oncall_world(window_end=datetime(2026, 5, 4, tzinfo=UTC))
    ownership = _make_ownership(world)
    window = TimeWindow(end=datetime(2026, 5, 4, tzinfo=UTC), days=14)
    specs1 = build_oncall_specs(world, ownership, window, seed=42, top_n=4)
    specs2 = build_oncall_specs(world, ownership, window, seed=42, top_n=4)
    for s1, s2 in zip(specs1, specs2, strict=True):
        assert s1.cast == s2.cast
