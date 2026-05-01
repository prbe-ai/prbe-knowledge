"""Tests for the scenario runner: TimeWindow + working_days + weekly_mondays + run_scenarios."""

from __future__ import annotations

from datetime import UTC, date, datetime

from scripts.synth.archetypes.base import Source
from scripts.synth.ownership import OwnershipIndex
from scripts.synth.profile import Profile
from scripts.synth.scenarios import (
    TimeWindow,
    run_scenarios,
    weekly_mondays,
    working_days,
)
from scripts.synth.world_model import (
    Person,
    Service,
    ServiceKind,
    Topic,
    TopicKind,
    WorldModel,
)


def _profile(raw_extras: dict | None = None) -> Profile:
    raw = {
        "customer_id": "cust-eval-test-01",
        "repos": [{"url": "github.com/x/y", "local_path": "/tmp/y"}],
        "preset": "tiny-test",
        "seed": 7,
    }
    if raw_extras:
        raw.update(raw_extras)
    return Profile(
        customer_id=raw["customer_id"],
        repos=(),
        preset=raw["preset"],
        seed=raw["seed"],
        raw=raw,
    )


def _build_test_world() -> WorldModel:
    """Tiny WorldModel with 3 people, 2 services, 5 topics."""
    people = (
        Person(
            canonical_id="gh:alice",
            gh_username="alice",
            display_name="Alice",
            email_aliases=("alice@x.com",),
            role_hint=None,
            repos_active_in=("github.com/x/y",),
            activity_score=10.0,
        ),
        Person(
            canonical_id="gh:bob",
            gh_username="bob",
            display_name="Bob",
            email_aliases=("bob@x.com",),
            role_hint=None,
            repos_active_in=("github.com/x/y",),
            activity_score=5.0,
        ),
        Person(
            canonical_id="gh:carol",
            gh_username="carol",
            display_name="Carol",
            email_aliases=("carol@x.com",),
            role_hint=None,
            repos_active_in=("github.com/x/y",),
            activity_score=2.0,
        ),
    )
    services = (
        Service(
            name="payments", qualified="payments", repo_url="github.com/x/y",
            kind=ServiceKind.API, description=None, owners=(), recent_activity=1.0,
            deploy_target=None,
        ),
        Service(
            name="billing", qualified="billing", repo_url="github.com/x/y",
            kind=ServiceKind.API, description=None, owners=(), recent_activity=1.0,
            deploy_target=None,
        ),
    )
    topics = (
        Topic(text="fix payments null", kind=TopicKind.COMMIT, repo_url="github.com/x/y",
              ts=datetime(2026, 4, 28, tzinfo=UTC),
              mentioned_services=("payments",), mentioned_people=("gh:alice",), weight=0.8),
        Topic(text="billing rate limit", kind=TopicKind.ISSUE, repo_url="github.com/x/y",
              ts=datetime(2026, 4, 27, tzinfo=UTC),
              mentioned_services=("billing",), mentioned_people=("gh:bob",), weight=0.7),
        Topic(text="payments retry logic", kind=TopicKind.PR, repo_url="github.com/x/y",
              ts=datetime(2026, 4, 26, tzinfo=UTC),
              mentioned_services=("payments",), mentioned_people=("gh:alice",), weight=0.9),
        Topic(text="billing dashboard", kind=TopicKind.PR, repo_url="github.com/x/y",
              ts=datetime(2026, 4, 25, tzinfo=UTC),
              mentioned_services=("billing",), mentioned_people=("gh:carol",), weight=0.6),
        Topic(text="db migration", kind=TopicKind.ISSUE, repo_url="github.com/x/y",
              ts=datetime(2026, 4, 24, tzinfo=UTC),
              mentioned_services=("payments", "billing"), mentioned_people=(), weight=0.5),
    )
    return WorldModel(
        repos=(),
        people=people,
        services=services,
        topic_pool=topics,
        channels=(),
        notion_sections=(),
        time_anchors=(),
        dep_graph=(),
        company_name="acme",
        seed=7,
        extracted_at=datetime(2026, 4, 30, tzinfo=UTC),
        sha_set={},
    )


def _ownership_full() -> OwnershipIndex:
    return OwnershipIndex(
        services_by_person={
            "gh:alice": ("payments",),
            "gh:bob": ("billing",),
            "gh:carol": ("billing",),
        },
        people_by_service={
            "payments": ("gh:alice",),
            "billing": ("gh:bob", "gh:carol"),
        },
    )


# --- working_days / weekly_mondays ----------------------------------------

def test_working_days_excludes_weekends() -> None:
    # Window: Mon Apr 27 -> Fri May 1 (5 weekdays). end is exclusive.
    window = TimeWindow(end=datetime(2026, 5, 2, tzinfo=UTC), days=5)
    days = list(working_days(window))
    assert days == [
        date(2026, 4, 27),
        date(2026, 4, 28),
        date(2026, 4, 29),
        date(2026, 4, 30),
        date(2026, 5, 1),
    ]


def test_working_days_zero_days_yields_empty() -> None:
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=0)
    assert list(working_days(window)) == []


def test_weekly_mondays_finds_mondays_in_window() -> None:
    # 30-day window ending 2026-05-01 spans 2026-04-01 .. 2026-04-30.
    # Mondays in that range: Apr 6, Apr 13, Apr 20, Apr 27 (4 mondays).
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=30)
    mondays = list(weekly_mondays(window))
    assert mondays == [
        date(2026, 4, 6),
        date(2026, 4, 13),
        date(2026, 4, 20),
        date(2026, 4, 27),
    ]


def test_weekly_mondays_window_starting_on_monday() -> None:
    # If start_date itself is a Monday, include it.
    # end=2026-04-14, days=8 → start=2026-04-06 (Monday).
    window = TimeWindow(end=datetime(2026, 4, 14, tzinfo=UTC), days=8)
    mondays = list(weekly_mondays(window))
    assert mondays == [date(2026, 4, 6), date(2026, 4, 13)]


# --- run_scenarios --------------------------------------------------------

def test_run_scenarios_yields_docs_for_full_library() -> None:
    world = _build_test_world()
    own = _ownership_full()
    p = _profile()
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=14)
    docs = list(run_scenarios(world, own, p, window))
    # Expect both archetypes to produce docs (count varies; just assert > 0).
    sources = {d.source for d in docs}
    assert Source.SLACK in sources
    assert Source.NOTION in sources


def test_run_scenarios_archetype_filter_restricts_output() -> None:
    world = _build_test_world()
    own = _ownership_full()
    p = _profile()
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=14)
    docs = list(run_scenarios(world, own, p, window, archetype_filter=("STANDUP_UPDATE",)))
    # STANDUP_UPDATE is slack-only; no notion docs.
    assert all(d.source == Source.SLACK for d in docs)
    assert all(d.archetype == "STANDUP_UPDATE" for d in docs)


def test_run_scenarios_scenario_limit_caps_per_archetype() -> None:
    world = _build_test_world()
    own = _ownership_full()
    p = _profile()
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=30)
    docs = list(run_scenarios(world, own, p, window, scenario_limit=2))
    # Each archetype contributes <= 2 scenarios (each scenario yields 1+ docs).
    standup_scenarios = {d.scenario_id for d in docs if d.archetype == "STANDUP_UPDATE"}
    oncall_scenarios = {d.scenario_id for d in docs if d.archetype == "ON_CALL_HANDOFF"}
    assert len(standup_scenarios) <= 2
    assert len(oncall_scenarios) <= 2
