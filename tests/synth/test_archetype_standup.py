"""STANDUP_UPDATE archetype builder tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scripts.synth.archetypes.base import Source
from scripts.synth.archetypes.standup import STANDUP_UPDATE, build_standup_specs
from scripts.synth.ownership import OwnershipIndex
from scripts.synth.world_model import (
    ChannelHint,
    Person,
    RepoSummary,
    Service,
    ServiceKind,
    Topic,
    TopicKind,
    WorldModel,
)


def _build_test_world(
    *,
    n_people: int = 3,
    n_topics: int = 5,
    window_end: datetime | None = None,
) -> WorldModel:
    now = window_end or datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
    people = tuple(
        Person(
            canonical_id=f"gh:person{i}",
            gh_username=f"person{i}",
            display_name=f"Person {i}",
            email_aliases=(f"person{i}@example.com",),
            role_hint=None,
            repos_active_in=("github.com/prbe-ai/prbe",),
            activity_score=float(n_people - i),
        )
        for i in range(n_people)
    )
    services = (
        Service(name="payments-api", qualified="payments-api",
                repo_url="github.com/prbe-ai/prbe", kind=ServiceKind.API,
                description=None, owners=(), recent_activity=5.0, deploy_target=None),
        Service(name="auth-service", qualified="auth-service",
                repo_url="github.com/prbe-ai/prbe", kind=ServiceKind.API,
                description=None, owners=(), recent_activity=3.0, deploy_target=None),
    )
    topics = tuple(
        Topic(
            text=f"fix issue in payments-api #{j}",
            kind=TopicKind.COMMIT,
            repo_url="github.com/prbe-ai/prbe",
            ts=now - timedelta(days=j),
            mentioned_services=("payments-api",),
            mentioned_people=(),
            weight=1.0 / (j + 1),
        )
        for j in range(n_topics)
    )
    channels = (ChannelHint(name="#standup", suggested_topic=None, related_services=()),)
    return WorldModel(
        repos=(RepoSummary(url="github.com/prbe-ai/prbe", sha="abc", default_branch="main"),),
        people=people,
        services=services,
        topic_pool=topics,
        channels=channels,
        notion_sections=(),
        time_anchors=(),
        dep_graph=(),
        company_name="prbe",
        seed=42,
        extracted_at=now,
        sha_set={"github.com/prbe-ai/prbe": "abc"},
    )


def _make_ownership(world: WorldModel) -> OwnershipIndex:
    """Give every person 1 service for testing."""
    services_by_person = {
        p.canonical_id: ("payments-api",)
        for p in world.people
    }
    people_by_service = {
        "payments-api": tuple(p.canonical_id for p in world.people)
    }
    return OwnershipIndex(
        services_by_person=services_by_person,
        people_by_service=people_by_service,
    )


def test_standup_archetype_metadata() -> None:
    assert STANDUP_UPDATE.name == "STANDUP_UPDATE"
    assert STANDUP_UPDATE.cadence.value == "daily"
    assert STANDUP_UPDATE.needs_planner_call is False
    assert Source.SLACK in STANDUP_UPDATE.sources_used


def test_spec_count_matches_working_days_times_personas() -> None:
    """30-day window has 22 working days x 3 people = 66 specs (approx)."""
    world = _build_test_world(n_people=3)
    ownership = _make_ownership(world)
    end = datetime(2026, 5, 1, tzinfo=UTC)
    # Use a known 5-working-day window: Mon 2026-04-27 to Fri 2026-05-01.
    from scripts.synth.scenarios import TimeWindow
    window = TimeWindow(end=end, days=7)
    specs = build_standup_specs(world, ownership, window, seed=42, top_n=3)
    # 5 working days x 3 personas = 15 specs
    assert len(specs) == 15


def test_each_spec_has_one_doc_spec() -> None:
    world = _build_test_world(n_people=2)
    ownership = _make_ownership(world)
    from scripts.synth.scenarios import TimeWindow
    window = TimeWindow(end=datetime(2026, 4, 28, tzinfo=UTC), days=3)
    specs = build_standup_specs(world, ownership, window, seed=42, top_n=2)
    for spec in specs:
        assert len(spec.doc_specs) == 1
        assert spec.doc_specs[0].source == Source.SLACK


def test_determinism() -> None:
    """Same inputs produce identical output."""
    world = _build_test_world(n_people=2)
    ownership = _make_ownership(world)
    from scripts.synth.scenarios import TimeWindow
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=7)
    specs1 = build_standup_specs(world, ownership, window, seed=42, top_n=2)
    specs2 = build_standup_specs(world, ownership, window, seed=42, top_n=2)
    assert len(specs1) == len(specs2)
    for s1, s2 in zip(specs1, specs2, strict=True):
        assert s1.id == s2.id
        assert s1.doc_specs[0].text == s2.doc_specs[0].text


def test_person_without_services_skipped() -> None:
    world = _build_test_world(n_people=3)
    # Only person0 has services; others are empty.
    ownership = OwnershipIndex(
        services_by_person={"gh:person0": ("payments-api",)},
        people_by_service={"payments-api": ("gh:person0",)},
    )
    from scripts.synth.scenarios import TimeWindow
    window = TimeWindow(end=datetime(2026, 4, 28, tzinfo=UTC), days=3)
    specs = build_standup_specs(world, ownership, window, seed=42, top_n=3)
    # Only person0 generates specs (1 working day x 1 person)
    personas_seen = {spec.cast[0] for spec in specs}
    assert personas_seen == {"gh:person0"}


def test_text_template_renders() -> None:
    world = _build_test_world(n_people=1)
    ownership = _make_ownership(world)
    from scripts.synth.scenarios import TimeWindow
    window = TimeWindow(end=datetime(2026, 4, 28, tzinfo=UTC), days=2)
    specs = build_standup_specs(world, ownership, window, seed=42, top_n=1)
    assert len(specs) >= 1
    text = specs[0].doc_specs[0].text
    assert "Yesterday:" in text
    assert "Today:" in text
    assert "Blockers:" in text
