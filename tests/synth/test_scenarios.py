"""Tests for the scenario runner: TimeWindow + working_days + weekly_mondays + run_scenarios."""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

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
        "preset": "tiny_test",
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

async def test_run_scenarios_yields_docs_for_full_library() -> None:
    world = _build_test_world()
    own = _ownership_full()
    p = _profile()
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=14)
    docs = [doc async for _, doc in run_scenarios(world, own, p, window)]
    # Expect both archetypes to produce docs (count varies; just assert > 0).
    sources = {d.source for d in docs}
    assert Source.SLACK in sources
    assert Source.NOTION in sources


async def test_run_scenarios_archetype_filter_restricts_output() -> None:
    world = _build_test_world()
    own = _ownership_full()
    p = _profile()
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=14)
    docs = [
        doc async for _, doc in run_scenarios(
            world, own, p, window, archetype_filter=("STANDUP_UPDATE",)
        )
    ]
    # STANDUP_UPDATE is slack-only; no notion docs.
    assert all(d.source == Source.SLACK for d in docs)
    assert all(d.archetype == "STANDUP_UPDATE" for d in docs)


async def test_run_scenarios_scenario_limit_caps_per_archetype() -> None:
    world = _build_test_world()
    own = _ownership_full()
    p = _profile()
    window = TimeWindow(end=datetime(2026, 5, 1, tzinfo=UTC), days=30)
    docs = [doc async for _, doc in run_scenarios(world, own, p, window, scenario_limit=2)]
    # Each archetype contributes <= 2 scenarios (each scenario yields 1+ docs).
    standup_scenarios = {d.scenario_id for d in docs if d.archetype == "STANDUP_UPDATE"}
    oncall_scenarios = {d.scenario_id for d in docs if d.archetype == "ON_CALL_HANDOFF"}
    assert len(standup_scenarios) <= 2
    assert len(oncall_scenarios) <= 2


# --- regen wiring ---------------------------------------------------------

@pytest.mark.asyncio
async def test_run_scenarios_regen_recovers_failing_plot_doc(monkeypatch) -> None:
    """End-to-end: a plot scenario fails Pass 1 in round 1, succeeds round 2 via regen.

    Asserts the scenario is yielded (not dropped) and that the final doc
    text reflects the regenerated content.
    """
    from scripts.synth.archetypes.base import (
        Archetype,
        Cadence,
        Category,
        DocSpec,
        ScenarioSpec,
        Source,
        ValidatorLevel,
    )
    from scripts.synth.output.base import SynthDoc
    from scripts.synth.scenarios import run_scenarios
    from scripts.synth.validator import CombinedValidatorResult, Violation

    # Build a one-doc plot scenario whose initial text fails Pass 1.
    spec = ScenarioSpec(
        id="scn-incident-1",
        archetype_name="INCIDENT",
        instance_ts=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        cast=("gh:alice",),
        affected_services=("payments",),
        doc_specs=(
            DocSpec(
                id="d0",
                source=Source.SLACK,
                occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
                channel="#incidents",
                page_section=None,
                text="",
                thread_parent_id=None,
                personas=("gh:alice",),
                services_mentioned=("payments",),
            ),
        ),
        title="x", summary="y", root_cause="z", eval_questions=(),
    )

    failing_doc = SynthDoc(
        id="d0",
        source=Source.SLACK,
        source_event_id="d0",
        text="auto-scaling broke",
        occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        channel="#incidents",
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-incident-1",
        archetype="INCIDENT",
        personas=("gh:alice",),
        services_mentioned=("payments",),
        priority=10,
    )

    # IMPORTANT: Archetype shape — see Task 6's findings. Use the REAL field
    # names: sources_used, cast_size, spec_template_path. Cadence has AD_HOC,
    # not RARE. eval_question_count defaults to 0.
    incident_archetype = Archetype(
        name="INCIDENT",
        category=Category.PLOT,
        cadence=Cadence.AD_HOC,
        validator_level=ValidatorLevel.STRICT,
        needs_planner_call=True,
        sources_used=(Source.SLACK,),
        cast_size=(1, 3),
        spec_template_path=None,
    )

    async def fake_plot_builder(**kwargs):
        yield spec, [failing_doc]

    monkeypatch.setattr(
        "scripts.synth.archetypes.library.PLOT_BUILDERS",
        {"INCIDENT": fake_plot_builder},
    )
    monkeypatch.setattr(
        "scripts.synth.archetypes.library.get_active",
        lambda profile, archetype_filter=None: {"INCIDENT": incident_archetype},
    )

    # Mock validator: fail outer + regen_loop's first internal call, pass after one
    # regeneration. (run_scenarios validates once before entering regen_loop, and
    # regen_loop validates again at the start of round 1; both must fail for the
    # writer.regenerate path to be exercised. Then round-2 validation must pass.)
    call_count = {"n": 0}

    async def fake_validate(docs, world, *, scenario, archetype, pass2_client, pass2_model, **_kw):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return CombinedValidatorResult(
                pass1_violations=(Violation(doc_id="d0", out_of_world=("auto-scaling",)),),
                pass2_result=None,
                failing_doc_ids=("d0",),
                should_drop=True,
            )
        return CombinedValidatorResult(
            pass1_violations=(),
            pass2_result=None,
            failing_doc_ids=(),
            should_drop=False,
        )

    monkeypatch.setattr("scripts.synth.validator.validate", fake_validate)

    # Mock writer with .regenerate that returns clean text
    mock_writer = MagicMock()
    mock_writer.regenerate = AsyncMock(return_value="payments service had errors")
    mock_planner = MagicMock()

    yielded: list[tuple] = []
    profile = _profile({"archetypes": {"INCIDENT": {"count": 1}}})
    company_ctx = MagicMock()
    world = _build_test_world()
    ownership = _ownership_full()
    time_window = TimeWindow(end=datetime(2026, 4, 13, tzinfo=UTC), days=7)

    async for s, doc in run_scenarios(
        world=world,
        ownership=ownership,
        profile=profile,
        time_window=time_window,
        company_ctx=company_ctx,
        planner=mock_planner,
        writer=mock_writer,
        validator_pass2_client=None,
        validator_pass2_model=None,
    ):
        yielded.append((s, doc))

    assert len(yielded) == 1
    assert yielded[0][1].text == "payments service had errors"
    assert mock_writer.regenerate.await_count == 1


@pytest.mark.asyncio
async def test_run_scenarios_regen_disabled_drops_immediately(monkeypatch) -> None:
    """When regen_enabled=False, a failing plot scenario drops on round 1
    without calling writer.regenerate."""
    from scripts.synth.archetypes.base import (
        Archetype,
        Cadence,
        Category,
        DocSpec,
        ScenarioSpec,
        Source,
        ValidatorLevel,
    )
    from scripts.synth.output.base import SynthDoc
    from scripts.synth.scenarios import run_scenarios
    from scripts.synth.validator import CombinedValidatorResult, Violation

    spec = ScenarioSpec(
        id="scn-1", archetype_name="INCIDENT",
        instance_ts=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        cast=("gh:alice",), affected_services=("payments",),
        doc_specs=(DocSpec(
            id="d0", source=Source.SLACK,
            occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
            channel="#incidents", page_section=None, text="",
            thread_parent_id=None, personas=("gh:alice",),
            services_mentioned=("payments",),
        ),),
        title="x", summary="y", root_cause="z", eval_questions=(),
    )
    doc = SynthDoc(
        id="d0", source=Source.SLACK, source_event_id="d0",
        text="auto-scaling", occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        channel="#incidents", page_id=None, thread_parent_id=None,
        scenario_id="scn-1", archetype="INCIDENT", personas=("gh:alice",),
        services_mentioned=("payments",), priority=10,
    )
    arch = Archetype(
        name="INCIDENT", category=Category.PLOT, cadence=Cadence.AD_HOC,
        validator_level=ValidatorLevel.STRICT, needs_planner_call=True,
        sources_used=(Source.SLACK,), cast_size=(1, 3), spec_template_path=None,
    )

    async def fake_plot_builder(**_kwargs):
        yield spec, [doc]

    monkeypatch.setattr(
        "scripts.synth.archetypes.library.PLOT_BUILDERS",
        {"INCIDENT": fake_plot_builder},
    )
    monkeypatch.setattr(
        "scripts.synth.archetypes.library.get_active",
        lambda profile, archetype_filter=None: {"INCIDENT": arch},
    )

    async def fake_validate(*_a, **_kw):
        return CombinedValidatorResult(
            pass1_violations=(Violation(doc_id="d0", out_of_world=("auto-scaling",)),),
            pass2_result=None, failing_doc_ids=("d0",), should_drop=True,
        )

    monkeypatch.setattr("scripts.synth.validator.validate", fake_validate)

    mock_writer = MagicMock()
    mock_writer.regenerate = AsyncMock(return_value="should not be called")

    yielded: list[tuple] = []
    profile = _profile({"archetypes": {"INCIDENT": {"count": 1}}})
    async for _ in run_scenarios(
        world=_build_test_world(),
        ownership=_ownership_full(),
        profile=profile,
        time_window=TimeWindow(end=datetime(2026, 4, 13, tzinfo=UTC), days=7),
        company_ctx=MagicMock(),
        planner=MagicMock(),
        writer=mock_writer,
        validator_pass2_client=None,
        validator_pass2_model=None,
        regen_enabled=False,
    ):
        yielded.append(_)

    assert yielded == []
    assert mock_writer.regenerate.await_count == 0


@pytest.mark.asyncio
async def test_run_scenarios_regen_skips_docs_with_empty_failure_context(monkeypatch) -> None:
    """Rare validator state: failing_doc_ids includes a doc that has no
    Pass 1 nor Pass 2 detail. regen_loop should NOT call writer.regenerate
    for it (saves LLM calls), and the scenario should still drop after
    max_rounds.
    """
    from scripts.synth.archetypes.base import (
        Archetype,
        Cadence,
        Category,
        DocSpec,
        ScenarioSpec,
        Source,
        ValidatorLevel,
    )
    from scripts.synth.output.base import SynthDoc
    from scripts.synth.scenarios import run_scenarios
    from scripts.synth.validator import CombinedValidatorResult

    spec = ScenarioSpec(
        id="scn-empty-ctx-1", archetype_name="INCIDENT",
        instance_ts=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        cast=("gh:alice",), affected_services=("payments",),
        doc_specs=(DocSpec(
            id="d0", source=Source.SLACK,
            occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
            channel="#incidents", page_section=None, text="",
            thread_parent_id=None, personas=("gh:alice",),
            services_mentioned=("payments",),
        ),),
        title="x", summary="y", root_cause="z", eval_questions=(),
    )
    doc = SynthDoc(
        id="d0", source=Source.SLACK, source_event_id="d0",
        text="some text", occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        channel="#incidents", page_id=None, thread_parent_id=None,
        scenario_id="scn-empty-ctx-1", archetype="INCIDENT",
        personas=("gh:alice",), services_mentioned=("payments",), priority=10,
    )
    arch = Archetype(
        name="INCIDENT", category=Category.PLOT, cadence=Cadence.AD_HOC,
        validator_level=ValidatorLevel.STRICT, needs_planner_call=True,
        sources_used=(Source.SLACK,), cast_size=(1, 3), spec_template_path=None,
    )

    async def fake_plot_builder(**_kwargs):
        yield spec, [doc]

    monkeypatch.setattr(
        "scripts.synth.archetypes.library.PLOT_BUILDERS",
        {"INCIDENT": fake_plot_builder},
    )
    monkeypatch.setattr(
        "scripts.synth.archetypes.library.get_active",
        lambda profile, archetype_filter=None: {"INCIDENT": arch},
    )

    # Validator says doc fails but provides NO Pass 1 violations and NO
    # Pass 2 result — so format_failure_context returns "".
    async def fake_validate(*_a, **_kw):
        return CombinedValidatorResult(
            pass1_violations=(),
            pass2_result=None,
            failing_doc_ids=("d0",),
            should_drop=True,
        )

    monkeypatch.setattr("scripts.synth.validator.validate", fake_validate)

    mock_writer = MagicMock()
    mock_writer.regenerate = AsyncMock(return_value="should not be called")

    yielded: list[tuple] = []
    profile = _profile({"archetypes": {"INCIDENT": {"count": 1}}})
    async for _ in run_scenarios(
        world=_build_test_world(),
        ownership=_ownership_full(),
        profile=profile,
        time_window=TimeWindow(end=datetime(2026, 4, 13, tzinfo=UTC), days=7),
        company_ctx=MagicMock(),
        planner=MagicMock(),
        writer=mock_writer,
        validator_pass2_client=None,
        validator_pass2_model=None,
    ):
        yielded.append(_)

    # Drops after max_rounds (default 3). Writer was never asked to regenerate
    # because the failure context was empty.
    assert yielded == []
    assert mock_writer.regenerate.await_count == 0
