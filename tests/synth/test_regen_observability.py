"""Pin the structured-log field shapes for regen observability.

These events feed dashboards and operator runbooks; renaming a field is a
breaking change. This test asserts on the exact field set + types so a
careless rename will fail CI.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from structlog.testing import capture_logs

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
from scripts.synth.ownership import OwnershipIndex
from scripts.synth.profile import Profile
from scripts.synth.scenarios import TimeWindow, run_scenarios
from scripts.synth.validator import CombinedValidatorResult, Violation
from scripts.synth.world_model import (
    ChannelHint,
    Person,
    RepoSummary,
    SectionHint,
    Service,
    ServiceKind,
    WorldModel,
)

# Self-contained fixture helpers. We deliberately do NOT import private
# helpers from tests/synth/test_scenarios.py — this file should fail loudly
# on its own if a shape it relies on changes.


def _profile(extras: dict | None = None) -> Profile:
    raw = {
        "customer_id": "cust-eval-test",
        "preset": "tiny_test",
        "seed": 1,
        "repos": [],
    }
    if extras:
        raw.update(extras)
    return Profile(
        customer_id=raw["customer_id"],
        repos=(),
        preset=raw["preset"],
        seed=raw["seed"],
        raw=raw,
    )


def _build_test_world() -> WorldModel:
    alice = Person(
        canonical_id="gh:alice",
        gh_username="alice",
        display_name="Alice",
        email_aliases=(),
        role_hint="backend",
        repos_active_in=(),
        activity_score=10.0,
    )
    svc = Service(
        name="payments",
        qualified="payments",
        repo_url="https://github.com/acme/payments",
        kind=ServiceKind.API,
        description="x",
        owners=("gh:alice",),
        recent_activity=5.0,
        deploy_target=None,
    )
    return WorldModel(
        repos=(
            RepoSummary(
                url="https://github.com/acme/payments",
                sha="abc",
                default_branch="main",
            ),
        ),
        people=(alice,),
        services=(svc,),
        topic_pool=(),
        channels=(
            ChannelHint(
                name="#incidents",
                suggested_topic=None,
                related_services=(),
            ),
        ),
        notion_sections=(
            SectionHint(title="Postmortems", related_services=()),
        ),
        time_anchors=(),
        dep_graph=(),
        company_name="Acme",
        seed=1,
        extracted_at=datetime(2026, 4, 1, tzinfo=UTC),
    )


def _ownership_full() -> OwnershipIndex:
    return OwnershipIndex(
        services_by_person={"gh:alice": ("payments",)},
        people_by_service={"payments": ("gh:alice",)},
    )


def _make_arch() -> Archetype:
    return Archetype(
        name="INCIDENT",
        category=Category.PLOT,
        cadence=Cadence.AD_HOC,
        validator_level=ValidatorLevel.STRICT,
        needs_planner_call=True,
        sources_used=(Source.SLACK,),
        cast_size=(1, 3),
        spec_template_path=None,
    )


def _make_scenario_spec() -> ScenarioSpec:
    ts = datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC)
    return ScenarioSpec(
        id="scn-obs-1",
        archetype_name="INCIDENT",
        instance_ts=ts,
        cast=("gh:alice",),
        affected_services=("payments",),
        doc_specs=(
            DocSpec(
                id="d0",
                source=Source.SLACK,
                occurred_at=ts,
                channel="#incidents",
                page_section=None,
                text="",
                thread_parent_id=None,
                personas=("gh:alice",),
                services_mentioned=("payments",),
            ),
        ),
        title="x",
        summary="y",
        root_cause="z",
        eval_questions=(),
    )


def _make_failing_doc() -> SynthDoc:
    return SynthDoc(
        id="d0",
        source=Source.SLACK,
        source_event_id="d0",
        text="auto-scaling broke",
        occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        channel="#incidents",
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-obs-1",
        archetype="INCIDENT",
        personas=("gh:alice",),
        services_mentioned=("payments",),
        priority=10,
    )


@pytest.mark.asyncio
async def test_regen_round_log_shape(monkeypatch) -> None:
    """plot_scenario_regen_round must include scenario_id, archetype, round
    (int), failing_doc_ids (list), passing_doc_ids (list), violation_reasons
    (list).
    """
    spec = _make_scenario_spec()
    doc = _make_failing_doc()
    arch = _make_arch()

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

    # Validator: fail calls #1 and #2 (outer + regen-round-1), pass call #3.
    call_count = {"n": 0}

    async def fake_validate(*_a, **_kw):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return CombinedValidatorResult(
                pass1_violations=(
                    Violation(doc_id="d0", out_of_world=("auto-scaling",)),
                ),
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

    mock_writer = MagicMock()
    mock_writer.regenerate = AsyncMock(return_value="payments service errors")

    profile = _profile({"archetypes": {"INCIDENT": {"count": 1}}})

    with capture_logs() as captured:
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
            pass

    round_events = [
        e for e in captured if e.get("event") == "plot_scenario_regen_round"
    ]
    assert len(round_events) == 1, f"expected 1 round event, got {round_events}"
    e = round_events[0]
    assert e["scenario_id"] == "scn-obs-1"
    assert e["archetype"] == "INCIDENT"
    assert isinstance(e["round"], int)
    assert e["round"] == 1
    assert isinstance(e["failing_doc_ids"], list)
    assert "d0" in e["failing_doc_ids"]
    assert isinstance(e["passing_doc_ids"], list)
    assert isinstance(e["violation_reasons"], list)


@pytest.mark.asyncio
async def test_regen_terminal_drop_log_shape(monkeypatch) -> None:
    """plot_scenario_dropped_after_regen must include scenario_id, archetype,
    rounds_attempted (int), survived_doc_ids (list), never_converged_doc_ids
    (list).
    """
    spec = _make_scenario_spec()
    doc = _make_failing_doc()
    arch = _make_arch()

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

    # Validator: never passes
    async def fake_validate(*_a, **_kw):
        return CombinedValidatorResult(
            pass1_violations=(
                Violation(doc_id="d0", out_of_world=("auto-scaling",)),
            ),
            pass2_result=None,
            failing_doc_ids=("d0",),
            should_drop=True,
        )

    monkeypatch.setattr("scripts.synth.validator.validate", fake_validate)

    mock_writer = MagicMock()
    mock_writer.regenerate = AsyncMock(return_value="still has auto-scaling")

    profile = _profile({"archetypes": {"INCIDENT": {"count": 1}}})

    with capture_logs() as captured:
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
            pass

    drop_events = [
        e
        for e in captured
        if e.get("event") == "plot_scenario_dropped_after_regen"
    ]
    assert len(drop_events) == 1, f"expected 1 drop event, got {drop_events}"
    e = drop_events[0]
    assert e["scenario_id"] == "scn-obs-1"
    assert e["archetype"] == "INCIDENT"
    assert isinstance(e["rounds_attempted"], int)
    assert e["rounds_attempted"] == 3
    assert isinstance(e["survived_doc_ids"], list)
    assert isinstance(e["never_converged_doc_ids"], list)
    assert "d0" in e["never_converged_doc_ids"]
