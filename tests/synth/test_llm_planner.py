"""Tests for LLMPlanner: structured output, world-model validation, retry logic."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.synth.archetypes.base import (
    Archetype,
    Cadence,
    Category,
    Source,
    ValidatorLevel,
)
from scripts.synth.llm.planner import (
    LLMPlanner,
    PlannerValidationError,
)
from scripts.synth.ownership import OwnershipIndex
from scripts.synth.world_model import (
    ChannelHint,
    Person,
    RepoSummary,
    SectionHint,
    Service,
    ServiceKind,
    WorldModel,
)


def _make_world() -> WorldModel:
    alice = Person(
        canonical_id="gh:alice",
        gh_username="alice",
        display_name="Alice",
        email_aliases=("alice@example.com",),
        role_hint="backend",
        repos_active_in=("https://github.com/acme/payments",),
        activity_score=10.0,
    )
    bob = Person(
        canonical_id="gh:bob",
        gh_username="bob",
        display_name="Bob",
        email_aliases=("bob@example.com",),
        role_hint="oncall",
        repos_active_in=("https://github.com/acme/payments",),
        activity_score=8.0,
    )
    svc = Service(
        name="payments",
        qualified="payments",
        repo_url="https://github.com/acme/payments",
        kind=ServiceKind.API,
        description="Payments service",
        owners=("gh:alice",),
        recent_activity=5.0,
        deploy_target=None,
    )
    return WorldModel(
        repos=(RepoSummary(url="https://github.com/acme/payments", sha="abc123", default_branch="main"),),
        people=(alice, bob),
        services=(svc,),
        topic_pool=(),
        channels=(ChannelHint(name="#incidents", suggested_topic=None, related_services=()),),
        notion_sections=(SectionHint(title="Postmortems", related_services=()),),
        time_anchors=(),
        dep_graph=(),
        company_name="Acme",
        seed=42,
        extracted_at=datetime(2026, 4, 1, tzinfo=UTC),
    )


def _make_ownership(world: WorldModel) -> OwnershipIndex:
    from scripts.synth.ownership import build_ownership_index
    return build_ownership_index([], world)


def _make_archetype() -> Archetype:
    return Archetype(
        name="INCIDENT",
        category=Category.PLOT,
        cadence=Cadence.AD_HOC,
        sources_used=(Source.SLACK, Source.NOTION, Source.LINEAR, Source.GITHUB, Source.SENTRY),
        cast_size=(2, 4),
        needs_planner_call=True,
        validator_level=ValidatorLevel.STRICT,
        eval_question_count=2,
        spec_template_path=None,  # not needed when client is mocked
    )


def _valid_planner_output(world: WorldModel) -> dict:
    ts = "2026-04-12T14:30:00+00:00"
    return {
        "title": "payments-svc 500s after feature-flag rollout",
        "summary": "The payments service went down after a bad feature flag.",
        "cast": [
            {"canonical_id": "gh:alice", "role_in_scenario": "fixer"},
            {"canonical_id": "gh:bob", "role_in_scenario": "reporter"},
        ],
        "affected_services": ["payments"],
        "affected_repos": ["https://github.com/acme/payments"],
        "root_cause": "feature flag default flipped on without backend ready",
        "decision": None,
        "outcome": None,
        "timeline": [
            {"ts": ts, "source": "sentry", "kind": "alert", "channel": None},
            {"ts": ts, "source": "slack", "kind": "message", "channel": "#incidents"},
        ],
        "source_emissions": {"sentry": 1, "slack": 1, "linear": 1, "notion": 1, "github": 1},
        "eval_questions": [
            {
                "input": "What caused the payments outage?",
                "answer_substring": "feature flag default flipped on without backend ready",
                "difficulty": "easy",
            },
            {
                "input": "Who reported and who fixed the payments outage?",
                "answer_substring": "gh:bob reported, gh:alice fixed",
                "difficulty": "medium-cross-source",
            },
        ],
    }


@pytest.mark.asyncio
async def test_plan_returns_scenario_spec_on_valid_output() -> None:
    world = _make_world()
    ownership = _make_ownership(world)
    archetype = _make_archetype()

    mock_client = MagicMock()
    mock_client.generate_structured = AsyncMock(return_value=_valid_planner_output(world))

    planner = LLMPlanner(client=mock_client, model="claude-opus-4-7")
    from scripts.synth.company_context import CompanyContext
    company_ctx = CompanyContext(name="Acme", stage="seed", headcount=10)

    spec = await planner.plan(
        archetype=archetype,
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        instance_ts=datetime(2026, 4, 12, 14, 30, tzinfo=UTC),
        rng_seed=42,
    )
    assert spec.archetype_name == "INCIDENT"
    assert "gh:alice" in spec.cast
    assert "gh:bob" in spec.cast


@pytest.mark.asyncio
async def test_plan_doc_specs_derived_from_source_emissions() -> None:
    world = _make_world()
    ownership = _make_ownership(world)
    archetype = _make_archetype()

    mock_client = MagicMock()
    mock_client.generate_structured = AsyncMock(return_value=_valid_planner_output(world))

    planner = LLMPlanner(client=mock_client, model="claude-opus-4-7")
    from scripts.synth.company_context import CompanyContext
    company_ctx = CompanyContext(name="Acme", stage="seed", headcount=10)

    spec = await planner.plan(
        archetype=archetype,
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        instance_ts=datetime(2026, 4, 12, 14, 30, tzinfo=UTC),
        rng_seed=42,
    )
    # source_emissions has 5 entries each with count 1 → 5 doc specs total
    assert len(spec.doc_specs) == 5
    sources_in_specs = {ds.source for ds in spec.doc_specs}
    assert Source.SLACK in sources_in_specs
    assert Source.SENTRY in sources_in_specs


@pytest.mark.asyncio
async def test_plan_retries_on_invalid_cast() -> None:
    world = _make_world()
    ownership = _make_ownership(world)
    archetype = _make_archetype()

    bad_output = _valid_planner_output(world)
    bad_output["cast"] = [{"canonical_id": "gh:ghost", "role_in_scenario": "reporter"}]

    good_output = _valid_planner_output(world)

    call_count = 0

    async def mock_generate_structured(req, schema):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return bad_output
        return good_output

    mock_client = MagicMock()
    mock_client.generate_structured = mock_generate_structured

    planner = LLMPlanner(client=mock_client, model="claude-opus-4-7", max_retries=2)
    from scripts.synth.company_context import CompanyContext
    company_ctx = CompanyContext(name="Acme", stage="seed", headcount=10)

    spec = await planner.plan(
        archetype=archetype,
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        instance_ts=datetime(2026, 4, 12, 14, 30, tzinfo=UTC),
        rng_seed=42,
    )
    assert call_count == 2
    assert spec is not None


@pytest.mark.asyncio
async def test_plan_retries_on_invalid_service() -> None:
    world = _make_world()
    ownership = _make_ownership(world)
    archetype = _make_archetype()

    bad_output = _valid_planner_output(world)
    bad_output["affected_services"] = ["nonexistent-svc"]

    good_output = _valid_planner_output(world)

    responses = [bad_output, good_output]
    mock_client = MagicMock()
    mock_client.generate_structured = AsyncMock(side_effect=responses)

    planner = LLMPlanner(client=mock_client, model="claude-opus-4-7", max_retries=2)
    from scripts.synth.company_context import CompanyContext
    company_ctx = CompanyContext(name="Acme", stage="seed", headcount=10)

    spec = await planner.plan(
        archetype=archetype,
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        instance_ts=datetime(2026, 4, 12, 14, 30, tzinfo=UTC),
        rng_seed=42,
    )
    assert spec is not None


@pytest.mark.asyncio
async def test_plan_raises_after_max_retries_exhausted() -> None:
    world = _make_world()
    ownership = _make_ownership(world)
    archetype = _make_archetype()

    bad_output = _valid_planner_output(world)
    bad_output["cast"] = [{"canonical_id": "gh:ghost", "role_in_scenario": "reporter"}]

    mock_client = MagicMock()
    mock_client.generate_structured = AsyncMock(return_value=bad_output)

    planner = LLMPlanner(client=mock_client, model="claude-opus-4-7", max_retries=2)
    from scripts.synth.company_context import CompanyContext
    company_ctx = CompanyContext(name="Acme", stage="seed", headcount=10)

    with pytest.raises(PlannerValidationError, match="max_retries"):
        await planner.plan(
            archetype=archetype,
            world=world,
            ownership=ownership,
            company_ctx=company_ctx,
            instance_ts=datetime(2026, 4, 12, 14, 30, tzinfo=UTC),
            rng_seed=42,
        )


@pytest.mark.asyncio
async def test_eval_questions_count_matches_archetype() -> None:
    world = _make_world()
    ownership = _make_ownership(world)
    archetype = _make_archetype()  # eval_question_count=2

    mock_client = MagicMock()
    mock_client.generate_structured = AsyncMock(return_value=_valid_planner_output(world))

    planner = LLMPlanner(client=mock_client, model="claude-opus-4-7")
    from scripts.synth.company_context import CompanyContext
    company_ctx = CompanyContext(name="Acme", stage="seed", headcount=10)

    spec = await planner.plan(
        archetype=archetype,
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        instance_ts=datetime(2026, 4, 12, 14, 30, tzinfo=UTC),
        rng_seed=42,
    )
    # ScenarioSpec must carry eval_questions matching the archetype's count
    assert hasattr(spec, "eval_questions")
    assert len(spec.eval_questions) == 2
