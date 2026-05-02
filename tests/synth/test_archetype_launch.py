"""LAUNCH plot archetype builder tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import BaseModel

from scripts.synth.archetypes.base import Cadence, Category, ScenarioSpec, Source, ValidatorLevel
from scripts.synth.archetypes.launch import LAUNCH, build_launch_scenarios
from scripts.synth.company_context import CompanyContext
from scripts.synth.llm.base import LlmRequest, LlmResponse
from scripts.synth.llm.planner import LLMPlanner
from scripts.synth.llm.writer import LLMWriter
from scripts.synth.output.base import SynthDoc
from scripts.synth.ownership import OwnershipIndex
from scripts.synth.scenarios import TimeWindow
from scripts.synth.world_model import (
    ChannelHint,
    Person,
    RepoSummary,
    Service,
    ServiceKind,
    TimeAnchor,
    Topic,
    TopicKind,
    WorldModel,
)

_NOW = datetime(2026, 3, 15, 10, 0, 0, tzinfo=UTC)


def _build_world() -> WorldModel:
    people = (
        Person(
            canonical_id="gh:carol",
            gh_username="carol",
            display_name="Carol",
            email_aliases=("carol@example.com",),
            role_hint="product",
            repos_active_in=("github.com/prbe-ai/prbe",),
            activity_score=9.0,
        ),
        Person(
            canonical_id="gh:dan",
            gh_username="dan",
            display_name="Dan",
            email_aliases=("dan@example.com",),
            role_hint="engineer",
            repos_active_in=("github.com/prbe-ai/prbe",),
            activity_score=8.0,
        ),
    )
    services = (
        Service(
            name="export-api",
            qualified="export-api",
            repo_url="github.com/prbe-ai/prbe",
            kind=ServiceKind.API,
            description="data export feature",
            owners=("gh:dan",),
            recent_activity=8.0,
            deploy_target=None,
        ),
    )
    topics = (
        Topic(
            text="export-api shipped v2",
            kind=TopicKind.COMMIT,
            repo_url="github.com/prbe-ai/prbe",
            ts=_NOW - timedelta(days=1),
            mentioned_services=("export-api",),
            mentioned_people=("gh:carol",),
            weight=1.0,
        ),
    )
    anchors = (
        TimeAnchor(
            label="active-2026-W11",
            start=_NOW - timedelta(days=3),
            end=_NOW + timedelta(days=4),
            activity_score=9.0,
        ),
    )
    channels = (
        ChannelHint(name="#announcements", suggested_topic=None, related_services=()),
    )
    return WorldModel(
        repos=(RepoSummary(url="github.com/prbe-ai/prbe", sha="def", default_branch="main"),),
        people=people,
        services=services,
        topic_pool=topics,
        channels=channels,
        notion_sections=(),
        time_anchors=anchors,
        dep_graph=(),
        company_name="prbe",
        seed=7,
        extracted_at=_NOW,
        sha_set={"github.com/prbe-ai/prbe": "def"},
    )


def _make_ownership(world: WorldModel) -> OwnershipIndex:
    return OwnershipIndex(
        services_by_person={"gh:carol": ("export-api",), "gh:dan": ("export-api",)},
        people_by_service={"export-api": ("gh:carol", "gh:dan")},
    )


def _make_company_ctx(*, with_customers: bool = False) -> CompanyContext:
    customers: tuple = ()
    if with_customers:
        from scripts.synth.company_context import Customer

        customers = (Customer(name="Acme Corp", type="design_partner"),)
    return CompanyContext(name="prbe", stage="seed", headcount=10, customers=customers)


_CANNED_PLAN_NO_CUSTOMERS = {
    "title": "export-api v2 ships",
    "summary": "The export-api v2 feature launched after 6 weeks of development.",
    "cast": [
        {"canonical_id": "gh:carol", "role_in_scenario": "product_owner"},
        {"canonical_id": "gh:dan", "role_in_scenario": "engineering_lead"},
    ],
    "affected_services": ["export-api"],
    "affected_repos": ["github.com/prbe-ai/prbe"],
    "root_cause": None,
    "decision": None,
    "outcome": "export-api v2 shipped successfully to all customers",
    "timeline": [
        {"ts": "2026-03-15T10:00:00Z", "source": "slack", "kind": "message", "channel": "#announcements"},
        {"ts": "2026-03-15T10:05:00Z", "source": "notion", "kind": "doc", "channel": None},
        {"ts": "2026-03-15T10:10:00Z", "source": "linear", "kind": "ticket", "channel": None},
        {"ts": "2026-03-15T10:15:00Z", "source": "github", "kind": "pr", "channel": None},
    ],
    "source_emissions": {"slack": 1, "notion": 1, "linear": 1, "github": 1},
    "eval_questions": [
        {
            "input": "When did export-api v2 ship?",
            "answer_substring": "2026-03-15",
            "difficulty": "easy",
        },
        {
            "input": "Who owned the export-api v2 launch?",
            "answer_substring": "carol",
            "difficulty": "medium-cross-source",
        },
    ],
}

_CANNED_PLAN_WITH_CUSTOMERS = {
    **_CANNED_PLAN_NO_CUSTOMERS,
    "eval_questions": [
        {
            "input": "When did export-api v2 ship?",
            "answer_substring": "2026-03-15",
            "difficulty": "easy",
        },
        {
            "input": "Which customer was the design partner for export-api v2?",
            "answer_substring": "Acme Corp",
            "difficulty": "medium-cross-source",
        },
    ],
}


class _FakeLlmClientNoCustomers:
    async def generate(self, req: LlmRequest) -> LlmResponse:
        return LlmResponse(text="Launch announcement text.")

    async def generate_structured(self, req: LlmRequest, schema: type[BaseModel]) -> dict:
        return _CANNED_PLAN_NO_CUSTOMERS

    async def close(self) -> None:
        pass


class _FakeLlmClientWithCustomers:
    async def generate(self, req: LlmRequest) -> LlmResponse:
        return LlmResponse(text="Launch announcement with design partner mention.")

    async def generate_structured(self, req: LlmRequest, schema: type[BaseModel]) -> dict:
        return _CANNED_PLAN_WITH_CUSTOMERS

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_launch_archetype_metadata() -> None:
    assert LAUNCH.name == "LAUNCH"
    assert LAUNCH.category == Category.PLOT
    assert LAUNCH.cadence == Cadence.AD_HOC
    assert LAUNCH.cast_size == (2, 3)
    assert LAUNCH.sources_used == (Source.SLACK, Source.NOTION, Source.LINEAR, Source.GITHUB)
    assert LAUNCH.validator_level == ValidatorLevel.STRICT
    assert LAUNCH.eval_question_count == 2
    assert LAUNCH.spec_template_path == "planner_launch.txt"
    assert LAUNCH.needs_planner_call is True


@pytest.mark.asyncio
async def test_build_launch_scenarios_yields_one_scenario() -> None:
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)
    client = _FakeLlmClientNoCustomers()
    planner = LLMPlanner(client=client, model="claude-opus-4-7")
    writer = LLMWriter(client=client, model="claude-sonnet-4-6")

    results: list[tuple[ScenarioSpec, list[SynthDoc]]] = []
    async for item in build_launch_scenarios(
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        time_window=window,
        seed=7,
        planner=planner,
        writer=writer,
        count=1,
    ):
        results.append(item)

    assert len(results) == 1


@pytest.mark.asyncio
async def test_launch_scenario_has_four_docs_all_llm() -> None:
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)
    client = _FakeLlmClientNoCustomers()
    planner = LLMPlanner(client=client, model="claude-opus-4-7")

    write_calls: list[Source] = []

    class _TrackingWriter(LLMWriter):
        async def write(  # type: ignore[override]
            self, spec, source, emission_index, prior_emitted_docs, world, company_ctx
        ):
            write_calls.append(source)
            return f"Written doc for {source}"

    writer = _TrackingWriter(client=client, model="claude-sonnet-4-6")

    async for _spec, docs in build_launch_scenarios(
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        time_window=window,
        seed=7,
        planner=planner,
        writer=writer,
        count=1,
    ):
        assert len(docs) == 4
        # All 4 are LLM-written; no templated doc
        assert len(write_calls) == 4
        assert Source.SLACK in write_calls
        assert Source.NOTION in write_calls
        assert Source.LINEAR in write_calls
        assert Source.GITHUB in write_calls


@pytest.mark.asyncio
async def test_launch_eval_question_easy() -> None:
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)
    client = _FakeLlmClientNoCustomers()
    planner = LLMPlanner(client=client, model="claude-opus-4-7")
    writer = LLMWriter(client=client, model="claude-sonnet-4-6")

    async for spec, _docs in build_launch_scenarios(
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        time_window=window,
        seed=7,
        planner=planner,
        writer=writer,
        count=1,
    ):
        assert spec.eval_questions[0].difficulty == "easy"


@pytest.mark.asyncio
async def test_launch_eval_question_medium_cross_source_present_both_cases() -> None:
    """Eval question 2 is medium-cross-source regardless of whether customers exist."""
    world = _build_world()
    ownership = _make_ownership(world)
    window = TimeWindow(end=_NOW, days=30)

    for with_customers in (False, True):
        company_ctx = _make_company_ctx(with_customers=with_customers)
        client = _FakeLlmClientWithCustomers() if with_customers else _FakeLlmClientNoCustomers()
        planner = LLMPlanner(client=client, model="claude-opus-4-7")
        writer = LLMWriter(client=client, model="claude-sonnet-4-6")

        async for spec, _docs in build_launch_scenarios(
            world=world,
            ownership=ownership,
            company_ctx=company_ctx,
            time_window=window,
            seed=7,
            planner=planner,
            writer=writer,
            count=1,
        ):
            assert len(spec.eval_questions) == 2
            assert spec.eval_questions[1].difficulty == "medium-cross-source"


@pytest.mark.asyncio
async def test_launch_cast_size_within_bounds() -> None:
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)
    client = _FakeLlmClientNoCustomers()
    planner = LLMPlanner(client=client, model="claude-opus-4-7")
    writer = LLMWriter(client=client, model="claude-sonnet-4-6")

    async for spec, _docs in build_launch_scenarios(
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        time_window=window,
        seed=7,
        planner=planner,
        writer=writer,
        count=1,
    ):
        assert 2 <= len(spec.cast) <= 3


@pytest.mark.asyncio
async def test_launch_determinism() -> None:
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)

    async def _collect(seed: int) -> list[str]:
        client = _FakeLlmClientNoCustomers()
        planner = LLMPlanner(client=client, model="claude-opus-4-7")
        writer = LLMWriter(client=client, model="claude-sonnet-4-6")
        ids = []
        async for spec, _ in build_launch_scenarios(
            world=world,
            ownership=ownership,
            company_ctx=company_ctx,
            time_window=window,
            seed=seed,
            planner=planner,
            writer=writer,
            count=2,
        ):
            ids.append(spec.id)
        return ids

    run1 = await _collect(seed=7)
    run2 = await _collect(seed=7)
    assert run1 == run2
