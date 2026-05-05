"""INCIDENT plot archetype builder tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import BaseModel

from scripts.synth.archetypes.base import Cadence, Category, ScenarioSpec, Source, ValidatorLevel
from scripts.synth.archetypes.incident import INCIDENT, build_incident_scenarios
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 12, 14, 30, 0, tzinfo=UTC)


def _build_world() -> WorldModel:
    people = (
        Person(
            canonical_id="gh:alice",
            gh_username="alice",
            display_name="Alice",
            email_aliases=("alice@example.com",),
            role_hint="oncall",
            repos_active_in=("github.com/prbe-ai/prbe",),
            activity_score=9.0,
        ),
        Person(
            canonical_id="gh:bob",
            gh_username="bob",
            display_name="Bob",
            email_aliases=("bob@example.com",),
            role_hint="owner",
            repos_active_in=("github.com/prbe-ai/prbe",),
            activity_score=8.0,
        ),
    )
    services = (
        Service(
            name="payments-svc",
            qualified="payments-svc",
            repo_url="github.com/prbe-ai/prbe",
            kind=ServiceKind.API,
            description="payment processing",
            owners=("gh:bob",),
            recent_activity=10.0,
            deploy_target=None,
        ),
    )
    topics = (
        Topic(
            text="payments-svc 500s after feature-flag rollout",
            kind=TopicKind.COMMIT,
            repo_url="github.com/prbe-ai/prbe",
            ts=_NOW - timedelta(days=1),
            mentioned_services=("payments-svc",),
            mentioned_people=("gh:alice",),
            weight=1.0,
        ),
    )
    anchors = (
        TimeAnchor(
            label="active-2026-W15",
            start=_NOW - timedelta(days=3),
            end=_NOW + timedelta(days=4),
            activity_score=10.0,
        ),
    )
    channels = (
        ChannelHint(name="#incidents", suggested_topic=None, related_services=("payments-svc",)),
    )
    return WorldModel(
        repos=(RepoSummary(url="github.com/prbe-ai/prbe", sha="abc", default_branch="main"),),
        people=people,
        services=services,
        topic_pool=topics,
        channels=channels,
        notion_sections=(),
        time_anchors=anchors,
        dep_graph=(),
        company_name="prbe",
        seed=42,
        extracted_at=_NOW,
        sha_set={"github.com/prbe-ai/prbe": "abc"},
    )


def _make_ownership(world: WorldModel) -> OwnershipIndex:
    return OwnershipIndex(
        services_by_person={"gh:alice": ("payments-svc",), "gh:bob": ("payments-svc",)},
        people_by_service={"payments-svc": ("gh:alice", "gh:bob")},
    )


def _make_company_ctx() -> CompanyContext:
    return CompanyContext(name="prbe", stage="seed", headcount=10)


_CANNED_PLAN = {
    "title": "payments-svc 500s after feature-flag rollout",
    "summary": "Feature flag enabled prematurely caused 500s on payments-svc.",
    "cast": [
        {"canonical_id": "gh:alice", "role_in_scenario": "reporter"},
        {"canonical_id": "gh:bob", "role_in_scenario": "fixer"},
    ],
    "affected_services": ["payments-svc"],
    "affected_repos": ["github.com/prbe-ai/prbe"],
    "root_cause": "feature flag default flipped on without backend ready",
    "decision": None,
    "outcome": None,
    "timeline": [
        {"ts": "2026-04-12T14:30:00Z", "source": "sentry", "kind": "alert", "channel": None},
        {"ts": "2026-04-12T14:32:00Z", "source": "slack", "kind": "message", "channel": "#incidents"},
        {"ts": "2026-04-12T14:40:00Z", "source": "linear", "kind": "ticket", "channel": None},
        {"ts": "2026-04-12T15:00:00Z", "source": "notion", "kind": "doc", "channel": None},
        {"ts": "2026-04-12T16:00:00Z", "source": "github", "kind": "pr", "channel": None},
    ],
    "source_emissions": {"sentry": 1, "slack": 1, "linear": 1, "notion": 1, "github": 1},
    "eval_questions": [
        {
            "input": "What caused the payments-svc outage on 2026-04-12?",
            "answer_substring": "feature flag default flipped on without backend ready",
            "difficulty": "easy",
        },
        {
            "input": "Who reported the payments-svc outage and who fixed it?",
            "answer_substring": "alice reported, bob fixed",
            "difficulty": "medium-cross-source",
        },
    ],
}


class _FakeLlmClient:
    """Fixture-keyed fake that returns canned planner dicts and writer text."""

    async def generate(self, req: LlmRequest) -> LlmResponse:
        return LlmResponse(text="Slack thread about the payments-svc incident.")

    async def generate_structured(self, req: LlmRequest, schema: type[BaseModel]) -> dict:
        return _CANNED_PLAN

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_incident_archetype_metadata() -> None:
    assert INCIDENT.name == "INCIDENT"
    assert INCIDENT.category == Category.PLOT
    assert INCIDENT.cadence == Cadence.AD_HOC
    assert INCIDENT.cast_size == (2, 4)
    assert INCIDENT.sources_used == (Source.SENTRY, Source.SLACK, Source.LINEAR, Source.NOTION, Source.GITHUB)
    assert INCIDENT.validator_level == ValidatorLevel.STRICT
    assert INCIDENT.eval_question_count == 2
    assert INCIDENT.spec_template_path == "planner_incident.txt"
    assert INCIDENT.needs_planner_call is True


@pytest.mark.asyncio
async def test_build_incident_scenarios_yields_one_scenario() -> None:
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)
    client = _FakeLlmClient()
    planner = LLMPlanner(client=client, model="claude-opus-4-7")
    writer = LLMWriter(client=client, model="claude-sonnet-4-6")

    results: list[tuple[ScenarioSpec, list[SynthDoc]]] = []
    async for spec, docs in build_incident_scenarios(
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        time_window=window,
        seed=42,
        planner=planner,
        writer=writer,
        count=1,
    ):
        results.append((spec, docs))

    assert len(results) == 1


@pytest.mark.asyncio
async def test_incident_scenario_has_five_docs() -> None:
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)
    client = _FakeLlmClient()
    planner = LLMPlanner(client=client, model="claude-opus-4-7")
    writer = LLMWriter(client=client, model="claude-sonnet-4-6")

    async for _spec, docs in build_incident_scenarios(
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        time_window=window,
        seed=42,
        planner=planner,
        writer=writer,
        count=1,
    ):
        assert len(docs) == 5
        sources = [d.source for d in docs]
        assert Source.SENTRY in sources
        assert Source.SLACK in sources
        assert Source.LINEAR in sources
        assert Source.NOTION in sources
        assert Source.GITHUB in sources


@pytest.mark.asyncio
async def test_sentry_doc_is_templated_no_writer_call() -> None:
    """The Sentry doc is constructed without calling writer.write()."""
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)
    client = _FakeLlmClient()
    planner = LLMPlanner(client=client, model="claude-opus-4-7")

    write_calls: list[Source] = []

    class _TrackingWriter(LLMWriter):
        async def write(self, spec, source, emission_index, prior_emitted_docs, world, company_ctx):  # type: ignore[override]
            write_calls.append(source)
            return f"Written doc for {source}"

    writer = _TrackingWriter(client=client, model="claude-sonnet-4-6")

    async for _spec, _docs in build_incident_scenarios(
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        time_window=window,
        seed=42,
        planner=planner,
        writer=writer,
        count=1,
    ):
        pass

    # writer.write called for 4 non-sentry sources; sentry is templated
    assert Source.SENTRY not in write_calls
    assert len(write_calls) == 4


@pytest.mark.asyncio
async def test_incident_eval_questions_count() -> None:
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)
    client = _FakeLlmClient()
    planner = LLMPlanner(client=client, model="claude-opus-4-7")
    writer = LLMWriter(client=client, model="claude-sonnet-4-6")

    async for spec, _docs in build_incident_scenarios(
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        time_window=window,
        seed=42,
        planner=planner,
        writer=writer,
        count=1,
    ):
        assert len(spec.eval_questions) == 2
        difficulties = {q.difficulty for q in spec.eval_questions}
        assert "easy" in difficulties
        assert "medium-cross-source" in difficulties


@pytest.mark.asyncio
async def test_incident_determinism() -> None:
    """Same seed + same canned client produces identical scenario IDs."""
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)

    async def _collect(seed: int) -> list[str]:
        client = _FakeLlmClient()
        planner = LLMPlanner(client=client, model="claude-opus-4-7")
        writer = LLMWriter(client=client, model="claude-sonnet-4-6")
        ids = []
        async for spec, _ in build_incident_scenarios(
            world=world, ownership=ownership, company_ctx=company_ctx,
            time_window=window, seed=seed, planner=planner, writer=writer, count=2,
        ):
            ids.append(spec.id)
        return ids

    run1 = await _collect(seed=42)
    run2 = await _collect(seed=42)
    assert run1 == run2
