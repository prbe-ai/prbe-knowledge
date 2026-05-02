"""BIG_REFACTOR plot archetype builder tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import BaseModel

from scripts.synth.archetypes.base import Cadence, Category, ScenarioSpec, Source, ValidatorLevel
from scripts.synth.archetypes.big_refactor import BIG_REFACTOR, build_big_refactor_scenarios
from scripts.synth.company_context import CompanyContext
from scripts.synth.llm.base import LlmRequest, LlmResponse
from scripts.synth.llm.planner import LLMPlanner
from scripts.synth.llm.writer import LLMWriter
from scripts.synth.output.base import SynthDoc
from scripts.synth.output.github import wrap as github_wrap
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

_NOW = datetime(2026, 2, 10, 9, 0, 0, tzinfo=UTC)


def _build_world() -> WorldModel:
    people = (
        Person(
            canonical_id="gh:eve",
            gh_username="eve",
            display_name="Eve",
            email_aliases=("eve@example.com",),
            role_hint="architect",
            repos_active_in=("github.com/prbe-ai/prbe",),
            activity_score=9.5,
        ),
        Person(
            canonical_id="gh:frank",
            gh_username="frank",
            display_name="Frank",
            email_aliases=("frank@example.com",),
            role_hint="engineer",
            repos_active_in=("github.com/prbe-ai/prbe",),
            activity_score=8.0,
        ),
        Person(
            canonical_id="gh:grace",
            gh_username="grace",
            display_name="Grace",
            email_aliases=("grace@example.com",),
            role_hint="engineer",
            repos_active_in=("github.com/prbe-ai/prbe",),
            activity_score=7.5,
        ),
    )
    services = (
        Service(
            name="ingestion-svc",
            qualified="ingestion-svc",
            repo_url="github.com/prbe-ai/prbe",
            kind=ServiceKind.API,
            description="data ingestion pipeline",
            owners=("gh:eve",),
            recent_activity=9.0,
            deploy_target=None,
        ),
    )
    topics = (
        Topic(
            text="RFC: migrate ingestion-svc from Kafka to NATS",
            kind=TopicKind.COMMIT,
            repo_url="github.com/prbe-ai/prbe",
            ts=_NOW - timedelta(days=2),
            mentioned_services=("ingestion-svc",),
            mentioned_people=("gh:eve",),
            weight=1.0,
        ),
    )
    anchors = (
        TimeAnchor(
            label="active-2026-W07",
            start=_NOW - timedelta(days=3),
            end=_NOW + timedelta(days=4),
            activity_score=9.5,
        ),
    )
    channels = (
        ChannelHint(
            name="#engineering",
            suggested_topic=None,
            related_services=("ingestion-svc",),
        ),
    )
    return WorldModel(
        repos=(RepoSummary(url="github.com/prbe-ai/prbe", sha="ghi", default_branch="main"),),
        people=people,
        services=services,
        topic_pool=topics,
        channels=channels,
        notion_sections=(),
        time_anchors=anchors,
        dep_graph=(),
        company_name="prbe",
        seed=13,
        extracted_at=_NOW,
        sha_set={"github.com/prbe-ai/prbe": "ghi"},
    )


def _make_ownership(world: WorldModel) -> OwnershipIndex:
    return OwnershipIndex(
        services_by_person={
            "gh:eve": ("ingestion-svc",),
            "gh:frank": ("ingestion-svc",),
            "gh:grace": ("ingestion-svc",),
        },
        people_by_service={"ingestion-svc": ("gh:eve", "gh:frank", "gh:grace")},
    )


def _make_company_ctx() -> CompanyContext:
    return CompanyContext(name="prbe", stage="seed", headcount=12)


_CANNED_PLAN = {
    "title": "RFC: migrate ingestion-svc from Kafka to NATS",
    "summary": "Eve proposes migrating ingestion-svc from Kafka to NATS for lower latency.",
    "cast": [
        {"canonical_id": "gh:eve", "role_in_scenario": "rfc_author"},
        {"canonical_id": "gh:frank", "role_in_scenario": "debater"},
        {"canonical_id": "gh:grace", "role_in_scenario": "debater"},
    ],
    "affected_services": ["ingestion-svc"],
    "affected_repos": ["github.com/prbe-ai/prbe"],
    "root_cause": None,
    "decision": "Adopt NATS for ingestion-svc; migrate by Q2 2026.",
    "outcome": None,
    "timeline": [
        {"ts": "2026-02-10T09:00:00Z", "source": "github", "kind": "issue", "channel": None},
        {"ts": "2026-02-10T09:05:00Z", "source": "notion", "kind": "doc", "channel": None},
        {"ts": "2026-02-10T09:10:00Z", "source": "slack", "kind": "message", "channel": "#engineering"},
        {"ts": "2026-02-10T09:15:00Z", "source": "slack", "kind": "message", "channel": "#engineering"},
        {"ts": "2026-02-10T09:20:00Z", "source": "slack", "kind": "message", "channel": "#engineering"},
        {"ts": "2026-02-10T10:00:00Z", "source": "linear", "kind": "ticket", "channel": None},
    ],
    "source_emissions": {"github": 1, "notion": 1, "slack": 3, "linear": 1},
    "eval_questions": [
        {
            "input": "Why did we move ingestion-svc from Kafka to NATS? Who proposed it?",
            "answer_substring": "lower latency; eve proposed",
            "difficulty": "hard-temporal",
        },
    ],
}


class _FakeLlmClient:
    async def generate(self, req: LlmRequest) -> LlmResponse:
        return LlmResponse(text="RFC debate message text.")

    async def generate_structured(self, req: LlmRequest, schema: type[BaseModel]) -> dict:
        return _CANNED_PLAN

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_big_refactor_archetype_metadata() -> None:
    assert BIG_REFACTOR.name == "BIG_REFACTOR"
    assert BIG_REFACTOR.category == Category.PLOT
    assert BIG_REFACTOR.cadence == Cadence.AD_HOC
    assert BIG_REFACTOR.cast_size == (3, 4)
    assert BIG_REFACTOR.sources_used == (Source.GITHUB, Source.NOTION, Source.SLACK, Source.LINEAR)
    assert BIG_REFACTOR.validator_level == ValidatorLevel.STRICT
    assert BIG_REFACTOR.eval_question_count == 1
    assert BIG_REFACTOR.spec_template_path == "planner_big_refactor.txt"
    assert BIG_REFACTOR.needs_planner_call is True


@pytest.mark.asyncio
async def test_build_big_refactor_scenarios_yields_one_scenario() -> None:
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)
    client = _FakeLlmClient()
    planner = LLMPlanner(client=client, model="claude-opus-4-7")
    writer = LLMWriter(client=client, model="claude-sonnet-4-6")

    results: list[tuple[ScenarioSpec, list[SynthDoc]]] = []
    async for item in build_big_refactor_scenarios(
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        time_window=window,
        seed=13,
        planner=planner,
        writer=writer,
        count=1,
    ):
        results.append(item)

    assert len(results) == 1


@pytest.mark.asyncio
async def test_big_refactor_doc_count_in_range() -> None:
    """BIG_REFACTOR emits 4-6 docs depending on Slack thread length."""
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)
    client = _FakeLlmClient()
    planner = LLMPlanner(client=client, model="claude-opus-4-7")
    writer = LLMWriter(client=client, model="claude-sonnet-4-6")

    async for _spec, docs in build_big_refactor_scenarios(
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        time_window=window,
        seed=13,
        planner=planner,
        writer=writer,
        count=1,
    ):
        assert 4 <= len(docs) <= 6


@pytest.mark.asyncio
async def test_slack_thread_replies_have_thread_parent_id() -> None:
    """Slack reply docs must carry thread_parent_id referencing the parent doc."""
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)
    client = _FakeLlmClient()
    planner = LLMPlanner(client=client, model="claude-opus-4-7")
    writer = LLMWriter(client=client, model="claude-sonnet-4-6")

    async for _spec, docs in build_big_refactor_scenarios(
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        time_window=window,
        seed=13,
        planner=planner,
        writer=writer,
        count=1,
    ):
        slack_docs = [d for d in docs if d.source == Source.SLACK]
        assert len(slack_docs) >= 2  # at least parent + 1 reply

        parent = slack_docs[0]
        assert parent.thread_parent_id is None  # parent has no thread_parent_id

        for reply in slack_docs[1:]:
            assert reply.thread_parent_id == parent.source_event_id


@pytest.mark.asyncio
async def test_github_rfc_is_issue_not_pr() -> None:
    """GitHub doc is an issue (RFC), not a pull request — verified via github wrapper."""
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)
    client = _FakeLlmClient()
    planner = LLMPlanner(client=client, model="claude-opus-4-7")
    writer = LLMWriter(client=client, model="claude-sonnet-4-6")

    async for _spec, docs in build_big_refactor_scenarios(
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        time_window=window,
        seed=13,
        planner=planner,
        writer=writer,
        count=1,
    ):
        github_docs = [d for d in docs if d.source == Source.GITHUB]
        assert len(github_docs) == 1

        # The doc should have archetype="BIG_REFACTOR" so the wrapper emits an issue
        assert github_docs[0].archetype == "BIG_REFACTOR"

        # Verify through the GitHub wrapper that it produces issues.opened, not pull_request
        payload = json.loads(github_wrap(github_docs[0]))
        assert payload["action"] == "opened"
        assert "issue" in payload
        assert "pull_request" not in payload


@pytest.mark.asyncio
async def test_big_refactor_eval_question_hard_temporal() -> None:
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)
    client = _FakeLlmClient()
    planner = LLMPlanner(client=client, model="claude-opus-4-7")
    writer = LLMWriter(client=client, model="claude-sonnet-4-6")

    async for spec, _docs in build_big_refactor_scenarios(
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        time_window=window,
        seed=13,
        planner=planner,
        writer=writer,
        count=1,
    ):
        assert len(spec.eval_questions) == 1
        assert spec.eval_questions[0].difficulty == "hard-temporal"


@pytest.mark.asyncio
async def test_big_refactor_cast_size_within_bounds() -> None:
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)
    client = _FakeLlmClient()
    planner = LLMPlanner(client=client, model="claude-opus-4-7")
    writer = LLMWriter(client=client, model="claude-sonnet-4-6")

    async for spec, _docs in build_big_refactor_scenarios(
        world=world,
        ownership=ownership,
        company_ctx=company_ctx,
        time_window=window,
        seed=13,
        planner=planner,
        writer=writer,
        count=1,
    ):
        assert 3 <= len(spec.cast) <= 4


@pytest.mark.asyncio
async def test_big_refactor_determinism() -> None:
    world = _build_world()
    ownership = _make_ownership(world)
    company_ctx = _make_company_ctx()
    window = TimeWindow(end=_NOW, days=30)

    async def _collect(seed: int) -> list[str]:
        client = _FakeLlmClient()
        planner = LLMPlanner(client=client, model="claude-opus-4-7")
        writer = LLMWriter(client=client, model="claude-sonnet-4-6")
        ids = []
        async for spec, _ in build_big_refactor_scenarios(
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

    run1 = await _collect(seed=13)
    run2 = await _collect(seed=13)
    assert run1 == run2
