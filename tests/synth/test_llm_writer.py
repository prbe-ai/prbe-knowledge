"""Tests for LLMWriter: prompt assembly, persona-view filtering, text return."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.synth.archetypes.base import DocSpec, ScenarioSpec, Source
from scripts.synth.llm.writer import LLMWriter
from scripts.synth.output.base import SynthDoc
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
        email_aliases=(),
        role_hint="backend",
        repos_active_in=(),
        activity_score=10.0,
    )
    bob = Person(
        canonical_id="gh:bob",
        gh_username="bob",
        display_name="Bob",
        email_aliases=(),
        role_hint="oncall",
        repos_active_in=(),
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


def _make_spec() -> ScenarioSpec:
    ts = datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC)
    ts2 = datetime(2026, 4, 12, 14, 30, 0, tzinfo=UTC)
    doc_a = DocSpec(
        id="scn-incident-slack-0",
        source=Source.SLACK,
        occurred_at=ts,
        channel="#incidents",
        page_section=None,
        text="",
        thread_parent_id=None,
        personas=("gh:bob",),
        services_mentioned=("payments",),
    )
    doc_b = DocSpec(
        id="scn-incident-notion-0",
        source=Source.NOTION,
        occurred_at=ts2,
        channel=None,
        page_section="Postmortems",
        text="",
        thread_parent_id=None,
        personas=("gh:alice",),
        services_mentioned=("payments",),
    )
    return ScenarioSpec(
        id="scn-incident-payments-2026-04-12",
        archetype_name="INCIDENT",
        instance_ts=ts,
        cast=("gh:alice", "gh:bob"),
        affected_services=("payments",),
        doc_specs=(doc_a, doc_b),
        title="payments down",
        summary="payments service 500s",
        root_cause="feature flag",
        eval_questions=(),
    )


def _make_prior_doc(doc_id: str, source: Source, occurred_at: datetime, persona: str) -> SynthDoc:
    return SynthDoc(
        id=doc_id,
        source=source,
        source_event_id=doc_id,
        text="prior content",
        occurred_at=occurred_at,
        channel="#incidents" if source == Source.SLACK else None,
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-incident-payments-2026-04-12",
        archetype="INCIDENT",
        personas=(persona,),
        services_mentioned=("payments",),
        priority=0,
    )


def _make_company_ctx():
    from scripts.synth.company_context import CompanyContext
    return CompanyContext(name="Acme", stage="seed", headcount=20)


@pytest.mark.asyncio
async def test_write_returns_plain_text(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "writer_slack.txt").write_text(
        "Write a Slack message: {scenario_summary} {persona_view} "
        "{allowed_services} {allowed_people} {allowed_channels} "
        "{current_emission} {instance_ts}"
    )

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(return_value=MagicMock(text="The payments service is down."))

    world = _make_world()
    writer = LLMWriter(client=mock_client, model="claude-sonnet-4-6", prompts_dir=prompts_dir)
    company_ctx = _make_company_ctx()
    spec = _make_spec()

    result = await writer.write(
        spec=spec,
        source=Source.SLACK,
        emission_index=0,
        prior_emitted_docs=(),
        world=world,
        company_ctx=company_ctx,
    )
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_write_prior_docs_filtered_by_timeline(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "writer_notion.txt").write_text(
        "{scenario_summary} {persona_view} {allowed_services} "
        "{allowed_people} {allowed_channels} {current_emission} {instance_ts}"
    )

    ts_early = datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC)
    ts_late = datetime(2026, 4, 12, 14, 30, 0, tzinfo=UTC)

    prior_early = _make_prior_doc("early-doc", Source.SLACK, ts_early, "gh:bob")
    prior_late = _make_prior_doc("late-doc", Source.NOTION, ts_late, "gh:alice")

    captured_prompts: list[str] = []

    async def capture_generate(req):
        captured_prompts.append(req.prompt)
        return MagicMock(text="postmortem content")

    mock_client = MagicMock()
    mock_client.generate = capture_generate

    world = _make_world()
    writer = LLMWriter(client=mock_client, model="claude-sonnet-4-6", prompts_dir=prompts_dir)
    company_ctx = _make_company_ctx()
    spec = _make_spec()

    # Writing the Notion doc (alice's doc at ts_late); alice's first emission is ts_late.
    # prior_early (bob's slack at ts_early) is before ts_late → included.
    # prior_late (alice's notion at ts_late) is NOT strictly before → excluded.
    await writer.write(
        spec=spec,
        source=Source.NOTION,
        emission_index=0,
        prior_emitted_docs=(prior_early, prior_late),
        world=world,
        company_ctx=company_ctx,
    )
    assert captured_prompts, "generate was not called"
    prompt = captured_prompts[0]
    assert "prior content" in prompt  # prior_early's text included
    assert "late-doc" not in prompt   # prior_late excluded


@pytest.mark.asyncio
async def test_write_allowlists_present_in_prompt(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "writer_slack.txt").write_text(
        "{scenario_summary} {persona_view} {allowed_services} "
        "{allowed_people} {allowed_channels} {current_emission} {instance_ts}"
    )

    captured_prompts: list[str] = []

    async def capture_generate(req):
        captured_prompts.append(req.prompt)
        return MagicMock(text="done")

    mock_client = MagicMock()
    mock_client.generate = capture_generate

    world = _make_world()
    writer = LLMWriter(client=mock_client, model="claude-sonnet-4-6", prompts_dir=prompts_dir)
    company_ctx = _make_company_ctx()

    await writer.write(
        spec=_make_spec(),
        source=Source.SLACK,
        emission_index=0,
        prior_emitted_docs=(),
        world=world,
        company_ctx=company_ctx,
    )
    assert captured_prompts
    prompt = captured_prompts[0]
    assert "payments" in prompt       # allowed_services
    assert "alice" in prompt          # allowed_people
    assert "#incidents" in prompt     # allowed_channels


@pytest.mark.asyncio
async def test_write_missing_prompt_template_raises(tmp_path: Path) -> None:
    empty_prompts_dir = tmp_path / "prompts"
    empty_prompts_dir.mkdir()  # no writer_slack.txt

    mock_client = MagicMock()
    world = _make_world()
    writer = LLMWriter(client=mock_client, model="claude-sonnet-4-6", prompts_dir=empty_prompts_dir)
    company_ctx = _make_company_ctx()

    with pytest.raises(FileNotFoundError):
        await writer.write(
            spec=_make_spec(),
            source=Source.SLACK,
            emission_index=0,
            prior_emitted_docs=(),
            world=world,
            company_ctx=company_ctx,
        )


@pytest.mark.asyncio
async def test_write_persona_view_excludes_docs_after_first_emission(tmp_path: Path) -> None:
    """Bob writes the Slack message first (ts_early); Alice writes notion later (ts_late).
    When writing Bob's Slack message, Alice's notion doc (ts_late) must be excluded."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "writer_slack.txt").write_text(
        "{scenario_summary} {persona_view} {allowed_services} "
        "{allowed_people} {allowed_channels} {current_emission} {instance_ts}"
    )

    ts_late = datetime(2026, 4, 12, 14, 30, 0, tzinfo=UTC)
    alice_notion = _make_prior_doc("alice-notion-0", Source.NOTION, ts_late, "gh:alice")

    captured_prompts: list[str] = []

    async def capture_generate(req):
        captured_prompts.append(req.prompt)
        return MagicMock(text="incidents channel post")

    mock_client = MagicMock()
    mock_client.generate = capture_generate

    world = _make_world()
    writer = LLMWriter(client=mock_client, model="claude-sonnet-4-6", prompts_dir=prompts_dir)
    company_ctx = _make_company_ctx()
    spec = _make_spec()  # bob's slack at ts_early, alice's notion at ts_late

    await writer.write(
        spec=spec,
        source=Source.SLACK,
        emission_index=0,
        prior_emitted_docs=(alice_notion,),
        world=world,
        company_ctx=company_ctx,
    )
    prompt = captured_prompts[0]
    # alice_notion occurred at ts_late which is AFTER bob's first emission at ts_early
    assert "alice-notion-0" not in prompt


@pytest.mark.asyncio
async def test_write_logs_llm_call(tmp_path: Path) -> None:
    import structlog
    import structlog.testing

    # If an earlier test in the suite called shared.logging.configure_logging,
    # structlog caches a filtering BoundLogger at INFO level (see
    # cache_logger_on_first_use=True in shared/logging.py). LLMWriter logs at
    # DEBUG, which then gets filtered before reaching capture_logs(). Reset
    # so capture_logs can install its own processor chain at any level.
    structlog.reset_defaults()

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "writer_slack.txt").write_text(
        "{scenario_summary} {persona_view} {allowed_services} "
        "{allowed_people} {allowed_channels} {current_emission} {instance_ts}"
    )

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(return_value=MagicMock(text="logged call"))

    world = _make_world()
    writer = LLMWriter(client=mock_client, model="claude-sonnet-4-6", prompts_dir=prompts_dir)
    company_ctx = _make_company_ctx()

    with structlog.testing.capture_logs() as cap_logs:
        await writer.write(
            spec=_make_spec(),
            source=Source.SLACK,
            emission_index=0,
            prior_emitted_docs=(),
            world=world,
            company_ctx=company_ctx,
        )
    # LLMWriter must emit at least one log record during a write call
    assert len(cap_logs) > 0
