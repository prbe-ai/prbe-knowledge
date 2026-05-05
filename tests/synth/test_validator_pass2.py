"""Tests for Validator Pass 2: LLM consistency check over a scenario's docs."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.synth.archetypes.base import ScenarioSpec, Source
from scripts.synth.llm.validator_pass2 import (
    StructuredOutputValidationError,
    validate_pass2,
)
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
        people=(alice,),
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
    return ScenarioSpec(
        id="scn-incident-payments-2026-04-12",
        archetype_name="INCIDENT",
        instance_ts=ts,
        cast=("gh:alice",),
        affected_services=("payments",),
        doc_specs=(),
        title="payments down",
        summary="payments service 500s",
    )


def _make_doc(doc_id: str, text: str) -> SynthDoc:
    return SynthDoc(
        id=doc_id,
        source=Source.SLACK,
        source_event_id=doc_id,
        text=text,
        occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        channel="#incidents",
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-incident-payments-2026-04-12",
        archetype="INCIDENT",
        personas=("gh:alice",),
        services_mentioned=("payments",),
        priority=0,
    )


@pytest.mark.asyncio
async def test_clean_docs_return_passed_true() -> None:
    world = _make_world()
    spec = _make_spec()
    docs = (
        _make_doc("doc-0", "The payments service is down."),
        _make_doc("doc-1", "Root cause: feature flag flipped."),
    )

    mock_client = MagicMock()
    mock_client.generate_structured = AsyncMock(
        return_value={"passed": True, "violations": []}
    )

    result = await validate_pass2(
        scenario=spec,
        docs=docs,
        world=world,
        client=mock_client,
        model="claude-haiku-4-5-20251001",
    )
    assert result.passed is True
    assert result.violations == ()


@pytest.mark.asyncio
async def test_llm_flagged_contradiction_returns_passed_false() -> None:
    world = _make_world()
    spec = _make_spec()
    docs = (
        _make_doc("doc-0", "The auth service is down."),  # contradiction: should be payments
    )

    mock_client = MagicMock()
    mock_client.generate_structured = AsyncMock(
        return_value={
            "passed": False,
            "violations": [
                {"doc_id": "doc-0", "issue": "mentions auth-svc but scenario is about payments"}
            ],
        }
    )

    result = await validate_pass2(
        scenario=spec,
        docs=docs,
        world=world,
        client=mock_client,
        model="claude-haiku-4-5-20251001",
    )
    assert result.passed is False
    assert len(result.violations) == 1
    assert result.violations[0].doc_id == "doc-0"


@pytest.mark.asyncio
async def test_violation_rate_29_percent_passes() -> None:
    """2 violations out of 7 docs = 28.6% → passes threshold."""
    world = _make_world()
    spec = _make_spec()
    docs = tuple(_make_doc(f"doc-{i}", f"content {i}") for i in range(7))

    violations_raw = [
        {"doc_id": "doc-0", "issue": "minor inconsistency"},
        {"doc_id": "doc-1", "issue": "another minor issue"},
    ]
    mock_client = MagicMock()
    mock_client.generate_structured = AsyncMock(
        return_value={"passed": True, "violations": violations_raw}
    )

    result = await validate_pass2(
        scenario=spec,
        docs=docs,
        world=world,
        client=mock_client,
        model="claude-haiku-4-5-20251001",
    )
    # 2/7 ≈ 0.286 ≤ 0.30 → passed stays True
    assert result.passed is True
    assert len(result.violations) == 2


@pytest.mark.asyncio
async def test_violation_rate_31_percent_fails() -> None:
    """3 violations out of 9 docs = 33.3% → forced passed=False."""
    world = _make_world()
    spec = _make_spec()
    docs = tuple(_make_doc(f"doc-{i}", f"content {i}") for i in range(9))

    violations_raw = [
        {"doc_id": f"doc-{i}", "issue": f"issue {i}"}
        for i in range(3)
    ]
    mock_client = MagicMock()
    mock_client.generate_structured = AsyncMock(
        return_value={"passed": True, "violations": violations_raw}
    )

    result = await validate_pass2(
        scenario=spec,
        docs=docs,
        world=world,
        client=mock_client,
        model="claude-haiku-4-5-20251001",
    )
    # 3/9 ≈ 0.333 > 0.30 → forced passed=False despite LLM saying True
    assert result.passed is False
    assert len(result.violations) == 3


@pytest.mark.asyncio
async def test_empty_docs_raises_value_error() -> None:
    world = _make_world()
    spec = _make_spec()
    mock_client = MagicMock()

    with pytest.raises(ValueError, match="docs"):
        await validate_pass2(
            scenario=spec,
            docs=(),
            world=world,
            client=mock_client,
            model="claude-haiku-4-5-20251001",
        )


@pytest.mark.asyncio
async def test_invalid_schema_raises_structured_output_validation_error() -> None:
    world = _make_world()
    spec = _make_spec()
    docs = (_make_doc("doc-0", "content"),)

    mock_client = MagicMock()
    # Return malformed dict missing required fields
    mock_client.generate_structured = AsyncMock(return_value={"wrong_field": "bad"})

    with pytest.raises(StructuredOutputValidationError):
        await validate_pass2(
            scenario=spec,
            docs=docs,
            world=world,
            client=mock_client,
            model="claude-haiku-4-5-20251001",
        )
