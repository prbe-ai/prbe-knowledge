"""Smoke tests for archetype base dataclasses and enums."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from scripts.synth.archetypes.base import (
    Archetype,
    Cadence,
    Category,
    DocSpec,
    ScenarioSpec,
    Source,
    ValidatorLevel,
)


def test_source_enum_values() -> None:
    assert Source.SLACK == "slack"
    assert Source.NOTION == "notion"
    assert Source.GRANOLA == "granola"
    assert Source.GITHUB == "github"
    assert Source.LINEAR == "linear"
    assert Source.SENTRY == "sentry"
    assert Source.CLAUDE_CODE == "claude_code"


def test_cadence_enum_values() -> None:
    assert Cadence.DAILY == "daily"
    assert Cadence.WEEKLY == "weekly"
    assert Cadence.BIWEEKLY == "biweekly"
    assert Cadence.MONTHLY == "monthly"
    assert Cadence.SPRINT == "sprint"
    assert Cadence.AD_HOC == "ad_hoc"


def test_archetype_construct_and_frozen() -> None:
    a = Archetype(
        name="STANDUP_UPDATE",
        category=Category.RECURRING,
        cadence=Cadence.DAILY,
        sources_used=(Source.SLACK,),
        cast_size=(1, 1),
        needs_planner_call=False,
        validator_level=ValidatorLevel.NAME_ONLY,
    )
    assert a.name == "STANDUP_UPDATE"
    assert a.cadence == Cadence.DAILY
    with pytest.raises(FrozenInstanceError):
        a.name = "OTHER"  # type: ignore[misc]


def test_doc_spec_construct_and_frozen() -> None:
    ts = datetime(2026, 5, 1, 9, 0, 0, tzinfo=UTC)
    doc = DocSpec(
        id="scn-standup-gh-alice-2026-05-01-slack-0",
        source=Source.SLACK,
        occurred_at=ts,
        channel="#standup",
        page_section=None,
        text="Yesterday: shipped payments. Today: auth-service - fix token refresh. Blockers: none.",
        thread_parent_id=None,
        personas=("gh:alice",),
        services_mentioned=("payments",),
    )
    assert doc.channel == "#standup"
    with pytest.raises(FrozenInstanceError):
        doc.text = "mutated"  # type: ignore[misc]


def test_scenario_spec_construct() -> None:
    ts = datetime(2026, 5, 1, 9, 0, 0, tzinfo=UTC)
    doc = DocSpec(
        id="scn-standup-gh-alice-2026-05-01-slack-0",
        source=Source.SLACK,
        occurred_at=ts,
        channel="#standup",
        page_section=None,
        text="Yesterday: shipped auth. Today: payments - fix retry. Blockers: none.",
        thread_parent_id=None,
        personas=("gh:alice",),
        services_mentioned=("auth",),
    )
    spec = ScenarioSpec(
        id="scn-standup-gh-alice-2026-05-01",
        archetype_name="STANDUP_UPDATE",
        instance_ts=ts,
        cast=("gh:alice",),
        affected_services=("auth",),
        doc_specs=(doc,),
    )
    assert len(spec.doc_specs) == 1
    assert spec.cast == ("gh:alice",)
