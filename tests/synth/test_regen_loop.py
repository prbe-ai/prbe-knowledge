"""Tests for regen_loop — per-scenario async orchestrator with budget."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

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
from scripts.synth.output.base import SynthDoc
from scripts.synth.regen import RegenResult, RoundReport, regen_loop
from scripts.synth.validator import CombinedValidatorResult, Violation


def _archetype() -> Archetype:
    return Archetype(
        name="INCIDENT",
        category=Category.PLOT,
        cadence=Cadence.AD_HOC,
        sources_used=(Source.SLACK,),
        cast_size=(1, 3),
        needs_planner_call=True,
        validator_level=ValidatorLevel.STRICT,
        eval_question_count=0,
        spec_template_path=None,
    )


def _spec() -> ScenarioSpec:
    ts = datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC)
    return ScenarioSpec(
        id="scn-1",
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


def _doc(doc_id: str, text: str) -> SynthDoc:
    return SynthDoc(
        id=doc_id,
        source=Source.SLACK,
        source_event_id=doc_id,
        text=text,
        occurred_at=datetime(2026, 4, 12, 14, 0, 0, tzinfo=UTC),
        channel="#incidents",
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-1",
        archetype="INCIDENT",
        personas=("gh:alice",),
        services_mentioned=("payments",),
        priority=10,
    )


@pytest.mark.asyncio
async def test_regen_loop_succeeds_on_round_1_when_initial_passes() -> None:
    """If validator passes on the very first call, regen_loop returns
    succeeded=True with rounds=[].

    NOTE: Callers (run_scenarios) only invoke regen_loop AFTER an initial
    failure, so this case is defensive — but the contract is clear.
    """
    docs = (_doc("d0", "ok"),)

    async def validator(_docs: tuple[SynthDoc, ...]) -> CombinedValidatorResult:
        return CombinedValidatorResult(
            pass1_violations=(),
            pass2_result=None,
            failing_doc_ids=(),
            should_drop=False,
        )

    writer = MagicMock()
    writer.regenerate = AsyncMock(return_value="should not be called")

    result = await regen_loop(
        spec=_spec(),
        archetype=_archetype(),
        docs=docs,
        max_rounds=3,
        writer=writer,
        validate_fn=validator,
        world=MagicMock(),
        company_ctx=MagicMock(),
    )
    assert isinstance(result, RegenResult)
    assert result.succeeded is True
    assert result.rounds == []
    assert writer.regenerate.await_count == 0


@pytest.mark.asyncio
async def test_regen_loop_succeeds_after_one_round() -> None:
    """Initial state: d0 fails Pass 1. Round 1: writer regenerates, validator passes."""
    initial_docs = (_doc("d0", "auto-scaling went bad"),)

    call_count = {"n": 0}

    async def validator(_docs: tuple[SynthDoc, ...]) -> CombinedValidatorResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: failing
            return CombinedValidatorResult(
                pass1_violations=(Violation(doc_id="d0", out_of_world=("auto-scaling",)),),
                pass2_result=None,
                failing_doc_ids=("d0",),
                should_drop=True,
            )
        # Subsequent: passing
        return CombinedValidatorResult(
            pass1_violations=(),
            pass2_result=None,
            failing_doc_ids=(),
            should_drop=False,
        )

    writer = MagicMock()
    writer.regenerate = AsyncMock(return_value="payments service spiked errors")

    result = await regen_loop(
        spec=_spec(),
        archetype=_archetype(),
        docs=initial_docs,
        max_rounds=3,
        writer=writer,
        validate_fn=validator,
        world=MagicMock(),
        company_ctx=MagicMock(),
    )
    assert result.succeeded is True
    assert len(result.rounds) == 1
    assert isinstance(result.rounds[0], RoundReport)
    assert result.rounds[0].round_num == 1
    assert result.rounds[0].failing_doc_ids == ("d0",)
    assert result.final_docs[0].text == "payments service spiked errors"
    assert writer.regenerate.await_count == 1


@pytest.mark.asyncio
async def test_regen_loop_exhausts_budget_and_fails() -> None:
    """Validator never passes. Loop tries 3 rounds, then returns succeeded=False."""
    docs = (_doc("d0", "auto-scaling"),)

    async def validator(_docs: tuple[SynthDoc, ...]) -> CombinedValidatorResult:
        return CombinedValidatorResult(
            pass1_violations=(Violation(doc_id="d0", out_of_world=("auto-scaling",)),),
            pass2_result=None,
            failing_doc_ids=("d0",),
            should_drop=True,
        )

    writer = MagicMock()
    writer.regenerate = AsyncMock(return_value="still has auto-scaling")

    result = await regen_loop(
        spec=_spec(),
        archetype=_archetype(),
        docs=docs,
        max_rounds=3,
        writer=writer,
        validate_fn=validator,
        world=MagicMock(),
        company_ctx=MagicMock(),
    )
    assert result.succeeded is False
    assert len(result.rounds) == 3
    assert writer.regenerate.await_count == 3
    # never_converged tracking
    assert "d0" in result.never_converged_doc_ids


@pytest.mark.asyncio
async def test_regen_loop_tracks_per_round_passing_and_failing_docs() -> None:
    """Two-doc scenario: d0 always passes; d1 fails round 1, passes round 2.

    Asserts the per-round trajectory captures both passing and failing
    doc ids per the user's 'what went wrong AND what went right' decision.
    """
    docs = (_doc("d0", "ok"), _doc("d1", "auto-scaling"))

    call_count = {"n": 0}

    async def validator(_docs: tuple[SynthDoc, ...]) -> CombinedValidatorResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return CombinedValidatorResult(
                pass1_violations=(Violation(doc_id="d1", out_of_world=("auto-scaling",)),),
                pass2_result=None,
                failing_doc_ids=("d1",),
                should_drop=True,
            )
        return CombinedValidatorResult(
            pass1_violations=(),
            pass2_result=None,
            failing_doc_ids=(),
            should_drop=False,
        )

    writer = MagicMock()
    writer.regenerate = AsyncMock(return_value="payments errors spiked")

    result = await regen_loop(
        spec=_spec(),
        archetype=_archetype(),
        docs=docs,
        max_rounds=3,
        writer=writer,
        validate_fn=validator,
        world=MagicMock(),
        company_ctx=MagicMock(),
    )
    assert result.succeeded is True
    assert result.rounds[0].failing_doc_ids == ("d1",)
    assert "d0" in result.rounds[0].passing_doc_ids
