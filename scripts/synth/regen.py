"""Validator regen loop — orchestrates per-doc regeneration when a plot
scenario fails strict validation.

Public surface:
- format_failure_context: build the failure block for a regen prompt
- splice_regenerated: merge regenerated text back into a docs tuple
- regen_loop: per-scenario async orchestrator (max N rounds)
- RegenResult, RoundReport: result shapes for callers + observability

Decisions (locked from 2026-05-05 handoff):
- Per-doc regen, not per-scenario.
- Pass 1 + Pass 2 violations feed a single unified prompt.
- Default round budget 3 (configurable via Profile.regen_max_rounds).
- On exhaustion: drop scenario, terminal log includes survival info.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from scripts.synth.llm.validator_pass2 import Pass2Result
from scripts.synth.output.base import SynthDoc
from scripts.synth.validator import Violation

if TYPE_CHECKING:
    from scripts.synth.archetypes.base import Archetype, ScenarioSpec
    from scripts.synth.company_context import CompanyContext
    from scripts.synth.llm.writer import LLMWriter
    from scripts.synth.validator import CombinedValidatorResult
    from scripts.synth.world_model import WorldModel


def format_failure_context(
    *,
    pass1_violations: tuple[Violation, ...],
    pass2_result: Pass2Result | None,
    target_doc_id: str,
) -> str:
    """Render Pass 1 + Pass 2 violations for `target_doc_id` as a prompt block.

    Returns "" when neither pass flagged the target. The string is meant to
    be interpolated into writer_regen.txt as the `{failure_context}` field.
    """
    lines: list[str] = []

    pass1_for_doc = [v for v in pass1_violations if v.doc_id == target_doc_id]
    if pass1_for_doc:
        tokens = sorted({t for v in pass1_for_doc for t in v.out_of_world})
        lines.append(
            "Pass 1 (out-of-world tokens): "
            + ", ".join(tokens)
            + " — these names are NOT in the WorldModel allowlist. "
            "Replace them or rephrase to avoid them."
        )

    if pass2_result is not None:
        pass2_for_doc = [v for v in pass2_result.violations if v.doc_id == target_doc_id]
        for v in pass2_for_doc:
            lines.append(f"Pass 2 (consistency issue): {v.issue}")

    return "\n".join(lines)


def splice_regenerated(
    original_docs: tuple[SynthDoc, ...],
    *,
    regenerated_text_by_doc_id: dict[str, str],
) -> tuple[SynthDoc, ...]:
    """Return a new tuple where named docs have new `text`, everything else is identical.

    Preserves doc order, count, and every SynthDoc field (id, source,
    source_event_id, occurred_at, channel, page_id, thread_parent_id,
    scenario_id, archetype, personas, services_mentioned, priority) — only
    `text` changes for entries listed in `regenerated_text_by_doc_id`.

    Raises:
        ValueError: if `regenerated_text_by_doc_id` references a doc id
            that is not present in `original_docs`.
    """
    if not regenerated_text_by_doc_id:
        return original_docs

    known_ids = {d.id for d in original_docs}
    unknown = set(regenerated_text_by_doc_id.keys()) - known_ids
    if unknown:
        raise ValueError(
            f"splice_regenerated: doc id(s) not in original scenario: {sorted(unknown)}"
        )

    out: list[SynthDoc] = []
    for d in original_docs:
        new_text = regenerated_text_by_doc_id.get(d.id)
        if new_text is None:
            out.append(d)
        else:
            out.append(replace(d, text=new_text))
    return tuple(out)


@dataclass(frozen=True)
class RoundReport:
    """One regen round's outcome — fed into structured logs."""

    round_num: int
    failing_doc_ids: tuple[str, ...]
    passing_doc_ids: tuple[str, ...]
    violation_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RegenResult:
    """Outcome of regen_loop for one scenario."""

    succeeded: bool
    final_docs: tuple[SynthDoc, ...]
    rounds: list[RoundReport]
    survived_doc_ids: tuple[str, ...]   # Docs that passed validation in the final state
    never_converged_doc_ids: tuple[str, ...]  # Docs failing in every round attempted


def _collect_violation_reasons(result: CombinedValidatorResult) -> tuple[str, ...]:
    reasons: list[str] = []
    for v in result.pass1_violations:
        reasons.append(f"{v.doc_id}: out_of_world={list(v.out_of_world)}")
    if result.pass2_result is not None:
        for v in result.pass2_result.violations:
            reasons.append(f"{v.doc_id}: {v.issue}")
    return tuple(reasons)


async def regen_loop(
    *,
    spec: ScenarioSpec,
    archetype: Archetype,
    docs: tuple[SynthDoc, ...],
    max_rounds: int,
    writer: LLMWriter,
    validate_fn: Callable[[tuple[SynthDoc, ...]], Awaitable[CombinedValidatorResult]],
    world: WorldModel,
    company_ctx: CompanyContext,
) -> RegenResult:
    """Run up to ``max_rounds`` regen rounds. Returns RegenResult.

    Each round:
      1. Call ``validate_fn(current_docs)``.
      2. If ``should_drop`` is False -> return ``RegenResult(succeeded=True, ...)``.
      3. Else build failure_context per failing doc, call ``writer.regenerate``
         once per failing doc, splice, increment round.
    On budget exhaustion -> return ``RegenResult(succeeded=False, ...)``.

    The caller (run_scenarios) is expected to handle structured logging
    based on the returned RoundReports + RegenResult fields. This keeps the
    loop pure and easier to unit-test.

    ``validate_fn`` is injected (not imported) so tests can substitute mocks
    without monkeypatching the validator module.
    """
    # archetype is accepted for future use (e.g. archetype-specific regen
    # policies / logging) and to keep run_scenarios' call site stable.
    del archetype

    current = docs
    rounds: list[RoundReport] = []
    ever_failed: set[str] = set()
    last_failing: tuple[str, ...] = ()

    for round_num in range(1, max_rounds + 1):
        result = await validate_fn(current)

        all_doc_ids = tuple(d.id for d in current)
        failing = result.failing_doc_ids
        failing_set = set(failing)
        passing = tuple(d_id for d_id in all_doc_ids if d_id not in failing_set)

        if not result.should_drop:
            # Success — early exit. Don't append a round report for the
            # passing call (callers can infer success from rounds list len
            # vs final state).
            return RegenResult(
                succeeded=True,
                final_docs=current,
                rounds=rounds,
                survived_doc_ids=all_doc_ids,
                never_converged_doc_ids=(),
            )

        # Failure: record this round, regenerate, splice
        ever_failed.update(failing)
        last_failing = failing
        violation_reasons = _collect_violation_reasons(result)
        rounds.append(
            RoundReport(
                round_num=round_num,
                failing_doc_ids=failing,
                passing_doc_ids=passing,
                violation_reasons=violation_reasons,
            )
        )

        # Regenerate each failing doc — but only if there's a non-empty
        # failure_context to feed the LLM. Skipping empty contexts avoids
        # spending tokens on a doc the validator flagged but neither pass
        # gave concrete feedback for.
        replacements: dict[str, str] = {}
        for failing_id in failing:
            target = next((d for d in current if d.id == failing_id), None)
            if target is None:
                continue
            failure_context = format_failure_context(
                pass1_violations=result.pass1_violations,
                pass2_result=result.pass2_result,
                target_doc_id=failing_id,
            )
            if not failure_context:
                continue
            new_text = await writer.regenerate(
                spec=spec,
                target_doc=target,
                prior_docs_full=current,
                failure_context=failure_context,
                world=world,
                company_ctx=company_ctx,
            )
            replacements[failing_id] = new_text

        current = splice_regenerated(current, regenerated_text_by_doc_id=replacements)

    # Budget exhausted: report what survived and what didn't
    final_passing = tuple(d.id for d in current if d.id not in set(last_failing))
    return RegenResult(
        succeeded=False,
        final_docs=current,
        rounds=rounds,
        survived_doc_ids=final_passing,
        never_converged_doc_ids=tuple(sorted(ever_failed & set(last_failing))),
    )
