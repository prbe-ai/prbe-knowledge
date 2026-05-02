"""LLMPlanner — produces a ScenarioSpec for plot archetypes via a single LLM call.

Validation:
  - Pydantic validates the structured output shape.
  - World-model validation checks cast, services, and channel references.
  - On failure: retry up to max_retries times with the violation in the user prompt.
  - After max_retries+1 total attempts: raise PlannerValidationError.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

from scripts.synth.archetypes.base import Archetype, DocSpec, ScenarioSpec, Source
from scripts.synth.archetypes.plot_base import CastMember, assemble_planner_prompt  # noqa: F401
from scripts.synth.llm.base import LlmClientProtocol, LlmRequest

if TYPE_CHECKING:
    from scripts.synth.company_context import CompanyContext
    from scripts.synth.ownership import OwnershipIndex
    from scripts.synth.world_model import WorldModel


@dataclass(frozen=True)
class TimelineEvent:
    ts: datetime
    source: str       # Source.value
    kind: str         # "alert", "message", "ticket", "doc", "pr"
    channel: str | None = None


@dataclass(frozen=True)
class EvalQuestion:
    input: str
    answer_substring: str
    difficulty: str   # "easy" | "medium-cross-source" | "hard-temporal"


class _CastMemberSchema(BaseModel):
    canonical_id: str
    role_in_scenario: str


class _TimelineEventSchema(BaseModel):
    ts: datetime
    source: str
    kind: str
    channel: str | None = None


class _EvalQuestionSchema(BaseModel):
    input: str
    answer_substring: str
    difficulty: str


class PlannerOutputSchema(BaseModel):
    """Pydantic schema validated against generate_structured output."""

    title: str
    summary: str
    cast: list[_CastMemberSchema]
    affected_services: list[str]
    affected_repos: list[str]
    root_cause: str | None = None
    decision: str | None = None
    outcome: str | None = None
    timeline: list[_TimelineEventSchema]
    source_emissions: dict[str, int]
    eval_questions: list[_EvalQuestionSchema]


class PlannerValidationError(Exception):
    """Raised when planner output references entities not in WorldModel/CompanyContext."""


def _validate_against_world(
    output: PlannerOutputSchema,
    world: WorldModel,
) -> str | None:
    """Return a violation description string, or None if the output is valid."""
    known_people = {p.canonical_id for p in world.people}
    for member in output.cast:
        if member.canonical_id not in known_people:
            return (
                f"cast member {member.canonical_id!r} does not exist in the world. "
                f"Valid canonical_ids: {sorted(known_people)}"
            )

    known_services = {s.qualified for s in world.services} | {s.name for s in world.services}
    for svc in output.affected_services:
        if svc not in known_services:
            return (
                f"affected_service {svc!r} is not a known service. "
                f"Valid services: {sorted(known_services)}"
            )

    known_channels = {ch.name for ch in world.channels}
    for event in output.timeline:
        if event.channel is not None and event.channel not in known_channels:
            return (
                f"timeline channel {event.channel!r} is not a known channel. "
                f"Valid channels: {sorted(known_channels)}"
            )

    return None


def _build_doc_specs(
    output: PlannerOutputSchema,
    scenario_id: str,
    instance_ts: datetime,
) -> tuple[DocSpec, ...]:
    """Derive DocSpec stubs from source_emissions (text left blank for Writer)."""
    specs: list[DocSpec] = []
    cast_ids = tuple(m.canonical_id for m in output.cast)
    affected = tuple(output.affected_services)
    for source_val, count in output.source_emissions.items():
        try:
            source = Source(source_val)
        except ValueError:
            continue
        for i in range(count):
            doc_id = f"{scenario_id}-{source_val}-{i}"
            # Find matching timeline event for channel/section hints
            channel: str | None = None
            page_section: str | None = None
            for event in output.timeline:
                if event.source == source_val:
                    channel = event.channel
                    break
            specs.append(
                DocSpec(
                    id=doc_id,
                    source=source,
                    occurred_at=instance_ts,
                    channel=channel,
                    page_section=page_section,
                    text="",  # filled by LLMWriter
                    thread_parent_id=None,
                    personas=cast_ids,
                    services_mentioned=affected,
                )
            )
    return tuple(specs)


class LLMPlanner:
    def __init__(
        self,
        client: LlmClientProtocol,
        model: str,
        max_retries: int = 2,
    ) -> None:
        self._client = client
        self._model = model
        self._max_retries = max_retries

    async def plan(
        self,
        archetype: Archetype,
        world: WorldModel,
        ownership: OwnershipIndex,
        company_ctx: CompanyContext,
        instance_ts: datetime,
        rng_seed: int,
    ) -> ScenarioSpec:
        """Single LLM call (with up-to-max_retries retries on validation failure).

        On retry, the violation is appended to the user prompt so the LLM corrects.
        Raises PlannerValidationError after max_retries+1 total attempts.
        """
        base_prompt = (
            assemble_planner_prompt(archetype, world, ownership, company_ctx, instance_ts, rng_seed)
            if archetype.spec_template_path
            else (
                f"Generate a {archetype.name} scenario. "
                f"Instance timestamp: {instance_ts.isoformat()}. "
                f"Cast pool: {', '.join(p.canonical_id for p in world.people)}. "
                f"Services: {', '.join(s.qualified for s in world.services)}."
            )
        )

        user_prompt = base_prompt
        violation: str | None = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0 and violation:
                user_prompt = (
                    base_prompt
                    + f"\n\n[CORRECTION REQUIRED — attempt {attempt + 1}]\n"
                    + f"Your previous response had this violation: {violation}\n"
                    + "Please correct it and regenerate the full output."
                )

            req = LlmRequest(
                model=self._model,
                system=(
                    f"You are a synthetic data generator. "
                    f"Produce a {archetype.name} scenario spec as structured JSON. "
                    f"Use only entities from the provided world model."
                ),
                prompt=user_prompt,
                max_tokens=4096,
                temperature=0.0,
            )

            raw = await self._client.generate_structured(req, PlannerOutputSchema)
            output = PlannerOutputSchema.model_validate(raw)

            violation = _validate_against_world(output, world)
            if violation is None:
                break
        else:
            raise PlannerValidationError(
                f"Planner failed after max_retries={self._max_retries} attempts. "
                f"Last violation: {violation}"
            )

        safe_title = (
            output.title.lower().replace(" ", "-")[:40].strip("-")
            if output.title
            else "scenario"
        )
        scenario_id = f"scn-{archetype.name.lower()}-{safe_title}-{instance_ts.date().isoformat()}"

        doc_specs = _build_doc_specs(output, scenario_id, instance_ts)

        eval_questions = tuple(
            EvalQuestion(
                input=q.input,
                answer_substring=q.answer_substring,
                difficulty=q.difficulty,
            )
            for q in output.eval_questions
        )

        return ScenarioSpec(
            id=scenario_id,
            archetype_name=archetype.name,
            instance_ts=instance_ts,
            cast=tuple(m.canonical_id for m in output.cast),
            affected_services=tuple(output.affected_services),
            doc_specs=doc_specs,
            title=output.title,
            summary=output.summary,
            root_cause=output.root_cause,
            decision=output.decision,
            outcome=output.outcome,
            eval_questions=eval_questions,
        )
