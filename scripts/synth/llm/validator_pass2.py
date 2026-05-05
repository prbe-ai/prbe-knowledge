"""Validator Pass 2 — LLM consistency check over a complete scenario's documents.

One generate_structured() call per scenario. The LLM is asked whether the docs
tell a consistent story with the ScenarioSpec. Violations are returned as
Pass2Violation objects. The threshold rule: if violations > 30% of docs,
passed is forced to False.

Uses generate_structured (not generate) so the output schema is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from scripts.synth.archetypes.base import ScenarioSpec
from scripts.synth.llm.base import LlmClientProtocol, LlmRequest
from scripts.synth.output.base import SynthDoc

if TYPE_CHECKING:
    from scripts.synth.world_model import WorldModel

_PROMPT_PATH = Path(__file__).parent / "prompts" / "validator_pass2.txt"

_VIOLATION_THRESHOLD = 0.30


@dataclass(frozen=True)
class Pass2Violation:
    doc_id: str
    issue: str


@dataclass(frozen=True)
class Pass2Result:
    passed: bool
    violations: tuple[Pass2Violation, ...]


class _ViolationSchema(BaseModel):
    doc_id: str
    issue: str


class Pass2OutputSchema(BaseModel):
    passed: bool
    violations: list[_ViolationSchema]


class StructuredOutputValidationError(Exception):
    """Raised when the LLM returns a dict that does not match Pass2OutputSchema."""


def _format_scenario_spec(scenario: ScenarioSpec) -> str:
    lines = [
        f"ID: {scenario.id}",
        f"Archetype: {scenario.archetype_name}",
        f"Title: {getattr(scenario, 'title', '(none)')}",
        f"Summary: {getattr(scenario, 'summary', '(none)')}",
        f"Cast: {', '.join(scenario.cast)}",
        f"Affected services: {', '.join(scenario.affected_services)}",
    ]
    for field in ("root_cause", "decision", "outcome"):
        val = getattr(scenario, field, None)
        if val:
            lines.append(f"{field.replace('_', ' ').title()}: {val}")
    return "\n".join(lines)


def _format_documents(docs: tuple[SynthDoc, ...]) -> str:
    parts: list[str] = []
    for doc in docs:
        source_val = doc.source.value if hasattr(doc.source, "value") else str(doc.source)
        parts.append(
            f"[{doc.id}] source={source_val}\n{doc.text[:2000]}"
        )
    return "\n\n".join(parts)


async def validate_pass2(
    scenario: ScenarioSpec,
    docs: tuple[SynthDoc, ...],
    world: WorldModel,
    client: LlmClientProtocol,
    model: str,
) -> Pass2Result:
    """Single LLM call asking 'do these docs tell a consistent story?'

    Raises:
        ValueError: if docs is empty.
        StructuredOutputValidationError: if the LLM returns an invalid schema.
    """
    if not docs:
        raise ValueError("validate_pass2 requires at least one docs entry; got empty tuple.")

    template = _PROMPT_PATH.read_text(encoding="utf-8")

    allowed_services = ", ".join(
        sorted({s.qualified for s in world.services} | {s.name for s in world.services})
    )
    allowed_people = ", ".join(
        sorted(
            {p.display_name for p in world.people if p.display_name}
            | {p.gh_username for p in world.people if p.gh_username}
        )
    )
    allowed_channels = ", ".join(sorted(ch.name for ch in world.channels))

    prompt = (
        template
        .replace("{scenario_spec}", _format_scenario_spec(scenario))
        .replace("{allowed_services}", allowed_services)
        .replace("{allowed_people}", allowed_people)
        .replace("{allowed_channels}", allowed_channels)
        .replace("{documents}", _format_documents(docs))
    )

    req = LlmRequest(
        model=model,
        system=(
            "You are a strict consistency validator. Output only valid JSON matching "
            "the specified schema. Do not add explanation outside the JSON."
        ),
        prompt=prompt,
        max_tokens=1024,
        temperature=0.0,
    )

    raw = await client.generate_structured(req, Pass2OutputSchema)

    try:
        output = Pass2OutputSchema.model_validate(raw)
    except ValidationError as exc:
        raise StructuredOutputValidationError(
            f"Pass 2 validator LLM returned invalid schema: {exc}"
        ) from exc

    violations = tuple(
        Pass2Violation(doc_id=v.doc_id, issue=v.issue)
        for v in output.violations
    )

    # Threshold enforcement: force passed=False if violation rate > 30%
    violation_rate = len(violations) / len(docs)
    passed = output.passed and violation_rate <= _VIOLATION_THRESHOLD

    return Pass2Result(passed=passed, violations=violations)
