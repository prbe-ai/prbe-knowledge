"""Shared helpers for all plot archetype builders.

Consumed by incident.py, launch.py, big_refactor.py. Not imported
by templated archetype builders (standup.py, oncall.py).
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.synth.archetypes.base import Archetype, ScenarioSpec
from scripts.synth.ownership import OwnershipIndex
from scripts.synth.world_model import WorldModel

# Imported at call sites only to avoid a circular import with company_context.
if TYPE_CHECKING:
    from scripts.synth.company_context import CompanyContext


def with_abs_prompt_path(archetype: Archetype) -> Archetype:
    """Return a copy of the archetype with spec_template_path resolved to an
    absolute path under scripts/synth/llm/prompts/.

    Plot archetypes ship their planner prompt templates as relative filenames
    in spec_template_path (e.g. "planner_incident.txt"). This helper resolves
    them to absolute paths so the planner's assemble_planner_prompt works
    regardless of cwd.
    """
    if archetype.spec_template_path is None:
        return archetype
    if Path(archetype.spec_template_path).is_absolute():
        return archetype
    abs_path = Path(__file__).parent.parent / "llm" / "prompts" / archetype.spec_template_path
    return dataclasses.replace(archetype, spec_template_path=str(abs_path))


@dataclass(frozen=True)
class CastMember:
    """A persona assigned to a specific role within a scenario."""

    canonical_id: str
    role_in_scenario: str


def pick_cast(
    world: WorldModel,
    ownership: OwnershipIndex,
    *,
    size: int,
    role_hints: tuple[str, ...],
    rng_seed: int,
) -> tuple[CastMember, ...]:
    """Select a deterministic cast from world.people respecting role_hints and seed.

    People are first sorted by activity_score descending (then canonical_id for
    tie-breaking) to prefer active personas. The RNG is seeded so the same
    (world, seed) always produces the same cast, even as people rotate.

    Role hints are assigned in order to the selected people. If size > len(role_hints),
    remaining personas receive a generic "participant" role.
    """
    rng = random.Random(rng_seed)
    candidates = sorted(world.people, key=lambda p: (-p.activity_score, p.canonical_id))
    if not candidates:
        return ()
    actual_size = min(size, len(candidates))
    chosen = rng.sample(candidates, k=actual_size)
    roles = list(role_hints) + ["participant"] * max(0, actual_size - len(role_hints))
    return tuple(
        CastMember(canonical_id=p.canonical_id, role_in_scenario=roles[i])
        for i, p in enumerate(chosen)
    )


def evidence_doc_keys(scenario: ScenarioSpec) -> dict[str, list[str]]:
    """Derive expected evidence doc paths from a ScenarioSpec.

    Returns a dict mapping source value (e.g. "slack") to a list of
    raw/<source>/<doc_id>.json paths, one per DocSpec with that source.
    """
    grouped: dict[str, list[str]] = {}
    for doc_spec in scenario.doc_specs:
        source_val = doc_spec.source.value if hasattr(doc_spec.source, "value") else str(doc_spec.source)
        path = f"raw/{source_val}/{doc_spec.id}.json"
        grouped.setdefault(source_val, []).append(path)
    return grouped


def _world_summary(world: WorldModel) -> str:
    """Compact text summary of the world for use in planner prompts."""
    people_lines = "\n".join(
        f"  - {p.canonical_id} ({p.display_name}, role_hint={p.role_hint or 'none'}, "
        f"activity={p.activity_score:.0f})"
        for p in world.people[:20]
    )
    return (
        f"Company: {world.company_name}\n"
        f"Seed: {world.seed}\n"
        f"People ({len(world.people)} total, showing up to 20):\n{people_lines}"
    )


def _cast_pool(world: WorldModel) -> str:
    """List of canonical_id values available for cast selection."""
    return "\n".join(p.canonical_id for p in world.people)


def _services_table(world: WorldModel) -> str:
    """Markdown table of services."""
    rows = ["| qualified | kind | owners |", "| --- | --- | --- |"]
    for svc in world.services:
        owners = ", ".join(svc.owners) if svc.owners else "—"
        rows.append(f"| {svc.qualified} | {svc.kind} | {owners} |")
    return "\n".join(rows)


def _recent_topics(world: WorldModel, *, limit: int = 10) -> str:
    """Top-weighted topics as a bulleted list."""
    sorted_topics = sorted(world.topic_pool, key=lambda t: -t.weight)[:limit]
    if not sorted_topics:
        return "(no recent topics)"
    return "\n".join(f"- {t.text}" for t in sorted_topics)


def assemble_planner_prompt(
    archetype: Archetype,
    world: WorldModel,
    ownership: OwnershipIndex,
    company_ctx: CompanyContext,
    instance_ts: datetime,
    rng_seed: int,
) -> str:
    """Read the prompt template at archetype.spec_template_path and substitute placeholders.

    Placeholders substituted (all must appear in the template):
      {world_summary}     — compact world summary
      {cast_pool}         — newline-separated list of canonical_ids
      {services_table}    — markdown table of services
      {recent_topics}     — bullet list of top-weighted topics
      {company_context}   — company name + customer list from CompanyContext
      {instance_ts}       — ISO 8601 timestamp of the scenario instance

    Raises:
        ValueError: if archetype.spec_template_path is None.
        FileNotFoundError: if the template file does not exist.
    """
    if archetype.spec_template_path is None:
        raise ValueError(
            f"Archetype {archetype.name!r} has no spec_template_path — "
            "cannot assemble planner prompt."
        )
    template_path = Path(archetype.spec_template_path)
    template = template_path.read_text(encoding="utf-8")

    customers_text = (
        ", ".join(str(c) for c in company_ctx.customers)
        if company_ctx.customers
        else "(none)"
    )
    company_context_block = (
        f"Company: {company_ctx.name}\n"
        f"Customers: {customers_text}"
    )

    return template.format(
        world_summary=_world_summary(world),
        cast_pool=_cast_pool(world),
        services_table=_services_table(world),
        recent_topics=_recent_topics(world),
        company_context=company_context_block,
        instance_ts=instance_ts.isoformat(),
    )
