"""ScenarioRunner — walks the active archetype set, builds specs, materializes
SynthDocs. Plan 2 only sees templated builders; Plan 3 will branch on
`archetype.needs_planner_call` to invoke an LLM Planner instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from scripts.synth.archetypes.base import DocSpec, ScenarioSpec, Source
from scripts.synth.output.base import SynthDoc

if TYPE_CHECKING:
    from scripts.synth.ownership import OwnershipIndex
    from scripts.synth.profile import Profile
    from scripts.synth.world_model import WorldModel


@dataclass(frozen=True)
class TimeWindow:
    """Half-open window [end - days, end). All times UTC."""

    end: datetime
    days: int


def working_days(window: TimeWindow) -> Iterator[date]:
    """Mon-Fri dates in the window, chronological."""
    start = (window.end - timedelta(days=window.days)).date()
    stop = window.end.date()
    cursor = start
    while cursor < stop:
        if cursor.weekday() < 5:  # 0=Mon ... 4=Fri
            yield cursor
        cursor = cursor + timedelta(days=1)


def weekly_mondays(window: TimeWindow) -> Iterator[date]:
    """Mondays inside the window, chronological.

    Advances `start` to the nearest Monday on or after it, then yields
    every 7th day while before `stop`.
    """
    start = (window.end - timedelta(days=window.days)).date()
    stop = window.end.date()
    cursor = start if start.weekday() == 0 else start + timedelta(days=(7 - start.weekday()) % 7)
    while cursor < stop:
        yield cursor
        cursor = cursor + timedelta(days=7)


def _materialize(doc_spec: DocSpec, scenario: ScenarioSpec) -> SynthDoc:
    """Convert a planner-emitted DocSpec into the wire-shaped SynthDoc."""
    return SynthDoc(
        id=doc_spec.id,
        source=doc_spec.source,
        source_event_id=doc_spec.id,
        text=doc_spec.text,
        occurred_at=doc_spec.occurred_at,
        channel=doc_spec.channel,
        page_id=doc_spec.id if doc_spec.source == Source.NOTION else None,
        thread_parent_id=doc_spec.thread_parent_id,
        scenario_id=scenario.id,
        archetype=scenario.archetype_name,
        personas=doc_spec.personas,
        services_mentioned=doc_spec.services_mentioned,
        priority=100,
    )


def run_scenarios(
    world: WorldModel,
    ownership: OwnershipIndex,
    profile: Profile,
    time_window: TimeWindow,
    *,
    archetype_filter: tuple[str, ...] | None = None,
    scenario_limit: int | None = None,
) -> Iterator[SynthDoc]:
    """Walk active archetypes, run their builders, materialize SynthDocs.

    `scenario_limit` caps PER ARCHETYPE (each builder's output is sliced).

    Imports are deferred to call time to avoid a circular import:
    scenarios -> library -> oncall/standup -> scenarios (for TimeWindow).
    """
    # Lazy import to break the circular dependency at module load time.
    from scripts.synth.archetypes.library import BUILDERS, get_active

    active = get_active(profile, archetype_filter=archetype_filter)
    for name in active:
        builder = BUILDERS[name]
        specs = builder(world, ownership, time_window, profile.seed)
        if scenario_limit is not None:
            specs = specs[:scenario_limit]
        for spec in specs:
            for doc_spec in spec.doc_specs:
                yield _materialize(doc_spec, spec)
