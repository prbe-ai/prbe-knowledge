"""Archetype library — central registry of recurring archetypes.

Plan 2 ships two: STANDUP_UPDATE (daily slack) and ON_CALL_HANDOFF (weekly
slack+notion). Plan 3 registers plot archetypes (INCIDENT, LAUNCH,
BIG_REFACTOR) here alongside their LLM-driven async builders.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

from scripts.synth.archetypes.base import Archetype, ScenarioSpec
from scripts.synth.archetypes.big_refactor import BIG_REFACTOR, build_big_refactor_scenarios
from scripts.synth.archetypes.incident import INCIDENT, build_incident_scenarios
from scripts.synth.archetypes.launch import LAUNCH, build_launch_scenarios
from scripts.synth.archetypes.oncall import ON_CALL_HANDOFF, build_oncall_specs
from scripts.synth.archetypes.standup import STANDUP_UPDATE, build_standup_specs

if TYPE_CHECKING:
    from scripts.synth.output.base import SynthDoc
    from scripts.synth.profile import Profile


ARCHETYPE_LIBRARY: dict[str, Archetype] = {
    "STANDUP_UPDATE": STANDUP_UPDATE,
    "ON_CALL_HANDOFF": ON_CALL_HANDOFF,
    "INCIDENT": INCIDENT,
    "LAUNCH": LAUNCH,
    "BIG_REFACTOR": BIG_REFACTOR,
}

# Builder signatures vary slightly across archetypes (top_n is a kwarg with
# default), so the registry is typed loosely. Callers (run_scenarios) pass
# only the positional args common to all builders.
#
# Only templated (non-planner) archetypes are registered here.
# Plot archetypes (INCIDENT, LAUNCH, BIG_REFACTOR) use PLOT_BUILDERS below;
# run_scenarios branches on archetype.needs_planner_call and never looks up
# plot archetype names in BUILDERS.
BUILDERS: dict[str, Callable[..., tuple[ScenarioSpec, ...]]] = {
    "STANDUP_UPDATE": build_standup_specs,
    "ON_CALL_HANDOFF": build_oncall_specs,
}

# Async plot builders — Plan 3. Each yields (ScenarioSpec, list[SynthDoc]).
# Signature: (world, ownership, company_ctx, time_window, seed, *, planner, writer, count)
PLOT_BUILDERS: dict[str, Callable[..., AsyncIterator[tuple[ScenarioSpec, list[SynthDoc]]]]] = {
    "INCIDENT": build_incident_scenarios,
    "LAUNCH": build_launch_scenarios,
    "BIG_REFACTOR": build_big_refactor_scenarios,
}


def get_active(
    profile: Profile,
    archetype_filter: tuple[str, ...] | None = None,
) -> dict[str, Archetype]:
    """Resolve the set of archetypes to run for this profile.

    Profile's optional `archetypes:` block lets the user disable a per-name
    archetype with `count: 0`. CLI's `--archetypes A,B` further restricts
    via `archetype_filter`. Both filters compose (intersection).
    """
    profile_arch = profile.raw.get("archetypes") or {}
    active: dict[str, Archetype] = {}
    for name, archetype in ARCHETYPE_LIBRARY.items():
        cfg = profile_arch.get(name) or {}
        count = cfg.get("count")
        if count == 0:
            continue
        active[name] = archetype
    if archetype_filter is not None:
        active = {k: v for k, v in active.items() if k in archetype_filter}
    return active
