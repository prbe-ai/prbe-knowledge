"""ScenarioRunner — walks the active archetype set, builds specs, materializes
SynthDocs. Plan 2 only sees templated builders; Plan 3 branches on
`archetype.needs_planner_call` to invoke async PLOT_BUILDERS driven by an
LLM Planner/Writer.

Re-exports: ScenarioSpec and EvalQuestion are importable from this module so
eval-artifact writers and tests can use a stable, top-level path.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from scripts.synth.archetypes.base import DocSpec, ScenarioSpec, Source
from scripts.synth.eval_question import EvalQuestion
from scripts.synth.output.base import SynthDoc
from shared.logging import get_logger

# Re-export so downstream code can do:
#   from scripts.synth.scenarios import ScenarioSpec, EvalQuestion
__all__ = [
    "EvalQuestion",
    "ScenarioSpec",
    "TimeWindow",
    "run_scenarios",
    "weekly_mondays",
    "working_days",
]

if TYPE_CHECKING:
    from scripts.synth.company_context import CompanyContext
    from scripts.synth.llm.base import LlmClientProtocol
    from scripts.synth.llm.planner import LLMPlanner
    from scripts.synth.llm.writer import LLMWriter
    from scripts.synth.ownership import OwnershipIndex
    from scripts.synth.profile import Profile
    from scripts.synth.world_model import WorldModel

log = get_logger(__name__)


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
    cursor = (
        start if start.weekday() == 0 else start + timedelta(days=(7 - start.weekday()) % 7)
    )
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


async def run_scenarios(
    world: WorldModel,
    ownership: OwnershipIndex,
    profile: Profile,
    time_window: TimeWindow,
    *,
    archetype_filter: tuple[str, ...] | None = None,
    scenario_limit: int | None = None,
    company_ctx: CompanyContext | None = None,
    planner: LLMPlanner | None = None,
    writer: LLMWriter | None = None,
    validator_pass2_client: LlmClientProtocol | None = None,
    validator_pass2_model: str | None = None,
) -> AsyncGenerator[tuple[ScenarioSpec, SynthDoc], None]:
    """Walk active archetypes, validate each scenario, yield (ScenarioSpec, SynthDoc) pairs.

    Each doc is paired with the scenario it came from. Callers that need
    per-scenario state (eval question writers, manifest tallies) deduplicate
    by spec.id.

    Templated archetypes (needs_planner_call=False) → call BUILDERS sync builder.
    Plot archetypes (needs_planner_call=True) → call PLOT_BUILDERS async builder.
      Requires planner, writer, company_ctx; warning logged & skipped if absent.

    Each scenario goes through validate(); dropped if should_drop is True.
    Regen loop deferred — see deferred-cleanup tracker.
    """
    # Lazy import to break the circular dependency at module load time.
    from scripts.synth.archetypes.library import BUILDERS, PLOT_BUILDERS, get_active
    from scripts.synth.llm.structured import StructuredOutputValidationError
    from scripts.synth.validator import validate as combined_validate

    active = get_active(profile, archetype_filter=archetype_filter)
    for name, archetype in active.items():
        archetype_cfg = (profile.raw.get("archetypes") or {}).get(name) or {}
        count = archetype_cfg.get("count")  # may be None for templated builders

        if not archetype.needs_planner_call:
            # Templated path
            specs = BUILDERS[name](world, ownership, time_window, profile.seed)
            if scenario_limit is not None:
                specs = specs[:scenario_limit]
            for spec in specs:
                docs = tuple(_materialize(ds, spec) for ds in spec.doc_specs)
                result = await combined_validate(
                    docs,
                    world,
                    scenario=spec,
                    archetype=archetype,
                    pass2_client=None,
                    pass2_model=None,
                )
                if result.should_drop:
                    log.warning(
                        "templated_scenario_dropped",
                        scenario_id=spec.id,
                        archetype=name,
                        failing=result.failing_doc_ids,
                    )
                    continue
                for doc in docs:
                    yield spec, doc
        else:
            # Plot path
            if planner is None or writer is None or company_ctx is None:
                log.warning(
                    "plot_archetype_skipped_no_llm",
                    archetype=name,
                    hint="Pass --mock-llm or configure LLM clients via profile.llm",
                )
                continue
            plot_builder = PLOT_BUILDERS.get(name)
            if plot_builder is None:
                log.error("no_plot_builder_registered", archetype=name)
                continue
            if count is None or count <= 0:
                continue
            if scenario_limit is not None:
                count = min(count, scenario_limit)
            try:
                async for spec, docs_list in plot_builder(
                    world=world,
                    ownership=ownership,
                    company_ctx=company_ctx,
                    time_window=time_window,
                    seed=profile.seed,
                    planner=planner,
                    writer=writer,
                    count=count,
                ):
                    docs = tuple(docs_list)
                    try:
                        result = await combined_validate(
                            docs,
                            world,
                            scenario=spec,
                            archetype=archetype,
                            pass2_client=validator_pass2_client,
                            pass2_model=validator_pass2_model,
                        )
                    except StructuredOutputValidationError:
                        log.exception("plot_validator_error", scenario_id=spec.id, archetype=name)
                        continue
                    if result.should_drop:
                        # TODO(plan3-cleanup): implement validator regen loop (max 2 rounds,
                        # surgical doc-level replacement preserving thread_parent_id wiring)
                        log.warning(
                            "plot_scenario_dropped",
                            scenario_id=spec.id,
                            archetype=name,
                            failing=result.failing_doc_ids,
                        )
                        continue
                    for doc in docs:
                        yield spec, doc
            except Exception:
                log.exception("plot_archetype_error", archetype=name)
                continue
