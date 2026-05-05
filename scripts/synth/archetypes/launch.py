"""LAUNCH plot archetype — async scenario builder.

Emits 4 docs per scenario (all LLM Writer — no templated doc):
  1. Slack #announcements — launch announcement (product owner)
  2. Notion product page  — one-pager (product owner)
  3. Linear feature ticket — marked done (engineering lead)
  4. GitHub merge PR      — the PR that shipped the feature (engineering lead)

The planner emits 2 eval questions:
  - easy:               "When did {feature} ship?"
  - medium-cross-source: "Which customer was the design partner for {feature}?"
                         (or "Who owned the {feature} launch?" if no customers)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime

from scripts.synth.archetypes.base import (
    Archetype,
    Cadence,
    Category,
    ScenarioSpec,
    Source,
    ValidatorLevel,
)
from scripts.synth.archetypes.plot_base import with_abs_prompt_path
from scripts.synth.company_context import CompanyContext
from scripts.synth.llm.planner import LLMPlanner
from scripts.synth.llm.writer import LLMWriter
from scripts.synth.output.base import SynthDoc
from scripts.synth.ownership import OwnershipIndex
from scripts.synth.scenarios import TimeWindow
from scripts.synth.world_model import WorldModel

LAUNCH = Archetype(
    name="LAUNCH",
    category=Category.PLOT,
    cadence=Cadence.AD_HOC,
    sources_used=(Source.SLACK, Source.NOTION, Source.LINEAR, Source.GITHUB),
    cast_size=(2, 3),
    needs_planner_call=True,
    validator_level=ValidatorLevel.STRICT,
    eval_question_count=2,
    spec_template_path="planner_launch.txt",
)

_LAUNCH_SOURCES = (Source.SLACK, Source.NOTION, Source.LINEAR, Source.GITHUB)

# Resolve prompt template path once at module load time.
_launch_archetype = with_abs_prompt_path(LAUNCH)


async def build_launch_scenarios(
    world: WorldModel,
    ownership: OwnershipIndex,
    company_ctx: CompanyContext,
    time_window: TimeWindow,
    seed: int,
    *,
    planner: LLMPlanner,
    writer: LLMWriter,
    count: int,
) -> AsyncIterator[tuple[ScenarioSpec, list[SynthDoc]]]:
    """Yield (spec, materialized_docs) for each LAUNCH scenario.

    All 4 docs are LLM Writer calls (no templated doc). Instance timestamps
    are picked deterministically from world.time_anchors ranked by
    activity_score descending, cycling with (seed + i) modulo spread.
    Uses anchor.end as the reference timestamp, matching Task 14 convention.
    """
    anchors = sorted(world.time_anchors, key=lambda a: a.activity_score, reverse=True)
    n_anchors = max(len(anchors), 1)

    for i in range(count):
        # Pick instance_ts deterministically: use anchor.end as the reference ts
        anchor_idx = (seed + i) % n_anchors
        instance_ts: datetime = anchors[anchor_idx].end if anchors else time_window.end

        # --- Planner call (1 LLM call) ---
        spec = await planner.plan(
            archetype=_launch_archetype,
            world=world,
            ownership=ownership,
            company_ctx=company_ctx,
            instance_ts=instance_ts,
            rng_seed=seed + i,
        )

        # --- LLM Writer calls for all 4 sources (no templated doc) ---
        prior_docs: tuple[SynthDoc, ...] = ()
        docs: list[SynthDoc] = []

        for emission_idx, source in enumerate(_LAUNCH_SOURCES):
            text = await writer.write(
                spec=spec,
                source=source,
                emission_index=emission_idx,
                prior_emitted_docs=prior_docs,
                world=world,
                company_ctx=company_ctx,
            )
            doc = SynthDoc(
                id=f"{spec.id}-{source.value}-{emission_idx}",
                source=source,
                source_event_id=f"{spec.id}-{source.value}-{emission_idx}",
                text=text,
                occurred_at=instance_ts,
                channel="#announcements" if source == Source.SLACK else None,
                page_id=f"{spec.id}-notion-0" if source == Source.NOTION else None,
                thread_parent_id=None,
                scenario_id=spec.id,
                archetype=LAUNCH.name,
                # spec.cast is tuple[str, ...] of canonical_ids
                personas=tuple(spec.cast),
                services_mentioned=tuple(spec.affected_services),
                priority=10 + emission_idx,
            )
            docs.append(doc)
            prior_docs = (*prior_docs, doc)

        yield spec, docs
