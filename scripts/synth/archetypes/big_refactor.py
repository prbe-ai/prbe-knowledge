"""BIG_REFACTOR plot archetype — async scenario builder.

Emits 4-6 docs per scenario (all LLM Writer — no templated doc):
  1. GitHub issue (RFC)         — long-form proposal (RFC author)
  2. Notion architecture doc   — decision record after debate (RFC author)
  3. Slack #engineering parent — RFC author opens debate thread
  4. Slack #engineering reply  — debater 1
  5. Slack #engineering reply  — debater 2 (optional; count from spec)
  6. Linear migration ticket   — follow-up work item (RFC author)

The planner emits 1 eval question:
  - hard-temporal: "Why did we move {X} from {A} to {B}? Who proposed it?"

Slack thread wiring: the parent doc's source_event_id is set as thread_parent_id
on all reply docs. The GitHub doc carries archetype="BIG_REFACTOR" which causes
the GitHub wrapper to emit an issues.opened envelope (not pull_request.opened).
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

BIG_REFACTOR = Archetype(
    name="BIG_REFACTOR",
    category=Category.PLOT,
    cadence=Cadence.AD_HOC,
    sources_used=(Source.GITHUB, Source.NOTION, Source.SLACK, Source.LINEAR),
    cast_size=(3, 4),
    needs_planner_call=True,
    validator_level=ValidatorLevel.STRICT,
    eval_question_count=1,
    spec_template_path="planner_big_refactor.txt",
)

# Resolve prompt template path once at module load time.
_big_refactor_archetype = with_abs_prompt_path(BIG_REFACTOR)


async def build_big_refactor_scenarios(
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
    """Yield (spec, materialized_docs) for each BIG_REFACTOR scenario.

    Slack thread is multi-message: parent + 1-2 replies. Reply docs carry
    thread_parent_id referencing the Slack parent doc's source_event_id.
    GitHub doc is an issue (RFC), not a pull request — the archetype field
    on the SynthDoc causes the GitHub wrapper to emit issues.opened.

    The number of Slack docs (2 or 3) is driven by the planner's
    source_emissions output, surfaced via the count of Slack DocSpecs in
    spec.doc_specs.

    Instance timestamps are picked deterministically from world.time_anchors
    ranked by activity_score descending, cycling with (seed + i) modulo spread.
    Uses anchor.end as the reference timestamp (Task 14 convention).
    """
    anchors = sorted(world.time_anchors, key=lambda a: a.activity_score, reverse=True)
    n_anchors = max(len(anchors), 1)

    for i in range(count):
        # Pick instance_ts deterministically: use anchor.end as the reference ts
        anchor_idx = (seed + i) % n_anchors
        instance_ts: datetime = anchors[anchor_idx].end if anchors else time_window.end

        # --- Planner call (1 LLM call) ---
        spec = await planner.plan(
            archetype=_big_refactor_archetype,
            world=world,
            ownership=ownership,
            company_ctx=company_ctx,
            instance_ts=instance_ts,
            rng_seed=seed + i,
        )

        # Determine Slack count from planner's doc_specs (source_emissions driven)
        slack_count = sum(1 for ds in spec.doc_specs if ds.source == Source.SLACK)
        if slack_count < 2:
            slack_count = 2
        elif slack_count > 3:
            slack_count = 3

        # Emission order: github (issue RFC), notion, slack (parent + replies), linear
        prior_docs: tuple[SynthDoc, ...] = ()
        docs: list[SynthDoc] = []

        # 1. GitHub issue (RFC)
        github_text = await writer.write(
            spec=spec,
            source=Source.GITHUB,
            emission_index=0,
            prior_emitted_docs=prior_docs,
            world=world,
            company_ctx=company_ctx,
        )
        github_doc = SynthDoc(
            id=f"{spec.id}-github-issue-0",
            source=Source.GITHUB,
            source_event_id=f"{spec.id}-github-issue-0",
            text=github_text,
            occurred_at=instance_ts,
            channel=None,
            page_id=None,
            thread_parent_id=None,
            scenario_id=spec.id,
            archetype=BIG_REFACTOR.name,
            personas=tuple(spec.cast),
            services_mentioned=tuple(spec.affected_services),
            priority=10,
        )
        docs.append(github_doc)
        prior_docs = (*prior_docs, github_doc)

        # 2. Notion architecture doc
        notion_text = await writer.write(
            spec=spec,
            source=Source.NOTION,
            emission_index=0,
            prior_emitted_docs=prior_docs,
            world=world,
            company_ctx=company_ctx,
        )
        notion_doc = SynthDoc(
            id=f"{spec.id}-notion-0",
            source=Source.NOTION,
            source_event_id=f"{spec.id}-notion-0",
            text=notion_text,
            occurred_at=instance_ts,
            channel=None,
            page_id=f"{spec.id}-notion-0",
            thread_parent_id=None,
            scenario_id=spec.id,
            archetype=BIG_REFACTOR.name,
            personas=tuple(spec.cast),
            services_mentioned=tuple(spec.affected_services),
            priority=20,
        )
        docs.append(notion_doc)
        prior_docs = (*prior_docs, notion_doc)

        # 3. Slack thread — parent + 1-2 replies
        slack_parent_event_id: str | None = None
        for slack_idx in range(slack_count):
            slack_text = await writer.write(
                spec=spec,
                source=Source.SLACK,
                emission_index=slack_idx,
                prior_emitted_docs=prior_docs,
                world=world,
                company_ctx=company_ctx,
            )
            slack_doc_id = f"{spec.id}-slack-{slack_idx}"
            slack_doc = SynthDoc(
                id=slack_doc_id,
                source=Source.SLACK,
                source_event_id=slack_doc_id,
                text=slack_text,
                occurred_at=instance_ts,
                channel="#engineering",
                page_id=None,
                thread_parent_id=slack_parent_event_id,  # None for parent; parent id for replies
                scenario_id=spec.id,
                archetype=BIG_REFACTOR.name,
                personas=tuple(spec.cast),
                services_mentioned=tuple(spec.affected_services),
                priority=30 + slack_idx,
            )
            docs.append(slack_doc)
            prior_docs = (*prior_docs, slack_doc)
            # Record parent event id after first Slack doc
            if slack_idx == 0:
                slack_parent_event_id = slack_doc_id

        # 4. Linear migration ticket
        linear_text = await writer.write(
            spec=spec,
            source=Source.LINEAR,
            emission_index=0,
            prior_emitted_docs=prior_docs,
            world=world,
            company_ctx=company_ctx,
        )
        linear_doc = SynthDoc(
            id=f"{spec.id}-linear-0",
            source=Source.LINEAR,
            source_event_id=f"{spec.id}-linear-0",
            text=linear_text,
            occurred_at=instance_ts,
            channel=None,
            page_id=None,
            thread_parent_id=None,
            scenario_id=spec.id,
            archetype=BIG_REFACTOR.name,
            personas=tuple(spec.cast),
            services_mentioned=tuple(spec.affected_services),
            priority=40,
        )
        docs.append(linear_doc)

        yield spec, docs
