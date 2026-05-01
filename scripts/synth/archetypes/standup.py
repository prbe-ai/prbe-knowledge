"""STANDUP_UPDATE archetype — daily Slack standup messages.

For each working day (Mon-Fri) in time_window, for each top-N persona by
activity_score who has at least one service in the OwnershipIndex, emit one
ScenarioSpec with one DocSpec (a Slack message to #standup).

Text template:
  "Yesterday: shipped {topic_a}. Today: {service} - {topic_b}. Blockers: none."

topic_a and topic_b are drawn from world.topic_pool filtered to topics whose
mentioned_services overlap the persona's services and ts >= day - 7 days.
If only one topic is available, topic_b falls back to "ongoing work".
If no topics are available, both slots use "ongoing work".
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from scripts.synth.archetypes.base import (
    Archetype,
    Cadence,
    Category,
    DocSpec,
    ScenarioSpec,
    Source,
    ValidatorLevel,
)
from scripts.synth.ownership import OwnershipIndex
from scripts.synth.scenarios import TimeWindow
from scripts.synth.world_model import WorldModel

STANDUP_UPDATE = Archetype(
    name="STANDUP_UPDATE",
    category=Category.RECURRING,
    cadence=Cadence.DAILY,
    sources_used=(Source.SLACK,),
    cast_size=(1, 1),
    needs_planner_call=False,
    validator_level=ValidatorLevel.NAME_ONLY,
)


def _working_days(window_end: datetime, days: int) -> list[date]:
    """Return Mon-Fri dates in [window_end - days, window_end], ascending."""
    result: list[date] = []
    end_date = window_end.date()
    start_date = end_date - timedelta(days=days - 1)
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # 0=Mon, 4=Fri
            result.append(current)
        current += timedelta(days=1)
    return result


def _safe_id(canonical_id: str) -> str:
    """Make canonical_id filesystem-safe by replacing ':' with '-'."""
    return canonical_id.replace(":", "-")


def build_standup_specs(
    world: WorldModel,
    ownership: OwnershipIndex,
    time_window: TimeWindow,
    seed: int,
    top_n: int = 5,
) -> tuple[ScenarioSpec, ...]:
    """Build STANDUP_UPDATE ScenarioSpecs for all working days in time_window.

    The `seed` argument is accepted for API symmetry across archetype builders;
    STANDUP_UPDATE achieves determinism via sorted iteration (no random selection),
    so the seed is currently unused.
    """
    end: datetime = time_window.end
    days: int = time_window.days
    work_days = _working_days(end, days)

    # Top-N personas by activity_score (descending), then canonical_id for tie-break.
    sorted_people = sorted(
        world.people,
        key=lambda p: (-p.activity_score, p.canonical_id),
    )[:top_n]

    specs: list[ScenarioSpec] = []

    for work_day in work_days:
        day_start = datetime(work_day.year, work_day.month, work_day.day, 9, 0, 0, tzinfo=UTC)
        lookback = day_start - timedelta(days=7)

        for person in sorted_people:
            person_services = ownership.services_by_person.get(person.canonical_id, ())
            if not person_services:
                continue

            primary_service = person_services[0]

            # Filter topics: mentioned_services overlaps person_services AND ts in lookback window.
            person_svc_set = set(person_services)
            relevant_topics = [
                t for t in world.topic_pool
                if t.ts is not None
                and lookback <= t.ts < day_start
                and (set(t.mentioned_services) & person_svc_set)
            ]

            # Sort by weight desc, then text for determinism.
            relevant_topics.sort(key=lambda t: (-t.weight, t.text))

            if len(relevant_topics) >= 2:
                topic_a_text = relevant_topics[0].text[:50]
                topic_b_text = relevant_topics[1].text[:50]
            elif len(relevant_topics) == 1:
                topic_a_text = relevant_topics[0].text[:50]
                topic_b_text = "ongoing work"
            else:
                topic_a_text = "ongoing work"
                topic_b_text = "ongoing work"

            text = (
                f"Yesterday: shipped {topic_a_text}. "
                f"Today: {primary_service} - {topic_b_text}. "
                f"Blockers: none."
            )

            safe_id = _safe_id(person.canonical_id)
            doc_id = f"scn-standup-{safe_id}-{work_day.isoformat()}-slack-0"
            scenario_id = f"scn-standup-{safe_id}-{work_day.isoformat()}"

            doc_spec = DocSpec(
                id=doc_id,
                source=Source.SLACK,
                occurred_at=day_start,
                channel="#standup",
                page_section=None,
                text=text,
                thread_parent_id=None,
                personas=(person.canonical_id,),
                services_mentioned=tuple(person_services),
            )
            spec = ScenarioSpec(
                id=scenario_id,
                archetype_name="STANDUP_UPDATE",
                instance_ts=day_start,
                cast=(person.canonical_id,),
                affected_services=tuple(person_services),
                doc_specs=(doc_spec,),
            )
            specs.append(spec)

    return tuple(specs)
