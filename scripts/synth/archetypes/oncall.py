"""ON_CALL_HANDOFF archetype — weekly Slack + Notion on-call handoff.

For each Monday in time_window:
  1. Pick outgoing = top_personas[week_index % top_n]
     incoming = top_personas[(week_index + 1) % top_n]
     week_index = ISO week number of the Monday.
  2. Pick up to 3 incidents = topics with kind==ISSUE (fallback COMMIT)
     and ts in [monday - 7d, monday).
  3. Emit 3 DocSpecs:
     - slack-0: Slack parent in #oncall from outgoing, summarizing incidents.
     - slack-1: Slack reply from incoming acknowledging.
     - notion-0: Notion page "On-call handoff <date>" with H2 per incident.
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
from scripts.synth.world_model import TopicKind, WorldModel

ON_CALL_HANDOFF = Archetype(
    name="ON_CALL_HANDOFF",
    category=Category.RECURRING,
    cadence=Cadence.WEEKLY,
    sources_used=(Source.SLACK, Source.NOTION),
    cast_size=(2, 2),
    needs_planner_call=False,
    validator_level=ValidatorLevel.NAME_ONLY,
)


def _mondays(window_end: datetime, days: int) -> list[date]:
    """Return all Mondays in [window_end - days, window_end], ascending."""
    result: list[date] = []
    end_date = window_end.date()
    start_date = end_date - timedelta(days=days - 1)
    current = start_date
    # Advance to first Monday
    while current.weekday() != 0:
        current += timedelta(days=1)
    while current <= end_date:
        result.append(current)
        current += timedelta(days=7)
    return result


def _incident_summary(incidents: list) -> str:
    """Render incident list to a bullet summary for the Slack parent message."""
    if not incidents:
        return "Quiet week, nothing on fire."
    lines = ["Incidents this week:"]
    for t in incidents[:3]:
        lines.append(f"- {t.text[:80]}")
    return "\n".join(lines)


def _notion_body(day: date, incidents: list, outgoing_id: str) -> str:
    """Render Notion page body with H2 per incident."""
    lines = [f"On-call handoff {day.isoformat()}", f"Outgoing: @{outgoing_id.replace('gh:', '')}"]
    if not incidents:
        lines.append("## Status")
        lines.append("Quiet week, nothing on fire.")
    else:
        for t in incidents[:3]:
            lines.append(f"## {t.text[:80]}")
            lines.append("Owner: TBD")
            lines.append("Status: resolved")
    return "\n".join(lines)


def build_oncall_specs(
    world: WorldModel,
    ownership: OwnershipIndex,
    time_window: TimeWindow,
    seed: int,
    top_n: int = 5,
) -> tuple[ScenarioSpec, ...]:
    """Build ON_CALL_HANDOFF ScenarioSpecs for each Monday in time_window.

    The `seed` argument is accepted for API symmetry across archetype builders;
    ON_CALL_HANDOFF achieves determinism via sorted iteration and ISO week
    indexing (no random selection), so the seed is currently unused.
    """
    end: datetime = time_window.end
    days: int = time_window.days
    mondays = _mondays(end, days)

    # Top-N personas by activity_score desc, canonical_id tie-break.
    sorted_people = sorted(
        world.people,
        key=lambda p: (-p.activity_score, p.canonical_id),
    )[:top_n]

    if len(sorted_people) < 2:
        return ()

    specs: list[ScenarioSpec] = []

    for monday in mondays:
        # Use ISO week number as rotation index for determinism.
        week_index = monday.isocalendar()[1]
        outgoing = sorted_people[week_index % len(sorted_people)]
        incoming = sorted_people[(week_index + 1) % len(sorted_people)]

        monday_dt = datetime(monday.year, monday.month, monday.day, 9, 0, 0, tzinfo=UTC)
        lookback = monday_dt - timedelta(days=7)

        # Pick incidents: ISSUE kind preferred, COMMIT as fallback.
        all_in_window = [
            t for t in world.topic_pool
            if t.ts is not None and lookback <= t.ts < monday_dt
        ]
        issues = [t for t in all_in_window if t.kind == TopicKind.ISSUE]
        incidents = issues[:3] if issues else [t for t in all_in_window if t.kind == TopicKind.COMMIT][:3]

        # Stable ordering for determinism: sort by (-weight, text).
        incidents.sort(key=lambda t: (-t.weight, t.text))

        date_str = monday.isoformat()
        slack_parent_id = f"scn-oncall-{date_str}-slack-0"
        slack_reply_id = f"scn-oncall-{date_str}-slack-1"
        notion_id = f"scn-oncall-{date_str}-notion-0"
        scenario_id = f"scn-oncall-{date_str}"

        # slack-0: parent from outgoing
        outgoing_text = _incident_summary(incidents)
        slack_parent = DocSpec(
            id=slack_parent_id,
            source=Source.SLACK,
            occurred_at=monday_dt,
            channel="#oncall",
            page_section=None,
            text=outgoing_text,
            thread_parent_id=None,
            personas=(outgoing.canonical_id,),
            services_mentioned=tuple(ownership.services_by_person.get(outgoing.canonical_id, ())),
        )

        # slack-1: reply from incoming
        incoming_text = "Got it, taking over. Will monitor and escalate if anything resurfaces."
        slack_reply = DocSpec(
            id=slack_reply_id,
            source=Source.SLACK,
            occurred_at=monday_dt,
            channel="#oncall",
            page_section=None,
            text=incoming_text,
            thread_parent_id=slack_parent_id,
            personas=(incoming.canonical_id,),
            services_mentioned=tuple(ownership.services_by_person.get(incoming.canonical_id, ())),
        )

        # notion-0: handoff page
        notion_text = _notion_body(monday, incidents, outgoing.canonical_id)
        notion_page = DocSpec(
            id=notion_id,
            source=Source.NOTION,
            occurred_at=monday_dt,
            channel=None,
            page_section="Engineering > On-call rotation",
            text=notion_text,
            thread_parent_id=None,
            personas=(outgoing.canonical_id, incoming.canonical_id),
            services_mentioned=tuple({
                s
                for cid in (outgoing.canonical_id, incoming.canonical_id)
                for s in ownership.services_by_person.get(cid, ())
            }),
        )

        all_cast = (outgoing.canonical_id, incoming.canonical_id)
        all_services = tuple({
            s
            for cid in all_cast
            for s in ownership.services_by_person.get(cid, ())
        })

        spec = ScenarioSpec(
            id=scenario_id,
            archetype_name="ON_CALL_HANDOFF",
            instance_ts=monday_dt,
            cast=all_cast,
            affected_services=all_services,
            doc_specs=(slack_parent, slack_reply, notion_page),
        )
        specs.append(spec)

    return tuple(specs)
