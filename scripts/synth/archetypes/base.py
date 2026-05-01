"""Archetype base dataclasses and enums.

These types are the shared vocabulary for the scenario layer. Every
archetype builder consumes WorldModel + OwnershipIndex and emits
ScenarioSpec objects composed of DocSpec leaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Source(StrEnum):
    SLACK = "slack"
    NOTION = "notion"
    GRANOLA = "granola"
    GITHUB = "github"
    LINEAR = "linear"
    SENTRY = "sentry"
    CLAUDE_CODE = "claude_code"


class Cadence(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    SPRINT = "sprint"
    AD_HOC = "ad_hoc"


class Category(StrEnum):
    RECURRING = "recurring"
    PLOT = "plot"
    SLOW_BURN = "slow_burn"


class ValidatorLevel(StrEnum):
    STRICT = "strict"
    NAME_ONLY = "name_only"
    NONE = "none"


@dataclass(frozen=True)
class Archetype:
    name: str
    category: Category
    cadence: Cadence
    sources_used: tuple[Source, ...]
    cast_size: tuple[int, int]       # (min, max) personas per scenario
    needs_planner_call: bool         # False for all Plan 2 archetypes
    validator_level: ValidatorLevel  # NAME_ONLY for all Plan 2 archetypes


@dataclass(frozen=True)
class DocSpec:
    """Specification for a single synthetic document before wrapping."""
    id: str
    source: Source
    occurred_at: datetime
    channel: str | None           # Slack channel name (e.g. "#standup")
    page_section: str | None      # Notion section path (e.g. "Engineering > On-call rotation")
    text: str
    thread_parent_id: str | None  # Slack thread parent doc_spec id, if this is a reply
    personas: tuple[str, ...]     # canonical_ids involved
    services_mentioned: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioSpec:
    """One instance of an archetype producing one or more DocSpecs."""
    id: str
    archetype_name: str
    instance_ts: datetime
    cast: tuple[str, ...]           # canonical_ids in this scenario
    affected_services: tuple[str, ...]
    doc_specs: tuple[DocSpec, ...]
