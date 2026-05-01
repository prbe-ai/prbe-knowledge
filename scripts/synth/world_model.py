"""WorldModel: immutable structure derived from input repos.

The deterministic layer's output. Every narrative-layer call (planner,
writer, validator) consumes this as cached prompt context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class ServiceKind(StrEnum):
    API = "api"
    WORKER = "worker"
    FRONTEND = "frontend"
    CLI = "cli"
    LIB = "lib"
    INFRA = "infra"
    UNKNOWN = "unknown"


class TopicKind(StrEnum):
    COMMIT = "commit"
    PR = "pr"
    ISSUE = "issue"
    README_SECTION = "readme_section"
    BRANCH = "branch"


@dataclass(frozen=True)
class RepoSummary:
    url: str
    sha: str
    default_branch: str


@dataclass(frozen=True)
class Person:
    canonical_id: str               # "gh:alice" if known, else hash-derived
    gh_username: str | None
    display_name: str
    email_aliases: tuple[str, ...]
    role_hint: str | None           # inferred from CODEOWNERS coverage
    repos_active_in: tuple[str, ...]
    activity_score: float


@dataclass(frozen=True)
class Service:
    name: str
    qualified: str                  # "repo/svc" if collision; else == name
    repo_url: str                   # primary owning repo
    kind: ServiceKind
    description: str | None
    owners: tuple[str, ...]         # canonical Person ids
    recent_activity: float
    deploy_target: str | None       # e.g. fly app name


@dataclass(frozen=True)
class Topic:
    text: str
    kind: TopicKind
    repo_url: str
    ts: datetime | None
    mentioned_services: tuple[str, ...]
    mentioned_people: tuple[str, ...]
    weight: float


@dataclass(frozen=True)
class ChannelHint:
    name: str                       # "#payments-deploys"
    suggested_topic: str | None
    related_services: tuple[str, ...]


@dataclass(frozen=True)
class SectionHint:
    title: str                      # "Engineering > Payments runbook"
    related_services: tuple[str, ...]


@dataclass(frozen=True)
class TimeAnchor:
    label: str                      # "active period 2026-W12"
    start: datetime
    end: datetime
    activity_score: float


@dataclass(frozen=True)
class DepEdge:
    from_service: str               # qualified name
    to_service: str                 # qualified name
    source_repo: str                # the repo whose manifest declared the dep


@dataclass(frozen=True)
class WorldModel:
    repos: tuple[RepoSummary, ...]
    people: tuple[Person, ...]
    services: tuple[Service, ...]
    topic_pool: tuple[Topic, ...]
    channels: tuple[ChannelHint, ...]
    notion_sections: tuple[SectionHint, ...]
    time_anchors: tuple[TimeAnchor, ...]
    dep_graph: tuple[DepEdge, ...]
    company_name: str
    seed: int
    extracted_at: datetime
    sha_set: dict[str, str] = field(default_factory=dict)  # repo_url → sha
