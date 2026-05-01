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


# ---------------------------------------------------------------------------
# WorldModelMerger — combines RepoSignals[] into a single WorldModel.
#
# Implemented across tasks 11-16. Each function is independently testable
# so the merger pipeline (Task 17) can compose them confidently.
# ---------------------------------------------------------------------------

from collections import defaultdict  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from scripts.synth.extractor.repo import RepoSignals


def canonicalize_people(
    signals: list[RepoSignals],
    *,
    min_threshold: int,
    max_personas: int,
) -> tuple[Person, ...]:
    """Merge committers + GH contributors into canonical Persons.

    Precedence for canonical_id:
      1. gh:<username> if a contributor entry mentions an email/name we see
      2. email:<lowercased> if no GH match
    Display name = the GH name if available, else the most-frequent commit name.
    Activity = total commit count across all repos.
    """
    # email -> gh_username (from contributor records)
    email_to_gh: dict[str, str] = {}
    # gh_username -> display_name + email_aliases
    gh_meta: dict[str, dict] = {}
    for sig in signals:
        for c in (sig.contributors or ()):
            for email in c.email_aliases:
                email_to_gh[email.lower()] = c.gh_username
            gh_meta.setdefault(
                c.gh_username,
                {"display_name": c.display_name, "emails": set()},
            )
            gh_meta[c.gh_username]["emails"].update(e.lower() for e in c.email_aliases)
            if c.display_name:
                gh_meta[c.gh_username]["display_name"] = c.display_name

    # Build a name -> gh_username map for secondary matching, but only when
    # the display name is unambiguous (maps to exactly one GH contributor).
    name_to_gh: dict[str, str] = {}
    name_counts: dict[str, int] = defaultdict(int)
    for _gh_username, meta in gh_meta.items():
        dn = meta.get("display_name")
        if dn:
            name_counts[dn] += 1
    for gh_username, meta in gh_meta.items():
        dn = meta.get("display_name")
        if dn and name_counts[dn] == 1:
            name_to_gh[dn] = gh_username

    # Pass 1: augment email_to_gh from commit name matches (order-independent —
    # only adds, never overwrites). Must complete before activity aggregation so
    # that every commit sees a stable email_to_gh regardless of commit order.
    for sig in signals:
        for commit in sig.commits:
            email = commit.author_email.lower()
            if email in email_to_gh:
                continue
            gh = name_to_gh.get(commit.author_name)
            if gh:
                email_to_gh[email] = gh
                gh_meta[gh]["emails"].add(email)

    # Pass 2: activity aggregation — pure read of stable email_to_gh.
    activity: dict[str, int] = defaultdict(int)
    repos_active_in: dict[str, set[str]] = defaultdict(set)
    aliases: dict[str, set[str]] = defaultdict(set)
    display_names: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sig in signals:
        for commit in sig.commits:
            email = commit.author_email.lower()
            gh = email_to_gh.get(email)
            cid = f"gh:{gh}" if gh else f"email:{email}"
            activity[cid] += 1
            repos_active_in[cid].add(sig.url)
            aliases[cid].add(commit.author_email.lower())
            display_names[cid][commit.author_name] += 1

    # Even contributors with zero recent commits should appear if they
    # show up in the GH contributor list — but only if the merger run
    # considers them above threshold. Per spec we drop low-activity, so
    # we leave the activity counter as is.

    rows: list[Person] = []
    for cid, count in activity.items():
        if count < min_threshold:
            continue
        gh_username: str | None = None
        if cid.startswith("gh:"):
            gh_username = cid.removeprefix("gh:")
            display = gh_meta.get(gh_username, {}).get("display_name") or gh_username
            aliases[cid].update(gh_meta.get(gh_username, {}).get("emails", set()))
        else:
            # most-frequent commit name
            display = max(display_names[cid].items(), key=lambda kv: kv[1])[0]

        rows.append(
            Person(
                canonical_id=cid,
                gh_username=gh_username,
                display_name=display,
                email_aliases=tuple(sorted(aliases[cid])),
                role_hint=None,                          # filled later by service-owner inference
                repos_active_in=tuple(sorted(repos_active_in[cid])),
                activity_score=float(count),
            )
        )

    rows.sort(key=lambda p: p.activity_score, reverse=True)
    return tuple(rows[:max_personas])
