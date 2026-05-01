"""WorldModel: immutable structure derived from input repos.

The deterministic layer's output. Every narrative-layer call (planner,
writer, validator) consumes this as cached prompt context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
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
    label: str                      # "active-2026-W12"
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
    from scripts.synth.extractor.manifests import Manifest
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


def infer_services(signals: list[RepoSignals]) -> tuple[Service, ...]:
    """Each repo contributes 1+ Service. Collisions on bare name get
    qualified by the repo's last URL segment (e.g. "A/payments")."""
    candidates: list[tuple[str, str, Manifest]] = []  # (svc_name, repo_url, manifest)
    for sig in signals:
        for m in sig.manifests:
            if m.name:
                candidates.append((m.name, sig.url, m))

    # Detect collisions across repos
    name_to_repos: dict[str, set[str]] = defaultdict(set)
    for name, repo_url, _ in candidates:
        name_to_repos[name].add(repo_url)

    services: list[Service] = []
    seen: set[tuple[str, str]] = set()  # (name, repo_url) — dedupe within a repo
    for name, repo_url, m in candidates:
        if (name, repo_url) in seen:
            continue
        seen.add((name, repo_url))
        if len(name_to_repos[name]) > 1:
            qualified = f"{repo_url.rsplit('/', 1)[-1]}/{name}"
        else:
            qualified = name
        services.append(
            Service(
                name=name,
                qualified=qualified,
                repo_url=repo_url,
                kind=_infer_kind(m),
                description=m.description,
                owners=(),
                recent_activity=0.0,
                deploy_target=None,
            )
        )
    return tuple(services)


import math  # noqa: E402


def _recency_decay(ts: datetime, now: datetime, half_life_days: float = 30.0) -> float:
    """Future-dated timestamps clamp to delta_days=0 so weight ≤ 1.0."""
    delta_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 0.5 ** (delta_days / half_life_days)


_TOPIC_KIND_WEIGHT = {
    TopicKind.PR: 1.0,
    TopicKind.ISSUE: 0.8,
    TopicKind.COMMIT: 0.5,
    TopicKind.README_SECTION: 0.3,
    TopicKind.BRANCH: 0.4,
}


def build_topic_pool(
    signals: list[RepoSignals],
    services: tuple[Service, ...],
    now: datetime,
) -> tuple[Topic, ...]:
    # Sorted for deterministic Topic.mentioned_services tuple order.
    service_names = sorted({s.name for s in services} | {s.qualified for s in services})

    topics: list[Topic] = []
    for sig in signals:
        for c in sig.commits:
            mentioned_services = tuple(
                n for n in service_names if n.lower() in c.subject.lower() or any(n in f for f in c.files_touched)
            )
            recency = _recency_decay(c.ts, now)
            weight = recency * _TOPIC_KIND_WEIGHT[TopicKind.COMMIT] * (
                1.0 + math.log1p(len(mentioned_services))
            )
            topics.append(
                Topic(
                    text=c.subject,
                    kind=TopicKind.COMMIT,
                    repo_url=sig.url,
                    ts=c.ts,
                    mentioned_services=mentioned_services,
                    mentioned_people=(),
                    weight=weight,
                )
            )
        for issue in sig.issues or ():
            mentioned = tuple(n for n in service_names if n.lower() in issue.title.lower())
            recency = _recency_decay(issue.updated_at, now)
            topics.append(
                Topic(
                    text=issue.title, kind=TopicKind.ISSUE,
                    repo_url=sig.url, ts=issue.updated_at,
                    mentioned_services=mentioned, mentioned_people=(),
                    weight=recency * _TOPIC_KIND_WEIGHT[TopicKind.ISSUE] * (
                        1.0 + math.log1p(len(mentioned))
                    ),
                )
            )
        for pr in sig.prs or ():
            mentioned = tuple(n for n in service_names if n.lower() in pr.title.lower())
            base_ts = pr.merged_at or pr.created_at
            recency = _recency_decay(base_ts, now)
            topics.append(
                Topic(
                    text=pr.title, kind=TopicKind.PR,
                    repo_url=sig.url, ts=base_ts,
                    mentioned_services=mentioned, mentioned_people=(),
                    weight=recency * _TOPIC_KIND_WEIGHT[TopicKind.PR] * (
                        1.0 + math.log1p(len(mentioned))
                    ),
                )
            )
        for branch in sig.branches:
            recency = _recency_decay(branch.last_commit_ts, now)
            topics.append(
                Topic(
                    text=branch.name, kind=TopicKind.BRANCH,
                    repo_url=sig.url, ts=branch.last_commit_ts,
                    mentioned_services=(), mentioned_people=(),
                    weight=recency * _TOPIC_KIND_WEIGHT[TopicKind.BRANCH],
                )
            )

    return tuple(topics)


def _infer_kind(manifest: Manifest) -> ServiceKind:
    """Heuristic: kind from manifest type. Plan 1 keeps it simple; richer
    inference (Dockerfile presence, asyncpg.listen detection, etc.) lives
    in plan 3 if needed."""
    from scripts.synth.extractor.manifests import ManifestKind
    if manifest.kind == ManifestKind.PACKAGE_JSON:
        return ServiceKind.FRONTEND
    if manifest.kind == ManifestKind.FLY_TOML:
        return ServiceKind.API
    if manifest.kind == ManifestKind.DOCKER_COMPOSE:
        return ServiceKind.INFRA
    return ServiceKind.LIB


_GENERIC_CHANNELS = ("#general", "#random", "#incidents", "#engineering", "#announcements")

_FIXED_NOTION_SECTIONS = (
    "Engineering", "Runbooks", "Postmortems", "Architecture",
    "Onboarding", "Product", "People & Hiring",
)


def synthesize_channels(
    services: tuple[Service, ...],
    codeowner_teams: set[str],
) -> tuple[ChannelHint, ...]:
    out: list[ChannelHint] = []

    # Per-service channel for api/worker/frontend
    for svc in services:
        if svc.kind in (ServiceKind.API, ServiceKind.WORKER, ServiceKind.FRONTEND):
            out.append(
                ChannelHint(
                    name=f"#{svc.name}",
                    suggested_topic=svc.description,
                    related_services=(svc.qualified,),
                )
            )

    # Top-5 deploy channels
    top_active = sorted(services, key=lambda s: s.recent_activity, reverse=True)[:5]
    for svc in top_active:
        out.append(
            ChannelHint(
                name=f"#{svc.name}-deploys",
                suggested_topic=None,
                related_services=(svc.qualified,),
            )
        )

    # Team channels
    for team in sorted(codeowner_teams):
        out.append(ChannelHint(name=f"#team-{team}", suggested_topic=None, related_services=()))

    # Generic
    for g in _GENERIC_CHANNELS:
        out.append(ChannelHint(name=g, suggested_topic=None, related_services=()))

    # Dedupe by name (preserve first occurrence)
    seen: set[str] = set()
    deduped: list[ChannelHint] = []
    for c in out:
        if c.name in seen:
            continue
        seen.add(c.name)
        deduped.append(c)
    return tuple(deduped)


def synthesize_sections(services: tuple[Service, ...]) -> tuple[SectionHint, ...]:
    out: list[SectionHint] = []
    for title in _FIXED_NOTION_SECTIONS:
        out.append(SectionHint(title=title, related_services=()))

    top10 = sorted(services, key=lambda s: s.recent_activity, reverse=True)[:10]
    for svc in top10:
        out.append(
            SectionHint(
                title=f"{svc.name} runbook",
                related_services=(svc.qualified,),
            )
        )
    return tuple(out)


def build_dep_graph(
    signals: list[RepoSignals],
    services: tuple[Service, ...],
) -> tuple[DepEdge, ...]:
    by_name = {s.name: s for s in services}

    edges: list[DepEdge] = []
    for sig in signals:
        for m in sig.manifests:
            if not m.name:
                continue
            from_svc = by_name.get(m.name)
            if not from_svc:
                continue
            for dep_name in m.dependencies:
                to_svc = by_name.get(dep_name)
                if not to_svc:
                    continue
                if to_svc.qualified == from_svc.qualified:
                    continue  # don't record self-edges
                edges.append(
                    DepEdge(
                        from_service=from_svc.qualified,
                        to_service=to_svc.qualified,
                        source_repo=sig.url,
                    )
                )
    return tuple(edges)


def compute_time_anchors(signals: list[RepoSignals]) -> tuple[TimeAnchor, ...]:
    """Cluster commit timestamps by ISO week. Each non-empty week becomes
    a TimeAnchor with activity_score = number of commits that week."""
    week_counts: dict[tuple[int, int], int] = defaultdict(int)
    for sig in signals:
        for c in sig.commits:
            year, week, _ = c.ts.isocalendar()
            week_counts[(year, week)] += 1

    anchors: list[TimeAnchor] = []
    for (year, week), count in sorted(week_counts.items()):
        # ISO week → start (Monday) of that week
        # date.fromisocalendar exists in 3.8+
        start_date = date.fromisocalendar(year, week, 1)
        start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=UTC)
        anchors.append(
            TimeAnchor(
                label=f"active-{year}-W{week:02d}",
                start=start,
                end=start + timedelta(days=7),
                activity_score=float(count),
            )
        )
    return tuple(anchors)


def merge_world_model(
    signals: list[RepoSignals],
    *,
    company_name: str,
    seed: int,
    min_threshold: int,
    max_personas: int,
    now: datetime,
) -> WorldModel:
    """Compose all merger steps into the immutable WorldModel."""
    people = canonicalize_people(signals, min_threshold=min_threshold, max_personas=max_personas)
    services = infer_services(signals)
    topic_pool = build_topic_pool(signals, services=services, now=now)

    # Codeowner team set: anything looking like @<team> with a slash (e.g. @org/team)
    codeowner_teams: set[str] = set()
    for sig in signals:
        for rule in sig.codeowners:
            for owner in rule.owners:
                if "/" in owner:  # @org/team
                    codeowner_teams.add(owner.split("/", 1)[1])

    channels = synthesize_channels(services, codeowner_teams=codeowner_teams)
    sections = synthesize_sections(services)
    dep_graph = build_dep_graph(signals, services)
    time_anchors = compute_time_anchors(signals)

    repos = tuple(
        RepoSummary(url=s.url, sha=s.latest_sha, default_branch=s.default_branch)
        for s in signals
    )
    sha_set = {s.url: s.latest_sha for s in signals}

    return WorldModel(
        repos=repos,
        people=people,
        services=services,
        topic_pool=topic_pool,
        channels=channels,
        notion_sections=sections,
        time_anchors=time_anchors,
        dep_graph=dep_graph,
        company_name=company_name,
        seed=seed,
        extracted_at=now,
        sha_set=sha_set,
    )
