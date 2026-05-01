"""Smoke tests for the immutable WorldModel dataclasses + their helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from scripts.synth.world_model import (
    DepEdge,
    Person,
    RepoSummary,
    Service,
    ServiceKind,
    Topic,
    TopicKind,
    WorldModel,
)


def test_worldmodel_is_frozen() -> None:
    wm = WorldModel(
        repos=(),
        people=(),
        services=(),
        topic_pool=(),
        channels=(),
        notion_sections=(),
        time_anchors=(),
        dep_graph=(),
        company_name="acme",
        seed=1,
        extracted_at=datetime.now(UTC),
        sha_set={},
    )
    try:
        wm.seed = 2  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("WorldModel must be frozen")


def test_person_canonical_id_is_required() -> None:
    p = Person(
        canonical_id="gh:alice",
        gh_username="alice",
        display_name="Alice",
        email_aliases=("alice@example.com",),
        role_hint=None,
        repos_active_in=("github.com/x/y",),
        activity_score=12.0,
    )
    assert p.canonical_id == "gh:alice"


def test_service_qualified_name_collision() -> None:
    s = Service(
        name="payments",
        qualified="repo-a/payments",
        repo_url="github.com/x/repo-a",
        kind=ServiceKind.API,
        description=None,
        owners=("gh:alice",),
        recent_activity=1.0,
        deploy_target=None,
    )
    assert s.qualified == "repo-a/payments"


def test_dep_edge_directional() -> None:
    e = DepEdge(from_service="api", to_service="lib", source_repo="x")
    assert e.from_service == "api"


def test_topic_recency_weighted() -> None:
    t = Topic(
        text="auth refactor",
        kind=TopicKind.PR,
        repo_url="github.com/x/y",
        ts=datetime(2026, 4, 1, tzinfo=UTC),
        mentioned_services=("auth-svc",),
        mentioned_people=("gh:alice",),
        weight=0.85,
    )
    assert 0 < t.weight <= 1.0


def test_repo_summary_records_sha() -> None:
    r = RepoSummary(url="github.com/x/y", sha="abcd1234", default_branch="main")
    assert r.sha == "abcd1234"
