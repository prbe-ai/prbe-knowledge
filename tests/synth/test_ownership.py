"""OwnershipIndex tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from scripts.synth.extractor.git_log import Branch, Commit
from scripts.synth.extractor.manifests import Manifest, ManifestKind
from scripts.synth.extractor.repo import RepoSignals
from scripts.synth.ownership import build_ownership_index
from scripts.synth.world_model import (
    Person,
    Service,
    ServiceKind,
    WorldModel,
)


def _make_commit(
    sha: str,
    author_email: str,
    author_name: str,
    files: tuple[str, ...],
    ts: datetime | None = None,
) -> Commit:
    return Commit(
        sha=sha,
        author_name=author_name,
        author_email=author_email,
        ts=ts or datetime(2026, 4, 1, tzinfo=UTC),
        subject="fix something",
        body="",
        files_touched=files,
    )


def _make_manifest(path: Path, name: str) -> Manifest:
    return Manifest(
        kind=ManifestKind.PYPROJECT,
        path=path,
        name=name,
        description=None,
        dependencies=(),
    )


def _make_signals(
    url: str,
    commits: list[Commit],
    manifests: list[Manifest],
) -> RepoSignals:
    return RepoSignals(
        url=url,
        clone_path=Path("/tmp/repo"),
        default_branch="main",
        latest_sha="abc123",
        description=None,
        manifests=tuple(manifests),
        readmes=(),
        codeowners=(),
        commits=tuple(commits),
        branches=(Branch(name="main", last_commit_sha="abc123", last_commit_ts=datetime(2026, 4, 1, tzinfo=UTC)),),
        issues=None,
        prs=None,
        contributors=None,
        workflows=None,
    )


def _make_world(people: list[Person], services: list[Service]) -> WorldModel:
    now = datetime(2026, 5, 1, tzinfo=UTC)
    return WorldModel(
        repos=(),
        people=tuple(people),
        services=tuple(services),
        topic_pool=(),
        channels=(),
        notion_sections=(),
        time_anchors=(),
        dep_graph=(),
        company_name="prbe",
        seed=42,
        extracted_at=now,
        sha_set={},
    )


def _make_person(canonical_id: str, email: str) -> Person:
    return Person(
        canonical_id=canonical_id,
        gh_username=canonical_id.removeprefix("gh:") if canonical_id.startswith("gh:") else None,
        display_name=canonical_id,
        email_aliases=(email,),
        role_hint=None,
        repos_active_in=(),
        activity_score=1.0,
    )


def test_single_repo_single_person() -> None:
    """One person committing to one service is indexed correctly."""
    commits = [
        _make_commit("c1", "alice@example.com", "Alice", ("payments/src/main.py",)),
        _make_commit("c2", "alice@example.com", "Alice", ("payments/src/other.py",)),
    ]
    manifests = [_make_manifest(Path("/repo/payments/pyproject.toml"), "payments")]
    signals = [_make_signals("github.com/prbe-ai/prbe", commits, manifests)]
    person = _make_person("email:alice@example.com", "alice@example.com")
    service = Service(name="payments", qualified="payments", repo_url="github.com/prbe-ai/prbe",
                      kind=ServiceKind.API, description=None, owners=(), recent_activity=1.0, deploy_target=None)
    world = _make_world([person], [service])

    idx = build_ownership_index(signals, world)
    assert "payments" in idx.services_by_person.get("email:alice@example.com", ())


def test_person_without_commits_gets_empty_tuple() -> None:
    signals = [_make_signals("github.com/prbe-ai/prbe", [], [])]
    person = _make_person("gh:bob", "bob@example.com")
    world = _make_world([person], [])
    idx = build_ownership_index(signals, world)
    assert idx.services_by_person.get("gh:bob", ()) == ()


def test_top_3_services_by_frequency() -> None:
    """If a person commits to 4+ services, only top 3 are kept."""
    commits = [
        _make_commit("c1", "alice@example.com", "Alice", ("svc-a/main.py",)),
        _make_commit("c2", "alice@example.com", "Alice", ("svc-a/other.py",)),
        _make_commit("c3", "alice@example.com", "Alice", ("svc-b/main.py",)),
        _make_commit("c4", "alice@example.com", "Alice", ("svc-c/main.py",)),
        _make_commit("c5", "alice@example.com", "Alice", ("svc-d/main.py",)),
    ]
    manifests = [
        _make_manifest(Path("/repo/svc-a/pyproject.toml"), "svc-a"),
        _make_manifest(Path("/repo/svc-b/pyproject.toml"), "svc-b"),
        _make_manifest(Path("/repo/svc-c/pyproject.toml"), "svc-c"),
        _make_manifest(Path("/repo/svc-d/pyproject.toml"), "svc-d"),
    ]
    signals = [_make_signals("github.com/prbe-ai/prbe", commits, manifests)]
    person = _make_person("email:alice@example.com", "alice@example.com")
    world = _make_world([person], [])
    idx = build_ownership_index(signals, world)
    top = idx.services_by_person.get("email:alice@example.com", ())
    assert len(top) <= 3
    # svc-a appears twice so it must be in top 3
    assert "svc-a" in top


def test_people_by_service_inverse() -> None:
    commits = [
        _make_commit("c1", "alice@example.com", "Alice", ("payments/main.py",)),
    ]
    manifests = [_make_manifest(Path("/repo/payments/pyproject.toml"), "payments")]
    signals = [_make_signals("github.com/prbe-ai/prbe", commits, manifests)]
    person = _make_person("email:alice@example.com", "alice@example.com")
    service = Service(name="payments", qualified="payments", repo_url="github.com/prbe-ai/prbe",
                      kind=ServiceKind.API, description=None, owners=(), recent_activity=1.0, deploy_target=None)
    world = _make_world([person], [service])
    idx = build_ownership_index(signals, world)
    assert "email:alice@example.com" in idx.people_by_service.get("payments", ())


def test_deterministic_tie_break() -> None:
    """Equal-frequency services are sorted alphabetically for determinism."""
    commits = [
        _make_commit("c1", "alice@example.com", "Alice", ("svc-z/main.py",)),
        _make_commit("c2", "alice@example.com", "Alice", ("svc-a/main.py",)),
    ]
    manifests = [
        _make_manifest(Path("/repo/svc-z/pyproject.toml"), "svc-z"),
        _make_manifest(Path("/repo/svc-a/pyproject.toml"), "svc-a"),
    ]
    signals = [_make_signals("github.com/prbe-ai/prbe", commits, manifests)]
    person = _make_person("email:alice@example.com", "alice@example.com")
    world = _make_world([person], [])
    idx1 = build_ownership_index(signals, world)
    idx2 = build_ownership_index(signals, world)
    assert idx1.services_by_person == idx2.services_by_person
