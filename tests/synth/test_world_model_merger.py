"""WorldModelMerger — turns RepoSignals[] into a single WorldModel.

Person canonicalization is the highest-stakes step (treating two people
as one produces threads where someone replies to themselves)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from scripts.synth.extractor.git_log import Commit
from scripts.synth.extractor.github_api import Contributor
from scripts.synth.extractor.repo import RepoSignals
from scripts.synth.world_model import canonicalize_people


def _commit(email: str, name: str, sha: str = "x") -> Commit:
    return Commit(
        sha=sha, author_name=name, author_email=email,
        ts=datetime(2026, 3, 1, tzinfo=UTC),
        subject="x", body="", files_touched=(),
    )


def _signals(commits, contributors=None) -> RepoSignals:
    return RepoSignals(
        url="github.com/x/y",
        clone_path=Path("/tmp/x"),
        default_branch="main",
        latest_sha="abcd",
        description=None,
        manifests=(),
        readmes=(),
        codeowners=(),
        commits=tuple(commits),
        branches=(),
        issues=None,
        prs=None,
        contributors=contributors,
        workflows=None,
    )


def test_canonicalize_uses_gh_username_when_available() -> None:
    sigs = [
        _signals(
            commits=[_commit("alice@work.com", "Alice"), _commit("alice@home.com", "Alice X")],
            contributors=(
                Contributor(gh_username="alice", display_name="Alice X",
                            email_aliases=("alice@work.com",), contributions=10),
            ),
        )
    ]
    people = canonicalize_people(sigs, min_threshold=1, max_personas=10)
    assert len(people) == 1
    assert people[0].canonical_id == "gh:alice"
    assert "alice@work.com" in people[0].email_aliases
    assert "alice@home.com" in people[0].email_aliases


def test_canonicalize_falls_back_to_email_when_no_gh() -> None:
    sigs = [_signals(commits=[
        _commit("a@x.com", "A", sha="1"),
        _commit("a@x.com", "A", sha="2"),
        _commit("b@x.com", "B", sha="3"),
    ])]
    people = canonicalize_people(sigs, min_threshold=1, max_personas=10)
    canon = sorted(p.canonical_id for p in people)
    assert canon == ["email:a@x.com", "email:b@x.com"]


def test_never_merges_by_display_name_alone() -> None:
    """Two 'John's at different companies must remain separate."""
    sigs = [_signals(commits=[
        _commit("john@a.com", "John"),
        _commit("john@b.com", "John"),
    ])]
    people = canonicalize_people(sigs, min_threshold=1, max_personas=10)
    assert len(people) == 2


def test_min_threshold_drops_low_activity_personas() -> None:
    sigs = [_signals(commits=[
        _commit("alice@x.com", "Alice", sha="1"),
        _commit("alice@x.com", "Alice", sha="2"),
        _commit("alice@x.com", "Alice", sha="3"),
        _commit("once@x.com", "Once-Off", sha="4"),
    ])]
    people = canonicalize_people(sigs, min_threshold=2, max_personas=10)
    assert {p.display_name for p in people} == {"Alice"}


def test_max_personas_caps_pool() -> None:
    sigs = [_signals(commits=[
        _commit(f"u{i}@x.com", f"User {i}", sha=f"s{i}") for i in range(40)
    ])]
    people = canonicalize_people(sigs, min_threshold=1, max_personas=5)
    assert len(people) == 5


def test_repos_active_in_recorded_per_person() -> None:
    sig_a = _signals(commits=[_commit("alice@x.com", "Alice", sha="a")])
    sig_a = RepoSignals(
        url="github.com/org/A", clone_path=sig_a.clone_path, default_branch="main",
        latest_sha=sig_a.latest_sha, description=None, manifests=(), readmes=(),
        codeowners=(), commits=sig_a.commits, branches=(),
        issues=None, prs=None, contributors=None, workflows=None,
    )
    sig_b = _signals(commits=[_commit("alice@x.com", "Alice", sha="b")])
    sig_b = RepoSignals(
        url="github.com/org/B", clone_path=sig_b.clone_path, default_branch="main",
        latest_sha=sig_b.latest_sha, description=None, manifests=(), readmes=(),
        codeowners=(), commits=sig_b.commits, branches=(),
        issues=None, prs=None, contributors=None, workflows=None,
    )

    [p] = canonicalize_people([sig_a, sig_b], min_threshold=1, max_personas=10)
    assert sorted(p.repos_active_in) == ["github.com/org/A", "github.com/org/B"]
