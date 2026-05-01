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


def test_canonicalize_is_order_independent_shared_email() -> None:
    """Order-of-commits must not change canonicalization output.

    Pre-fix bug: a shared-email commit pair where one commit's author_name
    resolves to a contributor and the other doesn't would split into two
    personas in one order but collapse into one in the other order, because
    `email_to_gh` was being mutated mid-loop.
    """
    contributor = Contributor(
        gh_username="alice",
        display_name="Alice X",
        email_aliases=("alice@work.com",),
        contributions=10,
    )
    # Two commits with the SAME email; one author_name resolves via name_to_gh,
    # one does not.
    commits_forward = [
        _commit("shared@x.com", "Bob", sha="1"),       # not in name_to_gh
        _commit("shared@x.com", "Alice X", sha="2"),   # resolves to gh:alice
    ]
    commits_reverse = list(reversed(commits_forward))

    a = canonicalize_people(
        [_signals(commits=commits_forward, contributors=(contributor,))],
        min_threshold=1, max_personas=10,
    )
    b = canonicalize_people(
        [_signals(commits=commits_reverse, contributors=(contributor,))],
        min_threshold=1, max_personas=10,
    )
    # Both must collapse to exactly one Person whose activity counts both commits.
    assert len(a) == len(b) == 1
    assert a[0].canonical_id == b[0].canonical_id == "gh:alice"
    assert a[0].activity_score == b[0].activity_score == 2.0


def test_canonicalize_lowercases_commit_emails_in_aliases() -> None:
    """Mixed-case commit emails must be normalized in email_aliases."""
    sigs = [_signals(commits=[
        _commit("Alice@Work.com", "Alice", sha="1"),
        _commit("alice@work.com", "Alice", sha="2"),
    ])]
    people = canonicalize_people(sigs, min_threshold=1, max_personas=10)
    assert len(people) == 1
    # Must be exactly one entry, lowercased.
    assert people[0].email_aliases == ("alice@work.com",)


from scripts.synth.world_model import infer_services  # noqa: E402


def test_infer_service_from_top_level_pyproject() -> None:
    """A repo with one top-level pyproject is one service named after it."""
    from scripts.synth.extractor.manifests import Manifest, ManifestKind
    sig = _signals(commits=[])
    sig = RepoSignals(
        url="github.com/x/payments-api", clone_path=sig.clone_path,
        default_branch="main", latest_sha="abc", description="Pay svc",
        manifests=(
            Manifest(kind=ManifestKind.PYPROJECT, path=Path("/x/pyproject.toml"),
                     name="payments-api", description="Pay svc", dependencies=()),
        ),
        readmes=(), codeowners=(), commits=(), branches=(),
        issues=None, prs=None, contributors=None, workflows=None,
    )
    services = infer_services([sig])
    assert {s.name for s in services} == {"payments-api"}


def test_infer_services_from_monorepo_subdirs() -> None:
    """Children of services/<name>/pyproject.toml each become a Service."""
    from scripts.synth.extractor.manifests import Manifest, ManifestKind
    sig = RepoSignals(
        url="github.com/x/mono", clone_path=Path("/tmp/mono"),
        default_branch="main", latest_sha="abc", description=None,
        manifests=(
            Manifest(kind=ManifestKind.PYPROJECT, path=Path("/tmp/mono/services/payments/pyproject.toml"),
                     name="payments", description=None, dependencies=()),
            Manifest(kind=ManifestKind.PYPROJECT, path=Path("/tmp/mono/services/billing/pyproject.toml"),
                     name="billing", description=None, dependencies=()),
        ),
        readmes=(), codeowners=(), commits=(), branches=(),
        issues=None, prs=None, contributors=None, workflows=None,
    )
    services = infer_services([sig])
    assert {s.name for s in services} == {"payments", "billing"}


def test_infer_services_qualifies_on_collision() -> None:
    """Two repos define a service named 'payments' → qualified names."""
    from scripts.synth.extractor.manifests import Manifest, ManifestKind

    def _sig_with(url, manifest_name):
        return RepoSignals(
            url=url, clone_path=Path("/tmp/x"),
            default_branch="main", latest_sha="abc", description=None,
            manifests=(
                Manifest(kind=ManifestKind.PYPROJECT, path=Path("/tmp/x/pyproject.toml"),
                         name=manifest_name, description=None, dependencies=()),
            ),
            readmes=(), codeowners=(), commits=(), branches=(),
            issues=None, prs=None, contributors=None, workflows=None,
        )

    services = infer_services([_sig_with("github.com/o/A", "payments"), _sig_with("github.com/o/B", "payments")])
    qualified = sorted(s.qualified for s in services)
    assert qualified == ["A/payments", "B/payments"]


from scripts.synth.world_model import build_topic_pool  # noqa: E402


def test_topic_pool_includes_recent_commits() -> None:
    sigs = [_signals(commits=[
        _commit("a@x.com", "A", sha="1"),
        _commit("b@x.com", "B", sha="2"),
    ])]
    sigs[0] = RepoSignals(  # type: ignore[index]
        url="github.com/x/y", clone_path=sigs[0].clone_path,
        default_branch="main", latest_sha="abc", description=None,
        manifests=(), readmes=(), codeowners=(), commits=(
            Commit(sha="1", author_name="A", author_email="a@x.com",
                   ts=datetime(2026, 4, 25, tzinfo=UTC),
                   subject="fix(payments): null pointer in checkout",
                   body="", files_touched=("services/payments/checkout.py",)),
        ),
        branches=(), issues=None, prs=None, contributors=None, workflows=None,
    )
    pool = build_topic_pool(sigs, services=(), now=datetime(2026, 4, 30, tzinfo=UTC))
    assert any("checkout" in t.text for t in pool)


def test_topic_recency_weighted_higher_for_recent_commits() -> None:
    old = Commit(sha="o", author_name="A", author_email="a@x.com",
                 ts=datetime(2026, 1, 1, tzinfo=UTC),
                 subject="old commit", body="", files_touched=())
    new = Commit(sha="n", author_name="A", author_email="a@x.com",
                 ts=datetime(2026, 4, 28, tzinfo=UTC),
                 subject="new commit", body="", files_touched=())
    sig = _signals(commits=[])
    sig = RepoSignals(
        url=sig.url, clone_path=sig.clone_path, default_branch="main", latest_sha=sig.latest_sha,
        description=None, manifests=(), readmes=(), codeowners=(), commits=(old, new),
        branches=(), issues=None, prs=None, contributors=None, workflows=None,
    )
    pool = build_topic_pool([sig], services=(), now=datetime(2026, 4, 30, tzinfo=UTC))
    new_weight = next(t.weight for t in pool if t.text == "new commit")
    old_weight = next(t.weight for t in pool if t.text == "old commit")
    assert new_weight > old_weight


from scripts.synth.world_model import (  # noqa: E402
    Service,
    ServiceKind,
    synthesize_channels,
    synthesize_sections,
)


def _svc(name: str, kind: ServiceKind = ServiceKind.API, recent: float = 1.0) -> Service:
    return Service(
        name=name, qualified=name, repo_url="github.com/x/y", kind=kind,
        description=None, owners=(), recent_activity=recent, deploy_target=None,
    )


def test_synthesize_channels_includes_per_service_and_generic() -> None:
    services = (_svc("payments"), _svc("billing"))
    channels = synthesize_channels(services, codeowner_teams=set())
    names = {c.name for c in channels}
    assert "#payments" in names
    assert "#billing" in names
    assert "#general" in names
    assert "#incidents" in names


def test_synthesize_channels_adds_deploy_channels_for_top_active() -> None:
    services = tuple(_svc(f"svc{i}", recent=float(i)) for i in range(8))
    channels = synthesize_channels(services, codeowner_teams=set())
    deploy_channels = {c.name for c in channels if c.name.endswith("-deploys")}
    # top-5 by activity → svc7,6,5,4,3
    assert deploy_channels == {"#svc7-deploys", "#svc6-deploys", "#svc5-deploys", "#svc4-deploys", "#svc3-deploys"}


def test_synthesize_sections_fixed_set_plus_per_service_runbooks() -> None:
    services = (_svc("payments"), _svc("billing"))
    sections = synthesize_sections(services)
    titles = {s.title for s in sections}
    assert "Engineering" in titles
    assert "Postmortems" in titles
    assert "payments runbook" in titles
    assert "billing runbook" in titles


from scripts.synth.world_model import build_dep_graph  # noqa: E402


def test_dep_edge_recorded_when_manifest_dep_matches_service() -> None:
    """A repo with manifest deps that name a Service produces a DepEdge
    from the manifest's owning service to the dependency service."""
    from scripts.synth.extractor.manifests import Manifest, ManifestKind

    services = (
        _svc("payments"),  # in repo A
        _svc("billing"),   # in repo B
    )
    services = (
        Service(name="payments", qualified="payments", repo_url="github.com/x/A",
                kind=ServiceKind.API, description=None, owners=(), recent_activity=1.0,
                deploy_target=None),
        Service(name="billing", qualified="billing", repo_url="github.com/x/B",
                kind=ServiceKind.LIB, description=None, owners=(), recent_activity=1.0,
                deploy_target=None),
    )
    sig_a = RepoSignals(
        url="github.com/x/A", clone_path=Path("/tmp/A"),
        default_branch="main", latest_sha="abc", description=None,
        manifests=(
            Manifest(kind=ManifestKind.PYPROJECT, path=Path("/tmp/A/pyproject.toml"),
                     name="payments", description=None, dependencies=("billing", "fastapi")),
        ),
        readmes=(), codeowners=(), commits=(), branches=(),
        issues=None, prs=None, contributors=None, workflows=None,
    )
    sig_b = RepoSignals(
        url="github.com/x/B", clone_path=Path("/tmp/B"),
        default_branch="main", latest_sha="abc", description=None,
        manifests=(
            Manifest(kind=ManifestKind.PYPROJECT, path=Path("/tmp/B/pyproject.toml"),
                     name="billing", description=None, dependencies=()),
        ),
        readmes=(), codeowners=(), commits=(), branches=(),
        issues=None, prs=None, contributors=None, workflows=None,
    )

    edges = build_dep_graph([sig_a, sig_b], services)
    assert len(edges) == 1
    assert edges[0].from_service == "payments"
    assert edges[0].to_service == "billing"
    assert edges[0].source_repo == "github.com/x/A"


def test_no_dep_edge_for_external_deps() -> None:
    """Manifest deps that don't match any Service are ignored."""
    from scripts.synth.extractor.manifests import Manifest, ManifestKind

    services = (_svc("payments"),)
    sig = RepoSignals(
        url="github.com/x/A", clone_path=Path("/tmp/A"),
        default_branch="main", latest_sha="abc", description=None,
        manifests=(
            Manifest(kind=ManifestKind.PYPROJECT, path=Path("/tmp/A/pyproject.toml"),
                     name="payments", description=None,
                     dependencies=("requests", "boto3", "stripe")),
        ),
        readmes=(), codeowners=(), commits=(), branches=(),
        issues=None, prs=None, contributors=None, workflows=None,
    )
    edges = build_dep_graph([sig], services)
    assert edges == ()
