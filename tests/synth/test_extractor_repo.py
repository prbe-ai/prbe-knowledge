"""End-to-end (local-only) RepoExtractor: walks a real tmp git repo,
returns RepoSignals."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from scripts.synth.extractor.repo import RepoExtractor, RepoSignals


def test_extracts_local_signals(tmp_repo: Path) -> None:
    extractor = RepoExtractor(github_client=None)

    signals = extractor.extract_local(
        url=f"file://{tmp_repo}",
        clone_path=tmp_repo,
        since=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(signals, RepoSignals)
    assert signals.url == f"file://{tmp_repo}"
    assert signals.latest_sha  # non-empty
    assert signals.default_branch == "main"

    # Manifests
    manifest_names = {m.name for m in signals.manifests if m.name}
    assert {"fake-repo", "payments", "billing"} <= manifest_names

    # CODEOWNERS
    assert signals.codeowners
    assert any("payments" in r.pattern for r in signals.codeowners)

    # Commits
    assert len(signals.commits) >= 5

    # Branches
    branch_names = {b.name for b in signals.branches}
    assert {"main", "feat/payments-refund"} <= branch_names

    # GitHub-only: None when no client
    assert signals.issues is None
    assert signals.prs is None
    assert signals.contributors is None


def test_latest_sha_matches_git(tmp_repo: Path) -> None:
    extractor = RepoExtractor(github_client=None)
    signals = extractor.extract_local(
        url=f"file://{tmp_repo}",
        clone_path=tmp_repo,
        since=datetime(2026, 1, 1, tzinfo=UTC),
    )
    expected = subprocess.run(
        ["git", "-C", str(tmp_repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert signals.latest_sha == expected
