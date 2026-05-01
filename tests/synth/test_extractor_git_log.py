"""Walk git log + branches via subprocess. Returns Commits + Branches."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.synth.extractor.git_log import (
    walk_branches,
    walk_commits,
)


def test_walk_commits_returns_recent_commits(tmp_repo: Path) -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    commits = walk_commits(tmp_repo, since=cutoff)
    assert len(commits) >= 5
    subjects = [c.subject for c in commits]
    assert any("scaffold service" in s for s in subjects)


def test_walk_commits_respects_since(tmp_repo: Path) -> None:
    # Cutoff after most fixture commits
    cutoff = datetime(2026, 4, 30, tzinfo=UTC)
    commits = walk_commits(tmp_repo, since=cutoff)
    assert commits == []


def test_walk_commits_captures_files_touched(tmp_repo: Path) -> None:
    commits = walk_commits(tmp_repo, since=datetime(2026, 1, 1, tzinfo=UTC))
    payments_commits = [c for c in commits if "payments" in c.subject.lower()]
    assert payments_commits
    for c in payments_commits:
        assert any("services/payments" in f for f in c.files_touched)


def test_walk_branches_lists_local_branches(tmp_repo: Path) -> None:
    branches = walk_branches(tmp_repo)
    names = [b.name for b in branches]
    assert "main" in names
    assert "feat/payments-refund" in names


def test_walk_branches_records_last_commit_ts(tmp_repo: Path) -> None:
    branches = walk_branches(tmp_repo)
    feat = next(b for b in branches if b.name == "feat/payments-refund")
    # Branch tip commit was authored 2026-04-01
    assert feat.last_commit_ts >= datetime(2026, 3, 31, tzinfo=UTC)
    assert feat.last_commit_ts <= datetime(2026, 4, 2, tzinfo=UTC) + timedelta(days=1)
