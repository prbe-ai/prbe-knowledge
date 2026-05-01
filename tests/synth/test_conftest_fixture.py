"""Smoke-test the tmp git repo fixture: it should produce a real
git repo with the expected commits and authors."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_tmp_repo_has_commits(tmp_repo: Path) -> None:
    log = subprocess.run(
        ["git", "-C", str(tmp_repo), "log", "--pretty=%H %ae"],
        check=True, capture_output=True, text=True,
    )
    lines = [ln for ln in log.stdout.splitlines() if ln.strip()]
    assert len(lines) >= 4  # multiple commits


def test_tmp_repo_has_distinct_authors(tmp_repo: Path) -> None:
    log = subprocess.run(
        ["git", "-C", str(tmp_repo), "log", "--pretty=%ae"],
        check=True, capture_output=True, text=True,
    )
    authors = {ln for ln in log.stdout.splitlines() if ln.strip()}
    assert len(authors) >= 2
