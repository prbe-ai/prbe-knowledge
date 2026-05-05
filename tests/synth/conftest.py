"""Shared fixtures for synth tests.

`tmp_repo` builds a tiny but realistic git repo in a tmp dir:
    - 2 services in `services/payments/` and `services/billing/`
    - Manifests (pyproject)
    - CODEOWNERS file
    - README at root + per-service
    - 6 commits across 3 distinct authors
    - 1 feature branch
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def _git(repo: Path, *args: str, env_extra: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    env.update({
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    })
    if env_extra:
        env.update(env_extra)
    subprocess.run(["git", "-C", str(repo), *args], check=True, env=env, capture_output=True)


def _commit(repo: Path, files: dict[str, str], message: str, author: str, email: str, date: str) -> None:
    for path, content in files.items():
        full = repo / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
    _git(repo, "add", "-A")
    _git(
        repo,
        "commit",
        "-m",
        message,
        env_extra={
            "GIT_AUTHOR_NAME": author,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": author,
            "GIT_COMMITTER_EMAIL": email,
            "GIT_AUTHOR_DATE": date,
            "GIT_COMMITTER_DATE": date,
        },
    )


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fake-repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "commit.gpgsign", "false")

    _commit(
        repo,
        {
            "README.md": "# fake-repo\n\nA tiny test repo for synth.\n",
            "pyproject.toml": (
                '[project]\nname = "fake-repo"\ndescription = "Test repo"\n'
            ),
            ".github/CODEOWNERS": (
                "/services/payments/  @alice\n"
                "/services/billing/   @bob @alice\n"
            ),
        },
        "feat: initial commit",
        "Alice", "alice@example.com",
        "2026-01-15T09:00:00",
    )

    _commit(
        repo,
        {"services/payments/pyproject.toml": '[project]\nname = "payments"\n',
         "services/payments/main.py": "def handle(): pass\n"},
        "feat(payments): scaffold service",
        "Alice", "alice@example.com",
        "2026-02-01T10:00:00",
    )

    _commit(
        repo,
        {"services/billing/pyproject.toml": '[project]\nname = "billing"\n',
         "services/billing/main.py": "def invoice(): pass\n"},
        "feat(billing): scaffold service",
        "Bob", "bob@example.com",
        "2026-02-15T14:30:00",
    )

    _commit(
        repo,
        {"services/payments/main.py": "def handle():\n    return 'ok'\n"},
        "fix(payments): return ok",
        "Alice", "alice@example.com",
        "2026-03-01T11:00:00",
    )

    _commit(
        repo,
        {"services/billing/invoice.py": "def total(): return 0\n"},
        "feat(billing): add invoice totaling",
        "Carol", "carol@example.com",
        "2026-03-20T16:00:00",
    )

    _git(repo, "checkout", "-b", "feat/payments-refund")
    _commit(
        repo,
        {"services/payments/refund.py": "def refund(): pass\n"},
        "wip(payments): refund logic",
        "Alice", "alice@example.com",
        "2026-04-01T09:00:00",
    )
    _git(repo, "checkout", "main")

    return repo


@pytest.fixture
def tmp_repo_profile_dir(tmp_repo: Path, tmp_path: Path) -> Path:
    """Build a profile YAML pointing at tmp_repo. Returns the dir."""
    profile_dir = tmp_path / "profile_dir"
    profile_dir.mkdir()
    profile_path = profile_dir / "profile.yaml"
    profile_path.write_text(
        f"""
customer_id: cust-eval-fake-01
preset: tiny_test
seed: 7
repos:
  - url: github.com/x/fake
    local_path: {tmp_repo}
world_model:
  min_commits_per_persona: 1
  topic_pool_lookback_days: 9999
""".strip()
    )
    return profile_dir
