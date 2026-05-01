"""RepoExtractor — orchestrates per-repo signal extraction.

Local extraction (mandatory): file walk + manifests + CODEOWNERS +
git log + branches.

GitHub extraction (optional): issues + PRs + contributors + workflows.
Pass a `GithubClient` to enable; pass `None` to skip.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from scripts.synth.extractor.codeowners import (
    CodeownerRule,
    find_codeowners_file,
    parse_codeowners,
)
from scripts.synth.extractor.git_log import Branch, Commit, walk_branches, walk_commits
from scripts.synth.extractor.github_api import (
    Contributor,
    GithubClient,
    Issue,
    PullRequest,
    Workflow,
    parse_repo_url,
)
from scripts.synth.extractor.manifests import Manifest, parse_manifests_in_repo


@dataclass(frozen=True)
class Readme:
    path: Path
    content: str


@dataclass(frozen=True)
class RepoSignals:
    url: str
    clone_path: Path
    default_branch: str
    latest_sha: str
    description: str | None
    manifests: tuple[Manifest, ...]
    readmes: tuple[Readme, ...]
    codeowners: tuple[CodeownerRule, ...]
    commits: tuple[Commit, ...]
    branches: tuple[Branch, ...]
    issues: tuple[Issue, ...] | None
    prs: tuple[PullRequest, ...] | None
    contributors: tuple[Contributor, ...] | None
    workflows: tuple[Workflow, ...] | None


class RepoExtractor:
    def __init__(self, github_client: GithubClient | None) -> None:
        self._gh = github_client

    def extract_local(self, url: str, clone_path: Path, since: datetime) -> RepoSignals:
        latest_sha = subprocess.run(
            ["git", "-C", str(clone_path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

        default_branch = subprocess.run(
            ["git", "-C", str(clone_path), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

        manifests = parse_manifests_in_repo(clone_path)
        readmes = _collect_readmes(clone_path)

        cof = find_codeowners_file(clone_path)
        codeowners = parse_codeowners(cof.read_text()) if cof else ()

        commits = walk_commits(clone_path, since=since)
        branches = walk_branches(clone_path)

        return RepoSignals(
            url=url,
            clone_path=clone_path,
            default_branch=default_branch,
            latest_sha=latest_sha,
            description=_top_level_description(manifests, clone_path),
            manifests=tuple(manifests),
            readmes=readmes,
            codeowners=codeowners,
            commits=tuple(commits),
            branches=tuple(branches),
            issues=None,
            prs=None,
            contributors=None,
            workflows=None,
        )

    async def extract(
        self, url: str, clone_path: Path, since: datetime, *, fetch_github: bool
    ) -> RepoSignals:
        local = self.extract_local(url, clone_path, since)
        if not fetch_github or self._gh is None:
            return local

        owner, repo = parse_repo_url(url)
        issues = await self._gh.fetch_issues(owner, repo)
        prs = await self._gh.fetch_prs(owner, repo)
        contributors = await self._gh.fetch_contributors(owner, repo)
        workflows = await self._gh.fetch_workflows(owner, repo)

        # Reconstruct with the GH fields filled in.
        return RepoSignals(
            url=local.url,
            clone_path=local.clone_path,
            default_branch=local.default_branch,
            latest_sha=local.latest_sha,
            description=local.description,
            manifests=local.manifests,
            readmes=local.readmes,
            codeowners=local.codeowners,
            commits=local.commits,
            branches=local.branches,
            issues=tuple(issues),
            prs=tuple(prs),
            contributors=tuple(contributors),
            workflows=tuple(workflows),
        )


def _collect_readmes(clone_path: Path) -> tuple[Readme, ...]:
    """Top-level README + first-level subdir READMEs."""
    found: list[Readme] = []
    for candidate in (
        clone_path / "README.md",
        clone_path / "README.rst",
        clone_path / "README",
    ):
        if candidate.is_file():
            found.append(Readme(path=candidate, content=candidate.read_text(errors="replace")))
            break
    for child in clone_path.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        for candidate in (child / "README.md", child / "README.rst"):
            if candidate.is_file():
                found.append(Readme(path=candidate, content=candidate.read_text(errors="replace")))
                break
    return tuple(found)


def _top_level_description(manifests: list[Manifest], clone_path: Path) -> str | None:
    """Pick the description from a top-level manifest, if any."""
    for m in manifests:
        if m.description and len(m.path.relative_to(clone_path).parts) <= 1:  # repo root manifest
            return m.description
    return None
