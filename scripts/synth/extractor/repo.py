"""RepoExtractor — orchestrates per-repo signal extraction.

Local extraction (mandatory): file walk + manifests + CODEOWNERS +
git log + branches.

GitHub extraction (optional): issues + PRs + contributors + workflows.
Pass a `GithubClient` to enable; pass `None` to skip.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from scripts.synth.cache import DiskCache
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
    def __init__(
        self,
        github_client: GithubClient | None,
        cache: DiskCache | None = None,
    ) -> None:
        self._gh = github_client
        self._cache = cache

    def extract_local(self, url: str, clone_path: Path, since: datetime) -> RepoSignals:
        latest_sha = subprocess.run(
            ["git", "-C", str(clone_path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

        cache_key = f"repo:{url}@{latest_sha}"
        if self._cache is not None:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return _signals_from_dict(hit)

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

        signals = RepoSignals(
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

        if self._cache is not None:
            self._cache.put(cache_key, _signals_to_dict(signals))

        return signals

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
        return replace(
            local,
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
    try:
        children = list(clone_path.iterdir())
    except OSError:
        return tuple(found)
    for child in children:
        if not child.is_dir() or child.name.startswith("."):
            continue
        for candidate in (child / "README.md", child / "README.rst"):
            if candidate.is_file():
                try:
                    content = candidate.read_text(errors="replace")
                except OSError:
                    break
                found.append(Readme(path=candidate, content=content))
                break
    return tuple(found)


def _top_level_description(manifests: list[Manifest], clone_path: Path) -> str | None:
    """Pick description from a manifest at the repo root (relative-parts == 1)."""
    for m in manifests:
        if not m.description:
            continue
        try:
            rel = m.path.relative_to(clone_path)
        except ValueError:
            continue
        if len(rel.parts) <= 1:
            return m.description
    return None


def _signals_to_dict(s: RepoSignals) -> dict:
    """Cache-serialize. We store enough to reconstruct: url, sha, default_branch,
    description, plus full commits + branches + manifests + codeowners + readmes.
    Datetimes are isoformatted; tuples become lists; Paths become strings."""
    import dataclasses

    def encode(v):
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {k: encode(getattr(v, k)) for k in (f.name for f in dataclasses.fields(v))}
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, tuple | list):
            return [encode(x) for x in v]
        if isinstance(v, dict):
            return {k: encode(val) for k, val in v.items()}
        return v

    out = encode(s)
    # Mark gh-only fields explicitly None so loader knows they're absent vs empty.
    return out


def _signals_from_dict(data: dict) -> RepoSignals:
    """Reconstruct RepoSignals from cached dict."""
    from scripts.synth.extractor.codeowners import CodeownerRule
    from scripts.synth.extractor.git_log import Branch, Commit
    from scripts.synth.extractor.manifests import Manifest, ManifestKind

    def commit(d):
        return Commit(
            sha=d["sha"], author_name=d["author_name"], author_email=d["author_email"],
            ts=datetime.fromisoformat(d["ts"]),
            subject=d["subject"], body=d["body"],
            files_touched=tuple(d["files_touched"]),
        )

    def branch(d):
        return Branch(name=d["name"], last_commit_sha=d["last_commit_sha"],
                      last_commit_ts=datetime.fromisoformat(d["last_commit_ts"]))

    def manifest(d):
        return Manifest(
            kind=ManifestKind(d["kind"]), path=Path(d["path"]),
            name=d["name"], description=d["description"],
            dependencies=tuple(d["dependencies"]),
            compose_service_names=tuple(d["compose_service_names"]),
        )

    def readme(d):
        return Readme(path=Path(d["path"]), content=d["content"])

    def rule(d):
        return CodeownerRule(pattern=d["pattern"], owners=tuple(d["owners"]))

    return RepoSignals(
        url=data["url"], clone_path=Path(data["clone_path"]),
        default_branch=data["default_branch"], latest_sha=data["latest_sha"],
        description=data["description"],
        manifests=tuple(manifest(m) for m in data["manifests"]),
        readmes=tuple(readme(r) for r in data["readmes"]),
        codeowners=tuple(rule(r) for r in data["codeowners"]),
        commits=tuple(commit(c) for c in data["commits"]),
        branches=tuple(branch(b) for b in data["branches"]),
        issues=None, prs=None, contributors=None, workflows=None,
    )
