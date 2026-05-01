"""Async GitHub API client. Just enough to populate `RepoSignals.{issues,
prs, contributors, workflows}`. No external lib — httpx is already a dep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx


@dataclass(frozen=True)
class Contributor:
    gh_username: str
    display_name: str | None
    email_aliases: tuple[str, ...]
    contributions: int


@dataclass(frozen=True)
class Issue:
    number: int
    title: str
    body: str
    state: str
    labels: tuple[str, ...]
    assignees: tuple[str, ...]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    body: str
    state: str
    labels: tuple[str, ...]
    author: str | None
    reviewers: tuple[str, ...]
    files_changed: tuple[str, ...]
    created_at: datetime
    merged_at: datetime | None


@dataclass(frozen=True)
class Workflow:
    name: str
    last_run_status: str | None
    last_run_at: datetime | None


_REPO_RE = re.compile(r"(?:https?://)?github\.com/([^/\s]+)/([^/\s]+?)(?:\.git)?/?$")


def parse_repo_url(url: str) -> tuple[str, str]:
    """github.com/owner/repo (with or without https:// and .git) → (owner, repo)."""
    m = _REPO_RE.match(url.strip())
    if not m:
        raise ValueError(f"not a recognized github URL: {url!r}")
    return m.group(1), m.group(2)


class GithubClient:
    def __init__(self, token: str | None, base_url: str = "https://api.github.com") -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any | None:
        try:
            r = await self._http.get(path, params=params)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError:
            return None

    async def fetch_contributors(self, owner: str, repo: str, limit: int = 100) -> list[Contributor]:
        rows = await self._get(f"/repos/{owner}/{repo}/contributors", {"per_page": limit})
        if not rows:
            return []
        out: list[Contributor] = []
        for row in rows:
            login = row.get("login")
            if not login:
                continue
            user = await self._get(f"/users/{login}") or {}
            email = user.get("email")
            out.append(
                Contributor(
                    gh_username=login,
                    display_name=user.get("name"),
                    email_aliases=(email,) if email else (),
                    contributions=row.get("contributions") or 0,
                )
            )
        return out

    async def fetch_issues(self, owner: str, repo: str, limit: int = 200) -> list[Issue]:
        # v1 returns at most one page (≤100); pagination via the Link header is a v2 concern.
        rows = await self._get(
            f"/repos/{owner}/{repo}/issues",
            {"state": "all", "per_page": min(limit, 100)},
        )
        if not rows:
            return []
        issues: list[Issue] = []
        for row in rows:
            if "pull_request" in row:  # GH lumps PRs into /issues
                continue
            issues.append(
                Issue(
                    number=row["number"],
                    title=row.get("title") or "",
                    body=row.get("body") or "",
                    state=row.get("state") or "open",
                    labels=tuple(lbl["name"] for lbl in row.get("labels", []) if lbl.get("name")),
                    assignees=tuple(a["login"] for a in row.get("assignees", []) if a.get("login")),
                    created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
                    updated_at=datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00")),
                )
            )
            if len(issues) >= limit:
                break
        return issues

    async def fetch_prs(self, owner: str, repo: str, limit: int = 200) -> list[PullRequest]:
        # v1 returns at most one page (≤100); pagination via the Link header is a v2 concern.
        rows = await self._get(
            f"/repos/{owner}/{repo}/pulls",
            {"state": "all", "per_page": min(limit, 100)},
        )
        if not rows:
            return []
        prs: list[PullRequest] = []
        for row in rows:
            merged_at_raw = row.get("merged_at")
            prs.append(
                PullRequest(
                    number=row["number"],
                    title=row.get("title") or "",
                    body=row.get("body") or "",
                    state=row.get("state") or "open",
                    labels=tuple(lbl["name"] for lbl in row.get("labels", []) if lbl.get("name")),
                    author=(row.get("user") or {}).get("login"),
                    reviewers=tuple(
                        u["login"] for u in row.get("requested_reviewers", []) if u.get("login")
                    ),
                    files_changed=(),  # cost-tradeoff: separate /files call; skip in v1
                    created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
                    merged_at=(
                        datetime.fromisoformat(merged_at_raw.replace("Z", "+00:00"))
                        if merged_at_raw
                        else None
                    ),
                )
            )
            if len(prs) >= limit:
                break
        return prs

    async def fetch_workflows(self, owner: str, repo: str) -> list[Workflow]:
        rows = await self._get(f"/repos/{owner}/{repo}/actions/workflows")
        if not rows:
            return []
        out: list[Workflow] = []
        for row in rows.get("workflows", []) or []:
            out.append(
                Workflow(
                    name=row.get("name") or row.get("path") or "",
                    # last_run_status requires /actions/runs; deferred to v2 (row["state"] is
                    # the workflow's enabled-state, not a run result — don't use it here).
                    last_run_status=None,
                    last_run_at=None,  # full run history is heavy; skip in v1
                )
            )
        return out
