"""Incremental-push file fetcher — pulls changed-file contents via GitHub
Contents API.

Used by the incremental code-graph path: a verified push event arrives
with `commits[].added/modified/removed`. We need the new file contents at
the SHA. Cloning is overkill for 1-50 files; the Contents API is exactly
the right tool.

Auth: GitHub App installation token (via `shared.github_auth`). Rate limit:
5000 reqs/hour per installation; typical push events touch <50 files, so
even pathological churn (50 pushes/hour * 50 files) lands at 2500 reqs/hr,
well under cap.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass

import httpx

from shared.exceptions import RateLimited, SourceAPIError
from shared.logging import get_logger

log = get_logger(__name__)

_GITHUB_API = "https://api.github.com"
# Per-call timeout. The worker will retry on transient errors via the
# normalizer's existing retry path.
_DEFAULT_TIMEOUT_SECONDS = 30.0
# Concurrent-fetch fan-out per repo. Tuned to avoid hammering the
# secondary rate limit (which kicks in on bursts even when the primary
# 5k/hour budget is fine).
_MAX_CONCURRENT_FETCHES = 8


@dataclass(slots=True)
class FetchedFile:
    rel_path: str
    content: bytes
    not_found: bool = False  # True if file was 404 (e.g., a removed file the diff missed)


async def fetch_files_at_sha(
    repo: str,
    sha: str,
    paths: list[str],
    token: str | None,
    *,
    client: httpx.AsyncClient | None = None,
) -> list[FetchedFile]:
    """Fetch raw contents for `paths` in `repo` at `sha`.

    Concurrent within a per-repo budget (`_MAX_CONCURRENT_FETCHES`) so a
    50-file push doesn't take 50 sequential round trips, but doesn't burst
    enough to trigger GitHub's secondary-rate-limit guards.
    """
    if not paths:
        return []

    sem = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_SECONDS)

    try:
        results = await asyncio.gather(
            *[_fetch_one(client, repo, sha, p, token, sem) for p in paths],
            return_exceptions=False,
        )
    finally:
        if own_client:
            await client.aclose()
    return results


async def _fetch_one(
    client: httpx.AsyncClient,
    repo: str,
    sha: str,
    rel_path: str,
    token: str | None,
    sem: asyncio.Semaphore,
) -> FetchedFile:
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{_GITHUB_API}/repos/{repo}/contents/{rel_path}"
    async with sem:
        try:
            resp = await client.get(url, params={"ref": sha}, headers=headers)
        except httpx.HTTPError as exc:
            raise SourceAPIError(f"GitHub Contents API request failed: {exc}") from exc

    if resp.status_code == 404:
        return FetchedFile(rel_path=rel_path, content=b"", not_found=True)
    if resp.status_code in (403, 429):
        # 403 with a rate-limit header == primary or secondary rate limit.
        if "rate limit" in resp.text.lower() or "abuse" in resp.text.lower():
            raise RateLimited(
                f"GitHub rate limited fetching {rel_path}: {resp.status_code}"
            )
        raise SourceAPIError(
            f"GitHub Contents API forbidden for {rel_path}: {resp.status_code}"
        )
    if resp.status_code >= 400:
        raise SourceAPIError(
            f"GitHub Contents API returned {resp.status_code} for {rel_path}"
        )

    data = resp.json()
    if isinstance(data, list):
        # `paths` should never name a directory; if it does, something
        # upstream filtered wrong. Treat as not_found rather than crash.
        log.warning("code_graph.fetch.directory_returned", repo=repo, path=rel_path)
        return FetchedFile(rel_path=rel_path, content=b"", not_found=True)

    encoding = data.get("encoding")
    raw = data.get("content", "") or ""
    if encoding == "base64":
        content = base64.b64decode(raw, validate=False)
    elif encoding is None and isinstance(raw, str):
        # GitHub returns no encoding for empty files.
        content = raw.encode("utf-8")
    else:
        log.warning(
            "code_graph.fetch.unknown_encoding",
            repo=repo,
            path=rel_path,
            encoding=encoding,
        )
        content = b""

    return FetchedFile(rel_path=rel_path, content=content, not_found=False)


__all__ = ["FetchedFile", "fetch_files_at_sha"]
