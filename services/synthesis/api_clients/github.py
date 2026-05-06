"""Read-only GitHub client used by the wiki bootstrap crawler (Lane E).

The ingestion-side `services/ingestion/handlers/github.py` keeps its own
inline pagination + cursor encoding because it has to interleave with
webhook replays. This module is a separate, simpler abstraction for the
synthesis crawler: a thin paginating async client with its own token
bucket, calibrated to ~70% of the GitHub App's 5000/hr quota.

All listing endpoints are recency-first (`sort=updated&direction=desc`)
so the crawler can stop early once signal dries up. The caller passes
the bearer token in — auth is resolved upstream by Lane C using
`shared.backend_client.fetch_github_installation_token`. Keeping this
layer auth-agnostic makes testing simple.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import Any

import httpx

from shared.logging import get_logger

log = get_logger(__name__)


GITHUB_API = "https://api.github.com"

# 70% of GitHub App's 5000/hr quota = 3500/hr ≈ 0.97 req/s. We default
# to 1 req/s (~3600/hr, 72%) which leaves ~1400/hr headroom for the
# ingestion-side handler to share the same installation.
_DEFAULT_TARGET_RPS = 1.0
_DEFAULT_BURST = 10

_BACKOFF_LADDER_S = (30.0, 60.0, 120.0, 240.0, 300.0)
_MAX_CONSECUTIVE_RATE_LIMITS = 5
_MAX_5XX_RETRIES = 3


class GitHubAPIError(Exception):
    """Non-retryable GitHub API failure (4xx other than 429)."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"GitHub API {status}: {body[:300]}")
        self.status = status
        self.body = body


class GitHubRateLimitExhausted(Exception):
    """Raised after too many consecutive 429s. Treat as a soft halt signal."""


class _AsyncTokenBucket:
    """Leaky-bucket limiter. Refills at ``rate_per_second`` up to ``capacity``.

    Clock is injectable so tests can drive the bucket without real sleeps.
    The default uses ``time.monotonic`` and ``asyncio.sleep`` for production.
    """

    def __init__(
        self,
        rate_per_second: float,
        capacity: int = _DEFAULT_BURST,
        *,
        now: Callable[[], float] | None = None,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self._rate = rate_per_second
        self._capacity = capacity
        self._now = now or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._tokens: float = float(capacity)
        self._last_refill = self._now()
        self._lock = asyncio.Lock()

    @property
    def tokens(self) -> float:
        """Current token count (after a refill). Test-only accessor."""
        self._refill()
        return self._tokens

    def _refill(self) -> None:
        t = self._now()
        delta = max(t - self._last_refill, 0.0)
        self._tokens = min(self._capacity, self._tokens + delta * self._rate)
        self._last_refill = t

    async def acquire(self) -> None:
        """Block until at least one token is available, then consume it."""
        async with self._lock:
            self._refill()
            if self._tokens < 1.0:
                deficit = 1.0 - self._tokens
                wait = deficit / self._rate
                await self._sleep(wait)
                self._refill()
            self._tokens -= 1.0


class GitHubAPIClient:
    """Read-only GitHub client used by the wiki bootstrap crawler.

    Holds a bearer token and shared httpx.AsyncClient. Surfaces
    paginating async iterators for the resources the crawler needs:
    repos, pulls, issues, commits, reviews. All resource listings
    are recency-first (newest updated first) so the crawler can
    stop early if signal dries up.

    Rate-limited via a token bucket calibrated to 70% of the GitHub
    App's published 5000/hr quota. On 429, sleeps until the
    ``X-RateLimit-Reset`` window with exp backoff + jitter.
    """

    def __init__(
        self,
        bearer: str,
        http: httpx.AsyncClient,
        *,
        target_rps: float = _DEFAULT_TARGET_RPS,
        burst: int = _DEFAULT_BURST,
        bucket: _AsyncTokenBucket | None = None,
        sleep: Callable[[float], Any] | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._bearer = bearer
        self._http = http
        self._sleep = sleep or asyncio.sleep
        self._now = now or time.time
        self._bucket = bucket or _AsyncTokenBucket(rate_per_second=target_rps, capacity=burst)

    @property
    def bucket(self) -> _AsyncTokenBucket:
        """Token bucket. Exposed for tests; production code should not touch."""
        return self._bucket

    # ------------------------------------------------------------------
    # public listing APIs
    # ------------------------------------------------------------------

    async def list_installation_repos(self) -> AsyncIterator[dict[str, Any]]:
        """Yield repos accessible to the installation, newest pushed first."""
        url: str | None = (
            f"{GITHUB_API}/installation/repositories?per_page=100&sort=pushed&direction=desc"
        )
        while url:
            payload, next_url = await self._get_page(url)
            repos = payload.get("repositories") if isinstance(payload, dict) else None
            if isinstance(repos, list):
                for repo in repos:
                    if isinstance(repo, dict):
                        yield repo
            url = next_url

    async def list_pulls(
        self,
        full_name: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield pulls newest-updated first. ``since``/``until`` filter client-side.

        GitHub's pulls endpoint doesn't accept ``since``/``until`` natively, so
        we filter after the fact. Because results are sorted desc by updated_at,
        we can stop iterating as soon as a row's ``updated_at`` falls below
        ``since`` — saves the rest of the pagination on big repos.
        """
        url: str | None = (
            f"{GITHUB_API}/repos/{full_name}/pulls"
            "?state=all&sort=updated&direction=desc&per_page=100"
        )
        while url:
            rows, next_url = await self._get_list_page(url)
            for row in rows:
                updated = _parse_iso(row.get("updated_at"))
                if until is not None and updated is not None and updated > until:
                    continue
                if since is not None and updated is not None and updated < since:
                    return  # desc order — older rows from here on
                yield row
            url = next_url

    async def list_issues(
        self,
        full_name: str,
        *,
        since: datetime | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield issues newest-updated first. PRs filtered out.

        GitHub's /issues endpoint conflates issues with pull requests — every
        PR also shows up here with a ``pull_request`` key. We strip those so
        the crawler only sees real issues; PRs come from ``list_pulls``.
        """
        params = "?state=all&sort=updated&direction=desc&per_page=100"
        if since is not None:
            params += f"&since={_iso(since)}"
        url: str | None = f"{GITHUB_API}/repos/{full_name}/issues{params}"
        while url:
            rows, next_url = await self._get_list_page(url)
            for row in rows:
                if row.get("pull_request") is not None:
                    continue
                yield row
            url = next_url

    async def list_commits(
        self,
        full_name: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield commits in the default branch. ``since``/``until`` are native."""
        params: list[str] = ["per_page=100"]
        if since is not None:
            params.append(f"since={_iso(since)}")
        if until is not None:
            params.append(f"until={_iso(until)}")
        url: str | None = f"{GITHUB_API}/repos/{full_name}/commits?" + "&".join(params)
        while url:
            rows, next_url = await self._get_list_page(url)
            for row in rows:
                yield row
            url = next_url

    async def list_pull_reviews(
        self,
        full_name: str,
        pr_number: int,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield reviews on a single PR."""
        url: str | None = f"{GITHUB_API}/repos/{full_name}/pulls/{pr_number}/reviews?per_page=100"
        while url:
            rows, next_url = await self._get_list_page(url)
            for row in rows:
                yield row
            url = next_url

    async def get_repo(self, full_name: str) -> dict[str, Any]:
        """Fetch a single repo's metadata."""
        payload, _ = await self._get_page(f"{GITHUB_API}/repos/{full_name}")
        if not isinstance(payload, dict):
            raise GitHubAPIError(200, f"unexpected repo body for {full_name}")
        return payload

    async def aclose(self) -> None:
        """No-op; the caller owns the httpx client."""

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._bearer}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _get_list_page(self, url: str) -> tuple[list[dict[str, Any]], str | None]:
        body, next_url = await self._get_page(url)
        if not isinstance(body, list):
            return [], next_url
        return [r for r in body if isinstance(r, dict)], next_url

    async def _get_page(self, url: str) -> tuple[Any, str | None]:
        """GET one page, applying rate-limit + 5xx retries. Return (body, next_url)."""
        consecutive_429 = 0
        consecutive_5xx = 0
        while True:
            await self._bucket.acquire()
            try:
                resp = await self._http.get(url, headers=self._headers())
            except httpx.HTTPError as exc:
                consecutive_5xx += 1
                if consecutive_5xx > _MAX_5XX_RETRIES:
                    raise GitHubAPIError(0, f"network error: {exc}") from exc
                await self._sleep(_backoff_seconds(consecutive_5xx))
                continue

            status = resp.status_code

            if _is_rate_limited(resp):
                consecutive_429 += 1
                if consecutive_429 > _MAX_CONSECUTIVE_RATE_LIMITS:
                    raise GitHubRateLimitExhausted(
                        f"GitHub returned 429/403 {consecutive_429} times in a row"
                    )
                delay = self._rate_limit_delay(resp, attempt=consecutive_429)
                log.warning(
                    "github.rate_limited",
                    url=url,
                    status=status,
                    delay=delay,
                    attempt=consecutive_429,
                )
                await self._sleep(delay)
                continue

            if 500 <= status < 600:
                consecutive_5xx += 1
                if consecutive_5xx > _MAX_5XX_RETRIES:
                    raise GitHubAPIError(status, resp.text)
                await self._sleep(_backoff_seconds(consecutive_5xx))
                continue

            if status >= 400:
                raise GitHubAPIError(status, resp.text)

            return resp.json(), _parse_next_link(
                resp.headers.get("link") or resp.headers.get("Link")
            )

    def _rate_limit_delay(self, resp: httpx.Response, *, attempt: int) -> float:
        """Compute sleep seconds for a 429/403 rate-limit response.

        First attempt honors ``Retry-After`` or ``X-RateLimit-Reset``. Repeated
        429s climb the exp-backoff ladder with jitter. Floor of 1s.
        """
        retry_after = resp.headers.get("retry-after")
        if retry_after is not None:
            try:
                base = float(retry_after)
            except ValueError:
                base = 5.0
            return max(base + random.uniform(1.0, 3.0), 1.0)

        reset = resp.headers.get("x-ratelimit-reset")
        if reset is not None:
            try:
                base = float(reset) - self._now()
            except ValueError:
                base = 5.0
            base = max(base, 1.0)
        else:
            base = 5.0

        if attempt > 1:
            ladder_idx = min(attempt - 2, len(_BACKOFF_LADDER_S) - 1)
            base = max(base, _BACKOFF_LADDER_S[ladder_idx])
        return base + random.uniform(1.0, 3.0)


# ----------------------------------------------------------------------
# module-level helpers (kept top-level so tests can hit them directly)
# ----------------------------------------------------------------------


def _parse_next_link(link_header: str | None) -> str | None:
    """Parse a GitHub Link header, returning the rel="next" URL (or None)."""
    if not link_header:
        return None
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="next"' not in part:
            continue
        if part.startswith("<") and ">" in part:
            return part.split(">", 1)[0][1:]
    return None


def _is_rate_limited(resp: httpx.Response) -> bool:
    """True if GitHub is rate-limiting us (429, or 403 with remaining=0)."""
    if resp.status_code == 429:
        return True
    if resp.status_code == 403:
        return resp.headers.get("x-ratelimit-remaining") == "0"
    return False


def _backoff_seconds(attempt: int) -> float:
    """Exp backoff with jitter for transient 5xx/network errors."""
    base = min(2 ** (attempt - 1), 30)
    return float(base) + random.uniform(0.0, 1.0)


def _iso(dt: datetime) -> str:
    """ISO-8601 with trailing Z (GitHub's expected form)."""
    s = dt.isoformat()
    if s.endswith("+00:00"):
        s = s[:-6] + "Z"
    return s


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


__all__ = [
    "GitHubAPIClient",
    "GitHubAPIError",
    "GitHubRateLimitExhausted",
]
