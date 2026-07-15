"""Tests for the synthesis-side read-only GitHub client.

Strategy: respx mocks all httpx traffic. We inject a fake clock + sleep
recorder into ``GitHubAPIClient`` so 429-backoff and token-bucket
behavior are deterministic and don't actually wait. The token-bucket
test feeds an injectable ``now()`` callable so we can advance virtual
time without ``asyncio`` involvement.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import pytest
import respx

from kb.synthesis.api_clients.github import (
    GitHubAPIClient,
    GitHubAPIError,
    GitHubRateLimitExhausted,
    _AsyncTokenBucket,
    _parse_next_link,
)


class _FakeClock:
    """Hand-cranked monotonic clock for deterministic rate-limit tests."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class _SleepRecorder:
    """Async sleep replacement that records durations + advances a clock."""

    def __init__(self, clock: _FakeClock | None = None) -> None:
        self.calls: list[float] = []
        self.clock = clock

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self.clock is not None:
            self.clock.advance(seconds)


def _link_header(next_url: str) -> str:
    return f'<{next_url}>; rel="next", <last>; rel="last"'


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


def _make_client(
    http: httpx.AsyncClient,
    *,
    sleep: _SleepRecorder | None = None,
    clock: _FakeClock | None = None,
) -> GitHubAPIClient:
    sleeper = sleep or _SleepRecorder()
    if clock is not None:
        bucket = _AsyncTokenBucket(rate_per_second=1000.0, capacity=100, now=clock, sleep=sleeper)
        return GitHubAPIClient("ghs_x", http, bucket=bucket, sleep=sleeper, now=clock)
    return GitHubAPIClient("ghs_x", http, target_rps=1000.0, burst=100, sleep=sleeper)


def test_parse_next_link_picks_next_only() -> None:
    header = (
        '<https://api.github.com/x?page=2>; rel="next", '
        '<https://api.github.com/x?page=10>; rel="last"'
    )
    assert _parse_next_link(header) == "https://api.github.com/x?page=2"
    assert _parse_next_link(None) is None
    assert _parse_next_link('<x>; rel="last"') is None


async def test_list_installation_repos_paginates_three_pages(
    http: httpx.AsyncClient,
) -> None:
    base = "https://api.github.com/installation/repositories"
    p1 = f"{base}?per_page=100"
    p2, p3 = f"{base}?page=2", f"{base}?page=3"
    with respx.mock(assert_all_called=True) as router:
        router.get(p1).mock(
            return_value=httpx.Response(
                200,
                json={"repositories": [{"full_name": "x/a"}, {"full_name": "x/b"}]},
                headers={"Link": _link_header(p2)},
            )
        )
        router.get(p2).mock(
            return_value=httpx.Response(
                200,
                json={"repositories": [{"full_name": "x/c"}]},
                headers={"Link": _link_header(p3)},
            )
        )
        router.get(p3).mock(
            return_value=httpx.Response(200, json={"repositories": [{"full_name": "x/d"}]})
        )
        client = _make_client(http)
        names = [r["full_name"] async for r in client.list_installation_repos()]
    assert names == ["x/a", "x/b", "x/c", "x/d"]


async def test_list_pulls_uses_recency_first_query(http: httpx.AsyncClient) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get(
            "https://api.github.com/repos/x/y/pulls",
            params={
                "state": "all",
                "sort": "updated",
                "direction": "desc",
                "per_page": "100",
            },
        ).mock(
            return_value=httpx.Response(
                200, json=[{"number": 1, "updated_at": "2026-04-01T00:00:00Z"}]
            )
        )
        client = _make_client(http)
        pulls = [p async for p in client.list_pulls("x/y")]
    assert pulls == [{"number": 1, "updated_at": "2026-04-01T00:00:00Z"}]
    assert route.called


async def test_list_pulls_stops_when_since_passed(http: httpx.AsyncClient) -> None:
    """Recency-desc means we can early-exit once updated_at < since."""
    with respx.mock(assert_all_called=True) as router:
        router.get("https://api.github.com/repos/x/y/pulls").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"number": 3, "updated_at": "2026-05-01T00:00:00Z"},
                    {"number": 2, "updated_at": "2026-04-15T00:00:00Z"},
                    {"number": 1, "updated_at": "2026-01-01T00:00:00Z"},
                ],
            )
        )
        client = _make_client(http)
        since = datetime(2026, 4, 1, tzinfo=UTC)
        numbers = [p["number"] async for p in client.list_pulls("x/y", since=since)]
    assert numbers == [3, 2]


async def test_list_issues_filters_pull_requests(http: httpx.AsyncClient) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get("https://api.github.com/repos/x/y/issues").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"number": 1, "title": "real issue"},
                    {"number": 2, "title": "PR", "pull_request": {"url": "..."}},
                    {"number": 3, "title": "another"},
                ],
            )
        )
        client = _make_client(http)
        issues = [i async for i in client.list_issues("x/y")]
    assert [i["number"] for i in issues] == [1, 3]


async def test_list_commits_passes_since_until(http: httpx.AsyncClient) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get(
            "https://api.github.com/repos/x/y/commits",
            params={
                "per_page": "100",
                "since": "2026-01-01T00:00:00Z",
                "until": "2026-02-01T00:00:00Z",
            },
        ).mock(return_value=httpx.Response(200, json=[{"sha": "abc"}]))
        client = _make_client(http)
        since = datetime(2026, 1, 1, tzinfo=UTC)
        until = datetime(2026, 2, 1, tzinfo=UTC)
        commits = [c async for c in client.list_commits("x/y", since=since, until=until)]
    assert commits == [{"sha": "abc"}]
    assert route.called


async def test_list_pull_reviews_hits_reviews_path(http: httpx.AsyncClient) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get("https://api.github.com/repos/x/y/pulls/42/reviews").mock(
            return_value=httpx.Response(200, json=[{"id": 99}])
        )
        client = _make_client(http)
        reviews = [r async for r in client.list_pull_reviews("x/y", 42)]
    assert reviews == [{"id": 99}]
    assert route.called


async def test_get_repo_returns_body(http: httpx.AsyncClient) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get("https://api.github.com/repos/x/y").mock(
            return_value=httpx.Response(200, json={"full_name": "x/y"})
        )
        client = _make_client(http)
        repo = await client.get_repo("x/y")
    assert repo["full_name"] == "x/y"


async def test_429_with_reset_sleeps_then_retries(http: httpx.AsyncClient) -> None:
    """First call 429s with reset 5s ahead; client sleeps and retries."""
    clock = _FakeClock(start=1_700_000_000.0)
    sleeper = _SleepRecorder(clock=clock)
    with respx.mock(assert_all_called=True) as router:
        route = router.get("https://api.github.com/repos/x/y")
        route.side_effect = [
            httpx.Response(429, headers={"x-ratelimit-reset": str(int(clock.t + 5))}, json={}),
            httpx.Response(200, json={"full_name": "x/y"}),
        ]
        client = _make_client(http, sleep=sleeper, clock=clock)
        repo = await client.get_repo("x/y")
    assert repo["full_name"] == "x/y"
    # Slept at least 5s (reset window) plus 1-3s jitter.
    assert any(c >= 5.0 for c in sleeper.calls), f"sleeps={sleeper.calls}"


async def test_403_with_remaining_zero_treated_as_rate_limit(
    http: httpx.AsyncClient,
) -> None:
    clock = _FakeClock()
    sleeper = _SleepRecorder(clock=clock)
    with respx.mock(assert_all_called=True) as router:
        route = router.get("https://api.github.com/repos/x/y")
        route.side_effect = [
            httpx.Response(
                403,
                json={},
                headers={
                    "x-ratelimit-remaining": "0",
                    "x-ratelimit-reset": str(int(clock.t + 2)),
                },
            ),
            httpx.Response(200, json={"full_name": "x/y"}),
        ]
        client = _make_client(http, sleep=sleeper, clock=clock)
        repo = await client.get_repo("x/y")
    assert repo["full_name"] == "x/y"
    assert sleeper.calls


async def test_five_consecutive_429s_raises_exhausted(http: httpx.AsyncClient) -> None:
    clock = _FakeClock()
    sleeper = _SleepRecorder(clock=clock)
    with respx.mock() as router:
        route = router.get("https://api.github.com/repos/x/y")
        route.side_effect = [httpx.Response(429, headers={"retry-after": "1"}) for _ in range(6)]
        client = _make_client(http, sleep=sleeper, clock=clock)
        with pytest.raises(GitHubRateLimitExhausted):
            await client.get_repo("x/y")


async def test_500_retries_three_times_then_raises(http: httpx.AsyncClient) -> None:
    clock = _FakeClock()
    sleeper = _SleepRecorder(clock=clock)
    with respx.mock() as router:
        route = router.get("https://api.github.com/repos/x/y")
        route.side_effect = [httpx.Response(500, text="boom") for _ in range(4)]
        client = _make_client(http, sleep=sleeper, clock=clock)
        with pytest.raises(GitHubAPIError) as exc:
            await client.get_repo("x/y")
    assert exc.value.status == 500


async def test_404_raises_immediately_no_retry(http: httpx.AsyncClient) -> None:
    with respx.mock() as router:
        route = router.get("https://api.github.com/repos/x/y").mock(
            return_value=httpx.Response(404, text="not found")
        )
        client = _make_client(http)
        with pytest.raises(GitHubAPIError) as exc:
            await client.get_repo("x/y")
    assert exc.value.status == 404
    assert route.call_count == 1


async def test_token_bucket_drains_then_blocks_until_refill() -> None:
    """12 acquires at rate=1/s capacity=10: last 2 must wait >=1s each.

    We drive a fake clock so virtual time is the only source of truth;
    every ``await sleep(s)`` advances the clock by exactly ``s``.
    """
    clock = _FakeClock()
    sleeper = _SleepRecorder(clock=clock)
    bucket = _AsyncTokenBucket(rate_per_second=1.0, capacity=10, now=clock, sleep=sleeper)
    start = clock.t
    for _ in range(12):
        await bucket.acquire()
    elapsed = clock.t - start
    # First 10 are free (burst). 11th and 12th each cost ~1s of wait.
    assert elapsed >= 1.5, f"elapsed={elapsed}, sleeps={sleeper.calls}"
    assert sum(sleeper.calls) >= 1.5


async def test_list_pulls_accepts_naive_datetime(http: httpx.AsyncClient) -> None:
    """Passing ``datetime(...)`` (naive) must not crash; coerced to UTC."""
    with respx.mock(assert_all_called=True) as router:
        router.get("https://api.github.com/repos/x/y/pulls").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"number": 3, "updated_at": "2026-05-01T00:00:00Z"},
                    {"number": 2, "updated_at": "2026-04-15T00:00:00Z"},
                    {"number": 1, "updated_at": "2026-01-01T00:00:00Z"},
                ],
            )
        )
        client = _make_client(http)
        since_naive = datetime(2026, 4, 1)  # naive on purpose, exercises coercion
        numbers = [p["number"] async for p in client.list_pulls("x/y", since=since_naive)]
    assert numbers == [3, 2]


async def test_list_issues_propagates_since_param(http: httpx.AsyncClient) -> None:
    """``since`` kwarg must serialize into the URL as ISO-Z."""
    with respx.mock(assert_all_called=True) as router:
        route = router.get(
            "https://api.github.com/repos/x/y/issues",
            params={
                "state": "all",
                "sort": "updated",
                "direction": "desc",
                "per_page": "100",
                "since": "2026-04-01T00:00:00Z",
            },
        ).mock(return_value=httpx.Response(200, json=[{"number": 1}]))
        client = _make_client(http)
        since = datetime(2026, 4, 1, tzinfo=UTC)
        issues = [i async for i in client.list_issues("x/y", since=since)]
    assert issues == [{"number": 1}]
    assert route.called


async def test_429_with_retry_after_header_sleeps(http: httpx.AsyncClient) -> None:
    """``retry-after: 3`` (no x-ratelimit-reset) sleeps ~3s then retries."""
    clock = _FakeClock()
    sleeper = _SleepRecorder(clock=clock)
    with respx.mock(assert_all_called=True) as router:
        route = router.get("https://api.github.com/repos/x/y")
        route.side_effect = [
            httpx.Response(429, headers={"retry-after": "3"}, json={}),
            httpx.Response(200, json={"full_name": "x/y"}),
        ]
        client = _make_client(http, sleep=sleeper, clock=clock)
        repo = await client.get_repo("x/y")
    assert repo["full_name"] == "x/y"
    # Slept >=3s (retry-after) plus 1-3s jitter; bounded above by 3+3=6.
    rate_limit_sleeps = [c for c in sleeper.calls if c >= 3.0]
    assert rate_limit_sleeps, f"sleeps={sleeper.calls}"
    assert rate_limit_sleeps[0] <= 7.0


async def test_500_then_200_recovers(http: httpx.AsyncClient) -> None:
    """Single 5xx then success: client retries once and returns body."""
    clock = _FakeClock()
    sleeper = _SleepRecorder(clock=clock)
    with respx.mock(assert_all_called=True) as router:
        route = router.get("https://api.github.com/repos/x/y")
        route.side_effect = [
            httpx.Response(500, text="boom"),
            httpx.Response(200, json={"full_name": "x/y"}),
        ]
        client = _make_client(http, sleep=sleeper, clock=clock)
        repo = await client.get_repo("x/y")
    assert repo["full_name"] == "x/y"
    assert sleeper.calls  # one backoff sleep happened


async def test_alternating_429_success_eventually_exhausts(http: httpx.AsyncClient) -> None:
    """Counters live on the instance: 429-success-429-... over many calls trips exhaust.

    With ``_MAX_CONSECUTIVE_RATE_LIMITS = 5`` and counter reset on 2xx, we
    construct a sequence where the counter never resets — five 429s in a row
    across calls — and confirm the client raises. A successful 2xx interleaved
    in the middle would reset the counter; we verify the no-reset case here.
    """
    clock = _FakeClock()
    sleeper = _SleepRecorder(clock=clock)
    # First call: 4x429 then success — counter ends at 0 (reset by 2xx).
    # Second call: 4x429 then success — counter ends at 0.
    # Third call: 5x429 — must raise on the 5th.
    with respx.mock() as router:
        route = router.get("https://api.github.com/repos/x/y")
        route.side_effect = [
            # call 1: 4x429, then 200
            *[httpx.Response(429, headers={"retry-after": "1"}, json={}) for _ in range(4)],
            httpx.Response(200, json={"full_name": "x/y"}),
            # call 2: 4x429, then 200
            *[httpx.Response(429, headers={"retry-after": "1"}, json={}) for _ in range(4)],
            httpx.Response(200, json={"full_name": "x/y"}),
            # call 3: 5x429 — exhausts
            *[httpx.Response(429, headers={"retry-after": "1"}, json={}) for _ in range(5)],
        ]
        client = _make_client(http, sleep=sleeper, clock=clock)
        # First two calls succeed (counter resets on each 2xx).
        await client.get_repo("x/y")
        await client.get_repo("x/y")
        # Third call: 5 consecutive 429s with no 2xx -> exhausted.
        with pytest.raises(GitHubRateLimitExhausted):
            await client.get_repo("x/y")


# ---------------------------------------------------------------------------
# Shared bucket registry (D6 — Phase 2 fan-out concurrency)
# ---------------------------------------------------------------------------


def test_get_shared_bucket_returns_same_instance_per_customer() -> None:
    """Multiple GitHubAPIClient instances on the same fly machine must
    share one ``_AsyncTokenBucket`` per (customer, source) so aggregate
    request rate stays at target_rps regardless of parallelism. The
    registry returns the same instance for repeated calls."""
    from kb.synthesis.api_clients.github import (
        _SHARED_BUCKETS,
        _AsyncTokenBucket,
        get_shared_bucket,
    )

    # Pop any prior test pollution so this test is hermetic.
    _SHARED_BUCKETS.clear()

    b1 = get_shared_bucket("cust-A")
    b2 = get_shared_bucket("cust-A")
    assert b1 is b2, "same key must return the same bucket"

    b3 = get_shared_bucket("cust-B")
    assert b1 is not b3, "different customer must get a different bucket"
    assert isinstance(b3, _AsyncTokenBucket)


def test_get_shared_bucket_ignores_target_rps_after_first_call() -> None:
    """The rate envelope is set on first creation; subsequent calls
    return the existing bucket as-is. A change to target_rps mid-run
    shouldn't shift an in-flight customer's quota."""
    from kb.synthesis.api_clients.github import _SHARED_BUCKETS, get_shared_bucket

    _SHARED_BUCKETS.clear()
    b1 = get_shared_bucket("cust-X", target_rps=2.0)
    b2 = get_shared_bucket("cust-X", target_rps=10.0)
    assert b1 is b2
    # ``_AsyncTokenBucket._rate`` is private API but the test asserts
    # we didn't replace the bucket with a faster one.
    assert b1._rate == 2.0
