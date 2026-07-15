"""Unit tests for the BackfillFanout protocol + registry + GitHub impl.

Covers the non-DB-dependent contract:
  - REGISTRY exposes the GitHub discoverer at module import.
  - GitHubBackfillFanout filters archived/disabled repos, sorts by
    pushed_at desc, caps at BACKFILL_MAX_TARGETS_PER_SOURCE.
  - Discovery failures swallow + return empty list (D4).

DB-dependent fan-out hook (`_maybe_fanout_phase2`) is exercised by the
integration tests that run against `live_db`; this file is fast +
hermetic.
"""

from __future__ import annotations

from typing import Any

import pytest

from engine.shared.constants import BACKFILL_MAX_TARGETS_PER_SOURCE
from kb.synthesis.fanout import REGISTRY, BackfillFanout
from kb.synthesis.fanout.github import GitHubBackfillFanout


def test_registry_has_github() -> None:
    assert "github" in REGISTRY
    assert isinstance(REGISTRY["github"], BackfillFanout)
    assert isinstance(REGISTRY["github"], GitHubBackfillFanout)


def test_github_fanout_source_classvar() -> None:
    assert GitHubBackfillFanout.source == "github"


@pytest.mark.asyncio
async def test_github_fanout_filters_archived_and_disabled(monkeypatch) -> None:
    repos = [
        {
            "full_name": "o/active",
            "pushed_at": "2026-05-06T00:00:00Z",
            "archived": False,
            "disabled": False,
        },
        {
            "full_name": "o/archived",
            "pushed_at": "2026-05-05T00:00:00Z",
            "archived": True,
            "disabled": False,
        },
        {
            "full_name": "o/disabled",
            "pushed_at": "2026-05-04T00:00:00Z",
            "archived": False,
            "disabled": True,
        },
        {
            "full_name": "o/older",
            "pushed_at": "2026-04-01T00:00:00Z",
            "archived": False,
            "disabled": False,
        },
    ]

    class _FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def list_installation_repos(self):
            for r in repos:
                yield r

    monkeypatch.setattr("kb.synthesis.fanout.github.GitHubAPIClient", _FakeClient)

    fanout = GitHubBackfillFanout()
    targets = await fanout.discover_targets(
        customer_id="c",
        bearer="bearer",
        http=None,  # type: ignore[arg-type]
    )

    assert "o/archived" not in targets
    assert "o/disabled" not in targets
    # Sorted by pushed_at desc — newer first.
    assert targets == ["o/active", "o/older"]


@pytest.mark.asyncio
async def test_github_fanout_caps_at_max(monkeypatch) -> None:
    cap = BACKFILL_MAX_TARGETS_PER_SOURCE
    repos = [
        {
            "full_name": f"o/r{idx:03d}",
            "pushed_at": f"2026-05-{(idx % 28) + 1:02d}T00:00:00Z",
            "archived": False,
            "disabled": False,
        }
        for idx in range(cap + 5)
    ]

    class _FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def list_installation_repos(self):
            for r in repos:
                yield r

    monkeypatch.setattr("kb.synthesis.fanout.github.GitHubAPIClient", _FakeClient)

    targets = await GitHubBackfillFanout().discover_targets(
        customer_id="c",
        bearer="b",
        http=None,  # type: ignore[arg-type]
    )
    assert len(targets) == cap


@pytest.mark.asyncio
async def test_github_fanout_swallows_api_error(monkeypatch) -> None:
    class _FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def list_installation_repos(self):
            raise RuntimeError("boom")
            yield  # pragma: no cover — make this an async generator

    monkeypatch.setattr("kb.synthesis.fanout.github.GitHubAPIClient", _FakeClient)

    targets = await GitHubBackfillFanout().discover_targets(
        customer_id="c",
        bearer="b",
        http=None,  # type: ignore[arg-type]
    )
    assert targets == []


@pytest.mark.asyncio
async def test_github_fanout_skips_repos_missing_full_name(monkeypatch) -> None:
    repos: list[dict[str, Any]] = [
        {"full_name": "o/r1", "pushed_at": "2026-05-06T00:00:00Z"},
        {"full_name": None, "pushed_at": "2026-05-06T00:00:00Z"},  # malformed
        {"full_name": "", "pushed_at": "2026-05-06T00:00:00Z"},  # malformed
        {"full_name": "o/r2", "pushed_at": "2026-05-05T00:00:00Z"},
    ]

    class _FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def list_installation_repos(self):
            for r in repos:
                yield r

    monkeypatch.setattr("kb.synthesis.fanout.github.GitHubAPIClient", _FakeClient)

    targets = await GitHubBackfillFanout().discover_targets(
        customer_id="c",
        bearer="b",
        http=None,  # type: ignore[arg-type]
    )
    assert targets == ["o/r1", "o/r2"]
