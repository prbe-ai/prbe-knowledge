"""Branch gating in github._fire_codegraph_incremental.

Without this filter, every push to every feature branch would re-extract
its changed files into a (customer, repo, file_path)-keyed cache that has
no branch dimension, so feature branch content would silently overwrite
main's cache rows. The gate compares the push's ref-stripped branch to
the per-(customer, repo) tracked branch (default: repo.default_branch,
override: customers.preferences.code_graph_branch_overrides[repo]).
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from engine.ingest.handlers.base import ConnectorContext
from engine.shared.config import Settings
from engine.shared.models import WebhookEvent
from kb.handlers import github as github_mod
from kb.handlers.github import (
    GitHubConnector,
    _push_branch_from_ref,
)


def _connector() -> GitHubConnector:
    return GitHubConnector(
        ConnectorContext(settings=Settings(), http=httpx.AsyncClient())
    )


def _push_event(
    *,
    customer_id: str = "c1",
    repo_full_name: str = "acme/api",
    default_branch: str = "main",
    ref: str = "refs/heads/main",
    sha: str = "deadbeef",
    files_added: list[str] | None = None,
    files_modified: list[str] | None = None,
) -> WebhookEvent:
    return WebhookEvent(
        customer_id=customer_id,
        source_system=github_mod.SourceSystem.GITHUB,
        source_event_id=f"push:{sha}",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/github/x.json",
        payload_s3_keys=["raw/github/x.json"],
        raw_payload={
            "ref": ref,
            "after": sha,
            "head_commit": {"id": sha},
            "repository": {
                "full_name": repo_full_name,
                "default_branch": default_branch,
            },
            "commits": [
                {
                    "added": files_added or [],
                    "modified": files_modified or ["src/foo.py"],
                    "removed": [],
                }
            ],
        },
        headers={},
    )


# ---- _push_branch_from_ref ----------------------------------------------


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("refs/heads/main", "main"),
        ("refs/heads/feat/some-thing", "feat/some-thing"),
        ("refs/tags/v1.0.0", None),
        ("refs/pull/42/merge", None),
        ("", None),
        (None, None),
        (123, None),
    ],
)
def test_push_branch_from_ref(ref, expected) -> None:
    assert _push_branch_from_ref(ref) == expected


# ---- branch gate ---------------------------------------------------------


@pytest.mark.asyncio
async def test_default_branch_push_fires_bridge(monkeypatch) -> None:
    """Push to the repo's default branch with no override → bridge fires."""
    enqueues: list[dict] = []

    async def fake_enqueue(*, customer_id, repo, sha, **kwargs):
        enqueues.append({"customer_id": customer_id, "repo": repo, "sha": sha})
        return True

    async def fake_branch_lookup(customer_id, repo, default_branch):
        return default_branch  # no override

    monkeypatch.setattr(
        github_mod.code_graph_bridge, "enqueue_incremental", fake_enqueue
    )
    monkeypatch.setattr(github_mod, "code_graph_indexed_branch", fake_branch_lookup)

    await _connector()._fire_codegraph_incremental(
        _push_event(ref="refs/heads/main", default_branch="main")
    )
    assert len(enqueues) == 1
    assert enqueues[0]["repo"] == "acme/api"


@pytest.mark.asyncio
async def test_feature_branch_push_skips_bridge(monkeypatch) -> None:
    """Push to a non-tracked branch → bridge does NOT fire. This is the
    main-branch filter the user requested; without it, feature branches
    poison the per-file content_hash cache.
    """
    enqueues: list[dict] = []

    async def fake_enqueue(**kwargs):
        enqueues.append(kwargs)
        return True

    async def fake_branch_lookup(customer_id, repo, default_branch):
        return default_branch

    monkeypatch.setattr(
        github_mod.code_graph_bridge, "enqueue_incremental", fake_enqueue
    )
    monkeypatch.setattr(github_mod, "code_graph_indexed_branch", fake_branch_lookup)

    await _connector()._fire_codegraph_incremental(
        _push_event(ref="refs/heads/feature/x", default_branch="main")
    )
    assert enqueues == []


@pytest.mark.asyncio
async def test_per_repo_override_redirects_tracking(monkeypatch) -> None:
    """When customer prefs override (acme/api → develop), pushes to develop
    fire the bridge; pushes to main are skipped.
    """
    enqueues: list[dict] = []

    async def fake_enqueue(**kwargs):
        enqueues.append(kwargs)
        return True

    async def fake_branch_lookup(customer_id, repo, default_branch):
        if repo == "acme/api":
            return "develop"
        return default_branch

    monkeypatch.setattr(
        github_mod.code_graph_bridge, "enqueue_incremental", fake_enqueue
    )
    monkeypatch.setattr(github_mod, "code_graph_indexed_branch", fake_branch_lookup)

    connector = _connector()
    await connector._fire_codegraph_incremental(
        _push_event(ref="refs/heads/develop", default_branch="main")
    )
    assert len(enqueues) == 1

    enqueues.clear()
    await connector._fire_codegraph_incremental(
        _push_event(ref="refs/heads/main", default_branch="main")
    )
    assert enqueues == []  # main is no longer tracked for this repo


@pytest.mark.asyncio
async def test_tag_push_skips_bridge(monkeypatch) -> None:
    """Tag pushes are not branch pushes — skip them outright."""
    enqueues: list[dict] = []

    async def fake_enqueue(**kwargs):
        enqueues.append(kwargs)
        return True

    async def fake_branch_lookup(customer_id, repo, default_branch):
        return default_branch

    monkeypatch.setattr(
        github_mod.code_graph_bridge, "enqueue_incremental", fake_enqueue
    )
    monkeypatch.setattr(github_mod, "code_graph_indexed_branch", fake_branch_lookup)

    await _connector()._fire_codegraph_incremental(
        _push_event(ref="refs/tags/v1.0.0", default_branch="main")
    )
    assert enqueues == []


@pytest.mark.asyncio
async def test_missing_default_branch_skips_bridge(monkeypatch) -> None:
    """A malformed payload missing repository.default_branch should not
    fire the bridge — we have no way to determine what to track.
    """
    enqueues: list[dict] = []

    async def fake_enqueue(**kwargs):
        enqueues.append(kwargs)
        return True

    monkeypatch.setattr(
        github_mod.code_graph_bridge, "enqueue_incremental", fake_enqueue
    )

    event = _push_event()
    event.raw_payload["repository"].pop("default_branch")
    await _connector()._fire_codegraph_incremental(event)
    assert enqueues == []
