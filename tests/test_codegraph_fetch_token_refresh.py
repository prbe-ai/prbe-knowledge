"""Token-refresh-on-401 behavior for code_graph.fetch.

Installation tokens last 60 minutes; long-tail incremental batches and
spillover-converted partial backfills can outlast a single token. The
fetcher must refresh exactly once on 401 and retry the failed request,
without minting N tokens for N concurrent 401s.
"""

from __future__ import annotations

import httpx
import pytest

from services.ingestion.code_graph import fetch as fetch_mod
from shared.exceptions import SourceAPIError


def _make_response(status: int, *, body: dict | None = None) -> httpx.Response:
    """Construct an httpx.Response without going through a real transport."""
    return httpx.Response(
        status_code=status,
        json=body or {"encoding": "base64", "content": ""},
        request=httpx.Request("GET", "https://api.github.com/x"),
    )


@pytest.mark.asyncio
async def test_no_refresh_when_customer_id_absent(monkeypatch) -> None:
    """Without customer_id, 401s surface as SourceAPIError. Backwards-compat
    check — the original API didn't refresh and existing callers must keep
    that behavior until they opt in.
    """
    calls: list[str] = []

    async def fake_get(self, url, **kwargs):
        calls.append(kwargs.get("headers", {}).get("Authorization", ""))
        return _make_response(401)

    async def boom(*args, **kwargs):
        raise AssertionError(
            "fetch_github_installation_token must NOT be called when customer_id is None"
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    monkeypatch.setattr(fetch_mod, "fetch_github_installation_token", boom)

    with pytest.raises(SourceAPIError):
        await fetch_mod.fetch_files_at_sha(
            repo="acme/api",
            sha="abc",
            paths=["src/foo.py"],
            token="old",
            customer_id=None,
        )

    assert len(calls) == 1
    assert calls[0] == "Bearer old"


@pytest.mark.asyncio
async def test_refresh_fires_once_on_401_then_retries(monkeypatch) -> None:
    """First request 401s, refresh mints a new token, retry uses it and
    succeeds. Single file, so only one refresh attempt.
    """
    bearers_seen: list[str] = []
    response_queue = [_make_response(401), _make_response(200)]

    async def fake_get(self, url, **kwargs):
        bearers_seen.append(kwargs.get("headers", {}).get("Authorization", ""))
        return response_queue.pop(0)

    refresh_calls = []

    async def fake_refresh(http, *, customer_id):
        refresh_calls.append(customer_id)
        return ("fresh-token", None)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    monkeypatch.setattr(fetch_mod, "fetch_github_installation_token", fake_refresh)

    results = await fetch_mod.fetch_files_at_sha(
        repo="acme/api",
        sha="abc",
        paths=["src/foo.py"],
        token="old",
        customer_id="c1",
    )

    assert len(results) == 1
    assert bearers_seen == ["Bearer old", "Bearer fresh-token"]
    assert refresh_calls == ["c1"]


@pytest.mark.asyncio
async def test_concurrent_401s_mint_token_only_once(monkeypatch) -> None:
    """Eight concurrent 401s on the same expired token must trigger
    exactly one fetch_github_installation_token call. Peers under the
    refresh lock observe state.value != stale and reuse the new token.

    Models production concurrency by having fake_get yield after popping
    a response (so all 8 first-sends interleave) and fake_refresh await
    a tiny sleep (so peer tasks queue on the lock while the holder mints,
    rather than each running serial with no overlap).
    """
    import asyncio as _asyncio

    response_queue: list[httpx.Response] = []
    # First 8 calls 401 (one per concurrent fetch); next 8 succeed
    # (the retries with the freshly-minted token).
    for _ in range(8):
        response_queue.append(_make_response(401))
    for _ in range(8):
        response_queue.append(_make_response(200))

    bearers_per_call: list[str] = []

    async def fake_get(self, url, **kwargs):
        bearers_per_call.append(kwargs.get("headers", {}).get("Authorization", ""))
        # Yield so other tasks' first-sends interleave before any retries.
        await _asyncio.sleep(0)
        return response_queue.pop(0)

    refresh_calls: list[str] = []

    async def fake_refresh(http, *, customer_id):
        refresh_calls.append(customer_id)
        # Yield while holding the lock so peer tasks finish their first send
        # and queue on lock.acquire — production prbe-backend latency is
        # tens of ms, plenty for peers to land on the lock.
        await _asyncio.sleep(0)
        return ("fresh-token", None)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    monkeypatch.setattr(fetch_mod, "fetch_github_installation_token", fake_refresh)

    paths = [f"src/file{i}.py" for i in range(8)]
    results = await fetch_mod.fetch_files_at_sha(
        repo="acme/api",
        sha="abc",
        paths=paths,
        token="old",
        customer_id="c1",
    )

    assert len(results) == 8
    # Exactly one mint call regardless of concurrent 401s.
    assert refresh_calls == ["c1"]
    # First 8 attempts use old token (the stale value); next 8 use fresh.
    assert bearers_per_call.count("Bearer old") == 8
    assert bearers_per_call.count("Bearer fresh-token") == 8


@pytest.mark.asyncio
async def test_refresh_failure_surfaces_original_401(monkeypatch) -> None:
    """If the refresher raises (e.g., prbe-backend down), the fetcher
    falls through to its existing 4xx handling on the original 401 —
    the queue row will retry later.
    """
    async def fake_get(self, url, **kwargs):
        return _make_response(401)

    async def fake_refresh(http, *, customer_id):
        raise RuntimeError("backend down")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    monkeypatch.setattr(fetch_mod, "fetch_github_installation_token", fake_refresh)

    with pytest.raises(SourceAPIError):
        await fetch_mod.fetch_files_at_sha(
            repo="acme/api",
            sha="abc",
            paths=["src/foo.py"],
            token="old",
            customer_id="c1",
        )
