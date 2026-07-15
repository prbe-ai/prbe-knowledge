"""POST /api/code-graph/reindex — manual reindex from the dashboard.

Covers the happy path (active github install → repos enqueued), the
"customer has no github installation" 404, and the auth gate.

The bridge's `enqueue_initial_backfill` is mocked because it writes the
event payload to R2 first; that integration is covered by the bridge's
own tests. Here we only verify the reindex helper enumerates repos and
calls the bridge with the right arguments.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from engine.shared.config import Settings, get_settings
from engine.shared.db import close_pool, init_pool, raw_conn
from kb.code_graph import reindex as reindex_module
from kb.ingestion_app import app

CUSTOMER = "cust-reindex-test"
INSTALLATION_ID = "12345"


def _stub_repos() -> list[dict]:
    return [
        {
            "full_name": "acme/svc-a",
            "default_branch": "main",
            "archived": False,
        },
        {
            "full_name": "acme/svc-b",
            "default_branch": "main",
            "archived": False,
        },
        {
            "full_name": "acme/legacy",
            "default_branch": "main",
            "archived": True,
        },
    ]


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture
def enqueued_calls() -> list[dict]:
    return []


@pytest.fixture(autouse=True)
def _stub_external_calls(monkeypatch, enqueued_calls: list[dict]) -> None:
    """Replace github, backend-token, and bridge calls with deterministic stubs.

    The reindex helper makes four external calls:
      - fetch_github_installation_token (prbe-backend)
      - _list_installation_repos        (api.github.com)
      - _resolve_head_sha               (api.github.com per repo)
      - enqueue_initial_backfill        (writes to R2 + queue row)
    Stub all four so the test stays hermetic.
    """

    async def _fake_token(http, *, customer_id):
        return ("stub-bearer", datetime.now(UTC) + timedelta(minutes=55))

    async def _fake_list_repos(http, bearer):
        return _stub_repos()

    async def _fake_resolve_head(http, bearer, full_name, default_branch):
        # Deterministic per-repo SHA so idempotency is testable across calls.
        return f"sha-{full_name.replace('/', '-')}-001"

    async def _fake_enqueue(**kwargs):
        enqueued_calls.append(kwargs)
        return True

    monkeypatch.setattr(
        reindex_module, "fetch_github_installation_token", _fake_token
    )
    monkeypatch.setattr(
        reindex_module, "_list_installation_repos", _fake_list_repos
    )
    monkeypatch.setattr(reindex_module, "_resolve_head_sha", _fake_resolve_head)
    monkeypatch.setattr(
        reindex_module, "enqueue_initial_backfill", _fake_enqueue
    )


@pytest_asyncio.fixture
async def client(live_db: None, settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'r', 'r-hash') ON CONFLICT DO NOTHING",
            CUSTOMER,
        )
        # access_token_encrypted is BYTEA NOT NULL; the reindex helper never
        # decrypts it (the github token is minted via prbe-backend), so an
        # empty blob is fine here.
        await conn.execute(
            "INSERT INTO integration_tokens "
            "(customer_id, source_system, status, scope, access_token_encrypted) "
            "VALUES ($1, 'github', 'active', $2, ''::bytea)",
            CUSTOMER,
            f"installation:{INSTALLATION_ID}",
        )

    await close_pool()
    transport = ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as c,
        app.router.lifespan_context(app),
    ):
        yield c
    await init_pool(settings)


def _hdr() -> dict[str, str]:
    return {"X-Internal-Knowledge-Key": "test-internal-key"}


@pytest.mark.asyncio
async def test_reindex_enqueues_per_repo_and_skips_archived(
    client: httpx.AsyncClient, enqueued_calls: list[dict],
) -> None:
    resp = await client.post(
        "/api/code-graph/reindex",
        json={"customer_id": CUSTOMER},
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 2 active repos enqueued, 1 archived skipped.
    assert body["enqueued"] == 2
    assert body["skipped"] == 1
    assert sorted(body["repos"]) == ["acme/svc-a", "acme/svc-b"]

    enqueued_repos = sorted(call["repo"] for call in enqueued_calls)
    assert enqueued_repos == ["acme/svc-a", "acme/svc-b"]
    for call in enqueued_calls:
        assert call["customer_id"] == CUSTOMER
        assert call["head_sha"].startswith("sha-")


@pytest.mark.asyncio
async def test_reindex_404s_when_no_github_install(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.post(
        "/api/code-graph/reindex",
        json={"customer_id": "cust-without-github"},
        headers=_hdr(),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_reindex_requires_internal_knowledge_key(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.post(
        "/api/code-graph/reindex",
        json={"customer_id": CUSTOMER},
        # No X-Internal-Knowledge-Key
    )
    assert resp.status_code == 401


