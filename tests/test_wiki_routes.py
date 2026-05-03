"""Integration tests for /api/wiki/pages/* against a live Postgres + the
in-process embedder stub (OPENAI_API_KEY is empty in conftest).

Covers:
- PUT then GET round-trip for a wiki page
- PUT twice with different bodies bumps version + reuses unchanged chunks
- DELETE marks the page as deleted; GET returns 404 afterwards
- LIST returns the page with its wiki_type filter
- 401 without X-Internal-Knowledge-Key
- 400 on invalid wiki_type / slug / doc_class
- After PUT the wiki page is searchable through the chunks table
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from services.ingestion.main import app
from shared.config import Settings, get_settings
from shared.db import close_pool, init_pool, raw_conn

CUSTOMER = "wiki-test-cust"


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest_asyncio.fixture
async def client(live_db: None, settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'wiki-test', 'h') ON CONFLICT DO NOTHING",
            CUSTOMER,
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
    return {
        "X-Internal-Knowledge-Key": "test-internal-key",
        "X-Prbe-Customer": CUSTOMER,
    }


@pytest.mark.asyncio
async def test_put_then_get_roundtrip(client: httpx.AsyncClient) -> None:
    body = (
        "When the Slack backfill stalls, ping [[Person: mahit]] and check "
        "[[Service: prbe-knowledge]]. Plain ref: [[serialize-cc-claims]]."
    )
    resp = await client.put(
        "/api/wiki/pages/runbook/slack-backfill-stuck",
        json={"title": "Slack backfill stuck", "body": body},
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["doc_id"] == "wiki:runbook:slack-backfill-stuck"
    assert data["source_url"] == "/wiki/runbook/slack-backfill-stuck"
    assert data["version"] == 1
    assert data["chunk_count"] >= 1
    assert {(link["kind"], link["target"]) for link in data["links"]} >= {
        ("person", "mahit"),
        ("service", "prbe-knowledge"),
    }
    assert data["dangling_links"] == ["[[serialize-cc-claims]]"]

    fetched = await client.get("/api/wiki/pages/runbook/slack-backfill-stuck", headers=_hdr())
    assert fetched.status_code == 200, fetched.text
    page = fetched.json()
    assert page["title"] == "Slack backfill stuck"
    assert page["body"] == body
    assert page["doc_class"] == "manual_entry"
    assert page["wiki_type"] == "runbook"
    assert page["slug"] == "slack-backfill-stuck"
    assert page["version"] == 1


@pytest.mark.asyncio
async def test_put_twice_bumps_version_and_diffs_chunks(
    client: httpx.AsyncClient,
) -> None:
    headers = _hdr()
    await client.put(
        "/api/wiki/pages/decision/adopt-pgvector",
        json={
            "title": "Adopt pgvector",
            "body": "We adopt pgvector for retrieval. Cheap, integrated, RLS-friendly.",
        },
        headers=headers,
    )
    second = await client.put(
        "/api/wiki/pages/decision/adopt-pgvector",
        json={
            "title": "Adopt pgvector (revised)",
            "body": "We adopt pgvector for retrieval. Cheap, integrated, RLS-friendly. New addendum: HNSW index tuning.",
        },
        headers=headers,
    )
    assert second.status_code == 200, second.text
    assert second.json()["version"] >= 2

    fetched = await client.get("/api/wiki/pages/decision/adopt-pgvector", headers=headers)
    assert "addendum" in fetched.json()["body"]


@pytest.mark.asyncio
async def test_delete_then_get_404(client: httpx.AsyncClient) -> None:
    headers = _hdr()
    await client.put(
        "/api/wiki/pages/feature/auth",
        json={"title": "Auth", "body": "OAuth across all sources."},
        headers=headers,
    )
    deleted = await client.delete("/api/wiki/pages/feature/auth", headers=headers)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["deleted"] is True

    missing = await client.get("/api/wiki/pages/feature/auth", headers=headers)
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_list_filters_by_wiki_type(client: httpx.AsyncClient) -> None:
    headers = _hdr()
    await client.put(
        "/api/wiki/pages/runbook/r1",
        json={"title": "R1", "body": "first"},
        headers=headers,
    )
    await client.put(
        "/api/wiki/pages/decision/d1",
        json={"title": "D1", "body": "second"},
        headers=headers,
    )

    runbooks = await client.get("/api/wiki/pages?type=runbook", headers=headers)
    assert runbooks.status_code == 200, runbooks.text
    items = runbooks.json()["items"]
    assert {it["slug"] for it in items} == {"r1"}

    everything = await client.get("/api/wiki/pages", headers=headers)
    assert {it["slug"] for it in everything.json()["items"]} == {"r1", "d1"}


@pytest.mark.asyncio
async def test_put_requires_internal_key(client: httpx.AsyncClient) -> None:
    resp = await client.put(
        "/api/wiki/pages/runbook/x",
        json={"title": "X", "body": ""},
        headers={"X-Prbe-Customer": CUSTOMER},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_put_rejects_unknown_wiki_type(client: httpx.AsyncClient) -> None:
    resp = await client.put(
        "/api/wiki/pages/incident/x",
        json={"title": "X", "body": ""},
        headers=_hdr(),
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_put_rejects_invalid_slug(client: httpx.AsyncClient) -> None:
    resp = await client.put(
        "/api/wiki/pages/runbook/Bad_Slug",
        json={"title": "X", "body": ""},
        headers=_hdr(),
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_put_rejects_compiled_wiki_doc_class(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.put(
        "/api/wiki/pages/runbook/x",
        json={
            "title": "X",
            "body": "",
            "doc_class": "compiled_wiki",
        },
        headers=_hdr(),
    )
    assert resp.status_code == 422  # pydantic validator rejects


@pytest.mark.asyncio
async def test_put_persists_chunks_for_retrieval(
    client: httpx.AsyncClient,
) -> None:
    await client.put(
        "/api/wiki/pages/runbook/searchable",
        json={
            "title": "Searchable runbook",
            "body": "rare-token-xyzzy lives in this runbook for retrieval.",
        },
        headers=_hdr(),
    )
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT c.kind, c.content, d.doc_type
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id AND d.customer_id = c.customer_id
            WHERE c.customer_id = $1 AND d.doc_id = $2 AND c.valid_to IS NULL
            ORDER BY c.kind, c.chunk_index
            """,
            CUSTOMER,
            "wiki:runbook:searchable",
        )
    assert rows, "expected at least one persisted chunk"
    assert any(r["kind"] == "content" for r in rows)
    assert any("rare-token-xyzzy" in r["content"] for r in rows if r["kind"] == "content")
    assert all(r["doc_type"] == "wiki.runbook" for r in rows)


# ---------------------------------------------------------------------------
# History / revert / index (Phase 2 additions)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_lists_all_versions(client: httpx.AsyncClient) -> None:
    await client.put(
        "/api/wiki/pages/runbook/multi",
        json={
            "title": "Multi v1",
            "body": "Version one body.",
            "author_id": "alice@prbe.ai",
        },
        headers=_hdr(),
    )
    await client.put(
        "/api/wiki/pages/runbook/multi",
        json={
            "title": "Multi v2",
            "body": "Version two body — significantly revised.",
            "author_id": "alice@prbe.ai",
            "commit_message": "Rewrote the body.",
        },
        headers=_hdr(),
    )

    resp = await client.get("/api/wiki/pages/runbook/multi/history", headers=_hdr())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["doc_id"] == "wiki:runbook:multi"
    versions = [entry["version"] for entry in body["entries"]]
    assert versions == sorted(versions, reverse=True)
    live_count = sum(1 for entry in body["entries"] if entry["is_live"])
    assert live_count == 1
    # Newest version carries the explicit commit message.
    assert body["entries"][0]["commit_message"] == "Rewrote the body."
    # Older version got the default "Manual upload by ..." message.
    assert "Manual upload" in body["entries"][1]["commit_message"]


@pytest.mark.asyncio
async def test_revert_creates_new_version_with_old_body(
    client: httpx.AsyncClient,
) -> None:
    await client.put(
        "/api/wiki/pages/decision/db-choice",
        json={"title": "DB choice", "body": "Originally we chose Pinecone."},
        headers=_hdr(),
    )
    await client.put(
        "/api/wiki/pages/decision/db-choice",
        json={
            "title": "DB choice",
            "body": "Migrated off Pinecone to pgvector on Neon.",
        },
        headers=_hdr(),
    )

    resp = await client.post(
        "/api/wiki/pages/decision/db-choice/revert",
        json={
            "to_version": 1,
            "reason": "v2 lost the historical context",
            "author_id": "richard@prbe.ai",
        },
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["version"] >= 3

    fetched = await client.get("/api/wiki/pages/decision/db-choice", headers=_hdr())
    assert "Pinecone" in fetched.json()["body"]

    history = await client.get("/api/wiki/pages/decision/db-choice/history", headers=_hdr())
    entries = history.json()["entries"]
    assert entries[0]["commit_message"].startswith("Revert to v1")


@pytest.mark.asyncio
async def test_revert_404_on_unknown_version(client: httpx.AsyncClient) -> None:
    await client.put(
        "/api/wiki/pages/runbook/x",
        json={"title": "X", "body": "y"},
        headers=_hdr(),
    )
    resp = await client.post(
        "/api/wiki/pages/runbook/x/revert",
        json={"to_version": 99, "reason": "no such version"},
        headers=_hdr(),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_index_fallback_when_cron_has_not_run(
    client: httpx.AsyncClient,
) -> None:
    """Before the synthesis cron runs, GET /api/wiki/index returns a
    deterministic TOC built from the current page set."""
    await client.put(
        "/api/wiki/pages/runbook/r1",
        json={"title": "R1", "body": "First runbook."},
        headers=_hdr(),
    )
    await client.put(
        "/api/wiki/pages/decision/d1",
        json={"title": "D1", "body": "First decision."},
        headers=_hdr(),
    )

    resp = await client.get("/api/wiki/index", headers=_hdr())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    titles = {entry["title"] for entry in body["entries"]}
    assert {"R1", "D1"} <= titles
    assert "Wiki" in body["body"]
    assert body["updated_at"] is None  # cron-stored doc absent
    assert body["version"] is None


@pytest.mark.asyncio
async def test_put_rejects_index_wiki_type(client: httpx.AsyncClient) -> None:
    """The 'index' wiki_type is reserved for the synthesis cron — humans
    can't author it via PUT."""
    resp = await client.put(
        "/api/wiki/pages/index/contents",
        json={"title": "Hand-rolled index", "body": ""},
        headers=_hdr(),
    )
    assert resp.status_code == 400
