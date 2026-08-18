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

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from engine.shared.config import Settings, get_settings
from engine.shared.db import close_pool, init_pool, raw_conn, with_tenant
from kb.ingestion_app import app
from kb.synthesis import persistence
from tests.wiki_fixtures import insert_document

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
        "/api/wiki/pages/runbook/auth",
        json={"title": "Auth", "body": "OAuth across all sources."},
        headers=headers,
    )
    deleted = await client.delete("/api/wiki/pages/runbook/auth", headers=headers)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["deleted"] is True

    missing = await client.get("/api/wiki/pages/runbook/auth", headers=headers)
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
async def test_put_rejects_invalid_wiki_type(client: httpx.AsyncClient) -> None:
    # wiki_type is free-form (the LLM picks slugs as it sees fit), so
    # `/api/wiki/pages/incident/x` is now valid. The route still rejects
    # the singleton 'index' type (cron-only) and any string that
    # violates the URL-safe regex `^[a-z][a-z0-9_]{0,31}$`.
    resp = await client.put(
        "/api/wiki/pages/index/x",
        json={"title": "X", "body": ""},
        headers=_hdr(),
    )
    assert resp.status_code == 400

    resp = await client.put(
        "/api/wiki/pages/Has-Hyphen/x",
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
async def test_put_rejects_index_wiki_type(client: httpx.AsyncClient) -> None:
    """The 'index' wiki_type is reserved for the synthesis cron — humans
    can't author it via PUT."""
    resp = await client.put(
        "/api/wiki/pages/index/contents",
        json={"title": "Hand-rolled index", "body": ""},
        headers=_hdr(),
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_revert_rejects_index_wiki_type(client: httpx.AsyncClient) -> None:
    """Same reservation, on the other write. Worth its own test because the
    read below deliberately lifts it: the two must not drift into each other.

    A COMPLETE body (`reason` is required too): an incomplete one is a 422
    from FastAPI before the route ever runs, which would pass a "not 200"
    check while proving nothing about the reservation.
    """
    resp = await client.post(
        "/api/wiki/pages/index/contents/revert",
        json={"to_version": 1, "reason": "trying to rewrite the generated index"},
        headers=_hdr(),
    )
    assert resp.status_code == 400, resp.text
    assert "reserved" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_history_is_readable_for_the_index_page(
    client: httpx.AsyncClient,
) -> None:
    """The index page's history is READABLE even though writing it is not.

    THIS SHIPPED BROKEN. The reservation was applied by one validator shared
    between the writes and this read, so `GET /pages/index/contents/history`
    answered 400 for every tenant — and with it research-os's
    `GET /v1/wiki/versions` and the MCP's `wiki_versions()`, both of which ask
    for exactly this page because it is the document `GET /v1/wiki` returns.

    The row is written through the app's own PUT and then RE-IDENTIFIED as the
    index page, rather than hand-inserted: `documents` has a dozen NOT NULL
    columns the writer fills, and a hand-rolled INSERT that keeps up with them
    is a second copy of the writer that rots.

    Asserts a REAL history, not just "not 400": a route that validated the type
    and then read the wrong doc_id would still pass a status check.
    """
    await client.put(
        "/api/wiki/pages/runbook/soon-to-be-index",
        json={"title": "Contents", "body": "A generated table of contents."},
        headers=_hdr(),
    )
    doc_id = "wiki:index:contents"
    async with raw_conn() as conn:
        await conn.execute(
            """
            UPDATE documents
               SET doc_id = $2, source_id = 'index:contents', doc_type = 'wiki.index'
             WHERE customer_id = $1 AND doc_id = $3
            """,
            CUSTOMER,
            doc_id,
            "wiki:runbook:soon-to-be-index",
        )

    resp = await client.get("/api/wiki/pages/index/contents/history", headers=_hdr())

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["doc_id"] == doc_id
    assert [entry["version"] for entry in body["entries"]] == [1]
    assert body["entries"][0]["is_live"] is True


@pytest.mark.asyncio
async def test_history_of_an_absent_index_page_is_404_not_400(
    client: httpx.AsyncClient,
) -> None:
    """A tenant whose synthesis has never run must get ABSENCE, not rejection.

    The two are different bugs with different fixes, and 400 was the one this
    route gave for every tenant regardless: "your wiki_type is invalid" sends a
    caller looking at their request when the answer is "there is nothing here
    yet". research-os maps this 404 to an empty history; it maps a 400 to a
    502, so the distinction reaches users.
    """
    resp = await client.get("/api/wiki/pages/index/contents/history", headers=_hdr())
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_history_still_rejects_a_malformed_wiki_type(
    client: httpx.AsyncClient,
) -> None:
    """Lifting the `index` reservation must not lift the SHAPE check with it.

    The shape check is what keeps a stray path segment from becoming a
    permanent garbage doc_id lookup; only the reservation is read-specific.
    """
    resp = await client.get("/api/wiki/pages/NOT-A-TYPE/contents/history", headers=_hdr())
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Bootstrap trigger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_trigger_requires_internal_key(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.post(
        "/api/wiki/bootstrap/trigger",
        json={"sources": ["github"]},
        headers={"X-Prbe-Customer": CUSTOMER},
    )
    assert resp.status_code == 401


@pytest.fixture
def _stub_bootstrap_registry(monkeypatch) -> None:
    """Make the trigger route's REGISTRY validation see a known set of
    sources during tests. Lane C ships the registry empty, so without
    this any /bootstrap/trigger payload with `sources=[...]` would 400.
    Lane D will register real crawlers; tests can then drop this stub."""
    from kb import wiki_routes as _wr

    monkeypatch.setattr(
        _wr,
        "BACKFILL_CRAWLER_REGISTRY",
        {"github": object, "slack": object},
        raising=False,
    )


@pytest.mark.asyncio
async def test_bootstrap_trigger_fires_pg_notify(
    client: httpx.AsyncClient, _stub_bootstrap_registry: None
) -> None:
    """POSTing the trigger fires payload-less pg_notify on
    WIKI_BACKFILL_CHANNEL and inserts pending rows the worker will
    claim. Body is now empty payload (workers claim via FOR UPDATE
    SKIP LOCKED), so the listener no longer routes per-payload."""
    import asyncio

    import asyncpg

    from engine.shared.config import get_settings as _get_settings
    from engine.shared.constants import WIKI_BACKFILL_CHANNEL

    notifications: list[str] = []
    listen_dsn = _get_settings().database_url
    listener_conn = await asyncpg.connect(listen_dsn)

    def _on_notify(_c, _pid, _channel, payload) -> None:
        notifications.append(payload)

    try:
        await listener_conn.add_listener(WIKI_BACKFILL_CHANNEL, _on_notify)

        resp = await client.post(
            "/api/wiki/bootstrap/trigger",
            json={
                "sources": ["github", "slack"],
                "wipe_first": True,
                "reason": "first run",
            },
            headers=_hdr(),
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["triggered"] is True
        assert isinstance(body["run_ids"], list)
        assert len(body["run_ids"]) == 2
        assert all(isinstance(rid, int) for rid in body["run_ids"])

        # Give the NOTIFY a moment to deliver — Postgres queues NOTIFY
        # at NOTIFY-time and delivers on commit; ASGITransport runs the
        # endpoint synchronously inside the same loop.
        for _ in range(20):
            if notifications:
                break
            await asyncio.sleep(0.05)
        assert notifications, "expected pg_notify on the bootstrap channel"
        # Wake hint is empty payload — workers claim rows directly,
        # no per-NOTIFY routing needed.
        assert notifications[0] == ""

        # The route inserts pending wiki_synthesis_runs rows; workers
        # claim via FOR UPDATE SKIP LOCKED and flip them to running.
        async with raw_conn() as conn:
            rows = await conn.fetch(
                """
                SELECT source, kind, stage, status FROM wiki_synthesis_runs
                WHERE customer_id = $1 AND kind = 'bootstrap'
                ORDER BY source
                """,
                CUSTOMER,
            )
        assert {r["source"] for r in rows} == {"github", "slack"}
        assert all(r["kind"] == "bootstrap" for r in rows)
        assert all(r["stage"] == "synthesis" for r in rows)
        # New invariant: trigger inserts at 'pending', not 'running'.
        assert all(r["status"] == "pending" for r in rows)
    finally:
        await listener_conn.close()


@pytest.mark.asyncio
async def test_bootstrap_trigger_defaults(
    client: httpx.AsyncClient, _stub_bootstrap_registry: None
) -> None:
    """Empty body defaults to all registered crawlers; wipe_first=True."""
    resp = await client.post(
        "/api/wiki/bootstrap/trigger",
        json={},
        headers=_hdr(),
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["triggered"] is True
    assert isinstance(body["run_ids"], list)
    # Stub registry has two entries (github + slack), so all-default
    # should pre-open one run per source.
    assert len(body["run_ids"]) == 2


@pytest.mark.asyncio
async def test_bootstrap_trigger_rejects_unknown_sources(
    client: httpx.AsyncClient, _stub_bootstrap_registry: None
) -> None:
    """An unknown source name returns 400, not a silent drop."""
    resp = await client.post(
        "/api/wiki/bootstrap/trigger",
        json={"sources": ["github", "definitely-not-real"]},
        headers=_hdr(),
    )
    assert resp.status_code == 400, resp.text
    assert "definitely-not-real" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_bootstrap_trigger_returns_409_on_in_flight(
    client: httpx.AsyncClient, _stub_bootstrap_registry: None
) -> None:
    """An in-flight (pending or running) row blocks a fresh trigger
    unless ``force=true``. The 409 body carries the in-flight run_ids
    + per-source status so the dashboard can render a structured
    cancel-and-restart prompt."""
    # Pre-insert a 'running' row for github.
    async with raw_conn() as conn:
        existing_id = int(
            await conn.fetchval(
                """
                INSERT INTO wiki_synthesis_runs
                    (customer_id, kind, stage, source, status)
                VALUES ($1, 'bootstrap', 'synthesis', 'github', 'running')
                RETURNING run_id
                """,
                CUSTOMER,
            )
        )

    resp = await client.post(
        "/api/wiki/bootstrap/trigger",
        json={"sources": ["slack"]},
        headers=_hdr(),
    )
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["status"] == "in_flight"
    assert existing_id in detail["run_ids"]
    assert detail["sources_running"] == ["github"]
    assert detail["sources_pending"] == []
    # No new pending rows were inserted (atomicity check).
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT source, status FROM wiki_synthesis_runs "
            "WHERE customer_id = $1 AND kind = 'bootstrap'",
            CUSTOMER,
        )
    assert {(r["source"], r["status"]) for r in rows} == {("github", "running")}


@pytest.mark.asyncio
async def test_bootstrap_trigger_force_proceeds_after_drain_timeout(
    client: httpx.AsyncClient,
    _stub_bootstrap_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``?force=true`` with an in-flight row + no worker registered:
    trigger marks the old row 'cancelled', sleeps the drain window,
    proceeds with wipe + new pending insert.

    Drain timeout patched down so the test runs fast."""
    from kb import wiki_routes as _wr

    monkeypatch.setattr(_wr, "BACKFILL_CANCEL_DRAIN_TIMEOUT_SECONDS", 0.05)

    async with raw_conn() as conn:
        old_id = int(
            await conn.fetchval(
                """
                INSERT INTO wiki_synthesis_runs
                    (customer_id, kind, stage, source, status)
                VALUES ($1, 'bootstrap', 'synthesis', 'github', 'running')
                RETURNING run_id
                """,
                CUSTOMER,
            )
        )

    resp = await client.post(
        "/api/wiki/bootstrap/trigger?force=true",
        json={"sources": ["github"], "wipe_first": False},
        headers=_hdr(),
    )
    assert resp.status_code == 202, resp.text
    new_run_ids = resp.json()["run_ids"]
    assert len(new_run_ids) == 1
    assert old_id not in new_run_ids

    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT run_id, status, error FROM wiki_synthesis_runs "
            "WHERE customer_id = $1 AND kind = 'bootstrap' ORDER BY run_id",
            CUSTOMER,
        )
    by_id = {int(r["run_id"]): r for r in rows}
    assert by_id[old_id]["status"] == "cancelled"
    assert "force-trigger" in (by_id[old_id]["error"] or "")
    new_id = next(rid for rid in by_id if rid != old_id)
    assert by_id[new_id]["status"] == "pending"


@pytest.mark.asyncio
async def test_bootstrap_trigger_force_fires_cancel_notify(
    client: httpx.AsyncClient,
    _stub_bootstrap_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``?force=true`` fires pg_notify on WIKI_BACKFILL_CANCEL_CHANNEL
    with a JSON payload carrying customer_id + cancelled run_ids so
    workers can cancel matching tasks."""
    import asyncio

    import asyncpg

    from engine.shared.config import get_settings as _get_settings
    from engine.shared.constants import WIKI_BACKFILL_CANCEL_CHANNEL
    from kb import wiki_routes as _wr

    monkeypatch.setattr(_wr, "BACKFILL_CANCEL_DRAIN_TIMEOUT_SECONDS", 0.05)

    listen_dsn = _get_settings().database_url
    listener_conn = await asyncpg.connect(listen_dsn)
    cancel_payloads: list[str] = []

    def _on_cancel(_c, _pid, _channel, payload) -> None:
        cancel_payloads.append(payload)

    try:
        await listener_conn.add_listener(WIKI_BACKFILL_CANCEL_CHANNEL, _on_cancel)

        async with raw_conn() as conn:
            old_id = int(
                await conn.fetchval(
                    """
                    INSERT INTO wiki_synthesis_runs
                        (customer_id, kind, stage, source, status)
                    VALUES ($1, 'bootstrap', 'synthesis', 'slack', 'pending')
                    RETURNING run_id
                    """,
                    CUSTOMER,
                )
            )

        resp = await client.post(
            "/api/wiki/bootstrap/trigger?force=true",
            json={"sources": ["slack"], "wipe_first": False},
            headers=_hdr(),
        )
        assert resp.status_code == 202, resp.text

        for _ in range(20):
            if cancel_payloads:
                break
            await asyncio.sleep(0.05)
        assert cancel_payloads, "expected pg_notify on the cancel channel"
        import orjson as _orjson

        decoded = _orjson.loads(cancel_payloads[0])
        assert decoded["customer_id"] == CUSTOMER
        assert old_id in decoded["run_ids"]
    finally:
        await listener_conn.close()


# ---------------------------------------------------------------------------
# Bootstrap status
# ---------------------------------------------------------------------------


async def _insert_bootstrap_run(
    *,
    source: str,
    status: str,
    pages_created: int = 0,
    pages_updated: int = 0,
    error: str | None = None,
    started_offset_seconds: int = 0,
) -> int:
    async with raw_conn() as conn:
        return int(
            await conn.fetchval(
                """
                INSERT INTO wiki_synthesis_runs
                    (customer_id, kind, stage, source, status,
                     pages_created, pages_updated, error,
                     started_at, finished_at)
                VALUES ($1, 'bootstrap', 'synthesis', $2, $3, $4, $5, $6,
                        NOW() - make_interval(secs => $7),
                        CASE WHEN $3 = 'running' THEN NULL
                             ELSE NOW() - make_interval(secs => $7) END)
                RETURNING run_id
                """,
                CUSTOMER,
                source,
                status,
                pages_created,
                pages_updated,
                error,
                started_offset_seconds,
            )
        )


@pytest.mark.asyncio
async def test_bootstrap_status_when_never_run(client: httpx.AsyncClient) -> None:
    """Empty payload when the customer has never bootstrapped."""
    resp = await client.get("/api/wiki/bootstrap/status", headers=_hdr())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {
        "in_progress": False,
        "started_at": None,
        "sources_attempted": [],
        "sources_succeeded": [],
        "sources_failed": {},
        "pages_created": 0,
        "pages_updated": 0,
        "targets": {},
    }


@pytest.mark.asyncio
async def test_bootstrap_status_aggregates_recent_burst(
    client: httpx.AsyncClient,
) -> None:
    """One burst with three sources: complete, partial, failed."""
    await _insert_bootstrap_run(
        source="github", status="complete", pages_created=3, pages_updated=2
    )
    await _insert_bootstrap_run(source="slack", status="partial", pages_created=1, pages_updated=4)
    await _insert_bootstrap_run(source="linear", status="failed", error="rate-limited")

    resp = await client.get("/api/wiki/bootstrap/status", headers=_hdr())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["in_progress"] is False
    assert sorted(body["sources_attempted"]) == ["github", "linear", "slack"]
    # complete + partial both count as "succeeded"; failed surfaces the error.
    assert sorted(body["sources_succeeded"]) == ["github", "slack"]
    assert body["sources_failed"] == {"linear": "rate-limited"}
    assert body["pages_created"] == 4
    assert body["pages_updated"] == 6
    assert body["started_at"] is not None


@pytest.mark.asyncio
async def test_bootstrap_status_in_progress(client: httpx.AsyncClient) -> None:
    """A 'running' row in the burst flips in_progress=True."""
    await _insert_bootstrap_run(source="github", status="complete")
    await _insert_bootstrap_run(source="slack", status="running")

    resp = await client.get("/api/wiki/bootstrap/status", headers=_hdr())
    assert resp.status_code == 200, resp.text
    assert resp.json()["in_progress"] is True


@pytest.mark.asyncio
async def test_bootstrap_status_ignores_old_runs_outside_burst(
    client: httpx.AsyncClient,
) -> None:
    """Rows older than 60s before the anchor are NOT in the current burst."""
    # Old burst from a prior trigger — anchor should ignore.
    await _insert_bootstrap_run(source="ancient", status="complete", started_offset_seconds=3600)
    # Recent burst.
    await _insert_bootstrap_run(source="github", status="complete", pages_created=1)
    resp = await client.get("/api/wiki/bootstrap/status", headers=_hdr())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["sources_attempted"] == ["github"]
    assert body["pages_created"] == 1


# ---------------------------------------------------------------------------
# Compare-and-swap on PUT + the per-page lock the synthesis agent takes
# ---------------------------------------------------------------------------
#
# The wiki has two writers -- the nightly synthesis agent and whoever runs
# `probe wiki write` -- and before this route took a version precondition,
# the second one to arrive silently won. These tests pin the contract that
# replaces that: equal version wins, stale gets 409 CARRYING THE BODY, and
# a write with no precondition still works so the dashboard BFF (which has
# never sent one) keeps functioning.


async def _put(client: httpx.AsyncClient, slug: str, body: str, **extra: object) -> httpx.Response:
    return await client.put(
        f"/api/wiki/pages/runbook/{slug}",
        json={"title": "CAS fixture", "body": body, **extra},
        headers=_hdr(),
    )


@pytest.mark.asyncio
async def test_put_without_expected_version_still_writes(
    client: httpx.AsyncClient,
) -> None:
    """BACKWARDS COMPATIBILITY, and the reason the engine's precondition is
    optional while research-os's is mandatory.

    The dashboard reaches this route through the prbe-backend BFF and has
    never sent a version. A mandatory precondition here would 428 every
    dashboard save on the day it shipped. research-os requires the version
    at ITS boundary instead, where every caller is new.
    """
    first = await _put(client, "no-precondition", "one")
    assert first.status_code == 200, first.text
    second = await _put(client, "no-precondition", "two")
    assert second.status_code == 200, second.text
    assert second.json()["version"] == 2


@pytest.mark.asyncio
async def test_put_with_matching_expected_version_wins(
    client: httpx.AsyncClient,
) -> None:
    """The writer that read the current version writes."""
    created = await _put(client, "matching", "one", expected_version=0)
    assert created.status_code == 200, created.text
    assert created.json()["version"] == 1

    updated = await _put(client, "matching", "two", expected_version=1)
    assert updated.status_code == 200, updated.text
    assert updated.json()["version"] == 2


@pytest.mark.asyncio
async def test_put_with_stale_expected_version_409_carries_current_body(
    client: httpx.AsyncClient,
) -> None:
    """THE CONTRACT WORTH PORTING. A loser is told what it lost to.

    The body in the 409 is the whole point: the other writer is usually the
    nightly agent, which the caller cannot see coming and cannot wait out.
    Handing back only a version number means every loser must re-read, and
    the re-read races the same way.
    """
    await _put(client, "stale", "original", expected_version=0)
    await _put(client, "stale", "moved on by someone else", expected_version=1)

    lost = await _put(client, "stale", "my edit", expected_version=1)
    assert lost.status_code == 409, lost.text
    detail = lost.json()["detail"]
    assert detail["expected_version"] == 1
    assert detail["current_version"] == 2
    assert detail["current_body"] == "moved on by someone else"

    # AND NOTHING WAS WRITTEN. A 409 that still persisted would be worse
    # than no check at all -- it would report a conflict while performing
    # the clobber the check exists to prevent.
    page = await client.get("/api/wiki/pages/runbook/stale", headers=_hdr())
    assert page.json()["body"] == "moved on by someone else"
    assert page.json()["version"] == 2


@pytest.mark.asyncio
async def test_put_expecting_absent_page_conflicts_when_it_exists(
    client: httpx.AsyncClient,
) -> None:
    """`expected_version=0` means "create it, and only if nobody beat me".

    Without this branch, two agents told to create the same page would both
    succeed and the second would silently replace the first's work.
    """
    await _put(client, "already-there", "first writer", expected_version=0)
    second = await _put(client, "already-there", "second writer", expected_version=0)
    assert second.status_code == 409, second.text
    assert second.json()["detail"]["current_body"] == "first writer"


# ---------------------------------------------------------------------------
# The two writers, together
# ---------------------------------------------------------------------------


async def _run_agent_update(slug: str, body_markdown: str) -> None:
    """Run the REAL agent persist path against one page.

    Not a stand-in for the agent: `_persist_update` is the function the
    nightly run calls, and it is the one that decides whether a page may be
    rewritten. Tests that mocked it would pass against a skip that no longer
    exists.
    """
    from kb.synthesis.wiki_agent import WikiAgentRuntime, _StagedUpdate

    runtime = WikiAgentRuntime(
        CUSTOMER,
        agent_run_id="test-agent-run",
        run_id=1,
        run_kind="synthesis",
    )
    await runtime._persist_update(
        _StagedUpdate(
            wiki_type="runbook",
            slug=slug,
            body_markdown=body_markdown,
            summary="nightly synthesis pass",
            commit_message="synthesis",
        )
    )


@pytest.mark.asyncio
async def test_hand_editing_a_page_does_not_stop_the_pipeline(
    client: httpx.AsyncClient,
) -> None:
    """THE DEFINITION-OF-DONE TEST, and it asserts the OPPOSITE of what it
    used to.

    Editing a page by hand used to freeze it forever: the route stamps every
    public write `doc_class=manual_entry`, and `_persist_update` skipped that
    class outright. No API, CLI or UI path could undo it. Nobody chose that --
    they fixed a typo and the page quietly stopped updating.

    So the guarantee now runs the other way: a hand-written page STILL
    receives nightly updates, because a human edit is evidence about what the
    page should say, not an instruction to abandon it. Freezing is a separate,
    explicit, reversible thing -- see the tests below.
    """
    human_text = "Written by a person through `probe wiki write`."
    created = await _put(client, "human-owned", human_text, expected_version=0)
    assert created.status_code == 200, created.text
    assert created.json()["version"] == 1, "precondition: the hand write is the live version"

    await _run_agent_update("human-owned", "THE AGENT UPDATED THIS PAGE.")

    page = await client.get("/api/wiki/pages/runbook/human-owned", headers=_hdr())
    assert page.status_code == 200, page.text
    assert page.json()["body"] == "THE AGENT UPDATED THIS PAGE."
    assert page.json()["version"] == 2, "the agent's write is a new version"
    # The default is ON and a hand edit does not change it. Asserted
    # explicitly: if a future writer starts stamping the setting on every PUT,
    # the body assertion above would still pass while the trapdoor came back.
    assert page.json()["pipeline_updates"] is True


@pytest.mark.asyncio
async def test_settings_default_is_on_for_a_page_nobody_configured(
    client: httpx.AsyncClient,
) -> None:
    """THE MIGRATION DECISION, asserted as behaviour.

    Migration 0103 inserts no rows: an absent `wiki_page_settings` row reads
    as pipeline_updates=TRUE. That is what returns every page frozen by the
    old trapdoor to receiving updates, without a backfill UPDATE that could
    get its WHERE clause wrong.

    So the default has to live in the reader, and it has to be ON.
    """
    from kb.synthesis import persistence

    await _put(client, "never-configured", "body", expected_version=0)

    page = await client.get("/api/wiki/pages/runbook/never-configured", headers=_hdr())
    assert page.json()["pipeline_updates"] is True

    # Asserted at the persistence layer too, because that is the function the
    # nightly agent consults -- a route that hardcoded True in its response
    # would satisfy the check above while the agent still skipped the page.
    assert (
        await persistence.fetch_page_pipeline_updates(CUSTOMER, "runbook", "never-configured")
        is True
    )
    # And for a page that does not exist at all: not frozen, because there is
    # nothing there to freeze.
    assert (
        await persistence.fetch_page_pipeline_updates(CUSTOMER, "runbook", "no-such-page") is True
    )


@pytest.mark.asyncio
async def test_freezing_a_page_stops_the_pipeline(client: httpx.AsyncClient) -> None:
    """The explicit setting is what stops a rewrite -- and only it."""
    frozen_text = "Frozen on purpose. The nightly run must leave this alone."
    await _put(client, "frozen", frozen_text, expected_version=0)

    resp = await client.put(
        "/api/wiki/pages/runbook/frozen/settings",
        json={"pipeline_updates": False, "author_id": "richard"},
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["pipeline_updates"] is False

    await _run_agent_update("frozen", "THE AGENT REWROTE THIS PAGE FROM SCRATCH.")

    page = await client.get("/api/wiki/pages/runbook/frozen", headers=_hdr())
    assert page.json()["body"] == frozen_text
    assert page.json()["version"] == 1, "a skipped write must not bump the version"
    assert page.json()["pipeline_updates"] is False


@pytest.mark.asyncio
async def test_freezing_is_reversible(client: httpx.AsyncClient) -> None:
    """THE POINT OF THE WHOLE CHANGE: a frozen page can be un-frozen, through
    the same API that froze it.

    The old freeze had no path back at any layer -- not the route, not the
    CLI, not the dashboard. Only hand-written SQL. A test that only proved
    freezing works would have passed against that design too.
    """
    await _put(client, "thawed", "original", expected_version=0)

    for state in (False, True):
        resp = await client.put(
            "/api/wiki/pages/runbook/thawed/settings",
            json={"pipeline_updates": state},
            headers=_hdr(),
        )
        assert resp.status_code == 200, resp.text

    await _run_agent_update("thawed", "THE AGENT UPDATED THIS PAGE AGAIN.")

    page = await client.get("/api/wiki/pages/runbook/thawed", headers=_hdr())
    assert page.json()["body"] == "THE AGENT UPDATED THIS PAGE AGAIN."
    assert page.json()["pipeline_updates"] is True


@pytest.mark.asyncio
async def test_toggling_the_setting_writes_no_version(
    client: httpx.AsyncClient,
) -> None:
    """Freezing changes no prose, so it must not appear in page history.

    This is why the setting lives in `wiki_page_settings` instead of on the
    `documents` row: a version-carried flag could only be changed by writing a
    version, and a history where half the entries changed nothing is a history
    nobody reads.
    """
    await _put(client, "quiet-toggle", "body text", expected_version=0)
    before = await client.get("/api/wiki/pages/runbook/quiet-toggle", headers=_hdr())

    await client.put(
        "/api/wiki/pages/runbook/quiet-toggle/settings",
        json={"pipeline_updates": False},
        headers=_hdr(),
    )

    after = await client.get("/api/wiki/pages/runbook/quiet-toggle", headers=_hdr())
    assert after.json()["version"] == before.json()["version"]
    assert after.json()["body"] == before.json()["body"]
    assert after.json()["pipeline_updates"] is False


@pytest.mark.asyncio
async def test_the_settings_route_takes_the_agents_page_lock(
    client: httpx.AsyncClient,
) -> None:
    """A FREEZE THAT RETURNS 200 MUST ACTUALLY BEAT THE AGENT.

    `_persist_update` holds the page lock across read-then-write: it reads
    `pipeline_updates`, decides, and only then rewrites the body -- a window
    that spans chunking and embedding, so seconds. A settings route outside
    that lock can return 200 inside the window and the page is rewritten
    anyway: a freeze the user was told succeeded, and did not.

    Asserted by holding the AGENT's key and showing the route BLOCKS on it. A
    route that ignored the lock would answer immediately, so the timeout is
    the assertion.

    The key is built here rather than imported from the route, for the reason
    `test_http_put_takes_the_page_lock` gives: importing `page_lock_key` would
    make the test agree with the route by construction, and a route that
    changed its namespace would still pass.
    """
    import asyncio

    from engine.shared.locks import advisory_lock_key

    await _put(client, "lock-me", "body", expected_version=0)
    agents_key = advisory_lock_key("page", CUSTOMER, "runbook:lock-me")

    async with with_tenant(CUSTOMER) as holder:
        await holder.execute("SELECT pg_advisory_xact_lock($1)", agents_key)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                client.put(
                    "/api/wiki/pages/runbook/lock-me/settings",
                    json={"pipeline_updates": False},
                    headers=_hdr(),
                ),
                timeout=1.5,
            )

    # Released: the same call now completes.
    resp = await client.put(
        "/api/wiki/pages/runbook/lock-me/settings",
        json={"pipeline_updates": False},
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_settings_on_a_missing_page_is_404(client: httpx.AsyncClient) -> None:
    """A typo'd slug must not silently succeed.

    Storing the setting for a page that does not exist would be harmless to
    the data and awful for the user: they would be told they froze a page,
    and the page they meant would keep updating.
    """
    resp = await client.put(
        "/api/wiki/pages/runbook/no-such-page/settings",
        json={"pipeline_updates": False},
        headers=_hdr(),
    )
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_deleting_a_page_forgets_its_setting(
    client: httpx.AsyncClient,
) -> None:
    """A recreated slug must not inherit a freeze decided about a page that no
    longer exists.

    `wiki_page_settings` has no FK to `documents` -- its PK includes `version`,
    so there is nothing to cascade from -- which means the row outlives the
    page unless the delete path clears it.
    """
    await _put(client, "recycled", "first life", expected_version=0)
    await client.put(
        "/api/wiki/pages/runbook/recycled/settings",
        json={"pipeline_updates": False},
        headers=_hdr(),
    )
    await client.delete("/api/wiki/pages/runbook/recycled", headers=_hdr())

    await _put(client, "recycled", "second life")
    page = await client.get("/api/wiki/pages/runbook/recycled", headers=_hdr())
    assert page.json()["pipeline_updates"] is True

    await _run_agent_update("recycled", "THE AGENT UPDATED THE RECREATED PAGE.")
    page = await client.get("/api/wiki/pages/runbook/recycled", headers=_hdr())
    assert page.json()["body"] == "THE AGENT UPDATED THE RECREATED PAGE."


@pytest.mark.asyncio
async def test_a_write_that_omits_the_setting_leaves_it_alone(
    client: httpx.AsyncClient,
) -> None:
    """A caller who has never heard of the setting cannot reset it.

    Every existing writer -- the dashboard BFF, `probe wiki write`, the
    research-os proxy -- sends a body with no `pipeline_updates` field. If the
    schema defaulted that to True, saving a typo fix on a frozen page would
    silently un-freeze it: the same class of bug as the trapdoor, pointing the
    other way.
    """
    await _put(client, "stays-frozen", "original", expected_version=0)
    await client.put(
        "/api/wiki/pages/runbook/stays-frozen/settings",
        json={"pipeline_updates": False},
        headers=_hdr(),
    )

    await _put(client, "stays-frozen", "edited by someone unaware of the setting")

    page = await client.get("/api/wiki/pages/runbook/stays-frozen", headers=_hdr())
    assert page.json()["pipeline_updates"] is False


@pytest.mark.asyncio
async def test_http_put_takes_the_page_lock(client: httpx.AsyncClient) -> None:
    """The route participates in the agent's per-page advisory lock.

    WHY THIS MATTERS BEYOND TIDINESS. The agent's manual-entry skip is a
    read-then-write: it re-reads the page, sees `doc_class=compiled_wiki`,
    and only then overwrites. Before this route took the lock, a PUT could
    land inside that window -- turning the page into a human-authored one
    the agent was already committed to destroying. The skip was real; it
    just was not atomic.

    Asserted by holding the agent's key and showing the route BLOCKS on it.
    A route that ignored the lock would answer immediately, so the timeout
    is the assertion.
    """
    import asyncio

    from engine.shared.locks import advisory_lock_key

    # THE KEY IS BUILT HERE, NOT IMPORTED FROM THE ROUTE. Calling the
    # route's own `page_lock_key` would make this test agree with the route
    # by construction: change the route's namespace to "httproute" and the
    # test would hold the SAME wrong key, block correctly, and pass while
    # the route no longer shares a lock with the agent at all. (Verified --
    # that mutation survived an earlier version of this test.)
    #
    # So the expression below is a deliberate literal copy of the one in
    # `WikiAgentRuntime._persist_update`. It is the AGENT's key, and the
    # test's claim is that the route blocks on it.
    key = advisory_lock_key("page", CUSTOMER, "runbook:contended")
    released = asyncio.Event()

    async def hold_the_lock() -> None:
        async with with_tenant(CUSTOMER) as conn:
            await conn.execute("SELECT pg_advisory_xact_lock($1)", key)
            await released.wait()

    holder = asyncio.create_task(hold_the_lock())
    put: asyncio.Task[httpx.Response] | None = None
    try:
        await asyncio.sleep(0.3)  # let the holder acquire

        put = asyncio.create_task(_put(client, "contended", "blocked?", expected_version=0))
        await asyncio.sleep(0.6)
        blocked = not put.done()
    finally:
        # RELEASE IN `finally`, ALWAYS. The holder sits in an open
        # transaction on a real connection; leaking it when the assertion
        # below fails would leave the fixture's TRUNCATE waiting on that
        # transaction forever. A test whose FAILURE mode is a hung suite
        # gets skipped rather than fixed -- and this is exactly the test a
        # mutation run is expected to fail.
        released.set()
        await holder

    assert blocked, (
        "the PUT completed while the agent's page lock was held -- the route "
        "is not participating in the lock, so the agent's manual-entry skip "
        "is decided in a window this write can slip through"
    )
    resp = await asyncio.wait_for(put, timeout=60)
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_a_page_kind_outside_the_closed_set_is_refused(
    client: httpx.AsyncClient,
) -> None:
    """`wiki_type` is a closed set, and the page routes are where that bites.

    It used to be free-form, with the agent explicitly told it "may invent new
    types if the corpus calls for it". The type is a path segment and a doc_id
    component with no rename route, so an invented one is a permanent page
    kind -- `repo` / `repository` / `codebase` become three sections of the
    same wiki that nothing can merge.

    `company` is checked because it reads as plausible -- it was named in the
    old prompt's list of suggestions -- and that is exactly the kind of value
    that used to slip through. It is deliberately NOT `feature`, which this
    test used until `feature` became a real page kind; a test pinned to a value
    that later gets added stops testing anything the day it is added.
    """
    resp = await client.put(
        "/api/wiki/pages/company/acme",
        json={"title": "Auth", "body": "OAuth across all sources."},
        headers=_hdr(),
    )
    assert resp.status_code == 400, resp.text


# ---------------------------------------------------------------------------
# Tenant-level generation setting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_is_off_until_someone_turns_it_on(
    client: httpx.AsyncClient,
) -> None:
    """A tenant that has never touched the setting reads as OFF.

    Matching every consumer: the queue drains and the nightly trigger compare
    `preferences->>'wiki_generation_enabled'` to the string `'true'`, so an
    absent key is off. A route that reported `null` or errored on the common
    case would make "is the wiki on" unanswerable for exactly the tenants that
    have never used it.
    """
    resp = await client.get("/api/wiki/settings", headers=_hdr())

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "customer_id": CUSTOMER,
        "generation_enabled": False,
        # GET never schedules a seed; the field only flips on the PUT
        # that transitions off→on.
        "catchup_started": False,
    }


@pytest.mark.asyncio
async def test_generation_can_be_turned_on_and_back_off(
    client: httpx.AsyncClient,
) -> None:
    """Both directions, and the stored value is what the PIPELINE reads.

    Asserted against the raw `preferences->>` text, not just the response: the
    route could return `true` while writing a JSON boolean, a nested object, or
    the wrong key, and every one of those reads back correctly through its own
    response while the nightly trigger sees nothing.
    """
    on = await client.put("/api/wiki/settings", json={"generation_enabled": True}, headers=_hdr())
    assert on.status_code == 200, on.text
    assert on.json()["generation_enabled"] is True

    async with raw_conn() as conn:
        stored = await conn.fetchval(
            "SELECT preferences->>'wiki_generation_enabled' FROM customers WHERE customer_id = $1",
            CUSTOMER,
        )
    assert stored == "true", "the pipeline compares this to the STRING 'true'"

    off = await client.put("/api/wiki/settings", json={"generation_enabled": False}, headers=_hdr())
    assert off.status_code == 200, off.text
    assert off.json()["generation_enabled"] is False
    assert (await client.get("/api/wiki/settings", headers=_hdr())).json()[
        "generation_enabled"
    ] is False


@pytest.mark.asyncio
async def test_turning_generation_on_preserves_other_preferences(
    client: httpx.AsyncClient,
) -> None:
    """The write must not clobber the rest of the JSONB column.

    `preferences` is shared. A naive `SET preferences = '{"wiki...": "true"}'`
    reads back perfectly through this route while silently discarding every
    other setting the tenant had -- which is the kind of loss nobody attributes
    to the wiki weeks later.
    """
    async with raw_conn() as conn:
        await conn.execute(
            'UPDATE customers SET preferences = \'{"keep_me": "yes"}\'::jsonb '
            "WHERE customer_id = $1",
            CUSTOMER,
        )

    await client.put("/api/wiki/settings", json={"generation_enabled": True}, headers=_hdr())

    async with raw_conn() as conn:
        kept = await conn.fetchval(
            "SELECT preferences->>'keep_me' FROM customers WHERE customer_id = $1",
            CUSTOMER,
        )
    assert kept == "yes"


@pytest.mark.asyncio
async def test_turning_generation_on_works_from_the_default_preferences(
    client: httpx.AsyncClient,
) -> None:
    """The actual common case: `preferences` at its `'{}'` default.

    A tenant that has never set anything has an EMPTY OBJECT, not NULL --
    `customers.preferences` is NOT NULL DEFAULT '{}'. That is why the write
    passes `create_missing`: `jsonb_set` without it leaves an absent key
    absent, reports success, and the operator then waits for a nightly run that
    will never include them.
    """
    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE customers SET preferences = '{}'::jsonb WHERE customer_id = $1",
            CUSTOMER,
        )

    resp = await client.put("/api/wiki/settings", json={"generation_enabled": True}, headers=_hdr())

    assert resp.status_code == 200, resp.text
    assert resp.json()["generation_enabled"] is True
    async with raw_conn() as conn:
        stored = await conn.fetchval(
            "SELECT preferences->>'wiki_generation_enabled' FROM customers WHERE customer_id = $1",
            CUSTOMER,
        )
    assert stored == "true"


@pytest.mark.asyncio
async def test_the_settings_routes_require_the_internal_key(
    client: httpx.AsyncClient,
) -> None:
    """Same trust boundary as every other route here: the key, then the tenant
    header. Without this a read of one team's setting is a header away."""
    for call in (
        client.get("/api/wiki/settings", headers={"X-Prbe-Customer": CUSTOMER}),
        client.put(
            "/api/wiki/settings",
            json={"generation_enabled": True},
            headers={"X-Prbe-Customer": CUSTOMER},
        ),
    ):
        assert (await call).status_code == 401


# ---------------------------------------------------------------------------
# POST /pages/{wiki_type}/{slug}/append
#
# Deliberately lean. A test earns its place here only where the failure is
# SILENT or unrecoverable -- a dropped paragraph, a duplicated decision, a
# leaked write past a reservation. Everything that fails loudly the first time
# anyone runs the route (empty text, a missing key, a bad slug) is covered by
# the shared validators and is not re-tested per route.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_creates_the_page_when_it_does_not_exist(
    client: httpx.AsyncClient,
) -> None:
    """An agent logging a decision should not have to check existence first.

    That check is the two-step this route exists to remove, and an agent that
    has to perform it will race anyway.
    """
    resp = await client.post(
        "/api/wiki/pages/runbook/dockq-scoring/append",
        json={"text": "Chose DockQ over TM-score for interface quality."},
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["created"] is True
    assert payload["replayed"] is False

    page = await client.get("/api/wiki/pages/runbook/dockq-scoring", headers=_hdr())
    assert "DockQ over TM-score" in page.json()["body"]


@pytest.mark.asyncio
async def test_two_concurrent_appends_both_survive(client: httpx.AsyncClient) -> None:
    """THE feature. A plain read-modify-write is last-one-wins, so two agents
    logging at the same moment silently lose one paragraph.

    Both are fired without awaiting in between so they contend for the page
    lock rather than running in sequence.
    """
    await client.post(
        "/api/wiki/pages/runbook/concurrent/append",
        json={"text": "seed"},
        headers=_hdr(),
    )

    first, second = await asyncio.gather(
        client.post(
            "/api/wiki/pages/runbook/concurrent/append",
            json={"text": "first agent"},
            headers=_hdr(),
        ),
        client.post(
            "/api/wiki/pages/runbook/concurrent/append",
            json={"text": "second agent"},
            headers=_hdr(),
        ),
    )
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    body = (await client.get("/api/wiki/pages/runbook/concurrent", headers=_hdr())).json()["body"]
    assert "first agent" in body
    assert "second agent" in body


@pytest.mark.asyncio
async def test_replaying_an_idempotency_key_appends_nothing(
    client: httpx.AsyncClient,
) -> None:
    """A timeout is exactly the case where the caller cannot know whether the
    write landed, and retrying is what an agent does next. Without this the
    retry duplicates the paragraph and one decision reads as two.

    The replay must also report the version the FIRST call produced, not the
    page's current version -- a retry should get the answer it was retrying
    for.
    """
    payload = {"text": "only once", "idempotency_key": "retry-me"}
    first = await client.post("/api/wiki/pages/runbook/idem/append", json=payload, headers=_hdr())
    assert first.status_code == 200, first.text

    second = await client.post("/api/wiki/pages/runbook/idem/append", json=payload, headers=_hdr())
    assert second.status_code == 200, second.text
    assert second.json()["replayed"] is True
    assert second.json()["version"] == first.json()["version"]

    body = (await client.get("/api/wiki/pages/runbook/idem", headers=_hdr())).json()["body"]
    assert body.count("only once") == 1


@pytest.mark.asyncio
async def test_append_rejects_index_wiki_type(client: httpx.AsyncClient) -> None:
    """The third write onto the reserved type, alongside PUT and revert.

    Worth its own test for the reason the revert one gives: the read validator
    deliberately lifts this reservation, and the writes must not drift into it.
    """
    resp = await client.post(
        "/api/wiki/pages/index/contents/append",
        json={"text": "sneaking a line onto the generated index"},
        headers=_hdr(),
    )
    assert resp.status_code == 400, resp.text
    assert "reserved" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_append_past_the_page_cap_refuses_with_413(
    client: httpx.AsyncClient,
) -> None:
    """A page that is appended to unattended eventually fills up.

    The refusal has to be a 413 that names the cap, not `ck_wiki_live_page_size`
    surfacing as a 500 -- the caller is an agent deciding what to do next, and
    a constraint violation tells it nothing actionable.
    """
    filler = "x" * 4000
    for _ in range(2):
        await client.post(
            "/api/wiki/pages/runbook/full/append",
            json={"text": filler},
            headers=_hdr(),
        )

    resp = await client.post(
        "/api/wiki/pages/runbook/full/append",
        json={"text": filler},
        headers=_hdr(),
    )
    assert resp.status_code == 413, resp.text
    assert "cap" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_append_works_on_a_page_frozen_against_the_pipeline(
    client: httpx.AsyncClient,
) -> None:
    """`pipeline_updates=false` freezes a page against the nightly AGENT, not
    against people. A deliberate write is the thing that setting protects, so
    blocking it here would invert the feature."""
    await client.put(
        "/api/wiki/pages/runbook/frozen",
        json={"title": "Frozen", "body": "original"},
        headers=_hdr(),
    )
    await client.put(
        "/api/wiki/pages/runbook/frozen/settings",
        json={"pipeline_updates": False},
        headers=_hdr(),
    )

    resp = await client.post(
        "/api/wiki/pages/runbook/frozen/append",
        json={"text": "a human still gets to write here"},
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    body = (await client.get("/api/wiki/pages/runbook/frozen", headers=_hdr())).json()["body"]
    assert "original" in body
    assert "a human still gets to write here" in body


@pytest.mark.asyncio
async def test_append_after_a_delete_starts_a_new_page(client: httpx.AsyncClient) -> None:
    """The serialization the handler documents, pinned.

    An append is not version-checked, so its behaviour against the OTHER
    writers has to be stated somewhere that fails when it changes. A delete
    that committed first must be respected: the append starts a fresh page
    rather than resurrecting the old body, and reports `created`. Silently
    restoring deleted content would make delete unreliable, and appending into
    a tombstone would lose the paragraph.
    """
    await client.put(
        "/api/wiki/pages/runbook/deleted-then-appended",
        json={"title": "Gone", "body": "secret that was deleted on purpose"},
        headers=_hdr(),
    )
    assert (
        await client.delete("/api/wiki/pages/runbook/deleted-then-appended", headers=_hdr())
    ).status_code == 200

    resp = await client.post(
        "/api/wiki/pages/runbook/deleted-then-appended/append",
        json={"text": "written after the delete"},
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["created"] is True

    body = (
        await client.get("/api/wiki/pages/runbook/deleted-then-appended", headers=_hdr())
    ).json()["body"]
    assert "written after the delete" in body
    assert "secret that was deleted on purpose" not in body


@pytest.mark.asyncio
async def test_references_publishes_what_the_engine_parsed(
    client: httpx.AsyncClient,
) -> None:
    """The engine is the only thing that understands `[[artifact:x]]`, and this
    route is how a consumer gets those facts without writing a second parser.

    A page's own wiki-to-wiki links must NOT appear here: the consumer wants
    research references, and mixing the two would make "which projects
    reference this one" count runbooks as projects.
    """
    await client.put(
        "/api/wiki/pages/runbook/dockq-method",
        json={
            "title": "DockQ method",
            "body": (
                "Score with [[artifact: dockq-scorer]] as run in "
                "[[experiment: fold-baselines]]. See also [[Person: mahit]]."
            ),
        },
        headers=_hdr(),
    )

    resp = await client.get("/api/wiki/references", headers=_hdr())
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]

    by_kind = {(i["kind"], i["name"]) for i in items}
    assert ("artifact", "dockq-scorer") in by_kind
    assert ("experiment", "fold-baselines") in by_kind
    # The person reference is not a research reference.
    assert not any(i["kind"] == "person" for i in items)

    one = next(i for i in items if i["name"] == "dockq-scorer")
    assert one["src_wiki_type"] == "runbook"
    assert one["src_slug"] == "dockq-method"
    assert one["canonical_id"] == "probe:artifact:dockq-scorer"


@pytest.mark.asyncio
async def test_references_can_be_filtered_to_one_kind(
    client: httpx.AsyncClient,
) -> None:
    await client.put(
        "/api/wiki/pages/runbook/mixed",
        json={
            "title": "Mixed",
            "body": "[[artifact: scorer]] and [[experiment: exp-one]].",
        },
        headers=_hdr(),
    )
    resp = await client.get("/api/wiki/references?kind=experiment", headers=_hdr())
    assert resp.status_code == 200, resp.text
    kinds = {i["kind"] for i in resp.json()["items"]}
    assert kinds == {"experiment"}


@pytest.mark.asyncio
async def test_reusing_a_key_with_different_text_is_a_409_not_a_silent_drop(
    client: httpx.AsyncClient,
) -> None:
    """Same key, different paragraph, is not a retry.

    Replaying the first answer here would report success and discard the new
    text -- a silently dropped decision, which is the exact failure the whole
    route exists to end. It has to be loud.
    """
    first = await client.post(
        "/api/wiki/pages/runbook/keyreuse/append",
        json={"text": "the original decision", "idempotency_key": "same-key"},
        headers=_hdr(),
    )
    assert first.status_code == 200, first.text

    second = await client.post(
        "/api/wiki/pages/runbook/keyreuse/append",
        json={"text": "a DIFFERENT decision", "idempotency_key": "same-key"},
        headers=_hdr(),
    )
    assert second.status_code == 409, second.text
    assert "different text" in second.json()["detail"]

    body = (await client.get("/api/wiki/pages/runbook/keyreuse", headers=_hdr())).json()["body"]
    assert "the original decision" in body
    assert "a DIFFERENT decision" not in body


# Settings toggle → catchup seed (the off→on transition)
# ---------------------------------------------------------------------------


async def _insert_live_doc(doc_id: str) -> None:
    await insert_document(CUSTOMER, doc_id)


async def _pending_docs() -> set[str]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT doc_id FROM wiki_synthesis_queue WHERE customer_id = $1 AND status = 'pending'",
            CUSTOMER,
        )
    return {r["doc_id"] for r in rows}


@pytest.mark.asyncio
async def test_turning_generation_on_seeds_the_existing_corpus(
    client: httpx.AsyncClient,
) -> None:
    """The off→on PUT schedules a seed of docs ingested BEFORE the flip.

    This is the product half of the retroactive-seed fix: without it a
    tenant who enables the wiki starts from "documents ingested after
    today" and their backfilled history never becomes wiki input.
    ASGITransport awaits Starlette's background tasks before returning,
    so the seed has run by the time the client call resolves.
    """
    await _insert_live_doc("doc:before-flip")

    resp = await client.put("/api/wiki/settings", json={"generation_enabled": True}, headers=_hdr())
    assert resp.status_code == 200, resp.text
    assert resp.json()["catchup_started"] is True
    assert await _pending_docs() == {"doc:before-flip"}


@pytest.mark.asyncio
async def test_on_to_on_put_does_not_reschedule_the_seed(
    client: httpx.AsyncClient,
) -> None:
    """Only the TRANSITION seeds — a repeated on-PUT is a plain flag write."""
    first = await client.put(
        "/api/wiki/settings", json={"generation_enabled": True}, headers=_hdr()
    )
    assert first.json()["catchup_started"] is True

    again = await client.put(
        "/api/wiki/settings", json={"generation_enabled": True}, headers=_hdr()
    )
    assert again.status_code == 200
    assert again.json()["catchup_started"] is False


@pytest.mark.asyncio
async def test_turning_generation_off_never_seeds(
    client: httpx.AsyncClient,
) -> None:
    await _insert_live_doc("doc:whatever")
    resp = await client.put(
        "/api/wiki/settings", json={"generation_enabled": False}, headers=_hdr()
    )
    assert resp.status_code == 200
    assert resp.json()["catchup_started"] is False
    assert await _pending_docs() == set()


@pytest.mark.asyncio
async def test_seed_failure_does_not_break_the_put(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The background seed is best-effort: the flag write already
    committed and the response already says 200, so a seeding crash must
    be logged and swallowed — the nightly reconcile is the retry."""

    await _insert_live_doc("doc:would-seed")

    async def exploding_seed(conn, customer_id, *, limit=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(persistence, "seed_missing_docs", exploding_seed)

    resp = await client.put("/api/wiki/settings", json={"generation_enabled": True}, headers=_hdr())
    assert resp.status_code == 200, resp.text
    assert resp.json()["catchup_started"] is True

    async with raw_conn() as conn:
        stored = await conn.fetchval(
            "SELECT preferences->>'wiki_generation_enabled' FROM customers WHERE customer_id = $1",
            CUSTOMER,
        )
    assert stored == "true"
    assert await _pending_docs() == set()


@pytest.mark.asyncio
async def test_reenabling_wakes_the_worker_even_with_nothing_new_to_seed(
    client: httpx.AsyncClient, settings: Settings
) -> None:
    """off→on with PENDING rows from before the flip-off must still fire
    the wake: inserted == 0 there, and gating the notify on it left the
    old rows waiting for the 30-minute periodic cycle."""
    import asyncio

    import asyncpg

    from engine.shared.constants import WIKI_PENDING_CHANNEL

    await _insert_live_doc("doc:old-pending")
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO wiki_synthesis_queue "
            "(customer_id, doc_id, doc_version, source_system, doc_type, status) "
            "VALUES ($1, 'doc:old-pending', 1, 'slack', 'slack.message', 'pending')",
            CUSTOMER,
        )

    received: list[str] = []
    notify_event = asyncio.Event()
    listener = await asyncpg.connect(settings.database_url)
    try:

        def _on_notify(_conn, _pid, channel, payload) -> None:
            received.append(payload)
            notify_event.set()

        await listener.add_listener(WIKI_PENDING_CHANNEL, _on_notify)

        resp = await client.put(
            "/api/wiki/settings", json={"generation_enabled": True}, headers=_hdr()
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["catchup_started"] is True

        await asyncio.wait_for(notify_event.wait(), timeout=5)
    finally:
        await listener.close()

    assert CUSTOMER in received


# ---------------------------------------------------------------------------
# Backfill trigger → queue reseed + preview (PR: rebuild recovers all sources)
# ---------------------------------------------------------------------------


async def _terminal_queue_row(doc_id: str, status: str = "done") -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO wiki_synthesis_queue
                (customer_id, doc_id, doc_version, source_system, doc_type,
                 status, triage_score, attempts)
            VALUES ($1, $2, 1, 'slack', 'slack.message', $3, 0.9, 2)
            """,
            CUSTOMER,
            doc_id,
            status,
        )


async def _compiled_page(doc_id: str) -> None:
    async with with_tenant(CUSTOMER) as conn:
        await conn.execute(
            """
            INSERT INTO documents
                (doc_id, version, customer_id, source_system, source_id,
                 source_url, doc_class, doc_type, content_hash, created_at,
                 updated_at, valid_from, acl)
            VALUES ($1, 1, $2, 'wiki', $1, '/wiki/x', 'compiled_wiki',
                    'wiki.page', $3, NOW(), NOW(), NOW(), '{}'::jsonb)
            """,
            doc_id,
            CUSTOMER,
            f"hash-{doc_id}",
        )


async def _queue_state() -> dict[str, str]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT doc_id, status FROM wiki_synthesis_queue WHERE customer_id = $1",
            CUSTOMER,
        )
    return {r["doc_id"]: r["status"] for r in rows}


@pytest.mark.asyncio
async def test_backfill_trigger_reseeds_the_daily_pipeline(
    client: httpx.AsyncClient,
) -> None:
    """The wipe's recovery path: crawlers only cover registered sources,
    so the trigger must seed missing docs and reset terminal rows in the
    same transaction — otherwise transcript-derived pages never return."""
    await _insert_live_doc("doc:unqueued")
    await _insert_live_doc("doc:done")
    await _terminal_queue_row("doc:done", status="done")
    await _compiled_page("wiki:page:old")

    resp = await client.post("/api/wiki/backfill/trigger", json={}, headers=_hdr())
    assert resp.status_code == 202, resp.text
    data = resp.json()
    assert data["eligible_documents"] == 2
    assert data["seeded"] == 1
    assert data["reset"] == 1
    assert data["run_ids"]

    assert await _queue_state() == {
        "doc:unqueued": "pending",
        "doc:done": "pending",
    }
    # And the wipe still happened -- as a RETIRE, not a delete.
    #
    # The distinction is the point. To every reader the wiki is just as empty:
    # each page read path filters `valid_to IS NULL`, so a retired page is as
    # gone as a deleted one. What survives is the version history that
    # `GET /v1/wiki/pages/{type}/{slug}/versions` and `POST .../revert` are
    # built on, and that `_restore_retired_pages` needs to undo a rebuild.
    async with raw_conn() as conn:
        live = await conn.fetchval(
            "SELECT count(*) FROM documents "
            "WHERE customer_id = $1 AND doc_class = 'compiled_wiki' "
            "AND valid_to IS NULL",
            CUSTOMER,
        )
        history = await conn.fetchval(
            "SELECT count(*) FROM documents WHERE customer_id = $1 AND doc_class = 'compiled_wiki'",
            CUSTOMER,
        )
    assert live == 0, "the wiki must read as empty after a wipe"
    assert history > 0, (
        "retired versions must survive -- deleting them is what made a failed "
        "rebuild unrecoverable and broke page history"
    )


@pytest.mark.asyncio
async def test_backfill_trigger_without_wipe_seeds_but_never_resets(
    client: httpx.AsyncClient,
) -> None:
    """wipe_first=false keeps existing pages, so terminal rows stay
    terminal — re-deriving them would only rewrite pages that still
    exist. Seeding gaps is still correct and harmless."""
    await _insert_live_doc("doc:unqueued")
    await _insert_live_doc("doc:done")
    await _terminal_queue_row("doc:done", status="done")

    resp = await client.post(
        "/api/wiki/backfill/trigger",
        json={"wipe_first": False},
        headers=_hdr(),
    )
    assert resp.status_code == 202, resp.text
    assert resp.json()["seeded"] == 1
    assert resp.json()["reset"] == 0
    assert (await _queue_state())["doc:done"] == "done"


@pytest.mark.asyncio
async def test_backfill_trigger_failure_after_wipe_rolls_everything_back(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wipe + reseed + run-insert live in ONE transaction: a failure
    after the deletes must leave the wiki exactly as it was, never a
    wiped tenant with no recovery rows."""
    import kb.wiki_routes as wiki_routes_module

    await _insert_live_doc("doc:a")
    await _compiled_page("wiki:page:keep")

    async def exploding_insert(conn, *, customer_id, sources):
        raise RuntimeError("boom after wipe")

    monkeypatch.setattr(wiki_routes_module, "_insert_pending_runs", exploding_insert)

    with pytest.raises(RuntimeError, match="boom after wipe"):
        await client.post("/api/wiki/backfill/trigger", json={}, headers=_hdr())

    # Nothing committed: page survives, queue still empty, no run rows.
    async with raw_conn() as conn:
        pages = await conn.fetchval(
            "SELECT count(*) FROM documents WHERE customer_id = $1 AND doc_class = 'compiled_wiki'",
            CUSTOMER,
        )
        runs = await conn.fetchval(
            "SELECT count(*) FROM wiki_synthesis_runs WHERE customer_id = $1",
            CUSTOMER,
        )
    assert pages == 1
    assert runs == 0
    assert await _queue_state() == {}


@pytest.mark.asyncio
async def test_backfill_preview_reports_counts_and_writes_nothing(
    client: httpx.AsyncClient,
) -> None:
    await _insert_live_doc("doc:unqueued")
    await _insert_live_doc("doc:done")
    await _terminal_queue_row("doc:done", status="done")
    await _compiled_page("wiki:page:old")

    resp = await client.get("/api/wiki/backfill/preview", headers=_hdr())
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "customer_id": CUSTOMER,
        "eligible_documents": 2,
        "would_seed": 1,
        "would_reset": 1,
        "compiled_pages": 1,
    }

    # Preview is read-only: queue unchanged, page still there, no runs.
    assert await _queue_state() == {"doc:done": "done"}
    async with raw_conn() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM wiki_synthesis_runs WHERE customer_id = $1",
                CUSTOMER,
            )
            == 0
        )


@pytest.mark.asyncio
async def test_backfill_preview_requires_internal_key(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.get("/api/wiki/backfill/preview", headers={"X-Prbe-Customer": CUSTOMER})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Undo: the recovery path that retiring (rather than deleting) makes possible.
# ---------------------------------------------------------------------------


async def _live_pages() -> set[str]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT doc_id FROM documents "
            "WHERE customer_id = $1 AND doc_class = 'compiled_wiki' "
            "AND valid_to IS NULL",
            CUSTOMER,
        )
    return {r["doc_id"] for r in rows}


@pytest.mark.asyncio
async def test_undo_restores_pages_the_rebuild_never_rederived(
    client: httpx.AsyncClient, _stub_bootstrap_registry: None
) -> None:
    """The case the whole retire-instead-of-delete change exists for.

    A rebuild retires every page and then re-derives them from the queue. But
    re-derivation is not guaranteed to give back what was there: a replayed row
    re-enters at TRIAGE, which is a scoring gate that legitimately rejects. So
    a rebuild can genuinely end with fewer pages than it started with. When the
    wipe deleted, those pages were gone for good and the only 'recovery' was to
    run the rebuild again and hope.
    """
    await _compiled_page("wiki:runbook:alpha")
    await _compiled_page("wiki:runbook:beta")
    assert await _live_pages() == {"wiki:runbook:alpha", "wiki:runbook:beta"}

    resp = await client.post(
        "/api/wiki/bootstrap/trigger", json={"sources": ["github"]}, headers=_hdr()
    )
    assert resp.status_code == 202, resp.text
    assert await _live_pages() == set(), "the wipe must leave the wiki reading empty"

    undo = await client.post("/api/wiki/backfill/undo", headers=_hdr())
    assert undo.status_code == 200, undo.text
    body = undo.json()
    assert body["undone"] is True
    assert body["restored_pages"] == 2
    assert await _live_pages() == {"wiki:runbook:alpha", "wiki:runbook:beta"}


@pytest.mark.asyncio
async def test_undo_leaves_pages_the_rebuild_already_rebuilt(
    client: httpx.AsyncClient, _stub_bootstrap_registry: None
) -> None:
    """Undo must not resurrect a predecessor underneath a rebuilt page.

    Two live versions of one doc would break every `valid_to IS NULL` read in
    the engine -- which is most of them -- and the page would then render
    whichever row the planner happened to return first.
    """
    await _compiled_page("wiki:runbook:alpha")
    resp = await client.post(
        "/api/wiki/bootstrap/trigger", json={"sources": ["github"]}, headers=_hdr()
    )
    assert resp.status_code == 202, resp.text

    # Stand in for the rebuild re-deriving the page: a fresh live version.
    async with with_tenant(CUSTOMER) as conn:
        await conn.execute(
            """
            INSERT INTO documents
                (doc_id, version, customer_id, source_system, source_id,
                 source_url, doc_class, doc_type, content_hash, created_at,
                 updated_at, valid_from, acl)
            VALUES ($1, 2, $2, 'wiki', $1, '/wiki/x', 'compiled_wiki',
                    'wiki.page', 'hash-rebuilt', NOW(), NOW(), NOW(), '{}'::jsonb)
            """,
            "wiki:runbook:alpha",
            CUSTOMER,
        )

    undo = await client.post("/api/wiki/backfill/undo", headers=_hdr())
    assert undo.status_code == 200, undo.text
    assert undo.json()["restored_pages"] == 0

    async with raw_conn() as conn:
        live = await conn.fetch(
            "SELECT version FROM documents "
            "WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL",
            CUSTOMER,
            "wiki:runbook:alpha",
        )
    assert [r["version"] for r in live] == [2], "exactly one live version, the rebuilt one"


@pytest.mark.asyncio
async def test_undo_cancels_the_runs_still_in_flight(
    client: httpx.AsyncClient, _stub_bootstrap_registry: None
) -> None:
    """Open runs are closed before the restore, not after.

    A crawler that committed a page between the restore and the response would
    leave the operator looking at a wiki they just asked to put back.
    """
    resp = await client.post(
        "/api/wiki/bootstrap/trigger", json={"sources": ["github"]}, headers=_hdr()
    )
    assert resp.status_code == 202, resp.text
    opened = resp.json()["run_ids"]

    undo = await client.post("/api/wiki/backfill/undo", headers=_hdr())
    assert undo.status_code == 200, undo.text
    assert sorted(undo.json()["cancelled_run_ids"]) == sorted(opened)

    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT status, error FROM wiki_synthesis_runs "
            "WHERE customer_id = $1 AND kind = 'bootstrap'",
            CUSTOMER,
        )
    assert {r["status"] for r in rows} == {"cancelled"}
    assert all("undone by admin" in (r["error"] or "") for r in rows)


@pytest.mark.asyncio
async def test_undo_with_no_rebuild_is_a_no_op_not_an_error(
    client: httpx.AsyncClient,
) -> None:
    """Idempotent, so a double-click and a stale dashboard both behave."""
    await _compiled_page("wiki:runbook:alpha")

    undo = await client.post("/api/wiki/backfill/undo", headers=_hdr())
    assert undo.status_code == 200, undo.text
    assert undo.json()["restored_pages"] == 0
    assert await _live_pages() == {"wiki:runbook:alpha"}


@pytest.mark.asyncio
async def test_undo_reports_when_there_was_nothing_to_undo(
    client: httpx.AsyncClient,
) -> None:
    """`undone` distinguishes "I stopped it" from "there was nothing to stop".

    It returned True unconditionally at first, which made those two answers
    identical -- and they are the one distinction a caller reaching for undo
    actually wants. Caught by smoking the deployed endpoint, where a no-op
    against a quiet tenant still answered `undone: true`.
    """
    undo = await client.post("/api/wiki/backfill/undo", headers=_hdr())
    assert undo.status_code == 200, undo.text
    body = undo.json()
    assert body["undone"] is False
    assert body["restored_pages"] == 0
    assert body["cancelled_run_ids"] == []
