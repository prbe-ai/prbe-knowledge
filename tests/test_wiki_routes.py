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
    from services.ingestion import wiki_routes as _wr

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

    from shared.config import get_settings as _get_settings
    from shared.constants import WIKI_BACKFILL_CHANNEL

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
    from services.ingestion import wiki_routes as _wr

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

    from services.ingestion import wiki_routes as _wr
    from shared.config import get_settings as _get_settings
    from shared.constants import WIKI_BACKFILL_CANCEL_CHANNEL

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
