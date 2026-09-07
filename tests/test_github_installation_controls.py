"""Real DB + native connector/normalizer tests for installation control v2.

Provider calls and embedding vectors are fake; HTTP handlers, leases, RLS,
queue admission, chunk/document SQL and cancellation barriers are real.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException
from pydantic import SecretStr

from engine.ingest.handlers.base import ConnectorContext
from engine.shared.config import get_settings
from engine.shared.constants import EMBEDDING_V2_DIM
from engine.shared.db import raw_conn, with_tenant
from engine.shared.embeddings import EmbeddedChunk, EmbedResult, FailedChunk
from kb.github_control import (
    admit_projection,
    cancel_job,
    create_job,
    enqueue_live,
    get_installation,
    normalize_scope,
)
from kb.github_control_purge import purge
from kb.github_control_routes import router
from kb.github_control_worker import GitHubControlWorker

TENANT = "github-control-test"
SCOPE = [{"external_id": "prbe/payments", "label": "prbe/payments"}]
HEADERS = {"X-Internal-Knowledge-Key": "test-internal-key", "X-Prbe-Customer": TENANT}
PREFIX = "/api/github/installations/101"


@pytest_asyncio.fixture
async def connected(live_db, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "internal_knowledge_api_key", SecretStr("test-internal-key"))
    async with raw_conn() as conn:
        for tenant in (TENANT, "github-other-tenant"):
            await conn.execute(
                "INSERT INTO customers(customer_id,display_name,api_key_hash) VALUES ($1,$1,$2)",
                tenant,
                tenant,
            )
        await conn.execute("DELETE FROM github_worker_capabilities")
    async with with_tenant(TENANT) as conn:
        for installation in ("101", "202"):
            await conn.execute(
                """INSERT INTO github_installations(customer_id,installation_id,managed,scope,sync_enabled)
                VALUES ($1,$2,TRUE,$3::jsonb,TRUE)""",
                TENANT,
                installation,
                json.dumps(SCOPE),
            )
    return TENANT


@pytest_asyncio.fixture
async def api(connected, monkeypatch):
    app = FastAPI()
    app.include_router(router)
    provider = httpx.AsyncClient()
    app.state.ctx = ConnectorContext(settings=get_settings(), http=provider)

    async def repos(*args):
        return [{"full_name": "prbe/payments"}, {"full_name": "other/project"}]

    monkeypatch.setattr("kb.github_control_routes.list_repositories", repos)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers=HEADERS
    ) as client:
        yield client
    await provider.aclose()


def envelope(*, title="Original", updated="2026-09-01T12:00:00Z", installation="101"):
    payload = json.loads(
        (Path(__file__).parents[1] / "fixtures/github/issue_opened.json").read_text()
    )
    payload["installation"] = {"id": int(installation)}
    payload["issue"]["title"] = title
    payload["issue"]["body"] = title + " content"
    payload["issue"]["updated_at"] = updated
    return {"_headers": {"X-GitHub-Event": "issues"}, "payload": payload}


class FakeEmbedder:
    async def embed_documents(self, items):
        return EmbedResult(
            embedded=[
                EmbeddedChunk(chunk_index=i, embedding=[0.0] * EMBEDDING_V2_DIM)
                for i, _ in enumerate(items)
            ],
            failed=[],
        )


@pytest_asyncio.fixture
async def worker(connected):
    async with httpx.AsyncClient() as http:
        result = GitHubControlWorker(ConnectorContext(settings=get_settings(), http=http))
        result.normalizer._embedder = FakeEmbedder()
        yield result


async def new_job(installation="101", key="test-launch"):
    async with with_tenant(TENANT) as conn:
        row = await get_installation(conn, TENANT, installation, lock=True)
        return await create_job(conn, row, SCOPE, key)


async def test_scope_canonical_and_no_traversal():
    assert normalize_scope(SCOPE + SCOPE) == SCOPE
    with pytest.raises(HTTPException):
        normalize_scope([{"external_id": "../private"}])


async def test_real_capability_heartbeat_not_constant(api):
    assert (await api.get("/api/github/capabilities")).json()["worker_ready"] is False
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO github_worker_capabilities(worker_id,protocol_version) VALUES ('test',2)"
        )
    assert (await api.get("/api/github/capabilities")).json()["worker_ready"] is True


async def test_launch_idempotency_scope_snapshot_and_two_installations(api):
    body = {"scope": SCOPE, "idempotency_key": "double-click"}
    first = await api.post(PREFIX + "/backfills", json=body)
    second = await api.post(PREFIX + "/backfills", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["id"] == second.json()["id"]
    other = await api.post("/api/github/installations/202/backfills", json=body)
    assert other.json()["id"] != first.json()["id"]
    conflict = await api.post(
        PREFIX + "/backfills", json={**body, "scope": [{"external_id": "other/project"}]}
    )
    assert conflict.status_code == 409
    assert len((await api.get(PREFIX + "/backfills")).json()["jobs"]) == 1


async def test_pause_preserves_history_and_exact_revision(api):
    job = await new_job()
    response = await api.patch(
        PREFIX + "/sync", json={"sync_enabled": False, "expected_revision": 1}
    )
    assert response.status_code == 200, response.text
    assert response.json()["sync_enabled"] is False
    assert (await api.get(PREFIX + f"/backfills/{job['id']}")).json()["state"] == "queued"
    assert (
        await api.patch(PREFIX + "/sync", json={"sync_enabled": True, "expected_revision": 1})
    ).status_code == 409
    async with with_tenant(TENANT) as conn:
        assert (await get_installation(conn, TENANT, "202"))["sync_enabled"] is True


async def test_empty_scope_rejected_only_when_enabling(api):
    response = await api.patch(PREFIX + "/sync", json={"scope": [], "sync_enabled": True})
    assert response.status_code == 422
    assert (
        await api.patch(PREFIX + "/sync", json={"scope": [], "sync_enabled": False})
    ).status_code == 200


async def test_new_protocol_is_not_old_worker_claimable(connected):
    assert await enqueue_live(TENANT, "101", envelope(), "event-a")
    async with with_tenant(TENANT) as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM ingestion_queue WHERE customer_id=$1 AND status='pending'",
                TENANT,
            )
            == 0
        )
        row = await conn.fetchrow("SELECT * FROM ingestion_queue WHERE customer_id=$1", TENANT)
        assert row["payload_s3_keys"] == []
        assert row["github_installation_id"] == "101"


async def test_native_worker_persists_body_chunks_and_source_version(worker):
    assert await enqueue_live(
        TENANT, "101", envelope(title="Newer", updated="2026-09-02T12:00:00Z"), "new"
    )
    await worker.queue_step(TENANT, live=True)
    async with with_tenant(TENANT) as conn:
        q = await conn.fetchrow(
            "SELECT status,error FROM ingestion_queue WHERE customer_id=$1", TENANT
        )
        assert q["status"] == "v2_completed", dict(q)
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM chunks WHERE customer_id=$1 AND content LIKE '%Newer%'",
                TENANT,
            )
            > 0
        )
    assert await enqueue_live(TENANT, "101", envelope(title="Older"), "old")
    await worker.queue_step(TENANT, live=True)
    async with with_tenant(TENANT) as conn:
        assert (
            await conn.fetchval(
                "SELECT title FROM documents WHERE customer_id=$1 AND valid_to IS NULL", TENANT
            )
            == "Newer"
        )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM ingestion_queue WHERE github_payload IS NOT NULL AND customer_id=$1",
                TENANT,
            )
            == 0
        )


async def test_cancel_waits_for_write_fence_then_acknowledges(connected):
    job = await new_job()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def accepted_write():
        async with with_tenant(TENANT) as conn:
            await get_installation(conn, TENANT, "101", lock=True)
            entered.set()
            await release.wait()

    write = asyncio.create_task(accepted_write())
    await entered.wait()
    cancel = asyncio.create_task(cancel_job(TENANT, "101", job["id"]))
    await asyncio.sleep(0.03)
    assert not cancel.done()
    release.set()
    await write
    assert (await cancel)["state"] == "canceled"


async def test_cancel_rejects_late_leased_history_apply(connected):
    from kb.github_control import enqueue_event

    job = await new_job()
    lease = uuid4()
    async with with_tenant(TENANT) as conn:
        installation = await get_installation(conn, TENANT, "101", lock=True)
        assert await enqueue_event(
            conn, installation=installation, envelope=envelope(), source_event_id="history", job=job
        )
        queue_id = await conn.fetchval(
            """UPDATE ingestion_queue SET status='v2_processing',github_lease_id=$2
            WHERE customer_id=$1 RETURNING queue_id""",
            TENANT,
            lease,
        )
    await cancel_job(TENANT, "101", job["id"])
    async with with_tenant(TENANT) as conn:
        assert await admit_projection(conn, TENANT, queue_id, lease) is False
        assert (
            await conn.fetchval(
                "SELECT github_payload FROM ingestion_queue WHERE queue_id=$1", queue_id
            )
            is None
        )


async def test_purge_one_installation_preserves_other_and_rejects_late_apply(worker):
    await enqueue_live(TENANT, "101", envelope(), "shared-a")
    await worker.queue_step(TENANT, live=True)
    await enqueue_live(TENANT, "202", envelope(installation="202"), "shared-b")
    await worker.queue_step(TENANT, live=True)
    result = await purge(TENANT, "101")
    assert result["verified"] is True
    assert result["documents"] == 0  # B holds the same stable source document
    async with with_tenant(TENANT) as conn:
        assert (
            await conn.fetchval("SELECT count(*) FROM documents WHERE customer_id=$1", TENANT) > 0
        )
        assert (await get_installation(conn, TENANT, "202"))["active"] is True
    result = await purge(TENANT, "202")
    assert result["documents"] == 1
    async with with_tenant(TENANT) as conn:
        assert (
            await conn.fetchval("SELECT count(*) FROM documents WHERE customer_id=$1", TENANT) == 0
        )


async def test_final_attempt_crash_terminalizes_and_erases_envelope(worker):
    job = await new_job()
    async with with_tenant(TENANT) as conn:
        await conn.execute(
            """UPDATE github_backfill_jobs SET state='running',attempts=5,lease_id=$2,
            heartbeat_at=now()-interval '1 hour' WHERE id=$1""",
            job["id"],
            uuid4(),
        )
    await worker.reconcile(TENANT)
    async with with_tenant(TENANT) as conn:
        row = await conn.fetchrow("SELECT * FROM github_backfill_jobs WHERE id=$1", job["id"])
        assert row["state"] == "failed"
        assert row["finished_at"] is not None


async def test_tenant_http_isolation(api):
    job = await new_job()
    response = await api.get(
        PREFIX + f"/backfills/{job['id']}",
        headers={**HEADERS, "X-Prbe-Customer": "github-other-tenant"},
    )
    assert response.status_code == 404
    response = await api.get(
        PREFIX + "/sync", headers={**HEADERS, "X-Internal-Knowledge-Key": "wrong"}
    )
    assert response.status_code == 401


async def test_native_history_is_durable_and_waits_for_indexing(worker, monkeypatch):
    from kb.handlers import _github_graphql as gql

    job = await new_job()
    repo = envelope()["payload"]["repository"]

    async def repos(*args):
        return [repo]

    async def bearer(*args, **kwargs):
        return "fake-test-bearer"

    async def query(http, headers, query, variables, **kwargs):
        if query == gql.BACKFILL_ISSUES_QUERY:
            nodes = [
                {
                    "number": n,
                    "title": f"History {n}",
                    "body": f"Older content {n}",
                    "state": "OPEN",
                    "createdAt": "2026-09-01T00:00:00Z",
                    "updatedAt": "2026-09-01T00:00:00Z",
                    "author": {"login": "tester"},
                }
                for n in range(1, 4)
            ]
            return {"repository": {"issues": {"nodes": nodes, "pageInfo": {"hasNextPage": False}}}}
        key = "pullRequests" if query == gql.BACKFILL_PULLS_QUERY else "releases"
        return {"repository": {key: {"nodes": [], "pageInfo": {"hasNextPage": False}}}}

    monkeypatch.setattr("kb.github_control_worker.list_repositories", repos)
    monkeypatch.setattr(worker.connector, "_resolve_installation_bearer", bearer)
    monkeypatch.setattr(gql, "run_graphql", query)
    assert await worker.history_step(TENANT)
    async with with_tenant(TENANT) as conn:
        row = await conn.fetchrow("SELECT * FROM github_backfill_jobs WHERE id=$1", job["id"])
        assert row["enumeration_complete"] is True
        assert row["state"] == "running", dict(row)
        assert row["processed_count"] == 3
    # A new worker instance resumes durable queued items, not an HTTP task.
    resumed = GitHubControlWorker(worker.ctx)
    resumed.normalizer._embedder = FakeEmbedder()
    while await resumed.queue_step(TENANT, live=False):
        pass
    await resumed.reconcile(TENANT)
    async with with_tenant(TENANT) as conn:
        row = await conn.fetchrow("SELECT * FROM github_backfill_jobs WHERE id=$1", job["id"])
        assert row["state"] == "completed", dict(row)
        assert json.loads(row["counts"])["indexed_events"] == 3


async def test_retry_does_not_change_live_cursor_or_another_job(api):
    first = await new_job(key="first")
    second = await new_job(key="second")
    await cancel_job(TENANT, "101", first["id"])
    response = await api.post(PREFIX + f"/backfills/{first['id']}/retry")
    assert response.status_code == 200
    assert response.json()["state"] == "queued"
    repeated = await api.post(PREFIX + f"/backfills/{first['id']}/retry")
    assert repeated.json()["attempts"] == response.json()["attempts"]
    assert (await api.get(PREFIX + f"/backfills/{second['id']}")).json()["state"] == "queued"
    assert (await api.get(PREFIX + "/sync")).json()["generation"] == 1


async def test_pause_cancels_catchup_not_explicit_history(api):
    explicit = await new_job()
    assert (await api.patch(PREFIX + "/sync", json={"sync_enabled": True})).status_code == 200
    assert (await api.patch(PREFIX + "/sync", json={"sync_enabled": False})).status_code == 200
    async with with_tenant(TENANT) as conn:
        assert (
            await conn.fetchval(
                "SELECT state FROM github_backfill_jobs WHERE customer_id=$1 AND kind='catchup'",
                TENANT,
            )
            == "canceled"
        )
        assert (
            await conn.fetchval(
                "SELECT state FROM github_backfill_jobs WHERE id=$1", explicit["id"]
            )
            == "queued"
        )


async def test_selected_token_binding_is_sent_to_backend(monkeypatch):
    from engine.shared.backend_client import fetch_github_installation_token
    from engine.shared.config import Settings

    settings = Settings(
        backend_base_url="http://backend", internal_backend_api_key=SecretStr("test")
    )
    monkeypatch.setattr("engine.shared.backend_client.get_settings", lambda: settings)
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"token": "test-bearer", "expires_at": "2026-09-07T12:00:00Z"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await fetch_github_installation_token(http, customer_id=TENANT, installation_id="202")
    assert seen == [{"customer_id": TENANT, "installation_id": "202"}]


async def test_force_rls_blocks_unscoped_and_foreign_tenant_reads(connected):
    # SET ROLE exercises a real non-owner role even when the fixture's DB
    # admin user bypasses RLS; this would catch a missing FORCE/policy.
    async with raw_conn() as conn:
        await conn.execute(
            "DO $$ BEGIN CREATE ROLE github_control_reader NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$"
        )
        await conn.execute("GRANT USAGE ON SCHEMA public TO github_control_reader")
        await conn.execute(
            "GRANT SELECT ON github_installations,github_backfill_jobs,github_document_bindings TO github_control_reader"
        )
        async with conn.transaction():
            await conn.execute("SET LOCAL ROLE github_control_reader")
            await conn.execute(
                "SELECT set_config('app.current_customer_id','github-other-tenant',true)"
            )
            assert await conn.fetchval("SELECT count(*) FROM github_installations") == 0
            await conn.execute("SELECT set_config('app.current_customer_id',$1,true)", TENANT)
            assert await conn.fetchval("SELECT count(*) FROM github_installations") == 2


def test_migration_and_bootstrap_share_exact_schema():
    root = Path(__file__).parents[1]
    assert (root / "kb/github_control_schema.sql").read_text() in (
        root / "db/schema.sql"
    ).read_text()


async def test_legacy_work_must_drain_before_adoption(api):
    async with with_tenant(TENANT) as conn:
        await conn.execute(
            "UPDATE github_installations SET managed=FALSE WHERE customer_id=$1 AND installation_id='101'",
            TENANT,
        )
        await conn.execute(
            "INSERT INTO backfill_state(customer_id,source_system,status) VALUES ($1,'github','running')",
            TENANT,
        )
    response = await api.patch(PREFIX + "/sync", json={"scope": SCOPE, "sync_enabled": True})
    assert response.status_code == 409
    async with with_tenant(TENANT) as conn:
        assert (
            await conn.fetchval(
                "SELECT managed FROM github_installations WHERE customer_id=$1 AND installation_id='101'",
                TENANT,
            )
            is False
        )


async def test_legacy_enqueue_cannot_bypass_adopted_controls(connected):
    from engine.shared.constants import SourceSystem
    from kb.backfill_runner import enqueue_backfill
    from kb.github_control import enqueue_legacy_webhook

    inserted, discard_raw = await enqueue_legacy_webhook(
        TENANT, "101", envelope(), "late-legacy", "raw/test"
    )
    assert inserted and discard_raw
    async with with_tenant(TENANT) as conn:
        assert (
            await conn.fetchval("SELECT status FROM ingestion_queue WHERE customer_id=$1", TENANT)
            == "v2_pending"
        )
    with pytest.raises(ValueError, match="installation-scoped"):
        await enqueue_backfill(TENANT, SourceSystem.GITHUB)
    await purge(TENANT, "101")
    assert await enqueue_legacy_webhook(
        TENANT, "101", envelope(), "late-after-purge", "raw/test"
    ) == (False, True)


async def test_partial_provider_page_is_not_success(worker, monkeypatch):
    from kb.handlers import _github_graphql as gql

    job = await new_job()

    async def repos(*args):
        return [envelope()["payload"]["repository"]]

    async def bearer(*args, **kwargs):
        return "test"

    async def unavailable(*args, **kwargs):
        return None

    monkeypatch.setattr("kb.github_control_worker.list_repositories", repos)
    monkeypatch.setattr(worker.connector, "_resolve_installation_bearer", bearer)
    monkeypatch.setattr(gql, "run_graphql", unavailable)
    await worker.history_step(TENANT)
    async with with_tenant(TENANT) as conn:
        row = await conn.fetchrow(
            "SELECT state,enumeration_complete FROM github_backfill_jobs WHERE id=$1", job["id"]
        )
        assert row["state"] == "failed"
        assert row["enumeration_complete"] is False


async def test_real_webhook_never_writes_r2_when_managed_or_paused(connected, monkeypatch):
    from types import SimpleNamespace

    from kb.ingestion_app import app

    class NoObjectStore:
        def __getattr__(self, name):
            raise AssertionError("Managed GitHub webhook must not touch R2")

    async def killswitch():
        return SimpleNamespace(enabled=True)

    monkeypatch.setattr("kb.ingestion_app.get_ingestion_killswitch", killswitch)
    async with httpx.AsyncClient() as provider:
        app.state.ctx = ConnectorContext(settings=get_settings(), http=provider)
        app.state.store = NoObjectStore()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", headers=HEADERS
        ) as client:
            response = await client.post(
                "/webhooks/github", json=envelope()["payload"], headers={"X-GitHub-Event": "issues"}
            )
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "accepted"
            await purge(TENANT, "101")
            response = await client.post(
                "/webhooks/github", json=envelope()["payload"], headers={"X-GitHub-Event": "issues"}
            )
            assert response.status_code == 200
            assert response.json()["status"] == "ignored"


async def test_purge_during_embedding_cannot_resurrect_documents(worker):
    entered = asyncio.Event()
    release = asyncio.Event()

    class DelayedEmbedder(FakeEmbedder):
        async def embed_documents(self, items):
            entered.set()
            await release.wait()
            return await super().embed_documents(items)

    worker.normalizer._embedder = DelayedEmbedder()
    await enqueue_live(TENANT, "101", envelope(), "slow-embedding")
    task = asyncio.create_task(worker.queue_step(TENANT, live=True))
    await asyncio.wait_for(entered.wait(), 5)
    assert (await purge(TENANT, "101"))["verified"] is True
    release.set()
    await task
    async with with_tenant(TENANT) as conn:
        assert (
            await conn.fetchval("SELECT count(*) FROM documents WHERE customer_id=$1", TENANT) == 0
        )
        assert await conn.fetchval("SELECT count(*) FROM chunks WHERE customer_id=$1", TENANT) == 0


async def test_additive_migration_preserves_legacy_mapping(connected):
    sql = (Path(__file__).parents[1] / "kb/github_control_schema.sql").read_text()
    async with raw_conn() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute("CREATE SCHEMA github_v2_migration_test")
            await conn.execute("SET LOCAL search_path=github_v2_migration_test,public")
            await conn.execute("CREATE TABLE customers(customer_id TEXT PRIMARY KEY)")
            await conn.execute(
                "CREATE TABLE customer_source_mapping(customer_id TEXT,external_id TEXT,source_system TEXT)"
            )
            await conn.execute(
                "CREATE TABLE ingestion_queue(customer_id TEXT,priority INT,enqueued_at TIMESTAMPTZ,status TEXT)"
            )
            await conn.execute("INSERT INTO customers VALUES ('legacy')")
            await conn.execute(
                "INSERT INTO customer_source_mapping VALUES ('legacy','123','github')"
            )
            await conn.execute(sql)
            row = await conn.fetchrow("SELECT * FROM github_installations")
            assert row["installation_id"] == "123"
            assert row["managed"] is False
            assert row["sync_enabled"] is True
            assert await conn.fetchval("SELECT count(*) FROM customer_source_mapping") == 1
        finally:
            await tx.rollback()


async def wait_for_advisory_wait():
    """Wait for an actual database lock conflict, not a scheduler sleep."""
    async with asyncio.timeout(5), raw_conn() as conn:
        while not await conn.fetchval(
            """SELECT EXISTS(SELECT 1 FROM pg_stat_activity
            WHERE datname=current_database() AND wait_event='advisory'
            AND pid<>pg_backend_pid())"""
        ):
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("action", ["created", "deleted"])
async def test_v2_repository_event_cannot_dispatch_bridge_after_purge(worker, monkeypatch, action):
    calls = []

    async def bridge(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("kb.handlers.github.code_graph_bridge.enqueue_initial_backfill", bridge)
    monkeypatch.setattr("kb.handlers.github.code_graph_bridge.enqueue_disconnect", bridge)
    normalize = worker.connector.normalize

    async def disconnect_before_normalize(event, hydrated):
        await purge(TENANT, "101")
        return await normalize(event, hydrated)

    monkeypatch.setattr(worker.connector, "normalize", disconnect_before_normalize)
    event = envelope()
    event["_headers"] = {"X-GitHub-Event": "repository"}
    event["payload"]["action"] = action
    assert await enqueue_live(TENANT, "101", event, "repository-lifecycle")
    await worker.queue_step(TENANT, live=True)
    assert calls == []
    async with with_tenant(TENANT) as conn:
        assert await conn.fetchval("SELECT status FROM ingestion_queue") == "v2_canceled"


async def test_binding_committed_during_purge_is_preserved(worker, monkeypatch):
    from engine.ingest import normalizer

    await enqueue_live(TENANT, "101", envelope(), "owner-a")
    await worker.queue_step(TENANT, live=True)
    await enqueue_live(TENANT, "202", envelope(installation="202"), "owner-b")
    entered = asyncio.Event()
    release = asyncio.Event()
    admit = normalizer._admit_ordered_write

    async def pause_with_document_lock(*args, **kwargs):
        entered.set()
        await release.wait()
        return await admit(*args, **kwargs)

    monkeypatch.setattr(normalizer, "_admit_ordered_write", pause_with_document_lock)
    writer = asyncio.create_task(worker.queue_step(TENANT, live=True))
    await asyncio.wait_for(entered.wait(), 5)
    purger = asyncio.create_task(purge(TENANT, "101"))
    try:
        await wait_for_advisory_wait()
        assert not purger.done()
    finally:
        release.set()
        await writer
    result = await purger
    assert result["documents"] == 0
    async with with_tenant(TENANT) as conn:
        assert await conn.fetchval("SELECT count(*) FROM documents") == 1
        assert await conn.fetchval("SELECT count(*) FROM chunks") > 0
        assert await conn.fetchval("SELECT installation_id FROM github_document_bindings") == "202"


async def test_purge_before_binding_retries_missing_reused_chunks(worker, monkeypatch):
    from kb import github_control_purge

    await enqueue_live(TENANT, "101", envelope(), "owner-a")
    await worker.queue_step(TENANT, live=True)
    await enqueue_live(TENANT, "202", envelope(installation="202"), "owner-b")
    entered = asyncio.Event()
    release = asyncio.Event()
    owned = github_control_purge._owned_doc_ids

    async def pause_with_ownership_locks(*args):
        ids = await owned(*args)
        entered.set()
        await release.wait()
        return ids

    monkeypatch.setattr(github_control_purge, "_owned_doc_ids", pause_with_ownership_locks)
    purger = asyncio.create_task(purge(TENANT, "101"))
    await asyncio.wait_for(entered.wait(), 5)
    writer = asyncio.create_task(worker.queue_step(TENANT, live=True))
    try:
        await wait_for_advisory_wait()
        assert not writer.done()
    finally:
        release.set()
        await purger
    await writer
    async with with_tenant(TENANT) as conn:
        assert (
            await conn.fetchval(
                "SELECT status FROM ingestion_queue WHERE github_installation_id='202'"
            )
            == "v2_pending"
        )
        assert await conn.fetchval("SELECT count(*) FROM documents") == 0
    # The healthy retry plans against the now-empty base instead of claiming
    # to reuse chunks that the completed purge deleted.
    await worker.queue_step(TENANT, live=True)
    async with with_tenant(TENANT) as conn:
        assert (
            await conn.fetchval(
                "SELECT status FROM ingestion_queue WHERE github_installation_id='202'"
            )
            == "v2_completed"
        )
        assert await conn.fetchval("SELECT count(*) FROM documents") == 1
        assert await conn.fetchval("SELECT count(*) FROM chunks") > 0
        assert await conn.fetchval("SELECT installation_id FROM github_document_bindings") == "202"


async def test_concurrent_shared_purges_leave_no_unowned_document(worker, monkeypatch):
    from kb import github_control_purge

    for installation in ("101", "202"):
        await enqueue_live(TENANT, installation, envelope(installation=installation), installation)
        await worker.queue_step(TENANT, live=True)
    entered = asyncio.Event()
    release = asyncio.Event()
    owned = github_control_purge._owned_doc_ids

    async def pause_first_purge(conn, customer_id, installation_id):
        ids = await owned(conn, customer_id, installation_id)
        if installation_id == "101":
            assert ids == []
            entered.set()
            await release.wait()
        return ids

    monkeypatch.setattr(github_control_purge, "_owned_doc_ids", pause_first_purge)
    first = asyncio.create_task(purge(TENANT, "101"))
    await asyncio.wait_for(entered.wait(), 5)
    second = asyncio.create_task(purge(TENANT, "202"))
    try:
        await wait_for_advisory_wait()
        assert not second.done()
    finally:
        release.set()
        results = await asyncio.gather(first, second)
    assert sum(result["documents"] for result in results) == 1
    async with with_tenant(TENANT) as conn:
        for table in ("documents", "chunks", "github_document_bindings"):
            assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0


async def test_legacy_enqueue_wins_before_purge_drain_check(connected, monkeypatch):
    from engine.shared.constants import SourceSystem
    from kb import github_control
    from kb.backfill_runner import enqueue_backfill

    async with with_tenant(TENANT) as conn:
        await conn.execute("UPDATE github_installations SET managed=FALSE")
    entered = asyncio.Event()
    release = asyncio.Event()
    lock = github_control.adoption_lock

    async def pause_legacy_enqueue(conn, customer_id):
        await lock(conn, customer_id)
        entered.set()
        await release.wait()

    monkeypatch.setattr(github_control, "adoption_lock", pause_legacy_enqueue)
    enqueuer = asyncio.create_task(enqueue_backfill(TENANT, SourceSystem.GITHUB))
    await asyncio.wait_for(entered.wait(), 5)
    purger = asyncio.create_task(purge(TENANT, "101"))
    try:
        await wait_for_advisory_wait()
    finally:
        release.set()
        await enqueuer
    with pytest.raises(HTTPException) as error:
        await purger
    assert error.value.status_code == 409
    async with with_tenant(TENANT) as conn:
        assert (await get_installation(conn, TENANT, "101"))["managed"] is False
        assert await conn.fetchval("SELECT status FROM backfill_state") == "pending"


async def test_purge_wins_before_legacy_enqueue(connected, monkeypatch):
    from engine.shared.constants import SourceSystem
    from kb import github_control_purge
    from kb.backfill_runner import enqueue_backfill

    async with with_tenant(TENANT) as conn:
        await conn.execute("UPDATE github_installations SET managed=FALSE")
    entered = asyncio.Event()
    release = asyncio.Event()
    owned = github_control_purge._owned_doc_ids

    async def pause_purge(*args):
        ids = await owned(*args)
        entered.set()
        await release.wait()
        return ids

    monkeypatch.setattr(github_control_purge, "_owned_doc_ids", pause_purge)
    purger = asyncio.create_task(purge(TENANT, "101"))
    await asyncio.wait_for(entered.wait(), 5)
    enqueuer = asyncio.create_task(enqueue_backfill(TENANT, SourceSystem.GITHUB))
    try:
        await wait_for_advisory_wait()
    finally:
        release.set()
        await purger
    with pytest.raises(ValueError, match="installation-scoped"):
        await enqueuer
    async with with_tenant(TENANT) as conn:
        assert await conn.fetchval("SELECT count(*) FROM backfill_state") == 0


@pytest.mark.parametrize("legacy_work", ["queue", "backfill"])
async def test_new_installation_must_drain_legacy_work(connected, monkeypatch, legacy_work):
    from kb import github_seed

    monkeypatch.setattr(github_seed, "github_mint_path", lambda settings: "hosted")

    async def unexpected_mint(*args, **kwargs):
        raise AssertionError("Token mint must not run before the legacy drain gate")

    monkeypatch.setattr(github_seed, "fetch_github_installation_token", unexpected_mint)
    async with with_tenant(TENANT) as conn:
        if legacy_work == "queue":
            await conn.execute(
                """INSERT INTO ingestion_queue(customer_id,source_system,source_event_id)
                VALUES ($1,'github','legacy')""",
                TENANT,
            )
        else:
            await conn.execute(
                """INSERT INTO backfill_state(customer_id,source_system,status)
                VALUES ($1,'github','running')""",
                TENANT,
            )
    with pytest.raises(github_seed.GitHubLegacyWorkPending):
        await github_seed.seed_github_installation(TENANT, "303", protocol_version=2)
    async with with_tenant(TENANT) as conn:
        assert not await conn.fetchval(
            "SELECT 1 FROM github_installations WHERE installation_id='303'"
        )


async def test_partial_embedding_retry_restores_chunks_before_job_completion(worker):
    from kb.github_control import enqueue_event

    expected_chunks = []

    class PartialEmbedder(FakeEmbedder):
        async def embed_documents(self, items):
            expected_chunks.append(len(items))
            result = await super().embed_documents(items)
            assert len(result.embedded) > 1
            return EmbedResult(
                embedded=result.embedded[1:],
                failed=[FailedChunk(0, "synthetic content", "temporary test failure")],
            )

    job = await new_job()
    async with with_tenant(TENANT) as conn:
        installation = await get_installation(conn, TENANT, "101", lock=True)
        await enqueue_event(
            conn, installation=installation, envelope=envelope(), source_event_id="partial", job=job
        )
        await conn.execute(
            "UPDATE github_backfill_jobs SET state='running',enumeration_complete=TRUE WHERE id=$1",
            job["id"],
        )
    worker.normalizer._embedder = PartialEmbedder()
    await worker.queue_step(TENANT, live=False)
    await worker.reconcile(TENANT)
    async with with_tenant(TENANT) as conn:
        assert await conn.fetchval("SELECT status FROM ingestion_queue") == "v2_pending"
        assert await conn.fetchval("SELECT state FROM github_backfill_jobs") == "running"
        assert await conn.fetchval("SELECT count(*) FROM documents") == 0
        assert await conn.fetchval("SELECT count(*) FROM chunks") == 0
    worker.normalizer._embedder = FakeEmbedder()
    await worker.queue_step(TENANT, live=False)
    await worker.reconcile(TENANT)
    async with with_tenant(TENANT) as conn:
        assert await conn.fetchval("SELECT status FROM ingestion_queue") == "v2_completed"
        assert await conn.fetchval("SELECT state FROM github_backfill_jobs") == "completed"
        assert (
            await conn.fetchval("SELECT count(*) FROM chunks WHERE valid_to IS NULL")
            == expected_chunks[0]
        )
        assert await conn.fetchval("SELECT count(*) FROM failed_chunks") == 0
        counts = json.loads(await conn.fetchval("SELECT counts FROM github_backfill_jobs"))
        assert counts["indexed_events"] == 1
        assert counts["failed_events"] == 0


async def test_exact_start_receipt_survives_provider_outage_and_scope_reordering(api, monkeypatch):
    body = {"scope": [*SCOPE, {"external_id": "other/project"}], "idempotency_key": "lost-response"}
    accepted = await api.post(PREFIX + "/backfills", json=body)
    assert accepted.status_code == 200

    async def unavailable(*args):
        raise HTTPException(502, "Provider unavailable")

    monkeypatch.setattr("kb.github_control_routes.list_repositories", unavailable)
    equivalent = {
        **body,
        "scope": [
            {"external_id": "OTHER/PROJECT", "label": "Different display label"},
            {"external_id": "PRBE/PAYMENTS"},
        ],
    }
    for suffix in ("/backfills/lookup", "/backfills"):
        recovered = await api.post(PREFIX + suffix, json=equivalent)
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["id"] == accepted.json()["id"]
        mismatch = await api.post(PREFIX + suffix, json={**body, "scope": SCOPE})
        assert mismatch.status_code == 409
    missing = await api.post(PREFIX + "/backfills/lookup", json={**body, "idempotency_key": "new"})
    assert missing.status_code == 404
    foreign = await api.post(
        PREFIX + "/backfills/lookup",
        json=body,
        headers={**HEADERS, "X-Prbe-Customer": "github-other-tenant"},
    )
    assert foreign.status_code == 404


async def test_retry_receipt_is_distinct_from_worker_attempts(api):
    job = await new_job()
    path = PREFIX + f"/backfills/{job['id']}/retry"
    async with with_tenant(TENANT) as conn:
        await conn.execute("UPDATE github_backfill_jobs SET attempts=3 WHERE id=$1", job["id"])
    assert (await api.post(path + "/lookup")).status_code == 404
    await cancel_job(TENANT, "101", job["id"])
    first = await api.post(path)
    assert first.json()["retry_count"] == 1
    repeated = await api.post(path)
    assert repeated.json()["retry_count"] == 1
    assert (await api.post(path + "/lookup")).json()["retry_count"] == 1
    async with with_tenant(TENANT) as conn:
        await conn.execute(
            "UPDATE github_backfill_jobs SET state='completed',counts='{}' WHERE id=$1", job["id"]
        )
    assert (await api.post(path + "/lookup")).json()["state"] == "completed"
    async with with_tenant(TENANT) as conn:
        await conn.execute("UPDATE github_backfill_jobs SET state='failed' WHERE id=$1", job["id"])
    assert (await api.post(path + "/lookup")).status_code == 404
    assert (await api.post(path)).json()["retry_count"] == 2
