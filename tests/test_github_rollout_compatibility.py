"""Startup ordering and hosted installation identity during mixed-version rollout."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import asyncpg
import httpx
import pytest
from pydantic import SecretStr

from engine.shared import backend_client, schema_readiness
from engine.shared.config import Settings
from engine.shared.exceptions import GitHubAuthError


def schema_connection(monkeypatch, outcomes):
    """A connection whose ownership can be asserted at every poll boundary."""
    state = SimpleNamespace(held=False, calls=0)
    values = iter(outcomes)

    async def fetchval(query):
        assert state.held
        assert "github_worker_capabilities" in query
        assert "github_lease_id" in query
        state.calls += 1
        return next(values)

    @asynccontextmanager
    async def connection():
        assert not state.held
        state.held = True
        try:
            yield SimpleNamespace(fetchval=fetchval)
        finally:
            state.held = False

    monkeypatch.setattr(schema_readiness, "raw_conn", connection)
    return state


async def test_migrated_schema_is_ready_without_sleep(monkeypatch):
    state = schema_connection(monkeypatch, [True])
    sleep = AsyncMock()
    monkeypatch.setattr(schema_readiness.asyncio, "sleep", sleep)
    await schema_readiness.wait_for_github_control_schema()
    assert state.calls == 1 and not state.held
    sleep.assert_not_called()


async def test_schema_wait_releases_connection_between_polls(monkeypatch):
    state = schema_connection(monkeypatch, [False, False, True])
    sleeps = []

    async def sleep(seconds):
        assert not state.held
        sleeps.append(seconds)

    monkeypatch.setattr(schema_readiness.asyncio, "sleep", sleep)
    await schema_readiness.wait_for_github_control_schema()
    assert state.calls == 3 and len(sleeps) == 2 and not state.held


@pytest.mark.parametrize("hung_query", [False, True])
async def test_missing_or_hung_schema_check_has_a_bounded_failure(monkeypatch, hung_query):
    state = SimpleNamespace(held=False)

    async def fetchval(_query):
        if hung_query:
            await asyncio.Event().wait()
        return False

    @asynccontextmanager
    async def connection():
        state.held = True
        try:
            yield SimpleNamespace(fetchval=fetchval)
        finally:
            state.held = False

    monkeypatch.setattr(schema_readiness, "raw_conn", connection)
    with pytest.raises(schema_readiness.GitHubSchemaNotReady, match="schema 0127"):
        await schema_readiness.wait_for_github_control_schema(
            timeout_seconds=0.02, poll_seconds=0.001
        )
    assert not state.held


async def test_schema_wait_remains_cancelable(monkeypatch):
    started = asyncio.Event()
    released = asyncio.Event()

    @asynccontextmanager
    async def connection():
        async def fetchval(_query):
            started.set()
            await asyncio.Event().wait()

        try:
            yield SimpleNamespace(fetchval=fetchval)
        finally:
            released.set()

    monkeypatch.setattr(schema_readiness, "raw_conn", connection)
    task = asyncio.create_task(schema_readiness.wait_for_github_control_schema())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released.is_set()


@pytest.mark.integration
async def test_schema_gate_uses_visible_committed_tables_and_all_new_columns(monkeypatch, settings):
    """Real catalogs: non-public/mixed schemas, DDL visibility, every missing field."""
    schema = f"github_rollout_{uuid4().hex}"
    fallback_schema = f"{schema}_fallback"
    tables = {
        "github_installations": (
            "history_lease_id", "history_heartbeat_at", "history_last_claimed_at"
        ),
        "github_backfill_jobs": (),
        "github_document_bindings": ("repository", "live_present", "history_present"),
        "github_worker_capabilities": (),
        "github_source_gates": (),
        "github_backfill_retry_receipts": (),
        "ingestion_queue": (
            "github_installation_id", "github_generation", "github_job_id",
            "github_payload", "github_lease_id",
        ),
    }
    writer = await asyncpg.connect(settings.database_url)
    reader = await asyncpg.connect(settings.database_url)
    gate = None

    @asynccontextmanager
    async def connection():
        yield reader

    monkeypatch.setattr(schema_readiness, "raw_conn", connection)
    try:
        await writer.execute(f'CREATE SCHEMA "{schema}"')
        await writer.execute(f'CREATE SCHEMA "{fallback_schema}"')
        for conn in (writer, reader):
            await conn.execute(f'SET search_path = "{schema}", "{fallback_schema}"')
        assert not await schema_readiness.github_control_schema_ready()
        transaction = writer.transaction()
        await transaction.start()
        for table, columns in tables.items():
            definitions = ", ".join(f"{column} text" for column in columns)
            await writer.execute(f"CREATE TABLE {table} ({definitions})")
        # An uncommitted migration must not admit replacement API/worker roles.
        gate = asyncio.create_task(schema_readiness.wait_for_github_control_schema(
            timeout_seconds=2, poll_seconds=0.001
        ))
        await asyncio.sleep(0.02)
        assert not gate.done()
        await transaction.commit()
        await gate
        # Not all resolved tables need to live in current_schema().
        await writer.execute(
            f'ALTER TABLE github_backfill_jobs SET SCHEMA "{fallback_schema}"'
        )
        assert await schema_readiness.github_control_schema_ready()
        for table, columns in tables.items():
            transaction = writer.transaction()
            await transaction.start()
            await writer.execute(f"DROP TABLE {table}")
            # Check with the writer so this transaction's missing relation is visible.
            assert not await writer.fetchval(schema_readiness._SCHEMA_READY_SQL), table
            await transaction.rollback()
            for column in columns:
                transaction = writer.transaction()
                await transaction.start()
                await writer.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
                assert not await writer.fetchval(schema_readiness._SCHEMA_READY_SQL), column
                await transaction.rollback()
        assert await schema_readiness.github_control_schema_ready()
    finally:
        if gate is not None:
            gate.cancel()
            await asyncio.gather(gate, return_exceptions=True)
        # This connection owns only the UUID-named fixture schemas above.
        await writer.execute("ROLLBACK")
        await writer.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await writer.execute(f'DROP SCHEMA IF EXISTS "{fallback_schema}" CASCADE')
        await reader.close()
        await writer.close()


@pytest.mark.parametrize("schema_ready", [True, False])
async def test_api_cannot_publish_readiness_before_schema_gate(monkeypatch, schema_ready):
    from engine.ingest import normalizer
    from kb import ingestion_app

    reached = asyncio.Event()
    release = asyncio.Event()
    entered = []

    async def gate():
        reached.set()
        await release.wait()
        if not schema_ready:
            raise schema_readiness.GitHubSchemaNotReady("schema 0127")

    initialize = AsyncMock()
    customer = AsyncMock()
    context = SimpleNamespace(http=SimpleNamespace(aclose=AsyncMock()))
    monkeypatch.setattr(ingestion_app, "init_pool", initialize)
    monkeypatch.setattr(ingestion_app, "wait_for_github_control_schema", gate)
    monkeypatch.setattr(ingestion_app, "ensure_default_customer", customer)
    monkeypatch.setattr(ingestion_app, "make_default_context", lambda: context)
    monkeypatch.setattr(ingestion_app, "get_store", Mock())
    monkeypatch.setattr(normalizer, "Normalizer", Mock())

    async def serve():
        app = SimpleNamespace(state=SimpleNamespace())
        async with ingestion_app.lifespan(app):
            entered.append(True)

    task = asyncio.create_task(serve())
    try:
        await asyncio.wait_for(reached.wait(), 1)
        initialize.assert_awaited_once()
        customer.assert_not_called()
        assert not entered and not task.done()
        release.set()
        if schema_ready:
            await task
            assert entered == [True]
            customer.assert_awaited_once()
            context.http.aclose.assert_awaited_once()
        else:
            with pytest.raises(schema_readiness.GitHubSchemaNotReady):
                await task
            assert not entered
            customer.assert_not_called()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("schema_ready", [True, False])
async def test_worker_cannot_start_any_drain_before_schema_gate(monkeypatch, schema_ready):
    from kb import worker

    class StopBeforeDrains(Exception):
        pass

    reached = asyncio.Event()
    release = asyncio.Event()

    async def gate():
        reached.set()
        await release.wait()
        if not schema_ready:
            raise schema_readiness.GitHubSchemaNotReady("schema 0127")

    initialize = AsyncMock()
    context = Mock(side_effect=StopBeforeDrains)
    monkeypatch.setattr(worker, "init_pool", initialize)
    monkeypatch.setattr(worker, "wait_for_github_control_schema", gate)
    monkeypatch.setattr(worker, "make_default_context", context)
    task = asyncio.create_task(worker.run_worker_forever())
    try:
        await asyncio.wait_for(reached.wait(), 1)
        initialize.assert_awaited_once()
        context.assert_not_called()
        assert not task.done()
        release.set()
        expected = StopBeforeDrains if schema_ready else schema_readiness.GitHubSchemaNotReady
        with pytest.raises(expected):
            await task
        assert context.call_count == int(schema_ready)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("returned_id", ["202", 202, "101", None])
async def test_hosted_token_requires_matching_installation_identity(monkeypatch, returned_id):
    settings = Settings(
        backend_base_url="http://backend.invalid",
        internal_backend_api_key=SecretStr("fixture-key"),
    )
    monkeypatch.setattr(backend_client, "get_settings", lambda: settings)
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        body = {"token": "fixture-bearer", "expires_at": "2099-12-31T00:00:00Z"}
        if returned_id is not None:
            body["installation_id"] = returned_id
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        if str(returned_id) == "202":
            token, _expiry = await backend_client.fetch_github_installation_token(
                http, customer_id="fixture-tenant", installation_id="202"
            )
            assert token == "fixture-bearer"
        else:
            with pytest.raises(GitHubAuthError, match="did not confirm") as error:
                await backend_client.fetch_github_installation_token(
                    http, customer_id="fixture-tenant", installation_id="202"
                )
            assert "fixture-bearer" not in str(error.value)
    assert seen == [{"customer_id": "fixture-tenant", "installation_id": "202"}]


async def test_legacy_token_request_keeps_the_old_optional_identity_contract(monkeypatch):
    settings = Settings(
        backend_base_url="http://backend.invalid",
        internal_backend_api_key=SecretStr("fixture-key"),
    )
    monkeypatch.setattr(backend_client, "get_settings", lambda: settings)

    def handler(request):
        assert json.loads(request.content) == {"customer_id": "fixture-tenant"}
        return httpx.Response(
            200, json={"token": "fixture-bearer", "expires_at": "2099-12-31T00:00:00Z"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        token, _expiry = await backend_client.fetch_github_installation_token(
            http, customer_id="fixture-tenant"
        )
    assert token == "fixture-bearer"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {
                "installation_id": "202",
                "token": "",
                "expires_at": "2099-12-31T00:00:00Z",
            },
            "invalid GitHub token response",
        ),
        (
            {
                "installation_id": "202",
                "token": "fixture-bearer",
                "expires_at": "not-a-timestamp",
            },
            "invalid GitHub token expiry",
        ),
        (
            {
                "installation_id": "202",
                "token": "fixture-bearer",
                "expires_at": "2000-01-01T00:00:00Z",
            },
            "expired GitHub installation token",
        ),
    ],
)
async def test_hosted_token_rejects_unusable_credentials(monkeypatch, body, message):
    settings = Settings(
        backend_base_url="http://backend.invalid",
        internal_backend_api_key=SecretStr("fixture-key"),
    )
    monkeypatch.setattr(backend_client, "get_settings", lambda: settings)

    def handler(_request):
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(GitHubAuthError, match=message):
            await backend_client.fetch_github_installation_token(
                http, customer_id="fixture-tenant", installation_id="202"
            )
