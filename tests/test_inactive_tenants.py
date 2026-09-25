"""A tenant that is not active is HELD: nothing is processed for it, and it
takes no new writes (engine/shared/tenant_status.py).

research-os terminates a team by setting its kb `customers.status` to
'terminated' and keeping the data for a hold before purging it ('deleted').
Before this the engine never looked: the idle-session sweep enumerated every
`customers` row and re-mined held sessions with a paid model, and the queue
claim took any pending row.

Every test seeds one ACTIVE tenant beside the HELD ones and checks both halves:
the held tenants' work is left exactly as it was AND the active tenant's goes
ahead. A filter that dropped everyone, or no one, fails either way.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from engine.shared.config import Settings, get_settings
from engine.shared.constants import BackfillStatus, CustomerStatus, SourceSystem
from engine.shared.db import raw_conn
from engine.shared.storage import reset_store
from engine.shared.tenant_status import TENANT_NOT_ACTIVE, active_tenant_sql, refusal_for

ACTIVE = "live-tenant"
HELD = {
    "held-terminated": CustomerStatus.TERMINATED.value,
    "held-deleted": CustomerStatus.DELETED.value,
}
EVERYONE = (ACTIVE, *HELD)
INTERNAL_KEY = "test-internal-key-32bytes-padding-padding"
_ROOT = Path(__file__).resolve().parent.parent


async def _seed_tenants() -> None:
    async with raw_conn() as conn:
        for customer, status in ((ACTIVE, CustomerStatus.ACTIVE.value), *HELD.items()):
            await conn.execute(
                "INSERT INTO customers (customer_id, display_name, api_key_hash, status) "
                "VALUES ($1, $1, $1 || '-hash', $2)",
                customer,
                status,
            )


@pytest.fixture
async def tenants(live_db) -> None:
    await _seed_tenants()


# ---- the shared predicate -----------------------------------------------------


async def test_the_predicate_is_not_captured_by_a_caller_alias(tenants) -> None:
    """Callers pass `c.customer_id` from queries that already call a table `c`
    (scripts/backfill_embedding_v2). Were the subquery's own alias `c`, it
    would compare the customers row with itself and admit every tenant."""
    async with raw_conn() as conn:
        rows = await conn.fetch(
            f"SELECT c.customer_id FROM customers c WHERE {active_tenant_sql('c.customer_id')}"
        )
    assert [r["customer_id"] for r in rows] == [ACTIVE]


async def test_refusal_is_for_held_tenants_only(tenants) -> None:
    assert await refusal_for(ACTIVE) is None
    # An unknown tenant keeps each door's existing behaviour.
    assert await refusal_for("nobody-here") is None
    for customer, status in HELD.items():
        assert await refusal_for(customer) == {"reason": TENANT_NOT_ACTIVE, "status": status}


# ---- the idle-session sweep (the paid re-mining this started from) -----------


async def _seed_session(conn, customer: str, keys: list[str], outcome: str | None = None) -> None:
    await conn.execute(
        """
        INSERT INTO ingestion_queue
            (customer_id, source_system, source_event_id, payload_s3_key,
             payload_s3_keys, status, enqueued_at, completed_at, priority, version,
             extraction_outcome)
        VALUES ($1, 'claude_code', 'sess', ($2::text[])[1], $2::text[], 'done',
                NOW() - INTERVAL '2 days', NOW() - INTERVAL '2 days', 60, 1, $3::jsonb)
        """,
        customer,
        keys,
        outcome,
    )


async def _sessions() -> dict[str, tuple]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT customer_id, status, version, payload_s3_keys, extraction_outcome "
            "FROM ingestion_queue ORDER BY customer_id"
        )
    return {r["customer_id"]: tuple(r.values())[1:] for r in rows}


async def test_the_idle_sweep_ends_only_an_active_tenants_session(tenants) -> None:
    from kb.session_completer import enqueue_idle_session_finalizers

    reset_store()
    async with raw_conn() as conn:
        for customer in EVERYONE:
            await _seed_session(conn, customer, [f"raw/claude_code/{customer}/2026/09/01/sess:0.json"])
    before = await _sessions()

    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 1

    after = await _sessions()
    assert after[ACTIVE][0] == "pending" and after[ACTIVE][2][-1].endswith("/finalize.marker")
    for customer in HELD:
        assert after[customer] == before[customer], f"{customer}'s session was touched"


async def test_a_held_tenants_partial_pass_is_not_retried(tenants) -> None:
    from kb.session_completer import enqueue_idle_session_finalizers

    outcome = '{"authoritative": false, "reason": "segment_failed", "keys": 2, "retries": 0}'
    async with raw_conn() as conn:
        for customer in EVERYONE:
            await _seed_session(
                conn,
                customer,
                [f"raw/claude_code/{customer}/2026/09/01/sess:0.json",
                 f"raw/claude_code/{customer}/sess/finalize.marker"],
                outcome,
            )
    before = await _sessions()

    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 1

    after = await _sessions()
    assert after[ACTIVE][0] == "pending"
    for customer in HELD:
        assert after[customer] == before[customer], f"{customer}'s pass was re-queued"


# ---- queue claims ---------------------------------------------------------------


async def _seed_pending(conn, customer: str, event: str, *, priority: int, age: str) -> None:
    await conn.execute(
        """
        INSERT INTO ingestion_queue
            (customer_id, source_system, source_event_id, payload_s3_keys,
             status, priority, enqueued_at)
        VALUES ($1, 'slack', $2, ARRAY['raw/slack/x.json'], 'pending', $3,
                NOW() - $4::text::interval)
        """,
        customer,
        event,
        priority,
        age,
    )


@pytest.mark.parametrize("coalesce", [1, 2], ids=["single_row_claim", "batch_claim"])
async def test_the_ingestion_claim_takes_only_an_active_tenants_rows(tenants, coalesce) -> None:
    """The held tenants' rows sit at the HEAD of the queue -- higher tier and
    older -- which is exactly where they end up during a hold."""
    from engine.ingest.handlers.base import make_default_context
    from engine.ingest.worker import Worker

    async with raw_conn() as conn:
        for customer in HELD:
            await _seed_pending(conn, customer, "held-1", priority=100, age="3 days")
            await _seed_pending(conn, customer, "held-2", priority=100, age="3 days")
        await _seed_pending(conn, ACTIVE, "live-1", priority=50, age="1 minute")

    ctx = make_default_context()
    try:
        worker = Worker(ctx, per_customer_max_inflight=3, claim_coalesce_max=coalesce)
        if coalesce == 1:
            first, second = await worker._claim_one(), await worker._claim_one()
            claimed = [first]
            assert second is None, "a held tenant's row was claimable"
        else:
            claimed = await worker._claim_batch()
            assert await worker._claim_batch() == [], "a held tenant's row was claimable"
    finally:
        await ctx.http.aclose()

    assert [(r["customer_id"], r["source_event_id"]) for r in claimed] == [(ACTIVE, "live-1")]
    async with raw_conn() as conn:
        held = await conn.fetch(
            "SELECT status, attempts FROM ingestion_queue WHERE customer_id <> $1", ACTIVE
        )
    assert len(held) == 4
    assert {(r["status"], r["attempts"]) for r in held} == {("pending", 0)}


async def test_the_post_write_claim_takes_only_an_active_tenants_rows(tenants) -> None:
    """Both legs: a fresh row, and a due one whose lease has expired."""
    from engine.ingest.post_write.worker import PostWriteWorker

    async with raw_conn() as conn:
        for n, customer in enumerate(HELD, start=1):
            await conn.execute(
                "INSERT INTO node_post_write_queue (customer_id, node_id, enqueued_at) "
                "VALUES ($1, $2, NOW() - INTERVAL '3 days')",
                customer, n,
            )
            await conn.execute(
                "INSERT INTO node_post_write_queue (customer_id, node_id, enqueued_at, locked_until) "
                "VALUES ($1, $2, NOW() - INTERVAL '3 days', NOW() - INTERVAL '1 hour')",
                customer, 100 + n,
            )
        await conn.execute(
            "INSERT INTO node_post_write_queue (customer_id, node_id) VALUES ($1, 7)", ACTIVE
        )

    worker = PostWriteWorker(concurrency=1)
    claimed = await worker._claim_one()
    assert (claimed["customer_id"], claimed["node_id"]) == (ACTIVE, 7)
    # With no fresh row left, each claim also runs the due leg.
    assert await worker._claim_one() is None, "a held tenant's row was claimable"

    async with raw_conn() as conn:
        leases = await conn.fetch(
            "SELECT node_id, locked_until < NOW() AS expired FROM node_post_write_queue "
            "WHERE customer_id <> $1 ORDER BY node_id", ACTIVE,
        )
    assert [(r["node_id"], r["expired"]) for r in leases] == [
        (1, None), (2, None), (101, True), (102, True)
    ]


async def test_the_inferred_edges_claim_takes_only_an_active_tenants_rows(tenants) -> None:
    from engine.ingest.inferred_edges.worker import InferredEdgesWorker

    async with raw_conn() as conn:
        for customer in EVERYONE:
            await conn.execute(
                "INSERT INTO inferred_edges_queue (customer_id, anchor_doc_id, extractor_id, enqueued_at) "
                "VALUES ($1, 'doc', 'x', NOW() - $2::text::interval)",
                customer, "1 minute" if customer == ACTIVE else "3 days",
            )

    worker = InferredEdgesWorker(concurrency=1)
    claimed = await worker._claim_one()
    assert claimed["customer_id"] == ACTIVE
    assert await worker._claim_one() is None, "a held tenant's row was claimable"


async def test_a_held_tenants_backfill_is_not_claimed(tenants) -> None:
    from kb.backfill_runner import claim_pending_backfill

    async with raw_conn() as conn:
        for customer in EVERYONE:
            await conn.execute(
                "INSERT INTO backfill_state (customer_id, source_system, status) "
                "VALUES ($1, 'slack', 'pending')",
                customer,
            )

    assert await claim_pending_backfill() == (ACTIVE, SourceSystem.SLACK)
    assert await claim_pending_backfill() is None
    async with raw_conn() as conn:
        held = await conn.fetch("SELECT status FROM backfill_state WHERE customer_id <> $1", ACTIVE)
    assert [r["status"] for r in held] == ["pending", "pending"]


# ---- producers that enumerate tenants -------------------------------------------


async def test_the_pollers_skip_a_held_tenant(tenants) -> None:
    from engine.ingest.handlers.base import PollConfig
    from kb.poller import IntegrationPoller
    from kb.polling.cursors import list_due_cursors

    config = PollConfig(
        interval_seconds=300,
        eligible_statuses=(BackfillStatus.COMPLETE, BackfillStatus.FAILED),
        notify_channel="granola_refresh",
    )
    async with raw_conn() as conn:
        for customer in EVERYONE:
            await conn.execute(
                "INSERT INTO ingestion_cursors (customer_id, source, resource_id, polled_at) "
                "VALUES ($1, 'linear', 'r', NOW() - INTERVAL '1 day')",
                customer,
            )
            await conn.execute(
                "INSERT INTO integration_tokens (customer_id, source_system, access_token_encrypted) "
                "VALUES ($1, 'granola', 'encrypted-stub')",
                customer,
            )
            await conn.execute(
                "INSERT INTO backfill_state (customer_id, source_system, status, last_progress_at) "
                "VALUES ($1, 'granola', 'complete', NOW() - INTERVAL '1 day')",
                customer,
            )

    due = await list_due_cursors(min_age_seconds=60, limit=10)
    assert [c.customer_id for c in due] == [ACTIVE]
    poller = IntegrationPoller(configs={SourceSystem.GRANOLA: config})
    assert await poller._fetch_due_customers(SourceSystem.GRANOLA, config) == [ACTIVE]


async def test_the_token_crons_skip_a_held_tenant(tenants) -> None:
    from engine.shared.tokens import list_tokens_expiring_within
    from scripts.cron_token_health_check import _list_active_tokens

    async with raw_conn() as conn:
        for customer in EVERYONE:
            await conn.execute(
                "INSERT INTO integration_tokens "
                "(customer_id, source_system, access_token_encrypted, expires_at) "
                "VALUES ($1, 'linear', 'encrypted-stub', NOW() + INTERVAL '5 minutes')",
                customer,
            )

    soon = datetime.now(UTC) + timedelta(hours=1)
    assert await list_tokens_expiring_within(soon) == [(ACTIVE, SourceSystem.LINEAR)]
    assert await _list_active_tokens() == [(ACTIVE, SourceSystem.LINEAR)]


async def test_the_queue_age_counts_only_an_active_tenants_backlog(tenants, monkeypatch) -> None:
    """A held tenant's rows wait out the whole hold; counted, they would pin
    the published backlog age at the hold's age for its whole length."""
    from engine.ingest import queue_age

    # sample_once() replaces a module-level snapshot other tests read.
    monkeypatch.setattr(queue_age, "_latest", queue_age.QueueAge())
    async with raw_conn() as conn:
        for customer in HELD:
            await _seed_pending(conn, customer, "held", priority=75, age="3 days")
        await _seed_pending(conn, ACTIVE, "live", priority=75, age="1 minute")
        await conn.execute("UPDATE ingestion_queue SET first_enqueued_at = enqueued_at")

    age = await queue_age.sample_once()
    assert age.pending == 1
    assert age.oldest_age_seconds < 3600


# ---- ingest doors ---------------------------------------------------------------


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch, settings: Settings):
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", INTERNAL_KEY)
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    get_settings.cache_clear()  # type: ignore[attr-defined]
    reset_store()
    yield
    get_settings.cache_clear()  # type: ignore[attr-defined]


def _headers(customer: str, **extra: str) -> dict[str, str]:
    return {"x-internal-knowledge-key": INTERNAL_KEY, "x-prbe-customer": customer, **extra}


_DOCUMENT = {
    "source_key": "notes",
    "documents": [{"id": "n-1", "type": "note", "title": "t", "body": "a research note"}],
}
_SESSION = {
    "protocol_version": 2, "session_id": "sess-held", "device_id": "d", "batch_seq": 0,
    "events": [{"line_no": 0, "raw": {"type": "user", "content": "hi"}}],
}


async def test_the_ingest_doors_refuse_a_held_tenant(tenants, gateway) -> None:
    """Custom ingest, every webhook (session batches and receipts included) and
    manual upload: 409 with a machine-readable reason, and nothing stored."""
    from kb.ingestion_app import app

    async with (
        httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        for customer, status in HELD.items():
            refused = [
                await client.post(
                    "/api/custom-ingest/documents",
                    json=_DOCUMENT,
                    headers=_headers(customer, **{"content-type": "application/json"}),
                ),
                await client.post("/webhooks/claude_code", json=_SESSION, headers=_headers(customer)),
                await client.post("/webhooks/slack", json={"type": "event_callback"},
                                  headers=_headers(customer)),
                await client.post(
                    "/api/manual-uploads",
                    files={"files": ("note.txt", b"a research note", "text/plain")},
                    headers=_headers(customer),
                ),
            ]
            for response in refused:
                assert response.status_code == 409, (response.request.url, response.text)
                assert response.json()["detail"] == {"reason": TENANT_NOT_ACTIVE, "status": status}

        # The same door still takes an active tenant's write.
        accepted = await client.post(
            "/api/custom-ingest/documents",
            json=_DOCUMENT,
            headers=_headers(ACTIVE, **{"content-type": "application/json"}),
        )
    assert accepted.status_code == 202, accepted.text

    async with raw_conn() as conn:
        stored = {
            table: await conn.fetchval(
                f"SELECT count(*) FROM {table} WHERE customer_id <> $1", ACTIVE
            )
            for table in ("ingestion_queue", "session_streams", "manual_uploads")
        }
    assert stored == {"ingestion_queue": 0, "session_streams": 0, "manual_uploads": 0}


# ---- no automatic path goes back to enumerating every tenant --------------------

#: `SELECT customer_id FROM customers` with no WHERE is how the idle sweep came
#: to re-mine held tenants. What may still enumerate every tenant, and why.
_ALL_TENANT_READERS = {
    # Retention deletes retired chunks of every tenant; a held tenant's
    # superseded rows are derived index data and go on schedule.
    "scripts/cron_chunk_retention.py",
    # Schema conversion: must move every tenant's rows or it loses data.
    "scripts/convert_chunks_to_partitioned.py",
    # Index maintenance over every tenant's rows.
    "scripts/swap_bm25_index.py",
    # Read-only report.
    "scripts/session_end_report.py",
    # Single-tenant detection (`LIMIT 2`), processes nothing.
    "engine/shared/customer_mapping.py",
}


def test_no_processing_path_enumerates_every_tenant() -> None:
    enumerate_all = re.compile(r"SELECT\s+customer_id\s+FROM\s+customers\b(?!\s+WHERE)")
    offenders = sorted(
        str(path.relative_to(_ROOT))
        for top in ("engine", "kb", "scripts", "services")
        for path in (_ROOT / top).rglob("*.py")
        if enumerate_all.search(path.read_text())
    )
    assert offenders == sorted(_ALL_TENANT_READERS), (
        "enumerate ACTIVE tenants (engine.shared.tenant_status.ACTIVE_TENANTS_SQL), "
        "or add the file above with the reason it must reach held tenants too"
    )
