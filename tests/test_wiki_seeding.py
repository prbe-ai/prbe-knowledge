"""Queue seeding: seed_missing_docs / reset_terminal_rows / the catchup CLI.

The seed is the retroactive counterpart of the Normalizer enqueue — the
same rows, minted after the fact. What is pinned here:

* the seed inserts exactly the live eligible docs (versioned + deleted +
  excluded-source docs stay out) and is idempotent under re-run;
* the dry-run counter reports the REAL would-insert split (the historical
  bug reported already_queued=eligible always, teaching operators the
  seed had already happened when it never had);
* the reset only resurrects terminal rows whose (doc_id, doc_version)
  still matches a live eligible document — superseded versions, deleted
  docs, excluded sources, in-flight rows, and (by default) dlq rows stay
  put, and per-row triage/synthesis state is actually cleared.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from engine.shared.db import raw_conn, with_tenant
from kb.synthesis import persistence
from scripts.wiki_synthesis_catchup import seed as cli_seed
from tests.wiki_fixtures import insert_customer, insert_document

CUSTOMER = "wiki-seed-cust"
_NOW = datetime(2026, 8, 1, tzinfo=UTC)


@pytest_asyncio.fixture
async def seeded_customer(live_db: None) -> str:
    await insert_customer(CUSTOMER, preferences='{"wiki_generation_enabled": true}')
    return CUSTOMER


async def _insert_doc(
    doc_id: str,
    *,
    source_system: str = "slack",
    version: int = 1,
    valid_to: datetime | None = None,
    deleted_at: datetime | None = None,
) -> None:
    await insert_document(
        CUSTOMER,
        doc_id,
        source_system=source_system,
        version=version,
        valid_to=valid_to,
        deleted_at=deleted_at,
        created_at=_NOW,
    )


async def _queue_row(
    doc_id: str,
    *,
    version: int = 1,
    status: str = "done",
    source_system: str = "slack",
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO wiki_synthesis_queue
                (customer_id, doc_id, doc_version, source_system, doc_type,
                 status, triage_score, attempts, dlq_reason)
            VALUES ($1, $2, $3, $4, $5, $6, 0.9, 3,
                    CASE WHEN $6 = 'dlq' THEN 'triage.boom' END)
            """,
            CUSTOMER,
            doc_id,
            version,
            source_system,
            f"{source_system}.message",
            status,
        )


async def _queue_statuses() -> dict[str, str]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT doc_id, status FROM wiki_synthesis_queue WHERE customer_id = $1",
            CUSTOMER,
        )
    return {r["doc_id"]: r["status"] for r in rows}


@pytest.mark.asyncio
async def test_seed_inserts_only_live_eligible_docs(seeded_customer: str) -> None:
    await _insert_doc("doc:live")
    await _insert_doc("doc:code", source_system="code_graph")
    await _insert_doc("doc:wiki", source_system="wiki")
    await _insert_doc("doc:old", valid_to=_NOW + timedelta(days=1))
    await _insert_doc("doc:gone", deleted_at=_NOW)

    async with with_tenant(CUSTOMER) as conn:
        eligible, inserted = await persistence.seed_missing_docs(conn, CUSTOMER)

    assert (eligible, inserted) == (1, 1)
    assert await _queue_statuses() == {"doc:live": "pending"}


@pytest.mark.asyncio
async def test_seed_is_idempotent(seeded_customer: str) -> None:
    await _insert_doc("doc:a")
    async with with_tenant(CUSTOMER) as conn:
        await persistence.seed_missing_docs(conn, CUSTOMER)
        eligible, inserted = await persistence.seed_missing_docs(conn, CUSTOMER)
    assert (eligible, inserted) == (1, 0)


@pytest.mark.asyncio
async def test_dry_run_counts_report_the_true_split(seeded_customer: str) -> None:
    """The would-insert number is an anti-join, not `eligible - 0`."""
    await _insert_doc("doc:queued")
    await _insert_doc("doc:missing")
    await _queue_row("doc:queued", status="pending")

    async with with_tenant(CUSTOMER) as conn:
        eligible, would_insert = await persistence.count_seedable_docs(
            conn, CUSTOMER
        )
    assert (eligible, would_insert) == (2, 1)

    stats = await cli_seed(
        CUSTOMER, dry_run=True, reset_terminal=False, notify=False
    )
    assert stats["would_insert"] == 1
    assert stats["already_queued"] == 1
    assert stats["inserted"] == 0
    # And a dry run wrote nothing.
    assert set(await _queue_statuses()) == {"doc:queued"}


@pytest.mark.asyncio
async def test_reset_touches_only_live_eligible_terminal_rows(
    seeded_customer: str,
) -> None:
    # Terminal row whose doc is live: resets.
    await _insert_doc("doc:redo")
    await _queue_row("doc:redo", status="done")
    # Terminal row for a superseded version: stays.
    await _insert_doc("doc:stale", valid_to=_NOW + timedelta(days=1))
    await _queue_row("doc:stale", status="done")
    # Terminal row whose doc is deleted: stays.
    await _insert_doc("doc:gone", deleted_at=_NOW)
    await _queue_row("doc:gone", status="rejected")
    # Terminal row for an excluded source: stays.
    await _insert_doc("doc:code", source_system="code_graph")
    await _queue_row("doc:code", status="rejected", source_system="code_graph")
    # In-flight row: stays.
    await _insert_doc("doc:busy")
    await _queue_row("doc:busy", status="synthesizing")
    # DLQ row: stays unless include_dlq.
    await _insert_doc("doc:poison")
    await _queue_row("doc:poison", status="dlq")

    async with with_tenant(CUSTOMER) as conn:
        assert await persistence.count_resettable_rows(conn, CUSTOMER) == 1
        reset = await persistence.reset_terminal_rows(conn, CUSTOMER)
    assert reset == 1
    assert await _queue_statuses() == {
        "doc:redo": "pending",
        "doc:stale": "done",
        "doc:gone": "rejected",
        "doc:code": "rejected",
        "doc:busy": "synthesizing",
        "doc:poison": "dlq",
    }

    # The reset row's per-row state was actually cleared.
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT triage_score, attempts, heartbeat_at "
            "FROM wiki_synthesis_queue WHERE customer_id=$1 AND doc_id='doc:redo'",
            CUSTOMER,
        )
    assert row["triage_score"] is None
    assert row["attempts"] == 0
    assert row["heartbeat_at"] is None


@pytest.mark.asyncio
async def test_include_dlq_redrives_poison_rows(seeded_customer: str) -> None:
    await _insert_doc("doc:poison")
    await _queue_row("doc:poison", status="dlq")

    async with with_tenant(CUSTOMER) as conn:
        assert (
            await persistence.count_resettable_rows(
                conn, CUSTOMER, include_dlq=True
            )
            == 1
        )
        reset = await persistence.reset_terminal_rows(
            conn, CUSTOMER, include_dlq=True
        )
    assert reset == 1
    statuses = await _queue_statuses()
    assert statuses["doc:poison"] == "pending"
    async with raw_conn() as conn:
        assert (
            await conn.fetchval(
                "SELECT dlq_reason FROM wiki_synthesis_queue "
                "WHERE customer_id=$1 AND doc_id='doc:poison'",
                CUSTOMER,
            )
            is None
        )


@pytest.mark.asyncio
async def test_cli_seed_inserts_and_opens_onboarding_run(
    seeded_customer: str,
) -> None:
    await _insert_doc("doc:a")
    await _insert_doc("doc:b")

    stats = await cli_seed(
        CUSTOMER, dry_run=False, reset_terminal=False, notify=False
    )
    assert stats["inserted"] == 2
    assert stats["run_id"] is not None
    assert stats["notified"] is False

    async with raw_conn() as conn:
        run = await conn.fetchrow(
            "SELECT kind, status, events_total FROM wiki_synthesis_runs "
            "WHERE run_id = $1",
            stats["run_id"],
        )
    assert dict(run) == {"kind": "onboarding", "status": "running", "events_total": 2}


@pytest.mark.asyncio
async def test_cli_reset_terminal_path(seeded_customer: str) -> None:
    await _insert_doc("doc:redo")
    await _queue_row("doc:redo", status="failed")

    stats = await cli_seed(
        CUSTOMER, dry_run=False, reset_terminal=True, notify=False
    )
    assert stats["reset"] == 1
    assert (await _queue_statuses())["doc:redo"] == "pending"


@pytest.mark.asyncio
async def test_seed_limit_bounds_one_call(seeded_customer: str) -> None:
    """The batching contract: `inserted == limit` means "call again"."""
    await _insert_doc("doc:a")
    await _insert_doc("doc:b")
    await _insert_doc("doc:c")

    async with with_tenant(CUSTOMER) as conn:
        eligible, first = await persistence.seed_missing_docs(
            conn, CUSTOMER, limit=2
        )
        # Batch mode skips the eligible count: computing it would force a
        # full scan per batch, defeating the LIMIT's early termination.
        assert (eligible, first) == (0, 2)
        _eligible, second = await persistence.seed_missing_docs(
            conn, CUSTOMER, limit=2
        )
        assert second == 1
        _eligible, third = await persistence.seed_missing_docs(
            conn, CUSTOMER, limit=2
        )
        assert third == 0
    assert len(await _queue_statuses()) == 3


def test_cli_rejects_invalid_flag_combinations() -> None:
    from scripts.wiki_synthesis_catchup import main

    with pytest.raises(SystemExit):
        main(["cust", "--include-dlq"])  # requires --reset-terminal
    with pytest.raises(SystemExit):
        main([])  # customer_id required unless --all-enabled


def test_concurrency_env_override_parses_defensively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed env value must not crash every importing service at
    boot; a non-positive one must not silently disable draining."""
    from engine.shared.constants import _env_positive_int

    monkeypatch.setenv("X_CONC", "3")
    assert _env_positive_int("X_CONC", 2) == 3
    for bad in ("", "two", "0", "-1"):
        monkeypatch.setenv("X_CONC", bad)
        assert _env_positive_int("X_CONC", 2) == 2
    monkeypatch.delenv("X_CONC")
    assert _env_positive_int("X_CONC", 2) == 2
