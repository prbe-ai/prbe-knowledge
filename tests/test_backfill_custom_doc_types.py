"""The `custom.<type>` doc_type backfill (E7 phase 1).

Rows ingested before ingest mapped a document's kind onto `doc_type` carry
the generic `custom.document`; the backfill rewrites exactly those, only
when the kind is well-formed AND the mapped value differs (a kind literally
named "document" maps to itself and must not be re-selected forever),
across every version of the doc, one transaction per batch, and does
nothing on a second run. `--revert` restores the legacy value.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from engine.shared import db as db_module
from engine.shared.custom_ingest import custom_doc_type
from scripts import backfill_custom_doc_types as backfill_mod
from scripts.backfill_custom_doc_types import _KIND_RE, _main, backfill_customer

_INSERT = """
INSERT INTO documents (
    doc_id, version, customer_id,
    source_system, source_id, source_url,
    doc_class, doc_type, content_type,
    content_hash, title, body_preview, body_size_bytes, body_token_count,
    created_at, updated_at, valid_from, valid_to, ingested_at, acl,
    metadata
) VALUES (
    $1, $2::int, $3, $4, $1, '/x/' || $1,
    'raw_source', $5, 'text/plain',
    'h-' || $1 || '-' || ($2::int)::text, $1, 'body', 4, 1,
    $6, $6, $6, $7, $6, '{}'::jsonb,
    $8::jsonb
)
"""


async def _seed(customer_id: str, *, status: str = "active") -> None:
    now = datetime.now(UTC)
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash, status)
            VALUES ($1, 'Backfill ' || $1, 'hash-' || $1, $2)
            ON CONFLICT (customer_id) DO UPDATE SET status = EXCLUDED.status
            """,
            customer_id,
            status,
        )
        rows = [
            # generic + well-formed kind, two versions (v1 dead, v2 live)
            ("run-1", 1, customer_id, "custom_ingest", "custom.document", now, now,
             json.dumps({"custom_document_type": "experiment.run"})),
            ("run-1", 2, customer_id, "custom_ingest", "custom.document", now, None,
             json.dumps({"custom_document_type": "experiment.run"})),
            # a second well-formed doc so batch_size=1 has to walk the keyset
            ("team-1", 1, customer_id, "custom_ingest", "custom.document", now, None,
             json.dumps({"custom_document_type": "team.note"})),
            # kind literally "document": maps to its own value; must NOT be
            # selected (the first draft looped on this row forever)
            ("doc-kind-1", 1, customer_id, "custom_ingest", "custom.document", now, None,
             json.dumps({"custom_document_type": "document"})),
            # generic + malformed kind: stays generic, as a fresh ingest would
            ("weird-1", 1, customer_id, "custom_ingest", "custom.document", now, None,
             json.dumps({"custom_document_type": "Bad Type!"})),
            # generic + no kind at all
            ("nokind-1", 1, customer_id, "custom_ingest", "custom.document", now, None,
             json.dumps({})),
            # already mapped: untouched by the backfill, reverted by --revert
            ("proj-1", 1, customer_id, "custom_ingest", "custom.project", now, None,
             json.dumps({"custom_document_type": "project"})),
            # not custom ingest: untouched even with a kind in metadata
            ("gh-1", 1, customer_id, "github", "custom.document", now, None,
             json.dumps({"custom_document_type": "experiment.run"})),
        ]
        await conn.executemany(_INSERT, rows)


async def _doc_types(customer_id: str) -> dict[tuple[str, int], str]:
    async with db_module.raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT doc_id, version, doc_type FROM documents WHERE customer_id = $1",
            customer_id,
        )
    return {(r["doc_id"], r["version"]): r["doc_type"] for r in rows}


async def test_backfill_maps_only_wellformed_generic_custom_rows(live_db: None) -> None:
    cid = "test-cust-doc-type-backfill"
    await _seed(cid)

    assert await backfill_customer(cid, dry_run=True) == 3  # run-1 v1 + v2, team-1
    assert (await _doc_types(cid))[("run-1", 2)] == "custom.document"  # dry run wrote nothing

    assert await backfill_customer(cid, batch_size=1) == 3  # batches of one walk the keyset
    types = await _doc_types(cid)
    assert types[("run-1", 1)] == "custom.experiment.run"
    assert types[("run-1", 2)] == "custom.experiment.run"
    assert types[("team-1", 1)] == "custom.team.note"
    assert types[("doc-kind-1", 1)] == "custom.document"
    assert types[("weird-1", 1)] == "custom.document"
    assert types[("nokind-1", 1)] == "custom.document"
    assert types[("proj-1", 1)] == "custom.project"
    assert types[("gh-1", 1)] == "custom.document"

    assert await backfill_customer(cid) == 0  # idempotent, and terminates


async def test_backfill_terminates_when_only_self_mapping_rows_remain(live_db: None) -> None:
    """The regression: a kind of "document" satisfied the old predicate after
    its own UPDATE, so the batch loop never drained."""
    cid = "test-cust-doc-type-backfill-selfmap"
    now = datetime.now(UTC)
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, $1, 'h') ON CONFLICT DO NOTHING",
            cid,
        )
        await conn.execute(
            _INSERT, "only-doc", 1, cid, "custom_ingest", "custom.document", now, None,
            json.dumps({"custom_document_type": "document"}),
        )
    assert await backfill_customer(cid, dry_run=True) == 0
    assert await backfill_customer(cid, batch_size=1) == 0
    assert (await _doc_types(cid))[("only-doc", 1)] == "custom.document"


async def test_revert_restores_the_legacy_value(live_db: None) -> None:
    cid = "test-cust-doc-type-backfill-revert"
    await _seed(cid)
    assert await backfill_customer(cid) == 3
    assert await backfill_customer(cid, dry_run=True, revert=True) == 4  # 3 + proj-1
    assert await backfill_customer(cid, revert=True, batch_size=2) == 4
    types = await _doc_types(cid)
    assert all(t == "custom.document" for (d, _), t in types.items() if d != "gh-1")
    assert await backfill_customer(cid, revert=True) == 0
    # and forward again: the kind survived in metadata, nothing was lost --
    # proj-1 joins this time, its pre-mapped value having been reverted too.
    assert await backfill_customer(cid) == 4


async def test_backfill_is_tenant_scoped(live_db: None) -> None:
    """Running for tenant A never touches tenant B's rows."""
    a, b = "test-cust-doc-type-backfill-a", "test-cust-doc-type-backfill-b"
    await _seed(a)
    await _seed(b)
    assert await backfill_customer(a) == 3
    assert (await _doc_types(b))[("run-1", 2)] == "custom.document"
    assert await backfill_customer(b) == 3


async def test_python_and_postgres_agree_on_which_kinds_are_wellformed(live_db: None) -> None:
    """Ingest validates the kind with Python `fullmatch`; the backfill applies
    the same body as a POSIX regex. They must accept and reject the same
    strings or new and old rows of one kind land on different doc_types."""
    corpus = [
        "experiment.run", "team.note", "a" * 79, "a" * 80, "Bad Type!", ".x", "x.",
        "a..b", "run\n", "", "UPPER", "with space", "ok-dash_underscore.dot",
    ]
    async with db_module.raw_conn() as conn:
        for kind in corpus:
            pg_ok = await conn.fetchval("SELECT $1::text ~ $2::text", kind, _KIND_RE)
            py_ok = custom_doc_type(kind) != "custom.document"
            assert bool(pg_ok) == py_ok, f"{kind!r}: postgres={pg_ok} python={py_ok}"


async def test_main_dry_run_and_all_tenants_skip_inactive(
    live_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `_main` owns a pool in production; here the live_db fixture owns it.
    monkeypatch.setattr(backfill_mod, "init_pool", AsyncMock())
    monkeypatch.setattr(backfill_mod, "close_pool", AsyncMock())
    active, inactive = "test-cust-doc-type-main-active", "test-cust-doc-type-main-inactive"
    await _seed(active)
    await _seed(inactive, status="inactive")
    assert await _main(["--customer", active, "--dry-run"]) == 0
    assert (await _doc_types(active))[("run-1", 2)] == "custom.document"
    assert await _main(["--all-tenants"]) == 0
    assert (await _doc_types(active))[("run-1", 2)] == "custom.experiment.run"
    assert (await _doc_types(inactive))[("run-1", 2)] == "custom.document"
