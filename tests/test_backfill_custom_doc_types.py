"""The `custom.<type>` doc_type backfill (E7 phase 1).

Rows ingested before ingest mapped a document's kind onto `doc_type` carry
the generic `custom.document`; the backfill rewrites exactly those, only
when the kind is well-formed, across every version of the doc, and does
nothing on a second run.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from engine.shared import db as db_module
from scripts.backfill_custom_doc_types import backfill_customer

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


async def _seed(customer_id: str) -> None:
    now = datetime.now(UTC)
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'Backfill ' || $1, 'hash-' || $1)
            ON CONFLICT DO NOTHING
            """,
            customer_id,
        )
        rows = [
            # generic + well-formed kind, two versions (v1 dead, v2 live)
            ("run-1", 1, customer_id, "custom_ingest", "custom.document", now, now,
             json.dumps({"custom_document_type": "experiment.run"})),
            ("run-1", 2, customer_id, "custom_ingest", "custom.document", now, None,
             json.dumps({"custom_document_type": "experiment.run"})),
            # generic + malformed kind: stays generic, as a fresh ingest would
            ("weird-1", 1, customer_id, "custom_ingest", "custom.document", now, None,
             json.dumps({"custom_document_type": "Bad Type!"})),
            # generic + no kind at all
            ("nokind-1", 1, customer_id, "custom_ingest", "custom.document", now, None,
             json.dumps({})),
            # already mapped: untouched
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

    assert await backfill_customer(cid, dry_run=True) == 2  # run-1 v1 + v2
    assert (await _doc_types(cid))[("run-1", 2)] == "custom.document"  # dry run wrote nothing

    assert await backfill_customer(cid, batch_size=1) == 2  # batches of one still drain
    types = await _doc_types(cid)
    assert types[("run-1", 1)] == "custom.experiment.run"
    assert types[("run-1", 2)] == "custom.experiment.run"
    assert types[("weird-1", 1)] == "custom.document"
    assert types[("nokind-1", 1)] == "custom.document"
    assert types[("proj-1", 1)] == "custom.project"
    assert types[("gh-1", 1)] == "custom.document"

    assert await backfill_customer(cid) == 0  # idempotent


async def test_backfill_is_tenant_scoped(live_db: None) -> None:
    """Running for tenant A never touches tenant B's rows."""
    a, b = "test-cust-doc-type-backfill-a", "test-cust-doc-type-backfill-b"
    await _seed(a)
    await _seed(b)
    assert await backfill_customer(a) == 2
    assert (await _doc_types(b))[("run-1", 2)] == "custom.document"
    assert await backfill_customer(b) == 2
