"""Backfill `documents.doc_type` for custom-ingest documents that predate
`custom.<type>` doc types.

Every custom-ingest document used to land as the generic `custom.document`
with its real kind only in `metadata.custom_document_type`. Ingest now maps
that kind to `custom.<kind>` (shared.custom_ingest.custom_doc_type) so a
caller's `doc_types` scope can exclude, say, note-less runs pre-search. This
script brings the rows already in the table in line.

Usage:
    .venv/bin/python -m scripts.backfill_custom_doc_types --customer cust-x
    .venv/bin/python -m scripts.backfill_custom_doc_types --all-tenants
    .venv/bin/python -m scripts.backfill_custom_doc_types --all-tenants --dry-run

Idempotent: only rows still carrying the generic family value AND a
well-formed kind are touched, so a re-run updates nothing. Every version of
a document is updated (dead versions included) so `doc_type` never
disagrees across versions of one doc_id. Runs per tenant under the tenant
GUC, in bounded batches, and never rewrites the kind itself -- a kind that
does not match CUSTOM_DOC_TYPE_KIND_RE stays `custom.document`, exactly as
a fresh ingest of it would.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from engine.shared.config import get_settings
from engine.shared.custom_ingest import CUSTOM_DOC_TYPE_FAMILY, CUSTOM_DOC_TYPE_KIND_RE
from engine.shared.db import close_pool, init_pool, raw_conn, with_tenant
from engine.shared.logging import configure_logging, get_logger

log = get_logger(__name__)

_GENERIC = f"{CUSTOM_DOC_TYPE_FAMILY}.document"
# Same shape custom_doc_type() admits, as a POSIX ERE for Postgres `~`.
_KIND_RE = CUSTOM_DOC_TYPE_KIND_RE.pattern

# Explicit casts: asyncpg cannot infer a parameter's type on the right of
# `~` or inside a CTE that is only referenced through a join.
_COUNT_SQL = f"""
SELECT count(*) FROM documents
WHERE customer_id = $1::text
  AND source_system = 'custom_ingest'
  AND doc_type = '{_GENERIC}'
  AND metadata->>'custom_document_type' ~ $2::text
"""

_UPDATE_SQL = f"""
WITH batch AS (
    SELECT doc_id, version FROM documents
    WHERE customer_id = $1::text
      AND source_system = 'custom_ingest'
      AND doc_type = '{_GENERIC}'
      AND metadata->>'custom_document_type' ~ $2::text
    ORDER BY doc_id, version
    LIMIT $3::int
)
UPDATE documents d
SET doc_type = '{CUSTOM_DOC_TYPE_FAMILY}.' || (d.metadata->>'custom_document_type')
FROM batch
WHERE d.customer_id = $1::text AND d.doc_id = batch.doc_id AND d.version = batch.version
RETURNING d.doc_id
"""


async def _list_customers() -> list[str]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT customer_id FROM customers WHERE status = 'active' ORDER BY customer_id"
        )
    return [r["customer_id"] for r in rows]


async def backfill_customer(
    customer_id: str, *, dry_run: bool = False, batch_size: int = 500
) -> int:
    """Return the number of rows updated (or, in dry-run, the number that
    would be)."""
    async with with_tenant(customer_id) as conn:
        if dry_run:
            n = await conn.fetchval(_COUNT_SQL, customer_id, _KIND_RE)
            log.info(
                "backfill_custom_doc_types.dry_run",
                customer_id=customer_id,
                pending=int(n or 0),
            )
            return int(n or 0)
        total = 0
        while True:
            rows = await conn.fetch(_UPDATE_SQL, customer_id, _KIND_RE, batch_size)
            if not rows:
                break
            total += len(rows)
            log.info(
                "backfill_custom_doc_types.batch",
                customer_id=customer_id,
                updated=len(rows),
                total=total,
            )
    log.info("backfill_custom_doc_types.done", customer_id=customer_id, updated=total)
    return total


async def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--customer", help="one tenant")
    target.add_argument("--all-tenants", action="store_true", help="every active tenant")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    try:
        customers = [args.customer] if args.customer else await _list_customers()
        grand = 0
        for cid in customers:
            grand += await backfill_customer(
                cid, dry_run=args.dry_run, batch_size=args.batch_size
            )
        log.info(
            "backfill_custom_doc_types.finished",
            tenants=len(customers),
            updated=grand,
            dry_run=args.dry_run,
        )
    finally:
        await close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
