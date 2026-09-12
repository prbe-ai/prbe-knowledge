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
    .venv/bin/python -m scripts.backfill_custom_doc_types --all-tenants --revert

Rollout order: run it AFTER the ingest change has fully rolled out on the
plane (old pods keep writing `custom.document`; the script is idempotent, so
simply re-run it once they are gone). After a code rollback, `--revert`
restores the single legacy value so a rolled-back ingest and the table agree
again; the rewrite is lossless because `metadata.custom_document_type` keeps
the kind. No other table denormalises doc_type (chunks carry doc_id only), so
`documents` is the whole blast radius.

Shape, and why:
- Idempotent: only rows whose doc_type differs from what ingest would write
  today are touched. That excludes the generic value itself: a document whose
  kind is literally "document" maps to `custom.document`, and a predicate
  that selected it would re-select it forever (the first draft of this script
  looped on exactly that row).
- Keyset-paginated on (doc_id, version): each batch walks the primary key
  forward from the last row it touched, so no row is ever re-selected and
  the tenant's match set is scanned once, not once per batch.
- One transaction PER BATCH: `with_tenant` opens a transaction, so opening it
  once around the loop would hold row locks on every updated document until
  the tenant finished and block concurrent ingest upserts of those doc_ids.
  Each batch commits on its own; a failure loses at most one batch.
- Every version of a document is updated (dead versions included) so
  doc_type never disagrees across versions of one doc_id.
- The kind is validated with the SAME pattern ingest uses
  (CUSTOM_DOC_TYPE_KIND_RE, applied here as a POSIX regex); a malformed kind
  stays `custom.document`, exactly as a fresh ingest of it would.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from engine.shared.config import get_settings
from engine.shared.constants import DocType
from engine.shared.custom_ingest import CUSTOM_DOC_TYPE_KIND_RE
from engine.shared.db import close_pool, init_pool, raw_conn, with_tenant
from engine.shared.logging import configure_logging, get_logger
from engine.shared.source_registry import DEFAULT_DOC_TYPE_PREFIX

log = get_logger(__name__)

_GENERIC = DocType.CUSTOM_DOCUMENT.value
# Anchored for Postgres `~` (Python uses fullmatch on the same body).
_KIND_RE = f"^{CUSTOM_DOC_TYPE_KIND_RE.pattern}$"

# A row is pending when ingest would write a different doc_type for it today:
# generic value, well-formed kind, and the mapped value differs (which rules
# out kind == "document").
_PENDING_WHERE = f"""
      customer_id = $1::text
      AND source_system = 'custom_ingest'
      AND doc_type = '{_GENERIC}'
      AND metadata->>'custom_document_type' ~ $2::text
      AND doc_type IS DISTINCT FROM '{DEFAULT_DOC_TYPE_PREFIX}' || (metadata->>'custom_document_type')
"""

_COUNT_SQL = f"SELECT count(*) FROM documents WHERE {_PENDING_WHERE}"

# Keyset batch: rows after the last (doc_id, version) touched, in PK order.
_UPDATE_SQL = f"""
WITH batch AS (
    SELECT doc_id, version FROM documents
    WHERE {_PENDING_WHERE}
      AND (doc_id, version) > ($4::text, $5::int)
    ORDER BY doc_id, version
    LIMIT $3::int
)
UPDATE documents d
SET doc_type = '{DEFAULT_DOC_TYPE_PREFIX}' || (d.metadata->>'custom_document_type')
FROM batch
WHERE d.customer_id = $1::text AND d.doc_id = batch.doc_id AND d.version = batch.version
RETURNING d.doc_id, d.version
"""

_REVERT_WHERE = f"""
      customer_id = $1::text
      AND source_system = 'custom_ingest'
      AND doc_type LIKE '{DEFAULT_DOC_TYPE_PREFIX}%'
      AND doc_type <> '{_GENERIC}'
"""

_REVERT_COUNT_SQL = f"SELECT count(*) FROM documents WHERE {_REVERT_WHERE}"

_REVERT_SQL = f"""
WITH batch AS (
    SELECT doc_id, version FROM documents
    WHERE {_REVERT_WHERE}
      AND (doc_id, version) > ($3::text, $4::int)
    ORDER BY doc_id, version
    LIMIT $2::int
)
UPDATE documents d
SET doc_type = '{_GENERIC}'
FROM batch
WHERE d.customer_id = $1::text AND d.doc_id = batch.doc_id AND d.version = batch.version
RETURNING d.doc_id, d.version
"""


async def _list_customers() -> list[str]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT customer_id FROM customers WHERE status = 'active' ORDER BY customer_id"
        )
    return [r["customer_id"] for r in rows]


async def backfill_customer(
    customer_id: str,
    *,
    dry_run: bool = False,
    batch_size: int = 500,
    revert: bool = False,
) -> int:
    """Return the number of rows updated (or, in dry-run, the number that
    would be). One transaction per batch; keyset-paginated."""
    if dry_run:
        async with with_tenant(customer_id) as conn:
            if revert:
                n = await conn.fetchval(_REVERT_COUNT_SQL, customer_id)
            else:
                n = await conn.fetchval(_COUNT_SQL, customer_id, _KIND_RE)
        log.info(
            "backfill_custom_doc_types.dry_run",
            customer_id=customer_id,
            pending=int(n or 0),
            revert=revert,
        )
        return int(n or 0)

    total = 0
    last_doc_id, last_version = "", 0
    while True:
        # A fresh transaction per batch: commit, release the row locks, let
        # live ingest through, then continue from the last key.
        async with with_tenant(customer_id) as conn:
            if revert:
                rows = await conn.fetch(
                    _REVERT_SQL, customer_id, batch_size, last_doc_id, last_version
                )
            else:
                rows = await conn.fetch(
                    _UPDATE_SQL, customer_id, _KIND_RE, batch_size, last_doc_id, last_version
                )
        if not rows:
            break
        total += len(rows)
        last = max(rows, key=lambda r: (r["doc_id"], r["version"]))
        last_doc_id, last_version = last["doc_id"], int(last["version"])
        log.info(
            "backfill_custom_doc_types.batch",
            customer_id=customer_id,
            updated=len(rows),
            total=total,
            revert=revert,
        )
        if len(rows) < batch_size:
            break
    log.info(
        "backfill_custom_doc_types.done",
        customer_id=customer_id,
        updated=total,
        revert=revert,
    )
    return total


async def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--customer", help="one tenant")
    target.add_argument("--all-tenants", action="store_true", help="every active tenant")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--revert",
        action="store_true",
        help="restore the legacy custom.document value (after a code rollback)",
    )
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
                cid, dry_run=args.dry_run, batch_size=args.batch_size, revert=args.revert
            )
        log.info(
            "backfill_custom_doc_types.finished",
            tenants=len(customers),
            updated=grand,
            dry_run=args.dry_run,
            revert=args.revert,
        )
    finally:
        await close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
