"""Re-type Codex and pi session documents out of the `claude_code.*` family.

Codex and pi connectors inherited `doc_type_prefix = "claude_code."` from
ClaudeCodeConnector. That was deliberate -- the parsing, coalescing and
staleness curve really are identical, and only the provenance label differs --
and it meant 10,191 documents carried a doc_type naming an agent that did not
produce them. `source_system` was always right; `doc_type` is the field a
reader sees on a hit, and every other source in the enum takes its prefix from
its own name.

The connectors now register `codex.` and `pi.`. This brings the rows already in
the table into line, so a `doc_types` filter and a rendered label agree with
each other and with `source_system`.

Usage:
    python -m scripts.backfill_session_doc_types --customer cust-x
    python -m scripts.backfill_session_doc_types --all-tenants
    python -m scripts.backfill_session_doc_types --all-tenants --dry-run
    python -m scripts.backfill_session_doc_types --all-tenants --revert

Shape, and why -- deliberately the same as backfill_custom_doc_types:

- SOURCE_SYSTEM IS THE TRUTH, not the doc_type. The rewrite is driven by
  `source_system IN ('codex','pi')`, never by parsing the existing doc_type,
  because the existing value is exactly the thing that is wrong. A document
  whose source_system says codex gets `codex.<kind>`, whatever it said before.
- Idempotent: only rows whose doc_type differs from what the connector would
  write today are touched, so a re-run scans and writes nothing.
- Keyset-paginated on (doc_id, version): each batch walks the primary key
  forward, so no row is re-selected and the match set is scanned once.
- ONE TRANSACTION PER BATCH. `with_tenant` opens a transaction, so wrapping
  the loop in one would hold row locks on every updated document until the
  tenant finished, blocking concurrent ingest upserts of those doc_ids.
- Every VERSION of a document is updated, dead ones included, so doc_type
  never disagrees across versions of one doc_id.
- `--revert` restores the `claude_code.` family, losslessly: source_system
  still says which agent it was, so the rewrite can be undone and redone.
- FORCE RLS: every statement runs inside `with_tenant`. Without the tenant GUC
  bound the policy reduces to `customer_id = NULL` and the UPDATE reports
  success having changed nothing. That is not hypothetical -- migration 0131's
  backfill did exactly that, and this script exists downstream of noticing.

Blast radius is `documents` alone: no other table denormalises doc_type
(chunks carry doc_id only).

EXIT CODES
  0  finished, or nothing to do.
  1  a database operation failed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from engine.shared.constants import DocType, SourceSystem
from engine.shared.db import close_pool, init_pool, raw_conn, with_tenant
from engine.shared.logging import configure_logging, get_logger

log = get_logger(__name__)

#: Rows per batch. Small enough that one transaction is short, large enough
#: that 10k documents is a handful of round trips.
BATCH = 500

#: The unit kinds every coding-agent family shares.
_KINDS = ("session", "qa", "code_change", "decision", "file_ref", "directive")

#: source_system -> the doc_type prefix that source should be using. Read from
#: the connectors rather than retyped, so this script cannot drift from what
#: ingest writes.
def _prefixes() -> dict[str, str]:
    from kb.handlers.claude_code import (
        ClaudeCodeConnector,
        CodexConnector,
        PiConnector,
    )

    return {
        SourceSystem.CODEX.value: CodexConnector.doc_type_prefix,
        SourceSystem.PI.value: PiConnector.doc_type_prefix,
        # Only used by --revert, as the family to restore.
        SourceSystem.CLAUDE_CODE.value: ClaudeCodeConnector.doc_type_prefix,
    }


def _mapping(*, revert: bool) -> dict[str, dict[str, str]]:
    """`source_system -> {old doc_type: new doc_type}` for every unit kind."""
    prefixes = _prefixes()
    legacy = prefixes[SourceSystem.CLAUDE_CODE.value]
    out: dict[str, dict[str, str]] = {}
    for source in (SourceSystem.CODEX.value, SourceSystem.PI.value):
        own = prefixes[source]
        pairs = {}
        for kind in _KINDS:
            old, new = f"{legacy}{kind}", f"{own}{kind}"
            # Both directions are validated against the enum, so a typo here
            # fails before it reaches a row.
            DocType(old), DocType(new)
            pairs[new if revert else old] = old if revert else new
        out[source] = pairs
    return out


async def _backfill_tenant(customer_id: str, *, revert: bool, dry_run: bool) -> int:
    updated = 0
    for source, pairs in _mapping(revert=revert).items():
        cursor_doc, cursor_ver = "", -1
        while True:
            async with with_tenant(customer_id) as conn:
                rows = await conn.fetch(
                    """
                    SELECT doc_id, version, doc_type
                      FROM documents
                     WHERE customer_id = $1
                       AND source_system = $2
                       AND doc_type = ANY($3::text[])
                       AND (doc_id, version) > ($4, $5)
                     ORDER BY doc_id, version
                     LIMIT $6
                    """,
                    customer_id, source, list(pairs), cursor_doc, cursor_ver, BATCH,
                )
                if not rows:
                    break
                cursor_doc, cursor_ver = rows[-1]["doc_id"], rows[-1]["version"]
                if dry_run:
                    updated += len(rows)
                    continue
                for old, new in pairs.items():
                    ids = [(r["doc_id"], r["version"]) for r in rows if r["doc_type"] == old]
                    if not ids:
                        continue
                    await conn.executemany(
                        """
                        UPDATE documents SET doc_type = $3
                         WHERE customer_id = $4 AND doc_id = $1 AND version = $2
                        """,
                        [(d, v, new, customer_id) for d, v in ids],
                    )
                    updated += len(ids)
            log.info(
                "backfill_session_doc_types.batch",
                customer_id=customer_id, source=source, updated=updated,
                revert=revert, dry_run=dry_run,
            )
    return updated


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--customer")
    group.add_argument("--all-tenants", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    configure_logging()
    await init_pool()
    try:
        if args.customer:
            tenants = [args.customer]
        else:
            async with raw_conn() as conn:
                tenants = [
                    r["customer_id"]
                    for r in await conn.fetch(
                        "SELECT customer_id FROM customers ORDER BY customer_id"
                    )
                ]
        total = 0
        for customer_id in tenants:
            total += await _backfill_tenant(
                customer_id, revert=args.revert, dry_run=args.dry_run
            )
        log.info(
            "backfill_session_doc_types.finished",
            tenants=len(tenants), updated=total,
            revert=args.revert, dry_run=args.dry_run,
        )
    except Exception as exc:
        log.error("backfill_session_doc_types.failed", error=str(exc))
        return 1
    finally:
        await close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
