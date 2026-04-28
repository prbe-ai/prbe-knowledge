"""Backfill the Document → narrowing-entity graph edges that the
list-pipeline entity filter joins through.

PR #13 added `Document → Repo/Channel/Ticket/ErrorGroup/Service` edges
to each ingestion handler so future docs land with the right edges.
But every document ingested before that PR is missing them — the
entity filter's `EXISTS (... graph_edges ...)` clause finds nothing
on existing data, so "last commit on prbe-backend" returns 0 rows.

This script walks live documents per tenant, infers the target entity
canonical_id from the doc's metadata + doc_id structure, and INSERTs
the missing graph_edges rows. The graph_nodes for the target entities
already exist (handlers always created them); only the edges between
the Document node and the entity node are absent.

Source coverage:
  - github  →  Document → Repo
                (parsed from doc_id: `github:{full_name}:...`)
  - slack   →  Document → Channel
                (parsed from doc_id: `slack:{team}:{channel}:{ts}`
                 with fallback to metadata.channel_id)
  - linear  →  Document → Ticket
                (issue: parsed from doc_id `linear:{org}:issue:{id}`;
                 comment: from metadata.issue_id)
  - sentry  →  Document → ErrorGroup + Document → Service
                (issue/event: parsed from doc_id `sentry:issue:{group_id}`
                 + metadata.project_slug)
  - notion / granola / claude_code: no narrowing entity, skipped.

Idempotent: graph_edges has a UNIQUE (customer_id, edge_type,
from_node_id, to_node_id, valid_from) constraint, so re-running does
nothing.

Usage:
    .venv/bin/python -m scripts.backfill_entity_edges --customer cust-X
    .venv/bin/python -m scripts.backfill_entity_edges --all-tenants
    .venv/bin/python -m scripts.backfill_entity_edges --customer X --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass

from shared.config import get_settings
from shared.constants import EdgeType, NodeLabel, SourceSystem
from shared.db import close_pool, init_pool, raw_conn, with_tenant
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class _EdgeSpec:
    edge_type: str
    target_label: str
    target_canonical_id: str


def _edges_for_doc(
    source_system: str,
    doc_id: str,
    metadata: dict,
) -> list[_EdgeSpec]:
    """Infer the Document → entity edges this doc should have.

    Returns an empty list when the source has no narrowing-entity edges
    (notion, granola, claude_code) or when the parse fails. Failures are
    logged at DEBUG only — a malformed doc_id shouldn't fail the whole
    backfill, just skip that row.
    """
    edges: list[_EdgeSpec] = []

    if source_system == SourceSystem.GITHUB.value:
        # github:{full_name}:{kind}:{rest...}
        parts = doc_id.split(":", 3)
        if len(parts) >= 3 and parts[0] == "github":
            edges.append(
                _EdgeSpec(
                    edge_type=EdgeType.TOUCHES.value,
                    target_label=NodeLabel.REPO.value,
                    target_canonical_id=parts[1],
                )
            )

    elif source_system == SourceSystem.SLACK.value:
        # slack:{team_id}:{channel}:{ts}
        parts = doc_id.split(":")
        channel = parts[2] if len(parts) >= 4 and parts[0] == "slack" else None
        # Fallback: per-doc metadata stores channel_id explicitly.
        if not channel:
            channel = metadata.get("channel_id")
        if channel:
            edges.append(
                _EdgeSpec(
                    edge_type=EdgeType.MEMBER_OF.value,
                    target_label=NodeLabel.CHANNEL.value,
                    target_canonical_id=channel,
                )
            )

    elif source_system == SourceSystem.LINEAR.value:
        # linear:{org_id}:issue:{issue_id}  OR  linear:{org_id}:comment:{comment_id}
        parts = doc_id.split(":", 3)
        ticket_id: str | None = None
        if len(parts) == 4 and parts[0] == "linear":
            kind = parts[2]
            if kind == "issue":
                ticket_id = parts[3]
            elif kind == "comment":
                ticket_id = metadata.get("issue_id") or metadata.get("issueId")
        if ticket_id:
            edges.append(
                _EdgeSpec(
                    edge_type=EdgeType.LINKED_FROM.value,
                    target_label=NodeLabel.TICKET.value,
                    target_canonical_id=ticket_id,
                )
            )

    elif source_system == SourceSystem.SENTRY.value:
        # sentry:issue:{group_id}  OR  sentry:issue:{group_id}:sample
        parts = doc_id.split(":")
        group_id: str | None = None
        if len(parts) >= 3 and parts[0] == "sentry" and parts[1] == "issue":
            group_id = parts[2]
        project_slug = metadata.get("project_slug")
        if group_id:
            edges.append(
                _EdgeSpec(
                    edge_type=EdgeType.LINKED_FROM.value,
                    target_label=NodeLabel.ERROR_GROUP.value,
                    target_canonical_id=group_id,
                )
            )
        if project_slug:
            edges.append(
                _EdgeSpec(
                    edge_type=EdgeType.LINKED_FROM.value,
                    target_label=NodeLabel.SERVICE.value,
                    target_canonical_id=project_slug,
                )
            )

    return edges


async def _list_customers() -> list[str]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT customer_id FROM customers WHERE status = 'active' "
            "ORDER BY customer_id"
        )
    return [r["customer_id"] for r in rows]


async def _backfill_tenant(
    customer_id: str, batch_size: int, dry_run: bool
) -> tuple[int, int, int, int]:
    """Returns (docs_seen, edges_inserted, edges_skipped_no_node, edges_already_present)."""
    docs_seen = 0
    edges_inserted = 0
    edges_skipped_no_node = 0
    edges_already_present = 0

    last_doc_id: str | None = None
    while True:
        async with with_tenant(customer_id) as conn:
            rows = await conn.fetch(
                """
                SELECT d.doc_id, d.source_system, d.created_at, d.metadata
                FROM documents d
                WHERE d.customer_id = $1
                  AND d.valid_to IS NULL
                  AND ($2::text IS NULL OR d.doc_id > $2)
                ORDER BY d.doc_id
                LIMIT $3
                """,
                customer_id,
                last_doc_id,
                batch_size,
            )

            if not rows:
                break

            for row in rows:
                docs_seen += 1
                last_doc_id = row["doc_id"]

                meta = row["metadata"]
                if isinstance(meta, str):
                    import json as _json

                    try:
                        meta = _json.loads(meta)
                    except Exception:
                        meta = {}
                if not isinstance(meta, dict):
                    meta = {}

                specs = _edges_for_doc(
                    source_system=row["source_system"],
                    doc_id=row["doc_id"],
                    metadata=meta,
                )

                for spec in specs:
                    if dry_run:
                        log.info(
                            "backfill_edges.dry_run",
                            customer=customer_id,
                            doc_id=row["doc_id"],
                            edge_type=spec.edge_type,
                            target_label=spec.target_label,
                            target_canonical_id=spec.target_canonical_id,
                        )
                        edges_inserted += 1
                        continue

                    # Look up node_ids; skip if either side is missing.
                    rec = await conn.fetchrow(
                        """
                        SELECT
                          (SELECT node_id FROM graph_nodes
                            WHERE customer_id = $1
                              AND label = 'Document'
                              AND canonical_id = $2) AS doc_node_id,
                          (SELECT node_id FROM graph_nodes
                            WHERE customer_id = $1
                              AND label = $3
                              AND canonical_id = $4) AS target_node_id
                        """,
                        customer_id,
                        row["doc_id"],
                        spec.target_label,
                        spec.target_canonical_id,
                    )
                    doc_node_id = rec["doc_node_id"] if rec else None
                    target_node_id = rec["target_node_id"] if rec else None

                    if not doc_node_id or not target_node_id:
                        edges_skipped_no_node += 1
                        log.debug(
                            "backfill_edges.skip_no_node",
                            customer=customer_id,
                            doc_id=row["doc_id"],
                            target_label=spec.target_label,
                            target_canonical_id=spec.target_canonical_id,
                            doc_node_present=bool(doc_node_id),
                            target_node_present=bool(target_node_id),
                        )
                        continue

                    # ON CONFLICT DO NOTHING — idempotent.
                    result = await conn.execute(
                        """
                        INSERT INTO graph_edges (
                            customer_id, edge_type,
                            from_node_id, to_node_id,
                            valid_from
                        )
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT DO NOTHING
                        """,
                        customer_id,
                        spec.edge_type,
                        doc_node_id,
                        target_node_id,
                        row["created_at"],
                    )
                    # asyncpg returns "INSERT 0 N" — count rows actually
                    # inserted (skipping duplicates).
                    if result.endswith(" 1"):
                        edges_inserted += 1
                    else:
                        edges_already_present += 1

    return docs_seen, edges_inserted, edges_skipped_no_node, edges_already_present


async def _amain() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--customer", help="Single customer to backfill")
    group.add_argument(
        "--all-tenants",
        action="store_true",
        help="Iterate every active customer",
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be inserted without writing",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    try:
        if args.all_tenants:
            customers = await _list_customers()
        else:
            customers = [args.customer]

        for customer in customers:
            log.info(
                "backfill_edges.start",
                customer=customer,
                dry_run=args.dry_run,
                batch_size=args.batch_size,
            )
            docs, inserted, skipped, dupes = await _backfill_tenant(
                customer, args.batch_size, args.dry_run
            )
            log.info(
                "backfill_edges.done",
                customer=customer,
                docs_seen=docs,
                edges_inserted=inserted,
                edges_skipped_no_node=skipped,
                edges_already_present=dupes,
            )
    finally:
        await close_pool()

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(_amain()))
