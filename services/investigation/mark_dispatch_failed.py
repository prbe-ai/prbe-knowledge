"""Stamp `metadata.investigation_dispatch_failed = true` on the live
INCIDENT doc when the orchestrator dispatch retry budget is exhausted.

The dashboard surfaces this flag as a "Re-trigger investigation" button
on the incident row — manual recovery for the rare case where
orchestrator is down for longer than our retry budget.

SCD2 caveat: the INCIDENT doc is `coalesce_into_live=True`, so we
UPDATE the live row in place rather than opening a new version. The
WHERE clause filters on `valid_to IS NULL` for the same reason.
"""
from __future__ import annotations

from shared.db import with_tenant


async def mark_dispatch_failed(
    *, customer_id: str, incident_doc_id: str,
) -> None:
    """Set metadata.investigation_dispatch_failed=true on the live
    INCIDENT doc row for this customer. No-op if the doc doesn't exist
    (e.g. it was deleted between dispatch attempt and exhaustion)."""
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE documents
            SET metadata = COALESCE(metadata, '{}'::jsonb)
                || jsonb_build_object('investigation_dispatch_failed', true),
                updated_at = NOW()
            WHERE doc_id = $1 AND customer_id = $2 AND valid_to IS NULL;
            """,
            incident_doc_id, customer_id,
        )
