"""CRUD over the incident_investigations table.

The `versions` JSONB column carries a list of per-pass entries; the row's
top-level `state` reflects the current lifecycle. Reviewer decisions
update the LATEST version entry in place, not the prior ones — re-runs
on reject leave the history intact for audit.

RLS: every query goes through `with_tenant(customer_id)`, which sets
`app.current_customer_id` and lets the `tenant_isolation` policy scope
the row set (and reject cross-tenant writes via WITH CHECK).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

from services.post_approval import dispatch as post_approval_dispatch
from shared.db import with_tenant
from shared.exceptions import InvestigationNotFound
from shared.investigation_schemas import (
    InvestigationDetail,
    InvestigationListItem,
    InvestigationMode,
    InvestigationState,
    InvestigationVersionEntry,
)


def _row_to_detail(row) -> InvestigationDetail:
    raw_versions = row["versions"] or "[]"
    if isinstance(raw_versions, str):
        raw_versions = json.loads(raw_versions)
    versions = [InvestigationVersionEntry(**v) for v in raw_versions]
    return InvestigationDetail(
        customer_id=row["customer_id"],
        incident_doc_id=row["incident_doc_id"],
        current_report_doc_id=row["current_report_doc_id"],
        state=row["state"],
        versions=versions,
        reviewer_id=row["reviewer_id"],
        reviewed_at=row["reviewed_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def upsert_pending_review(
    customer_id: str,
    incident_doc_id: str,
    report_doc_id: str,
    version: int,
    mode: InvestigationMode,
) -> InvestigationDetail:
    new_version_entry = {
        "version": version,
        "doc_id": report_doc_id,
        "mode": mode,
        "created_at": datetime.now(UTC).isoformat(),
        "decision": "pending",
        "reviewed_by": None,
        "reviewed_at": None,
        "feedback": None,
    }
    payload = json.dumps(new_version_entry)
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO incident_investigations
                (customer_id, incident_doc_id, current_report_doc_id,
                 state, versions)
            VALUES ($1, $2, $3, 'pending_review',
                    jsonb_build_array($4::jsonb))
            ON CONFLICT (customer_id, incident_doc_id) DO UPDATE
              SET current_report_doc_id = EXCLUDED.current_report_doc_id,
                  state = 'pending_review',
                  versions = incident_investigations.versions || $4::jsonb,
                  updated_at = now()
            RETURNING *;
            """,
            customer_id, incident_doc_id, report_doc_id, payload,
        )
    return _row_to_detail(row)


async def mark_approved(
    *,
    customer_id: str,
    incident_doc_id: str,
    reviewer_id: str,
) -> InvestigationDetail:
    now = datetime.now(UTC)
    now_iso = now.isoformat()
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            UPDATE incident_investigations
            SET state = 'approved',
                reviewer_id = $3,
                reviewed_at = $4::timestamptz,
                versions = jsonb_set(
                    versions,
                    ARRAY[(jsonb_array_length(versions) - 1)::text],
                    (versions -> (jsonb_array_length(versions) - 1))
                      || jsonb_build_object(
                           'decision', 'approved',
                           'reviewed_by', $3::text,
                           'reviewed_at', $5::text
                         )
                ),
                updated_at = now()
            WHERE customer_id = $1 AND incident_doc_id = $2
            RETURNING *;
            """,
            customer_id, incident_doc_id, reviewer_id, now, now_iso,
        )
    if row is None:
        raise InvestigationNotFound(f"no investigation for {incident_doc_id}")
    detail = _row_to_detail(row)
    # Signal the post-approval dispatch seam. on_approval is idempotent
    # and a no-op when the incident has not yet resolved — safe to call
    # unconditionally after every successful approve. The seam itself
    # fires the orchestrator only on the (approved ∧ resolved) edge.
    # Placed AFTER the UPDATE so a missing-row raise short-circuits
    # before we touch the dispatch seam.
    await post_approval_dispatch.on_approval(
        customer_id=customer_id,
        incident_doc_id=incident_doc_id,
    )
    return detail


async def mark_rejected(
    *,
    customer_id: str,
    incident_doc_id: str,
    reviewer_id: str,
    feedback: str,
) -> InvestigationDetail:
    now = datetime.now(UTC)
    now_iso = now.isoformat()
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            UPDATE incident_investigations
            SET state = 'rejected',
                reviewer_id = $3,
                reviewed_at = $4::timestamptz,
                versions = jsonb_set(
                    versions,
                    ARRAY[(jsonb_array_length(versions) - 1)::text],
                    (versions -> (jsonb_array_length(versions) - 1))
                      || jsonb_build_object(
                           'decision', 'rejected',
                           'reviewed_by', $3::text,
                           'reviewed_at', $5::text,
                           'feedback', $6::text
                         )
                ),
                updated_at = now()
            WHERE customer_id = $1 AND incident_doc_id = $2
            RETURNING *;
            """,
            customer_id, incident_doc_id, reviewer_id, now, now_iso, feedback,
        )
    if row is None:
        raise InvestigationNotFound(f"no investigation for {incident_doc_id}")
    return _row_to_detail(row)


async def get_detail(
    customer_id: str, incident_doc_id: str,
) -> InvestigationDetail | None:
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            "SELECT * FROM incident_investigations "
            "WHERE customer_id = $1 AND incident_doc_id = $2",
            customer_id, incident_doc_id,
        )
    return _row_to_detail(row) if row else None


async def list_for_customer(
    customer_id: str,
    *,
    state: InvestigationState | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[InvestigationListItem]:
    sql = (
        "SELECT incident_doc_id, current_report_doc_id, state, updated_at "
        "FROM incident_investigations WHERE customer_id = $1"
    )
    args: list = [customer_id]
    if state is not None:
        sql += " AND state = $2"
        args.append(state)
    sql += f" ORDER BY updated_at DESC LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}"
    args.extend([limit, offset])
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(sql, *args)
    return [
        InvestigationListItem(
            incident_doc_id=r["incident_doc_id"],
            current_report_doc_id=r["current_report_doc_id"],
            state=r["state"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]
