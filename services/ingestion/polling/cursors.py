"""Read / write helpers around the ``ingestion_cursors`` table.

Every per-tenant read / write is wrapped in ``with_tenant`` so the
``ingestion_cursors_tenant_isolation`` RLS policy is enforced; the
scheduler's cross-tenant walk uses ``raw_conn`` and reads the
``customer_id`` column directly (the policy permits SELECTs that
match the GUC, and the scheduler doesn't set the GUC when scanning,
so it sees rows whose customer_id matches an empty GUC — which is
none — UNLESS the policy is non-FORCE). The migration uses
non-FORCE RLS for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from shared.constants import SourceSystem
from shared.db import raw_conn, with_tenant


@dataclass
class CursorRow:
    """One ingestion_cursors row, surfaced to the scheduler / pollers."""

    customer_id: str
    source: SourceSystem
    resource_id: str
    cursor_value: str | None
    polled_at: datetime
    created_at: datetime
    last_error: str | None
    last_error_at: datetime | None


def _row_to_cursor(row: Any) -> CursorRow:
    return CursorRow(
        customer_id=row["customer_id"],
        source=SourceSystem(row["source"]),
        resource_id=row["resource_id"],
        cursor_value=row["cursor_value"],
        polled_at=row["polled_at"],
        created_at=row["created_at"],
        last_error=row["last_error"],
        last_error_at=row["last_error_at"],
    )


async def list_due_cursors(
    *,
    min_age_seconds: int,
    limit: int,
) -> list[CursorRow]:
    """Cross-tenant: return cursor rows whose ``polled_at`` is older
    than ``min_age_seconds`` ago, oldest first, up to ``limit``.

    Called by the scheduler at the top of each tick to decide which
    (customer, source, resource) to poll next. The RLS policy on
    ``ingestion_cursors`` is non-FORCE (migration 0072), so this SELECT
    sees every tenant's rows — the scheduler itself doesn't have a
    tenant scope. Per-row processing later wraps the cursor update in
    ``with_tenant`` so writes ARE policy-gated.
    """
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT customer_id, source, resource_id, cursor_value,
                   polled_at, created_at, last_error, last_error_at
              FROM ingestion_cursors
             WHERE polled_at < (now() - make_interval(secs => $1))
             ORDER BY polled_at ASC
             LIMIT $2
            """,
            min_age_seconds,
            limit,
        )
    return [_row_to_cursor(r) for r in rows]


async def load_cursor(
    *,
    customer_id: str,
    source: SourceSystem,
    resource_id: str,
) -> CursorRow | None:
    """Per-tenant: load a single cursor row. Returns None if no row
    exists yet (first-poll case — the per-source poller decides what
    that means)."""
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            SELECT customer_id, source, resource_id, cursor_value,
                   polled_at, created_at, last_error, last_error_at
              FROM ingestion_cursors
             WHERE customer_id = $1
               AND source = $2
               AND resource_id = $3
            """,
            customer_id,
            source.value,
            resource_id,
        )
    return _row_to_cursor(row) if row is not None else None


async def advance_cursor(
    *,
    customer_id: str,
    source: SourceSystem,
    resource_id: str,
    new_cursor_value: str | None,
) -> None:
    """Per-tenant: update an existing cursor row (or upsert if it doesn't
    exist yet — first poll case). Stamps ``polled_at`` to now and clears
    any previous ``last_error``. The composite PK
    (customer_id, source, resource_id) makes the ON CONFLICT safe even
    under two concurrent ticks on the same resource."""
    now = datetime.now(UTC)
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            INSERT INTO ingestion_cursors (
                customer_id, source, resource_id, cursor_value,
                polled_at, created_at, last_error, last_error_at
            ) VALUES ($1, $2, $3, $4, $5, $5, NULL, NULL)
            ON CONFLICT (customer_id, source, resource_id) DO UPDATE
              SET cursor_value = COALESCE(EXCLUDED.cursor_value, ingestion_cursors.cursor_value),
                  polled_at = EXCLUDED.polled_at,
                  last_error = NULL,
                  last_error_at = NULL
            """,
            customer_id,
            source.value,
            resource_id,
            new_cursor_value,
            now,
        )


async def stamp_error(
    *,
    customer_id: str,
    source: SourceSystem,
    resource_id: str,
    error: str,
) -> None:
    """Per-tenant: mark a poll attempt as failed. Updates polled_at
    (so the scheduler doesn't immediately re-poll a known-broken
    resource) and stamps the error string + timestamp. Does NOT touch
    cursor_value — the next successful poll uses the same cursor."""
    now = datetime.now(UTC)
    # Truncate to fit a reasonable column width while preserving the
    # head of the error (which is usually the most diagnostic).
    truncated = error[:2048]
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            INSERT INTO ingestion_cursors (
                customer_id, source, resource_id, cursor_value,
                polled_at, created_at, last_error, last_error_at
            ) VALUES ($1, $2, $3, NULL, $4, $4, $5, $4)
            ON CONFLICT (customer_id, source, resource_id) DO UPDATE
              SET polled_at = EXCLUDED.polled_at,
                  last_error = EXCLUDED.last_error,
                  last_error_at = EXCLUDED.last_error_at
            """,
            customer_id,
            source.value,
            resource_id,
            now,
            truncated,
        )
