"""Resolve (source_system, external_id) → customer_id.

Populated at OAuth install time via `record_mapping`; read at webhook time
via `resolve_customer`. Both operations are cheap — the table is small
(one row per connected workspace per customer) and the PK covers both.
"""

from __future__ import annotations

from typing import Any

import orjson

from shared.constants import SourceSystem
from shared.db import get_pool
from shared.exceptions import SourceAlreadyConnectedError


async def record_mapping(
    customer_id: str,
    source_system: SourceSystem,
    external_id: str,
    external_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Upsert a source-external-id → customer mapping.

    Called after OAuth exchange, once per workspace the token grants access to.
    Re-installing under the same `customer_id` is safe (refreshes name and
    metadata). A *different* `customer_id` raises `SourceAlreadyConnectedError`
    rather than silently overwriting — that overwrite is what previously let
    two members of the same Linear org route each other's webhooks across
    tenants and split chunks across customer_ids.

    Race note: the SELECT-then-INSERT here is not atomic against a parallel
    install, but the PK on (source_system, external_id) ensures only one row
    exists in the end. The loser of a race fails on either this check or the
    underlying PK violation — same outcome from the caller's perspective.
    """
    async with get_pool().acquire() as conn:
        existing = await conn.fetchrow(
            """
            SELECT customer_id FROM customer_source_mapping
            WHERE source_system = $1 AND external_id = $2
            """,
            source_system.value,
            external_id,
        )
        if existing is not None and existing["customer_id"] != customer_id:
            raise SourceAlreadyConnectedError(
                source_system=source_system.value,
                external_id=external_id,
                existing_customer_id=existing["customer_id"],
                attempted_customer_id=customer_id,
                external_name=external_name,
            )

        await conn.execute(
            """
            INSERT INTO customer_source_mapping
                (source_system, external_id, customer_id, external_name, metadata, updated_at)
            VALUES ($1, $2, $3, $4, $5::jsonb, NOW())
            ON CONFLICT (source_system, external_id)
            DO UPDATE SET external_name = EXCLUDED.external_name,
                          metadata      = customer_source_mapping.metadata || EXCLUDED.metadata,
                          updated_at    = NOW()
            """,
            source_system.value,
            external_id,
            customer_id,
            external_name,
            orjson.dumps(metadata or {}).decode("utf-8"),
        )


async def resolve_customer(
    source_system: SourceSystem, external_id: str
) -> str | None:
    """Return the customer_id that owns this external_id for this source."""
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT customer_id FROM customer_source_mapping
            WHERE source_system = $1 AND external_id = $2
            """,
            source_system.value,
            external_id,
        )
    return row["customer_id"] if row else None


async def list_mappings_for_customer(
    customer_id: str, source_system: SourceSystem | None = None
) -> list[tuple[str, str, str | None]]:
    """Return (source_system, external_id, external_name) tuples for a customer."""
    async with get_pool().acquire() as conn:
        if source_system is None:
            rows = await conn.fetch(
                """
                SELECT source_system, external_id, external_name
                FROM customer_source_mapping
                WHERE customer_id = $1
                ORDER BY source_system, external_id
                """,
                customer_id,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT source_system, external_id, external_name
                FROM customer_source_mapping
                WHERE customer_id = $1 AND source_system = $2
                ORDER BY external_id
                """,
                customer_id,
                source_system.value,
            )
    return [(r["source_system"], r["external_id"], r["external_name"]) for r in rows]


async def load_source_metadata(
    source_system: SourceSystem, external_id: str
) -> dict[str, Any]:
    """Read the metadata JSONB for one (source_system, external_id) row.

    Returns an empty dict when the row is missing or the column is NULL.
    Used by connectors that persist per-workspace state alongside the mapping
    (e.g. Slack stores its top-N display-name cache here).
    """
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT metadata FROM customer_source_mapping
            WHERE source_system = $1 AND external_id = $2
            """,
            source_system.value,
            external_id,
        )
    if row is None or row["metadata"] is None:
        return {}
    metadata = row["metadata"]
    # asyncpg returns jsonb as str when no codec is registered; normalize.
    if isinstance(metadata, str):
        metadata = orjson.loads(metadata)
    return metadata if isinstance(metadata, dict) else {}


async def patch_source_metadata(
    source_system: SourceSystem,
    external_id: str,
    patch: dict[str, Any],
) -> None:
    """Shallow-merge `patch` into the row's metadata JSONB column.

    Uses Postgres `||` operator, so top-level keys in `patch` REPLACE existing
    top-level keys. Missing rows are silently ignored (best-effort caching;
    callers can't usefully recover from "row got deleted while flushing").
    """
    if not patch:
        return
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            UPDATE customer_source_mapping
            SET metadata   = COALESCE(metadata, '{}'::jsonb) || $3::jsonb,
                updated_at = NOW()
            WHERE source_system = $1 AND external_id = $2
            """,
            source_system.value,
            external_id,
            orjson.dumps(patch).decode("utf-8"),
        )


async def single_customer_fallback() -> str | None:
    """Return the only customer_id if exactly one exists, else None.

    Used by the webhook handler as a solo-tenant convenience: when the
    payload can't be mapped (e.g. misconfigured OAuth) but there's only
    one tenant in the database, route the webhook there instead of 400-ing.
    Never triggered once you have two or more customers.
    """
    async with get_pool().acquire() as conn:
        rows = await conn.fetch("SELECT customer_id FROM customers LIMIT 2")
    return rows[0]["customer_id"] if len(rows) == 1 else None
