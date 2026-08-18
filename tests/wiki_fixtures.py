"""Shared row builders for the wiki seeding/reconcile tests.

One 13-column documents INSERT instead of a copy per test file: a schema
change to `documents` (the acl NOT NULL column already caught one round
of copies) breaks exactly one helper.
"""

from __future__ import annotations

from datetime import UTC, datetime

from engine.shared.db import raw_conn, with_tenant

DEFAULT_DOC_TS = datetime(2026, 8, 1, tzinfo=UTC)


async def insert_customer(
    customer_id: str,
    *,
    preferences: str = "{}",
    status: str = "active",
    display_name: str | None = None,
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash, "
            "preferences, status) VALUES ($1, $2, 'h', $3::jsonb, $4) "
            "ON CONFLICT (customer_id) DO UPDATE "
            "SET preferences = EXCLUDED.preferences, status = EXCLUDED.status",
            customer_id,
            display_name or customer_id,
            preferences,
            status,
        )


async def insert_document(
    customer_id: str,
    doc_id: str,
    *,
    source_system: str = "slack",
    version: int = 1,
    valid_to: datetime | None = None,
    deleted_at: datetime | None = None,
    created_at: datetime = DEFAULT_DOC_TS,
) -> None:
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            INSERT INTO documents
                (doc_id, version, customer_id, source_system, source_id,
                 source_url, doc_type, content_hash, created_at, updated_at,
                 valid_from, valid_to, deleted_at, acl)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9, $9, $10, $11,
                    '{}'::jsonb)
            """,
            doc_id,
            version,
            customer_id,
            source_system,
            doc_id,
            f"https://example.test/{doc_id}",
            f"{source_system}.message",
            f"hash-{doc_id}-{version}",
            created_at,
            valid_to,
            deleted_at,
        )
