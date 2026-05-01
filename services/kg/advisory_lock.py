"""Per-tenant Postgres advisory lock for single-writer enforcement.

Wraps ``pg_advisory_xact_lock(hashtextextended(customer_id, 0))`` inside an
existing asyncpg transaction. Auto-released at transaction commit/rollback.
Must be called inside an active transaction (e.g., obtained via
``shared.db.with_tenant()`` or ``async with conn.transaction():``).

See spec §7.2 — single-writer enforcement.

``hashtextextended(text, seed)`` returns ``int8`` (64-bit), which is what
``pg_advisory_xact_lock(bigint)`` accepts natively. The plain ``hashtext()``
function returns ``int4`` (32-bit), which collides more easily — prefer the
extended variant.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

import asyncpg


@contextlib.asynccontextmanager
async def tenant_xact_lock(
    conn: asyncpg.Connection, *, customer_id: str
) -> AsyncIterator[None]:
    """Acquire an xact-scoped advisory lock keyed on ``customer_id``.

    Blocks until released by any other holder. Auto-released when the
    transaction commits or rolls back. The connection MUST already be
    inside a transaction (e.g., obtained via ``shared.db.with_tenant()``
    or ``async with conn.transaction():``).

    Args:
        conn: An asyncpg connection that is already inside a transaction.
        customer_id: The tenant key the lock is hashed on. Must be non-empty.

    Raises:
        ValueError: If ``customer_id`` is empty.
        RuntimeError: If ``conn`` is not inside an active transaction.
    """
    if not customer_id:
        raise ValueError("tenant_xact_lock requires a non-empty customer_id")
    if not conn.is_in_transaction():
        raise RuntimeError(
            "tenant_xact_lock must be called inside an active transaction"
        )
    await conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
        customer_id,
    )
    yield
