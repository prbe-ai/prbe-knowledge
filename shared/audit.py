"""Audit log helper.

Writes append-only audit entries to `audit_log`. Every dashboard mutation
that touches sensitive surface (member role change, member removal,
team deletion, API key rotation, integration disconnect) writes one row
in the same transaction as the mutation.

Usage:

    async with raw_conn() as conn:
        async with conn.transaction():
            await conn.execute("...mutation...")
            await audit.record(
                conn,
                customer_id=customer_id,
                actor_id=user_id,
                action=AuditAction.MEMBER_REMOVED,
                resource_type="member",
                resource_id=member_user_id,
                details={"removed_email": email},
            )

The connection is passed in by the caller so the audit row commits or
rolls back atomically with the underlying mutation. Callers MUST pass
a connection that already has the surrounding transaction open.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import asyncpg


class AuditAction(StrEnum):
    """Canonical action verbs. Add new ones here, never inline strings."""

    # Customer / team lifecycle
    CUSTOMER_CREATED = "customer.created"
    CUSTOMER_SOFT_DELETED = "customer.soft_deleted"
    CUSTOMER_KEY_ROTATED = "customer.key_rotated"
    CUSTOMER_LINKED_TO_ORG = "customer.linked_to_org"

    # Membership (Better Auth-driven; we audit our side)
    MEMBER_INVITED = "member.invited"
    MEMBER_INVITE_REVOKED = "member.invite_revoked"
    MEMBER_INVITE_ACCEPTED = "member.invite_accepted"
    MEMBER_ROLE_CHANGED = "member.role_changed"
    MEMBER_REMOVED = "member.removed"
    MEMBER_LEFT = "member.left"

    # Integrations
    INTEGRATION_CONNECTED = "integration.connected"
    INTEGRATION_DISCONNECTED = "integration.disconnected"
    INTEGRATION_REFRESHED = "integration.refreshed"


async def record(
    conn: asyncpg.Connection,
    *,
    customer_id: str,
    actor_id: str,
    action: AuditAction | str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Append a row to audit_log on the given connection.

    The caller controls the surrounding transaction. This function does
    NOT open or commit a transaction — it inserts a single row.
    """
    action_str = action.value if isinstance(action, AuditAction) else action
    await conn.execute(
        """
        INSERT INTO audit_log
            (customer_id, actor_id, action, resource_type, resource_id, details)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        """,
        customer_id,
        actor_id,
        action_str,
        resource_type,
        resource_id,
        details or {},
    )
