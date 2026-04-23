"""Centralized OAuth token persistence.

Every connector writes tokens through here so encryption, expiry tracking,
and refresh-error surfacing stay consistent. Connectors are pure mappers;
this module does the DB + Fernet work.
"""

from __future__ import annotations

from datetime import datetime

from shared.constants import IntegrationStatus, SourceSystem
from shared.db import get_pool
from shared.encryption import decrypt_token, encrypt_token
from shared.models import IntegrationToken


async def save_token(token: IntegrationToken) -> None:
    """Insert-or-update an integration_tokens row with encrypted credentials."""
    access_enc = encrypt_token(token.access_token)
    refresh_enc = encrypt_token(token.refresh_token) if token.refresh_token else None
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO integration_tokens (
                customer_id, source_system,
                access_token_encrypted, refresh_token_encrypted,
                expires_at, scope, webhook_secret, status, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            ON CONFLICT (customer_id, source_system)
            DO UPDATE SET
                access_token_encrypted  = EXCLUDED.access_token_encrypted,
                refresh_token_encrypted = EXCLUDED.refresh_token_encrypted,
                expires_at              = EXCLUDED.expires_at,
                scope                   = EXCLUDED.scope,
                webhook_secret          = COALESCE(EXCLUDED.webhook_secret,
                                                   integration_tokens.webhook_secret),
                status                  = EXCLUDED.status,
                last_refresh_error      = NULL,
                updated_at              = NOW()
            """,
            token.customer_id,
            token.source_system.value,
            access_enc,
            refresh_enc,
            token.expires_at,
            token.scope,
            token.webhook_secret,
            IntegrationStatus.ACTIVE.value,
        )


async def load_token(
    customer_id: str, source_system: SourceSystem
) -> IntegrationToken | None:
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT access_token_encrypted, refresh_token_encrypted, expires_at,
                   scope, webhook_secret
            FROM integration_tokens
            WHERE customer_id = $1 AND source_system = $2 AND status = 'active'
            """,
            customer_id,
            source_system.value,
        )
    if row is None:
        return None
    return IntegrationToken(
        customer_id=customer_id,
        source_system=source_system,
        access_token=decrypt_token(row["access_token_encrypted"]),
        refresh_token=(
            decrypt_token(row["refresh_token_encrypted"])
            if row["refresh_token_encrypted"]
            else None
        ),
        expires_at=row["expires_at"],
        scope=row["scope"],
        webhook_secret=row["webhook_secret"],
    )


async def mark_refresh_error(
    customer_id: str, source_system: SourceSystem, error: str
) -> None:
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            UPDATE integration_tokens
            SET last_refresh_error = $1, last_refresh_at = NOW(), updated_at = NOW()
            WHERE customer_id = $2 AND source_system = $3
            """,
            error[:500],
            customer_id,
            source_system.value,
        )


async def list_tokens_expiring_within(
    window: datetime,
) -> list[tuple[str, SourceSystem]]:
    """Return (customer_id, source_system) for tokens whose expiry ≤ window
    — used by the refresh cron to decide who to reauth."""
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT customer_id, source_system
            FROM integration_tokens
            WHERE status = 'active'
              AND expires_at IS NOT NULL
              AND expires_at <= $1
            ORDER BY expires_at ASC
            """,
            window,
        )
    return [(r["customer_id"], SourceSystem(r["source_system"])) for r in rows]
