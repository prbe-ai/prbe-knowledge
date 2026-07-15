"""Centralized OAuth token persistence.

Every connector writes tokens through here so encryption, expiry tracking,
and refresh-error surfacing stay consistent. Connectors are pure mappers;
this module does the DB + Fernet work.

Two access paths exist on the same `integration_tokens` table:

- Singleton (non-device) helpers — `save_token` / `load_token` /
  `mark_refresh_error` / `list_tokens_expiring_within`. One row per
  (customer, source); used by every OAuth/API-key connector. All queries
  pin `device_id IS NULL` so device-scoped rows are invisible.
- Device-scoped helpers — `save_device_token` / `load_device_token` /
  `revoke_device_token` / `update_device_heartbeat` /
  `list_devices_for_customer`. Many rows per customer, keyed by `device_id`
  (uniqueness still enforced per-source by the partial unique index in
  db/schema.sql, but the helpers themselves are source-agnostic).
  Used by the claude_code and codex connectors for
  per-laptop bearer-token credentials. The mutation/list helpers are
  source-agnostic — `(customer_id, device_id)` already uniquely identifies
  a device, and the dashboard surfaces all sources in one list.

  TODO(per-device-stats): the schema's partial unique index is on
  `(customer_id, source_system, device_id)`, NOT `(customer_id, device_id)`,
  so two rows with the same (customer_id, device_id) under different
  source_systems are technically permitted by the DB — even though our
  uuid4 generation in /agent-tap/pair never produces them. CodeRabbit on
  PR #70 flagged the read/write asymmetry. When we tighten the schema,
  drop the old partial index and add `(customer_id, device_id) WHERE
  device_id IS NOT NULL`, then update `save_device_token`'s ON CONFLICT
  target to match.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import orjson

from engine.shared.constants import IntegrationStatus, SourceSystem
from engine.shared.db import get_pool
from engine.shared.encryption import decrypt_token, encrypt_token
from engine.shared.models import IntegrationToken


async def save_token(token: IntegrationToken) -> None:
    """Insert-or-update a singleton (non-device) integration_tokens row.

    For device-scoped credentials use save_device_token.
    """
    if token.device_id is not None:
        raise ValueError(
            "save_token is the singleton helper; use save_device_token for device-scoped tokens"
        )
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
            ON CONFLICT (customer_id, source_system) WHERE device_id IS NULL
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
    """Load the singleton (non-device) row for this (customer, source).

    Returns None if no row exists, or if the only existing rows are
    device-scoped. For device-scoped lookups use load_device_token.
    """
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT access_token_encrypted, refresh_token_encrypted, expires_at,
                   scope, webhook_secret
            FROM integration_tokens
            WHERE customer_id = $1
              AND source_system = $2
              AND status = 'active'
              AND device_id IS NULL
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


async def mark_token_auth_failed(
    customer_id: str, source_system: SourceSystem, error: str
) -> bool:
    """Flip a singleton token row from `active` to `auth_failed`.

    Returns True when a row was actually updated (the token was active and
    is now flagged failed); False when no active row existed or the row
    was already non-active (idempotent re-calls are safe).

    Distinct from `revoked`:
      - `revoked` is for user-initiated disconnects in the dashboard.
      - `auth_failed` is for upstream-detected invalidation — the token
        was working, then the upstream provider started returning 401.
        Common causes: user revoked the OAuth grant in the upstream's
        Connected Apps UI; provider rotated their OAuth app's secret;
        provider auto-revoked after long inactivity (Linear behavior).

    Caller is expected to flag this in the dashboard so the user knows
    to re-OAuth. Until then, `load_token` returns None (filters on
    `status='active'`) so ingestion stops trying to use the dead token.
    """
    async with get_pool().acquire() as conn:
        result = await conn.execute(
            """
            UPDATE integration_tokens
            SET status = $1,
                last_refresh_error = $2,
                last_refresh_at = NOW(),
                updated_at = NOW()
            WHERE customer_id = $3
              AND source_system = $4
              AND device_id IS NULL
              AND status = $5
            """,
            IntegrationStatus.AUTH_FAILED.value,
            error[:500],
            customer_id,
            source_system.value,
            IntegrationStatus.ACTIVE.value,
        )
    return result.endswith(" 1")


async def mark_refresh_error(
    customer_id: str, source_system: SourceSystem, error: str
) -> None:
    """Mark a refresh error on the singleton row only."""
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            UPDATE integration_tokens
            SET last_refresh_error = $1, last_refresh_at = NOW(), updated_at = NOW()
            WHERE customer_id = $2
              AND source_system = $3
              AND device_id IS NULL
            """,
            error[:500],
            customer_id,
            source_system.value,
        )


async def list_tokens_expiring_within(
    window: datetime,
) -> list[tuple[str, SourceSystem]]:
    """Return (customer_id, source_system) for *singleton* tokens expiring soon.

    Device-scoped rows are excluded; device tokens don't have OAuth-style expiry.
    """
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT customer_id, source_system
            FROM integration_tokens
            WHERE status = 'active'
              AND device_id IS NULL
              AND expires_at IS NOT NULL
              AND expires_at <= $1
            ORDER BY expires_at ASC
            """,
            window,
        )
    return [(r["customer_id"], SourceSystem(r["source_system"])) for r in rows]


# ---------------------------------------------------------------------------
# Device-scoped helpers (claude_code per-laptop bearer tokens).
#
# Row shape on integration_tokens:
#   - source_system  = 'claude_code'
#   - device_id      = stable UUID generated at pair time
#   - webhook_secret = SHA-256 hash of the device token (plaintext returned
#                      to the BFF gateway exactly once, never persisted here)
#   - device_metadata = {"os": "...", "hostname": "...", "paired_at": "...",
#                        "last_heartbeat_at": "..."}
# ---------------------------------------------------------------------------


async def save_device_token(token: IntegrationToken) -> None:
    """Insert-or-update a device-scoped integration_tokens row.

    The row's webhook_secret holds the *hash* of the device token, never the
    plaintext. The plaintext is returned to the pairing caller (the BFF gateway)
    exactly once at pair time; this helper is invoked from the internal
    /api/devices/register endpoint after the gateway hashes the device token.

    Re-pair behavior: if a row already exists for (customer_id, source_system,
    device_id), the UPSERT resets status to 'active', overwrites webhook_secret,
    and merges the new device_metadata into the existing JSONB. The pair
    endpoint must enforce its own authorization gate on re-pairs (e.g., consume
    the pairing token via the jti table) to prevent unauthorized reactivation.
    """
    if token.device_id is None:
        raise ValueError("save_device_token requires device_id")
    access_enc = encrypt_token(token.access_token)
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO integration_tokens (
                customer_id, source_system,
                access_token_encrypted, webhook_secret, status,
                device_id, device_metadata, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, NOW())
            ON CONFLICT (customer_id, source_system, device_id) WHERE device_id IS NOT NULL
            DO UPDATE SET
                access_token_encrypted = EXCLUDED.access_token_encrypted,
                webhook_secret         = EXCLUDED.webhook_secret,
                status                 = EXCLUDED.status,
                device_metadata        = COALESCE(integration_tokens.device_metadata, '{}'::jsonb)
                                          || EXCLUDED.device_metadata,
                updated_at             = NOW()
            """,
            token.customer_id,
            token.source_system.value,
            access_enc,
            token.webhook_secret,
            IntegrationStatus.ACTIVE.value,
            token.device_id,
            orjson.dumps(token.device_metadata or {}).decode("utf-8"),
        )


async def load_device_token(
    customer_id: str,
    source_system: SourceSystem,
    device_id: str,
) -> IntegrationToken | None:
    """Fetch the active device-scoped token row for this (customer, source, device).

    Returns None if no row exists, or if the row exists but has been revoked.
    Note: webhook_secret on the returned IntegrationToken is the SHA-256 *hash*
    of the device token, NOT the plaintext. The plaintext is only ever known
    to the BFF gateway at pair time and is never persisted on prbe-knowledge.
    """
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT access_token_encrypted, webhook_secret, device_metadata
            FROM integration_tokens
            WHERE customer_id = $1
              AND source_system = $2
              AND device_id = $3
              AND status = 'active'
            """,
            customer_id,
            source_system.value,
            device_id,
        )
    if row is None:
        return None
    return IntegrationToken(
        customer_id=customer_id,
        source_system=source_system,
        access_token=decrypt_token(row["access_token_encrypted"]),
        webhook_secret=row["webhook_secret"],
        device_id=device_id,
        device_metadata=_load_jsonb(row["device_metadata"]),
    )


async def revoke_device_token(
    customer_id: str,
    device_id: str,
) -> bool:
    """Mark a device row revoked. Returns True if a row was updated.

    Source-agnostic: `(customer_id, device_id)` already uniquely identifies
    a device row, so we don't filter on source_system. This keeps the
    endpoint working uniformly for claude_code and codex devices (and any
    future per-laptop source).
    """
    async with get_pool().acquire() as conn:
        result = await conn.execute(
            """
            UPDATE integration_tokens
            SET status = $1, updated_at = NOW()
            WHERE customer_id = $2
              AND device_id = $3
              AND status != $1
            """,
            IntegrationStatus.REVOKED.value,
            customer_id,
            device_id,
        )
    return result.endswith(" 1")


async def update_device_heartbeat(
    customer_id: str,
    device_id: str,
) -> bool:
    """Stamp last_heartbeat_at into device_metadata. Returns True if a row was found.

    Source-agnostic: see `revoke_device_token` for rationale.
    """
    now_iso = datetime.now().astimezone().isoformat()
    async with get_pool().acquire() as conn:
        result = await conn.execute(
            """
            UPDATE integration_tokens
            SET device_metadata = COALESCE(device_metadata, '{}'::jsonb)
                                   || jsonb_build_object('last_heartbeat_at', $3::text),
                updated_at = NOW()
            WHERE customer_id = $1
              AND device_id = $2
              AND status = 'active'
            """,
            customer_id,
            device_id,
            now_iso,
        )
    return result.endswith(" 1")


async def list_devices_for_customer(
    customer_id: str,
) -> list[dict[str, Any]]:
    """Return per-device summary rows for this customer across all sources.

    The dashboard surfaces a single "your devices" list spanning claude_code
    and codex, so we intentionally do not filter by source_system here.
    """
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT device_id, source_system, status, device_metadata,
                   created_at, updated_at
            FROM integration_tokens
            WHERE customer_id = $1
              AND device_id IS NOT NULL
            ORDER BY created_at ASC
            """,
            customer_id,
        )
    return [
        {
            "device_id": r["device_id"],
            "source_system": r["source_system"],
            "status": r["status"],
            "metadata": _load_jsonb(r["device_metadata"]),
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
        }
        for r in rows
    ]


def _load_jsonb(value: Any) -> dict[str, Any] | None:
    """asyncpg returns JSONB as either str or dict depending on codec config.

    Normalize both to a dict so callers don't have to care.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, bytes, bytearray)):
        return orjson.loads(value)
    return None
