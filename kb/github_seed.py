"""Reusable core for connecting a GitHub App installation.

GitHub Apps don't round-trip through the standard OAuth callback the way the
other connectors do: after installing the App on an org, GitHub redirects with
`installation_id` as a query param, separate from any user OAuth code. Seeding
the installation means writing a `customer_source_mapping` row plus an
`integration_tokens` row scoped `installation:<id>` (the token itself is minted
on demand), then dry-run-fetching one token to prove the mint path is live.

This module is the shared core. Two callers wrap it:
  - `scripts.github_seed_token` — operator one-off CLI (manages the pool,
    maps errors to stderr + exit codes).
  - `POST /api/github/connect` (`kb.admin_routes`) — the research-os connect
    flow (pool already up, maps errors to HTTP status codes).

It assumes the DB pool is already initialised and raises typed errors instead
of exiting, so each caller maps them to its own surface.
"""

from __future__ import annotations

from datetime import datetime

import httpx

from engine.shared.backend_client import (
    fetch_github_installation_token,
    github_mint_path,
)
from engine.shared.config import get_settings
from engine.shared.constants import (
    GITHUB_INSTALLATION_SCOPE_PREFIX,
    IntegrationStatus,
    SourceSystem,
)
from engine.shared.customer_mapping import record_mapping
from engine.shared.db import raw_conn
from engine.shared.encryption import encrypt_token
from engine.shared.logging import get_logger

log = get_logger(__name__)


class GitHubSeedError(Exception):
    """Base for github-seed failures a caller maps to its own surface."""


class CustomerNotFoundError(GitHubSeedError):
    """The customer_id has no `customers` row (bootstrap it first)."""

    def __init__(self, customer_id: str) -> None:
        super().__init__(f"customer {customer_id!r} not found")
        self.customer_id = customer_id


class GitHubMintNotConfigured(GitHubSeedError):
    """No installation-token mint path is configured on this deployment."""

    def __init__(self) -> None:
        super().__init__(
            "no GitHub token mint path configured — set BACKEND_BASE_URL + "
            "INTERNAL_BACKEND_API_KEY (hosted; prbe-backend mints) or "
            "GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY (standalone; local minting)"
        )


async def _customer_exists(customer_id: str) -> bool:
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM customers WHERE customer_id = $1",
            customer_id,
        )
    return row is not None


async def seed_github_installation(
    customer_id: str, installation_id: str
) -> datetime:
    """Upsert the mapping + installation-scoped token row for a GitHub App
    installation and validate the mint path by fetching one installation token.

    Returns the first minted token's expiry. Assumes the DB pool is already
    initialised. Idempotent: re-seeding the same installation re-activates the
    token row.

    Raises
    ------
    CustomerNotFoundError
        No `customers` row for `customer_id`.
    GitHubMintNotConfigured
        Neither the hosted nor the standalone mint path is configured.
    (httpx / PrbeError)
        The dry-run token fetch failed — the mint path is broken.
    """
    settings = get_settings()
    if not await _customer_exists(customer_id):
        raise CustomerNotFoundError(customer_id)
    if github_mint_path(settings) is None:
        raise GitHubMintNotConfigured()

    scope = f"{GITHUB_INSTALLATION_SCOPE_PREFIX}{installation_id}"
    # access_token_encrypted is NOT NULL in the schema. The value is never read
    # for installation-scoped rows -- `_resolve_installation_bearer` mints
    # through the configured path whenever scope starts with `installation:` --
    # so store an opaque placeholder that reads clearly in a DB dump.
    placeholder = encrypt_token("installation-minted-on-demand")

    # Record the mapping FIRST so the mint path can resolve
    # customer -> installation during the dry-run fetch below (standalone reads
    # integration_tokens.scope; hosted resolves via prbe-backend).
    await record_mapping(
        customer_id=customer_id,
        source_system=SourceSystem.GITHUB,
        external_id=installation_id,
        external_name=None,
        metadata={"installation_id": installation_id},
    )
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO integration_tokens
                (customer_id, source_system, access_token_encrypted,
                 refresh_token_encrypted, expires_at, scope, status)
            VALUES ($1, 'github', $2, NULL, NULL, $3, $4)
            ON CONFLICT (customer_id, source_system) WHERE device_id IS NULL DO UPDATE SET
                scope      = EXCLUDED.scope,
                status     = 'active',
                updated_at = NOW()
            """,
            customer_id,
            placeholder,
            scope,
            IntegrationStatus.ACTIVE.value,
        )

    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http:
        _token, expires_at = await fetch_github_installation_token(
            http,
            customer_id=customer_id,
        )

    log.info(
        "github_seed.ok",
        customer=customer_id,
        installation=installation_id,
        expires_at=expires_at.isoformat(),
    )
    return expires_at
