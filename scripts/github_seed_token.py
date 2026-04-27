"""Seed the integration_tokens row for a GitHub App installation.

GitHub Apps don't round-trip through the standard OAuth callback the way the
other connectors do: after installing the App on an org, GitHub redirects
with `installation_id` as a query param, separate from any user OAuth code.
This one-off CLI lets an operator drop that installation_id in so the
backfill + CODEOWNERS hydration paths can mint installation tokens on demand.

Usage:
    .venv/bin/python -m scripts.github_seed_token \\
        --customer prbe-internal \\
        --installation-id 87654321

What it does:
    1. Verifies the customer exists.
    2. Verifies GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY are configured.
    3. Dry-run mints an installation token (via shared.github_auth) to
       confirm the App credentials + installation_id are reachable.
    4. Upserts an `integration_tokens` row with scope=`installation:<id>`.
    5. Upserts a `customer_source_mapping` row so live webhooks route.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

from shared.config import get_settings
from shared.constants import (
    GITHUB_INSTALLATION_SCOPE_PREFIX,
    IntegrationStatus,
    SourceSystem,
)
from shared.customer_mapping import record_mapping
from shared.db import close_pool, init_pool, raw_conn
from shared.encryption import encrypt_token
from shared.github_auth import mint_installation_token
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)


async def _customer_exists(customer_id: str) -> bool:
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM customers WHERE customer_id = $1",
            customer_id,
        )
    return row is not None


async def seed(customer_id: str, installation_id: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    try:
        if not await _customer_exists(customer_id):
            sys.stderr.write(
                f"error: customer {customer_id!r} not found — run "
                "`scripts.bootstrap_customer --id <id> --display-name <name>` first\n"
            )
            raise SystemExit(1)

        if settings.github_app_id is None or settings.github_app_private_key is None:
            sys.stderr.write(
                "error: GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY must be set in .env\n"
            )
            raise SystemExit(1)

        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http:
            _token, expires_at = await mint_installation_token(
                http,
                settings.github_app_id,
                settings.github_app_private_key.get_secret_value(),
                installation_id,
            )

        log.info(
            "github_seed_token.dry_run_ok",
            customer=customer_id,
            installation=installation_id,
            expires_at=expires_at.isoformat(),
        )

        scope = f"{GITHUB_INSTALLATION_SCOPE_PREFIX}{installation_id}"
        # access_token_encrypted is NOT NULL in the schema. The actual value
        # is never read — `_resolve_installation_bearer` triggers the mint path
        # whenever scope starts with `installation:` — so we store an opaque
        # encrypted placeholder that makes the intent obvious in a DB dump.
        placeholder = encrypt_token("installation-minted-on-demand")

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

        await record_mapping(
            customer_id=customer_id,
            source_system=SourceSystem.GITHUB,
            external_id=installation_id,
            external_name=None,
            metadata={"installation_id": installation_id},
        )
    finally:
        await close_pool()

    print("=" * 72)
    print(f"GitHub installation seeded for customer {customer_id}")
    print(f"  installation_id: {installation_id}")
    print(f"  scope:           {scope}")
    print(f"  expires_at:      {expires_at.isoformat()} (first mint, refreshes on demand)")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--customer", required=True, help="existing customer_id")
    ap.add_argument(
        "--installation-id",
        required=True,
        help="GitHub App installation id (from the install redirect URL)",
    )
    args = ap.parse_args()
    asyncio.run(seed(args.customer, args.installation_id))


if __name__ == "__main__":
    main()
