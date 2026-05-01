"""Seed the integration_tokens row for a GitHub App installation.

GitHub Apps don't round-trip through the standard OAuth callback the way the
other connectors do: after installing the App on an org, GitHub redirects
with `installation_id` as a query param, separate from any user OAuth code.
This one-off CLI lets an operator drop that installation_id in so the
backfill + CODEOWNERS hydration paths can fetch installation tokens on demand.

Usage:
    .venv/bin/python -m scripts.github_seed_token \\
        --customer prbe-internal \\
        --installation-id 87654321

What it does:
    1. Verifies the customer exists.
    2. Upserts a `customer_source_mapping` row so prbe-backend can resolve
       the installation_id from this customer when it mints a token.
    3. Verifies BACKEND_BASE_URL + INTERNAL_BACKEND_API_KEY are configured.
    4. Dry-run fetches an installation token via prbe-backend's
       /internal/github/installation_token endpoint to confirm the
       end-to-end mint path is live.
    5. Upserts an `integration_tokens` row with scope=`installation:<id>`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

from shared.backend_client import fetch_github_installation_token
from shared.config import get_settings
from shared.constants import (
    GITHUB_INSTALLATION_SCOPE_PREFIX,
    IntegrationStatus,
    SourceSystem,
)
from shared.customer_mapping import record_mapping
from shared.db import close_pool, init_pool, raw_conn
from shared.encryption import encrypt_token
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

        if not settings.backend_base_url or not settings.internal_backend_api_key.get_secret_value():
            sys.stderr.write(
                "error: BACKEND_BASE_URL and INTERNAL_BACKEND_API_KEY must be set in .env\n"
            )
            raise SystemExit(1)

        # Record the mapping FIRST so backend can resolve customer → installation
        # when we dry-run-fetch the token below.
        await record_mapping(
            customer_id=customer_id,
            source_system=SourceSystem.GITHUB,
            external_id=installation_id,
            external_name=None,
            metadata={"installation_id": installation_id},
        )

        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http:
            _token, expires_at = await fetch_github_installation_token(
                http,
                customer_id=customer_id,
            )

        log.info(
            "github_seed_token.dry_run_ok",
            customer=customer_id,
            installation=installation_id,
            expires_at=expires_at.isoformat(),
        )

        scope = f"{GITHUB_INSTALLATION_SCOPE_PREFIX}{installation_id}"
        # access_token_encrypted is NOT NULL in the schema. The actual value
        # is never read — `_resolve_installation_bearer` fetches via backend
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
