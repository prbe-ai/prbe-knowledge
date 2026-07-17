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
    2. Upserts a `customer_source_mapping` row so the mint path can resolve
       the installation_id from this customer when it mints a token.
    3. Verifies a token mint path is configured: either hosted
       (BACKEND_BASE_URL + INTERNAL_BACKEND_API_KEY, prbe-backend mints) or
       standalone (GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY, local minting via
       engine.shared.github_app).
    4. Upserts an `integration_tokens` row with scope=`installation:<id>`.
    5. Dry-run fetches an installation token through
       `fetch_github_installation_token` (which selects the same hosted vs
       standalone branch the connectors use) to confirm the end-to-end mint
       path is live.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

from engine.shared.backend_client import fetch_github_installation_token, github_mint_path
from engine.shared.config import get_settings
from engine.shared.constants import (
    GITHUB_INSTALLATION_SCOPE_PREFIX,
    IntegrationStatus,
    SourceSystem,
)
from engine.shared.customer_mapping import record_mapping
from engine.shared.db import close_pool, init_pool, raw_conn
from engine.shared.encryption import encrypt_token
from engine.shared.logging import configure_logging, get_logger

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

        path = github_mint_path(settings)
        if path is None:
            sys.stderr.write(
                "error: no GitHub token mint path configured — set either\n"
                "       BACKEND_BASE_URL + INTERNAL_BACKEND_API_KEY (hosted; prbe-backend mints)\n"
                "       or GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY (standalone; local minting)\n"
            )
            raise SystemExit(1)

        scope = f"{GITHUB_INSTALLATION_SCOPE_PREFIX}{installation_id}"
        # access_token_encrypted is NOT NULL in the schema. The actual value
        # is never read — `_resolve_installation_bearer` fetches through the
        # configured mint path whenever scope starts with `installation:` — so
        # we store an opaque encrypted placeholder that makes the intent
        # obvious in a DB dump.
        placeholder = encrypt_token("installation-minted-on-demand")

        # Record the mapping + token row FIRST so the mint path can resolve
        # customer -> installation when we dry-run-fetch the token below
        # (hosted resolves via prbe-backend; standalone reads the
        # integration_tokens.scope written here).
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
            "github_seed_token.dry_run_ok",
            customer=customer_id,
            installation=installation_id,
            expires_at=expires_at.isoformat(),
        )
    finally:
        await close_pool()

    print("=" * 72)
    print(f"GitHub installation seeded for customer {customer_id}")
    print(f"  installation_id: {installation_id}")
    print(f"  scope:           {scope}")
    print(f"  mint path:       {path}")
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
