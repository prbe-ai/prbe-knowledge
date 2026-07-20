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

What it does (via `kb.github_seed.seed_github_installation`):
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

The same core also backs `POST /api/github/connect` (kb.admin_routes), which
the research-os connect flow calls so a claimed installation backfills
immediately instead of via this manual runbook.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from engine.shared.backend_client import github_mint_path
from engine.shared.config import get_settings
from engine.shared.constants import GITHUB_INSTALLATION_SCOPE_PREFIX
from engine.shared.db import close_pool, init_pool
from engine.shared.logging import configure_logging
from kb.github_seed import (
    CustomerNotFoundError,
    GitHubMintNotConfigured,
    seed_github_installation,
)


async def seed(customer_id: str, installation_id: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    try:
        try:
            expires_at = await seed_github_installation(customer_id, installation_id)
        except CustomerNotFoundError:
            sys.stderr.write(
                f"error: customer {customer_id!r} not found — run "
                "`scripts.bootstrap_customer --id <id> --display-name <name>` first\n"
            )
            raise SystemExit(1) from None
        except GitHubMintNotConfigured as exc:
            sys.stderr.write(f"error: {exc}\n")
            raise SystemExit(1) from None
    finally:
        await close_pool()

    scope = f"{GITHUB_INSTALLATION_SCOPE_PREFIX}{installation_id}"
    path = github_mint_path(settings)
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
