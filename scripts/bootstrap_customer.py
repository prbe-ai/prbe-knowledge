"""Provision a new customer.

Creates:
    - `customers` row with hashed API key
    - Per-tenant R2 bucket
    - Five OAuth install URLs (printed for the operator to hand to the customer)

Usage:
    .venv/bin/python -m scripts.bootstrap_customer --id acme --display-name "Acme Corp"

API key is generated, stored hashed in `customers.api_key_hash`, and printed
exactly once on stdout. Lose it and you rotate.
"""

from __future__ import annotations

import argparse
import asyncio

from shared.config import get_settings
from shared.constants import SourceSystem
from shared.db import close_pool, init_pool
from shared.logging import configure_logging, get_logger
from shared.provisioning import (
    CustomerAlreadyExists,
    create_customer,
    ensure_bucket_for,
)

log = get_logger(__name__)


async def bootstrap(
    customer_id: str,
    display_name: str,
    redirect_uri: str,
) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    try:
        api_key = await create_customer(customer_id, display_name)
    except CustomerAlreadyExists:
        print(f"Customer '{customer_id}' already exists. Use the admin API to rotate the key.")
        await close_pool()
        return

    bucket = await ensure_bucket_for(customer_id)

    # Best-effort OAuth install URLs — connectors without registered OAuth skip.
    from services.ingestion.handlers.base import make_default_context
    from services.ingestion.handlers.registry import build_connector, list_registered

    ctx = make_default_context()
    urls: dict[str, str] = {}
    for src in list_registered():
        connector = build_connector(src, ctx)
        try:
            urls[src.value] = connector.oauth_install_url(customer_id, redirect_uri)
        except Exception as exc:
            urls[src.value] = f"<not configured: {exc}>"

    await ctx.http.aclose()
    await close_pool()

    print("=" * 72)
    print(f"Customer provisioned: {customer_id} ({display_name})")
    print(f"  R2 bucket: {bucket}")
    print(f"  API key:   {api_key}   ← store this, it is not recoverable")
    print()
    print("OAuth install URLs:")
    for src in SourceSystem:
        print(f"  {src.value:10s} {urls.get(src.value, '<handler not registered>')}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="stable customer_id slug")
    ap.add_argument("--display-name", required=True)
    ap.add_argument(
        "--redirect-uri",
        default="https://api.prbe.ai/oauth/callback",
        help="OAuth redirect URI used in install URLs",
    )
    args = ap.parse_args()
    asyncio.run(bootstrap(args.id, args.display_name, args.redirect_uri))


if __name__ == "__main__":
    main()
