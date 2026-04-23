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
import hashlib
import secrets

from shared.config import get_settings
from shared.constants import SourceSystem
from shared.db import close_pool, init_pool, raw_conn
from shared.logging import configure_logging, get_logger
from shared.storage import get_store

log = get_logger(__name__)


async def bootstrap(
    customer_id: str,
    display_name: str,
    redirect_uri: str,
) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)

    # 1. API key — 32 url-safe bytes. Store sha256 hash.
    api_key = secrets.token_urlsafe(32)
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, $2, $3)
            ON CONFLICT (customer_id)
            DO UPDATE SET display_name = EXCLUDED.display_name,
                          api_key_hash = EXCLUDED.api_key_hash
            """,
            customer_id,
            display_name,
            api_key_hash,
        )

    # 2. R2 bucket
    store = get_store()
    bucket = store.bucket_for(customer_id)
    await store.ensure_bucket(bucket)

    # 3. OAuth install URLs (best-effort — connectors without registered OAuth skip)
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
