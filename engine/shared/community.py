"""Single-tenant community-mode bootstrap.

When ``DEFAULT_CUSTOMER_ID`` is set (standalone self-host), ensure a
``customers`` row exists so ingestion/retrieval work out of the box. The row's
``api_key_hash`` is ``sha256(KNOWLEDGE_API_TOKEN)``, so the existing bearer-auth
path in ``services/retrieval/auth.py`` resolves that static token to
``DEFAULT_CUSTOMER_ID`` with no special-casing.

No-op when ``DEFAULT_CUSTOMER_ID`` is unset (hosted / multi-tenant), so the
control-plane provisioning path is unaffected.
"""
from __future__ import annotations

import hashlib

from engine.shared.config import get_settings
from engine.shared.db import raw_conn
from engine.shared.logging import get_logger

log = get_logger(__name__)


async def ensure_default_customer() -> None:
    """Idempotently seed the single community-mode tenant. No-op when unset."""
    settings = get_settings()
    customer_id = settings.default_customer_id
    if not customer_id:
        return  # hosted / multi-tenant: provisioning owns the customers table

    token = settings.knowledge_api_token.get_secret_value()
    # Hash the static token so the existing customers.api_key_hash bearer
    # lookup resolves it to this tenant. Empty token => an unmatchable hash
    # (the user must set KNOWLEDGE_API_TOKEN before /query auth succeeds);
    # ingestion still works since the webhook path scopes to DEFAULT_CUSTOMER_ID.
    api_key_hash = hashlib.sha256(token.encode()).hexdigest() if token else ""
    r2_bucket = settings.bucket_for(customer_id)

    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash, r2_bucket)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (customer_id) DO UPDATE
              SET api_key_hash = EXCLUDED.api_key_hash
              WHERE customers.api_key_hash IS DISTINCT FROM EXCLUDED.api_key_hash
            """,
            customer_id,
            "Community Self-Host",
            api_key_hash,
            r2_bucket,
        )
    log.info("community.default_customer_ensured", customer_id=customer_id)


__all__ = ["ensure_default_customer"]
