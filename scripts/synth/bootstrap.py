"""TenantBootstrap — idempotent customer init + prefix-guarded clean.

`init_tenant` ensures the customer row, R2 bucket, and stub integration_tokens
rows exist. `clean_tenant` is the dangerous one: hard-guarded by customer_id
prefix to refuse production tenants, then DELETE per known table + R2 prefix.

The customers row is NOT deleted — it stays as a "tenant exists" marker so
init can re-bind without race.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from engine.shared.encryption import encrypt_token
from engine.shared.logging import get_logger

if TYPE_CHECKING:
    from engine.shared.storage import ObjectStore
    from scripts.synth.profile import Profile


log = get_logger(__name__)


# Tables whose rows belong to a customer. Order matters for the explicit
# DELETEs (children before parents) — though the FK CASCADE would handle
# it transparently if any single DELETE were skipped. Keeping explicit
# DELETEs for visibility + idempotency.
CUSTOMER_OWNED_TABLES: tuple[str, ...] = (
    "ingestion_queue",
    "chunks",
    "documents",
    "graph_edges",
    "graph_nodes",
    "graph_node_provenance",
    "integration_tokens",
    "acl_snapshots",
    "failed_chunks",
    "ingestion_events",
    "audit_log",
    "customer_source_mapping",
    "usage_events",
    "backfill_state",
)


async def init_tenant(profile: Profile, db, bucket: ObjectStore) -> None:
    """Idempotent tenant bootstrap.

    Creates the customers row, ensures the R2 bucket, and writes stub
    integration_tokens for each source the profile uses. Re-running is safe
    via ON CONFLICT DO NOTHING.
    """
    customer_id = profile.customer_id
    display_name = profile.raw.get("display_name") or f"synth-{customer_id}"
    sources = profile.raw.get("sources") or ["slack", "notion"]

    # customers.api_key_hash is NOT NULL. Synth tenants don't have a real
    # bearer key — stamp a deterministic placeholder so the row inserts.
    # Bearer auth never resolves to this row because no caller can know
    # the pre-hash input. ON CONFLICT DO NOTHING below preserves any
    # real api_key_hash already on an existing tenant.
    placeholder_hash = hashlib.sha256(
        f"synth-stub-no-bearer-{customer_id}".encode()
    ).hexdigest()

    await db.execute(
        """
        INSERT INTO customers (customer_id, display_name, api_key_hash, status)
        VALUES ($1, $2, $3, 'active')
        ON CONFLICT (customer_id) DO NOTHING
        """,
        customer_id,
        display_name,
        placeholder_hash,
    )

    bucket_name = await bucket.bucket_for(customer_id)
    await bucket.ensure_bucket(bucket_name)

    # The worker decrypts integration_tokens.access_token_encrypted with the
    # Fernet key from TOKEN_ENCRYPTION_KEY. Storing a literal placeholder
    # ('synth-stub') breaks decrypt; encrypt with the active key so any caller
    # that has the same key (the worker reading config) can round-trip it.
    # The plaintext is meaningless — synth connectors never make outbound
    # OAuth calls — but the bytes have to be valid Fernet ciphertext.
    encrypted_stub = encrypt_token("synth-stub")

    for source in sources:
        await db.execute(
            """
            INSERT INTO integration_tokens
              (customer_id, source_system, access_token_encrypted, status)
            VALUES ($1, $2, $3, 'active')
            ON CONFLICT (customer_id, source_system)
              WHERE device_id IS NULL
            DO UPDATE SET
              access_token_encrypted = EXCLUDED.access_token_encrypted,
              status = 'active'
            """,
            customer_id,
            source,
            encrypted_stub,
        )

    log.info("tenant_init_complete", customer_id=customer_id, sources=sources)


async def clean_tenant(customer_id: str, db, bucket: ObjectStore) -> None:
    """Prefix-guarded teardown of a synthetic tenant.

    Refuses any customer_id NOT starting with cust-eval- or cust-synth-.
    Transactionally DELETEs from every CUSTOMER_OWNED_TABLES row matching
    customer_id, then list-and-delete R2 keys under raw/.../<customer_id>/synth/.
    The customers row is preserved as a tenant marker.
    """
    if not customer_id.startswith(("cust-eval-", "cust-synth-")):
        raise ValueError(
            f"refuse to clean non-synthetic customer: {customer_id!r}"
        )

    # asyncpg's Pool has .execute() (auto-acquires per call) but NOT
    # .transaction() — transactions are scoped to a single Connection so
    # every statement in them sees the same MVCC snapshot and rolls back
    # together. Acquire a Connection from the Pool, then run the
    # transaction on that one connection.
    async with db.acquire() as conn, conn.transaction():
        for table in CUSTOMER_OWNED_TABLES:
            await conn.execute(
                f"DELETE FROM {table} WHERE customer_id = $1",
                customer_id,
            )

    bucket_name = await bucket.bucket_for(customer_id)
    needle = f"/{customer_id}/synth/"
    keys = await bucket.list_keys(bucket_name, "raw/")
    synth_keys = [k for k in keys if needle in k]
    for key in synth_keys:
        await bucket.delete(bucket_name, key)

    log.info(
        "tenant_clean_complete",
        customer_id=customer_id,
        rows_deleted_per_table=len(CUSTOMER_OWNED_TABLES),
        r2_keys_deleted=len(synth_keys),
    )
