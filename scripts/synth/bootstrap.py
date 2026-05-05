"""TenantBootstrap — idempotent customer init + prefix-guarded clean.

`init_tenant` ensures the customer row, R2 bucket, and stub integration_tokens
rows exist. `clean_tenant` is the dangerous one: hard-guarded by customer_id
prefix to refuse production tenants, then DELETE per known table + R2 prefix.

The customers row is NOT deleted — it stays as a "tenant exists" marker so
init can re-bind without race.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shared.logging import get_logger

if TYPE_CHECKING:
    from scripts.synth.profile import Profile
    from shared.storage import ObjectStore


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

    await db.execute(
        """
        INSERT INTO customers (customer_id, display_name, status)
        VALUES ($1, $2, 'active')
        ON CONFLICT (customer_id) DO NOTHING
        """,
        customer_id,
        display_name,
    )

    bucket_name = bucket.bucket_for(customer_id)
    await bucket.ensure_bucket(bucket_name)

    for source in sources:
        await db.execute(
            """
            INSERT INTO integration_tokens
              (customer_id, source_system, access_token_encrypted, status)
            VALUES ($1, $2, 'synth-stub', 'active')
            ON CONFLICT (customer_id, source_system)
              WHERE device_id IS NULL
            DO NOTHING
            """,
            customer_id,
            source,
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

    async with db.transaction():
        for table in CUSTOMER_OWNED_TABLES:
            await db.execute(
                f"DELETE FROM {table} WHERE customer_id = $1",
                customer_id,
            )

    bucket_name = bucket.bucket_for(customer_id)
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
