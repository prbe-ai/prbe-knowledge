"""Customer provisioning primitives.

Shared between the `scripts/bootstrap_customer.py` CLI and the admin API
routes. Each function is idempotent-where-sensible but refuses silent data
destruction (creating a customer that already exists raises, rather than
rotating the API key behind an operator's back).
"""

from __future__ import annotations

import hashlib
import secrets

from asyncpg.exceptions import UniqueViolationError

from shared.db import raw_conn
from shared.exceptions import PrbeError
from shared.logging import get_logger
from shared.storage import get_store

log = get_logger(__name__)


class CustomerAlreadyExists(PrbeError):
    """Raised when create_customer() is called with an existing customer_id."""


class CustomerNotFound(PrbeError):
    """Raised when rotate_customer_key() targets a non-existent customer."""


def _hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def _generate_api_key() -> str:
    # 32 url-safe bytes → ~43 chars of entropy. Matches bootstrap_customer.py.
    return secrets.token_urlsafe(32)


async def create_customer(customer_id: str, display_name: str) -> str:
    """Insert a fresh customer row and return the plaintext API key.

    Returns the plaintext key exactly once — it is not recoverable afterward.
    Raises CustomerAlreadyExists on duplicate customer_id; callers wanting to
    reset a key should use rotate_customer_key() explicitly.
    """
    api_key = _generate_api_key()
    api_key_hash = _hash_api_key(api_key)
    try:
        async with raw_conn() as conn:
            await conn.execute(
                """
                INSERT INTO customers (customer_id, display_name, api_key_hash)
                VALUES ($1, $2, $3)
                """,
                customer_id,
                display_name,
                api_key_hash,
            )
    except UniqueViolationError as exc:
        raise CustomerAlreadyExists(
            "customer already exists", customer_id=customer_id
        ) from exc
    log.info("provisioning.customer_created", customer=customer_id)
    return api_key


async def rotate_customer_key(customer_id: str) -> str:
    """Issue a new API key for an existing customer. Returns the plaintext key."""
    api_key = _generate_api_key()
    api_key_hash = _hash_api_key(api_key)
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            UPDATE customers SET api_key_hash = $1
            WHERE customer_id = $2
            RETURNING customer_id
            """,
            api_key_hash,
            customer_id,
        )
    if row is None:
        raise CustomerNotFound("customer not found", customer_id=customer_id)
    log.info("provisioning.customer_rotated", customer=customer_id)
    return api_key


async def ensure_bucket_for(customer_id: str) -> str:
    """Create the per-tenant R2/MinIO bucket if it doesn't exist. Returns bucket name."""
    store = get_store()
    bucket = await store.bucket_for(customer_id)
    await store.ensure_bucket(bucket)
    return bucket


async def delete_customer(customer_id: str) -> None:
    """Hard-delete a customer and all their data.

    Every child table FKs to customers with ON DELETE CASCADE, so one
    DELETE nukes the whole tenant. Bucket cleanup is best-effort — a
    failed bucket delete leaves an orphan we can clean up manually, but
    a failed DB delete would leave a much worse state.
    """
    async with raw_conn() as conn:
        result = await conn.execute(
            "DELETE FROM customers WHERE customer_id = $1", customer_id
        )
    if result.rsplit(" ", 1)[-1] == "0":
        raise CustomerNotFound("customer not found", customer_id=customer_id)

    store = get_store()
    bucket = await store.bucket_for(customer_id)
    try:
        await store.delete_bucket_recursive(bucket)
    except Exception as exc:
        log.warning(
            "provisioning.bucket_delete_failed",
            customer=customer_id,
            bucket=bucket,
            error=str(exc),
        )
    log.info("provisioning.customer_deleted", customer=customer_id)
