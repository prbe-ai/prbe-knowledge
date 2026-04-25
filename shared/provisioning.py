"""Customer provisioning primitives.

Shared between the `scripts/bootstrap_customer.py` CLI, the legacy admin
API routes, and the dashboard service. Each function is idempotent-
where-sensible but refuses silent data destruction (creating a customer
that already exists raises, rather than rotating the API key behind an
operator's back).

Soft vs hard delete:
  * `soft_delete_customer` flips status to 'deleted' and is what the
    dashboard's "Delete team" surface calls. It is reversible during
    the recovery window (offline reaper handles eventual hard delete).
  * `delete_customer` is the hard, irreversible cascade — reserved for
    the reaper or for ops break-glass. The dashboard never calls it.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

from asyncpg.exceptions import ForeignKeyViolationError, UniqueViolationError

from shared.db import raw_conn
from shared.exceptions import PrbeError
from shared.logging import get_logger
from shared.storage import get_store

log = get_logger(__name__)


class CustomerAlreadyExists(PrbeError):
    """Raised when create_customer() is called with an existing customer_id."""


class CustomerNotFound(PrbeError):
    """Raised when rotate_customer_key() targets a non-existent customer."""


class OrganizationAlreadyClaimed(PrbeError):
    """Raised when an organization already has a customer linked to it."""


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
    bucket = store.bucket_for(customer_id)
    await store.ensure_bucket(bucket)
    return bucket


async def delete_customer(customer_id: str) -> None:
    """Hard-delete a customer and all their data.

    Every child table FKs to customers with ON DELETE CASCADE, so one
    DELETE nukes the whole tenant. Bucket cleanup is best-effort — a
    failed bucket delete leaves an orphan we can clean up manually, but
    a failed DB delete would leave a much worse state.

    The dashboard never calls this directly. It is reserved for the
    offline reaper that purges customers that have been soft-deleted
    longer than the recovery window.
    """
    async with raw_conn() as conn:
        result = await conn.execute(
            "DELETE FROM customers WHERE customer_id = $1", customer_id
        )
    if result.rsplit(" ", 1)[-1] == "0":
        raise CustomerNotFound("customer not found", customer_id=customer_id)

    store = get_store()
    bucket = store.bucket_for(customer_id)
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


async def soft_delete_customer(customer_id: str) -> None:
    """Mark a customer 'deleted' without removing data.

    All read paths (retrieval, ingestion, dashboard) filter on
    `status = 'active'`, so a soft-deleted tenant becomes invisible
    immediately while data persists for the recovery window. The
    offline reaper later calls `delete_customer` for hard purge.
    """
    async with raw_conn() as conn:
        result = await conn.execute(
            """
            UPDATE customers
            SET status = 'deleted'
            WHERE customer_id = $1 AND status != 'deleted'
            """,
            customer_id,
        )
    affected = result.rsplit(" ", 1)[-1]
    if affected == "0":
        # Either the customer doesn't exist or it's already soft-deleted.
        # Distinguish so the caller can return the right HTTP status.
        async with raw_conn() as conn:
            row = await conn.fetchrow(
                "SELECT status FROM customers WHERE customer_id = $1", customer_id
            )
        if row is None:
            raise CustomerNotFound("customer not found", customer_id=customer_id)
        # Already soft-deleted — idempotent.
        return
    log.info("provisioning.customer_soft_deleted", customer=customer_id)


async def create_customer_for_organization(
    customer_id: str,
    organization_id: str,
    display_name: str,
) -> str:
    """Create a customer linked to a Better Auth organization.

    Used by the dashboard bridge endpoint after authClient.organization.create
    succeeds. The customer_id is caller-supplied (typically derived from the
    organization slug or id); the FK to neon_auth.organization is enforced
    at the DB level via ON DELETE RESTRICT, so the link cannot dangle.

    Returns the plaintext API key once. Raises CustomerAlreadyExists for
    a duplicate customer_id, OrganizationAlreadyClaimed if the org already
    has a customer, or ValueError if the organization_id doesn't exist.
    """
    api_key = _generate_api_key()
    api_key_hash = _hash_api_key(api_key)
    try:
        async with raw_conn() as conn:
            await conn.execute(
                """
                INSERT INTO customers
                    (customer_id, display_name, api_key_hash, organization_id)
                VALUES ($1, $2, $3, $4::uuid)
                """,
                customer_id,
                display_name,
                api_key_hash,
                organization_id,
            )
    except UniqueViolationError as exc:
        # Distinguish customer_id collision from organization_id collision.
        # The partial unique index `customers_organization_id_unique` fires
        # on org collision; the table PK fires on customer_id collision.
        if "customers_organization_id_unique" in str(exc):
            raise OrganizationAlreadyClaimed(
                "organization already claimed by another customer",
                organization_id=organization_id,
            ) from exc
        raise CustomerAlreadyExists(
            "customer already exists", customer_id=customer_id
        ) from exc
    except ForeignKeyViolationError as exc:
        raise ValueError(
            f"organization {organization_id} not found in neon_auth.organization"
        ) from exc
    log.info(
        "provisioning.customer_linked_to_org",
        customer=customer_id,
        organization=organization_id,
    )
    return api_key


async def get_customer_by_organization(organization_id: str) -> dict[str, Any] | None:
    """Look up the active customer for an organization, if any.

    Returns None when the org has no customer or the customer is soft-deleted.
    Used by the dashboard backend to resolve an authenticated user's session
    (org_id from JWT) to the customer_id that scopes their data.
    """
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT customer_id, display_name, status, organization_id
            FROM customers
            WHERE organization_id = $1::uuid AND status = 'active'
            """,
            organization_id,
        )
    if row is None:
        return None
    return dict(row)
