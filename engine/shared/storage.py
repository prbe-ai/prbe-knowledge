"""R2 (S3-compatible) client wrapper. Works against R2 in prod, MinIO locally.

boto3 is sync; we run calls in a thread executor to stay out of the asyncio path.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from engine.shared.config import Settings, get_settings
from engine.shared.exceptions import StorageNotFound, StorageUnavailable

log = logging.getLogger(__name__)

# Per-pod cache of customer_id -> r2_bucket. The bucket name is immutable
# once a customer is provisioned (renaming would orphan all uploads), so
# there's no TTL — set on first read, kept until pod restart. ~50 bytes/
# entry * a few thousand tenants = trivial memory.
_BUCKET_CACHE: dict[str, str] = {}
_BUCKET_CACHE_LOCK = asyncio.Lock()


async def _load_bucket(customer_id: str, settings: Settings) -> str:
    """Resolve r2_bucket from the customers row. Raises if the row is
    missing or the column is unexpectedly NULL — both should be impossible
    after migration 0075 (NOT NULL on r2_bucket) and the CP-side mirror
    populating r2_bucket on every INSERT.

    The ``settings`` arg is kept on the signature so test stubs / fakes
    that monkey-patched ``_load_bucket`` upstream don't break, but the
    legacy prefix-formula fallback that 0073 and 0074 propped up is
    intentionally gone here — silently writing to the wrong bucket is
    worse than 5xx-ing the request and letting the retry land in the
    correct place once the DB is back. db import is lazy to avoid a
    module-load cycle (shared.db imports shared.config which can be
    imported before storage by tests)."""
    from engine.shared.db import raw_conn  # local import: avoid module-load cycle

    del settings  # no longer consulted; kept for signature compat
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT r2_bucket FROM customers WHERE customer_id = $1",
            customer_id,
        )
    if not row:
        raise StorageUnavailable(
            f"no customers row for customer_id={customer_id!r}; "
            "mirror should have created it before any upload"
        )
    bucket = row["r2_bucket"]
    if not bucket:
        raise StorageUnavailable(
            f"customers.r2_bucket is NULL for customer_id={customer_id!r}; "
            "migration 0075 should have backfilled every row"
        )
    return str(bucket)


def _reset_bucket_cache_for_tests() -> None:
    """Test-only hook so fixtures can swap out the customers DB between cases
    without seeing stale cached bucket names from a prior test's customer_id."""
    _BUCKET_CACHE.clear()


@dataclass(slots=True)
class ObjectLocation:
    bucket: str
    key: str


def _make_client(settings: Settings) -> Any:
    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key.get_secret_value(),
        region_name=settings.r2_region,
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


class ObjectStore:
    """Thin async wrapper so callers don't block the event loop on boto3 calls."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = _make_client(self._settings)

    # ---- bucket ops ---------------------------------------------------------

    async def ensure_bucket(self, bucket: str) -> None:
        def _ensure() -> None:
            try:
                self._client.head_bucket(Bucket=bucket)
                return
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in {"404", "NoSuchBucket", "NotFound"}:
                    raise StorageUnavailable(f"head_bucket failed: {exc}") from exc
            try:
                self._client.create_bucket(Bucket=bucket)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                    return
                raise StorageUnavailable(f"create_bucket failed: {exc}") from exc

        await asyncio.to_thread(_ensure)

    async def bucket_for(self, customer_id: str) -> str:
        """Per-tenant R2 bucket name.

        Reads ``customers.r2_bucket`` (cached, immutable once a tenant
        is provisioned). Migration 0075 made the column NOT NULL and
        the CP-side mirror populates it on every customer INSERT, so
        a NULL/missing value here is a real bug rather than a normal
        rollout state — ``_load_bucket`` raises ``StorageUnavailable``
        rather than guessing a bucket name. Caller surfaces that as a
        5xx; the upload retries.
        """
        cached = _BUCKET_CACHE.get(customer_id)
        if cached is not None:
            return cached
        async with _BUCKET_CACHE_LOCK:
            # Re-check under the lock — another waiter may have populated it.
            cached = _BUCKET_CACHE.get(customer_id)
            if cached is not None:
                return cached
            bucket = await _load_bucket(customer_id, self._settings)
            _BUCKET_CACHE[customer_id] = bucket
            return bucket

    # ---- object ops ---------------------------------------------------------

    async def put(
        self,
        bucket: str,
        key: str,
        body: bytes,
        content_type: str = "application/json",
    ) -> ObjectLocation:
        def _put() -> None:
            try:
                self._client.put_object(
                    Bucket=bucket, Key=key, Body=body, ContentType=content_type
                )
            except (ClientError, BotoCoreError) as exc:
                raise StorageUnavailable(f"put_object failed: {exc}") from exc

        await asyncio.to_thread(_put)
        return ObjectLocation(bucket=bucket, key=key)

    async def get(self, bucket: str, key: str) -> bytes:
        def _get() -> bytes:
            try:
                resp = self._client.get_object(Bucket=bucket, Key=key)
                return resp["Body"].read()
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in {"NoSuchKey", "404"}:
                    raise StorageNotFound(f"{bucket}/{key}") from exc
                raise StorageUnavailable(f"get_object failed: {exc}") from exc

        return await asyncio.to_thread(_get)

    async def exists(self, bucket: str, key: str) -> bool:
        def _head() -> bool:
            try:
                self._client.head_object(Bucket=bucket, Key=key)
                return True
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in {"404", "NoSuchKey", "NotFound"}:
                    return False
                raise StorageUnavailable(f"head_object failed: {exc}") from exc

        return await asyncio.to_thread(_head)

    async def delete(self, bucket: str, key: str) -> None:
        """Delete one object. Missing buckets/keys are treated as already gone."""
        def _delete() -> None:
            try:
                self._client.delete_object(Bucket=bucket, Key=key)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in {"NoSuchBucket", "NoSuchKey", "404", "NotFound"}:
                    return
                raise StorageUnavailable(f"delete_object failed: {exc}") from exc
            except BotoCoreError as exc:
                raise StorageUnavailable(f"delete_object failed: {exc}") from exc

        await asyncio.to_thread(_delete)

    async def delete_bucket_recursive(self, bucket: str) -> None:
        """Delete every object in a bucket, then delete the bucket.

        Missing buckets are swallowed silently. Used by customer delete.
        """
        def _delete() -> None:
            try:
                paginator = self._client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=bucket):
                    contents = page.get("Contents") or []
                    if not contents:
                        continue
                    self._client.delete_objects(
                        Bucket=bucket,
                        Delete={"Objects": [{"Key": c["Key"]} for c in contents]},
                    )
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in {"NoSuchBucket", "404", "NotFound"}:
                    raise StorageUnavailable(f"list/delete_objects failed: {exc}") from exc
                return
            try:
                self._client.delete_bucket(Bucket=bucket)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in {"NoSuchBucket", "404", "NotFound"}:
                    raise StorageUnavailable(f"delete_bucket failed: {exc}") from exc

        await asyncio.to_thread(_delete)

    async def list_keys(self, bucket: str, prefix: str) -> list[str]:
        """Return all keys in `bucket` starting with `prefix`. Paginated.

        Used by Claude Code's fetch_supplementary to assemble per-session batches.
        """
        def _list() -> list[str]:
            keys: list[str] = []
            paginator = self._client.get_paginator("list_objects_v2")
            try:
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    for obj in page.get("Contents", []) or []:
                        keys.append(obj["Key"])
            except (BotoCoreError, ClientError) as exc:
                code = exc.response.get("Error", {}).get("Code", "") if isinstance(exc, ClientError) else ""
                if code in {"NoSuchBucket", "404", "NotFound"}:
                    return []
                raise StorageUnavailable(f"list_objects_v2 failed: {exc}") from exc
            return keys

        return await asyncio.to_thread(_list)


_store: ObjectStore | None = None


def get_store() -> ObjectStore:
    global _store
    if _store is None:
        _store = ObjectStore()
    return _store


def reset_store() -> None:
    """Tests call this to force re-init with patched settings."""
    global _store
    _store = None
