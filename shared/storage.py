"""R2 (S3-compatible) client wrapper. Works against R2 in prod, MinIO locally.

boto3 is sync; we run calls in a thread executor to stay out of the asyncio path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from shared.config import Settings, get_settings
from shared.exceptions import StorageNotFound, StorageUnavailable


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

    def bucket_for(self, customer_id: str) -> str:
        return self._settings.bucket_for(customer_id)

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
