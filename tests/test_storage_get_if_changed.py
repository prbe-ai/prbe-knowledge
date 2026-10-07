"""ObjectStore.get_if_changed against a real S3 API (MinIO here, R2 in prod).

The trajectory cache's freshness rests on it: the object's body when the ETag
sent is not current, nothing (a 304) when it is, StorageNotFound when the
object is gone -- whatever ETag was sent.
"""

from __future__ import annotations

import pytest

from engine.shared.storage import StorageNotFound, get_store


@pytest.mark.asyncio
async def test_conditional_get_follows_the_object() -> None:
    store = get_store()
    bucket = "prbe-getifchanged-test"
    key = "raw/claude_code/cust/sess/trajectory.json"
    await store.ensure_bucket(bucket)
    try:
        await store.put(bucket, key, b'{"v": 1}')
        first = await store.get_if_changed(bucket, key, None)
        assert first.body == b'{"v": 1}' and first.etag

        same = await store.get_if_changed(bucket, key, first.etag)
        assert same.body is None and same.etag == first.etag

        await store.put(bucket, key, b'{"v": 2}')
        changed = await store.get_if_changed(bucket, key, first.etag)
        assert changed.body == b'{"v": 2}' and changed.etag != first.etag

        stale_etag = await store.get_if_changed(bucket, key, '"not-the-etag"')
        assert stale_etag.body == b'{"v": 2}'

        await store.delete(bucket, key)
        with pytest.raises(StorageNotFound):
            await store.get_if_changed(bucket, key, changed.etag)
    finally:
        await store.delete_bucket_recursive(bucket)
