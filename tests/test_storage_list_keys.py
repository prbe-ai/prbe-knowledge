import pytest

from engine.shared.storage import get_store


@pytest.mark.asyncio
async def test_list_keys_returns_only_matching_prefix() -> None:
    store = get_store()
    # Hardcode the bucket name: this is a list_keys behavior test and
    # doesn't need a real customers row. ``store.bucket_for`` now does a
    # DB lookup that would require a pool the test doesn't set up.
    bucket = "prbe-listkeys-test-cust"
    await store.ensure_bucket(bucket)
    await store.put(bucket, "raw/claude_code/listkeys-test-cust/sess-1/0.jsonl", b'{"a": 1}\n')
    await store.put(bucket, "raw/claude_code/listkeys-test-cust/sess-1/1.jsonl", b'{"a": 2}\n')
    await store.put(bucket, "raw/claude_code/listkeys-test-cust/sess-2/0.jsonl", b'{"b": 1}\n')

    keys = await store.list_keys(bucket, "raw/claude_code/listkeys-test-cust/sess-1/")
    assert sorted(keys) == [
        "raw/claude_code/listkeys-test-cust/sess-1/0.jsonl",
        "raw/claude_code/listkeys-test-cust/sess-1/1.jsonl",
    ]

    # Cleanup
    await store.delete_bucket_recursive(bucket)
